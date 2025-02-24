# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
from pathlib import Path
import yaml
from packaging import version
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
import torchvision
import wandb
import accelerate
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.training_utils import EMAModel
from einops import rearrange
# from diffusers import AutoencoderKLTemporalDecoder
from src.diffusers import AutoencoderKLTemporalDecoder
from src.diffusers.optimization import get_scheduler

# from openvidsr import RealVSRCSVVideoDataset  # Import your custom dataset
from dataset import DatasetFromLMDB, get_transforms_video  # Add this import

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a temporal autoencoder model.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the RealBasicVSR yaml config file",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="temporal-ae-finetuned",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.000001,
        help="Weight for KL divergence loss",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Interval between frames",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution of input videos",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default="/data/42-julia-hpc-rz-cv/sig95vg/OpenVid-1M/dataset_index.lmdb",
        help="Path to the LMDB database",
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="/data/42-julia-hpc-rz-cv/sig95vg/OpenVid-1M/keys.pkl",
        help="Path to the pickle file containing keys",
    )
    parser.add_argument(
        "--validation_videos",
        nargs="+",
        type=str,
        default=None,
        help="Optional list of validation video paths for image generation during training.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
        help="Where to report training metrics.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="Path to config file for model architecture.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether to enable xformers memory efficient attention.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    
    args = parser.parse_args()
    return args

class VAELoss(torch.nn.Module):
    def __init__(self, kl_weight=0.000001):
        super().__init__()
        self.kl_weight = kl_weight
        
    def forward(self, reconstructed, pixel_values, posterior):
        """
        Args:
            reconstructed: reconstructed video [B, C, T, H, W]
            pixel_values: original video [B, C, T, H, W]
            posterior: VAE posterior distribution
        """
        # Reconstruction loss (L2/MSE)
        recon_loss = F.mse_loss(reconstructed, pixel_values, reduction="mean")
        
        # KL divergence loss
        kl_loss = posterior.kl().mean()
        
        # Total loss
        loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
@torch.no_grad()
def log_validation(model, args, validation_videos, accelerator, global_step, epoch):
    """Validate and log predictions from the temporal autoencoder."""
    if accelerator.is_main_process:
        logger.info("Running validation...")
        
    model.eval()
    with torch.no_grad():
        # Load and process validation videos
        val_videos = []
        for video_path in validation_videos:
            vframes, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
            vframes = vframes.permute(0, 3, 1, 2)
            
            # Select only 16 evenly spaced frames
            total_frames = vframes.shape[0]
            if total_frames > 16:
                indices = torch.linspace(0, total_frames-1, 16).long()
                vframes = vframes[indices]
            elif total_frames < 16:
                # If video is too short, loop the frames
                indices = torch.arange(16) % total_frames
                vframes = vframes[indices]
            
            # # Resize frames first to save memory
            # if vframes.shape[1] > args.resolution or vframes.shape[2] > args.resolution:
            #     resize_transform = torchvision.transforms.Resize(
            #         (args.resolution, args.resolution),
            #         antialias=True
            #     )
            #     vframes = resize_transform(vframes.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
            # Apply same transforms as training
            transform = get_transforms_video(resolution=args.resolution)
            video = transform(vframes)
            
            # video = video.unsqueeze(0)  # [1, T, C, H, W]
            
            val_videos.append(video)
        
        val_videos = torch.stack(val_videos)
        # from B T C H W to (B*T) C H W
        val_videos = rearrange(val_videos, 'b t c h w -> (b t) c h w')
        # Ensure the stacked tensor is on the correct device
        val_videos = val_videos.to(accelerator.device)
        
        # Generate reconstructions
        with accelerator.autocast():
            outputs = model(val_videos)
            reconstructed = outputs.sample
        
        # Log original and reconstructed videos
        if accelerator.is_main_process:
            # Save a grid of frames from original and reconstructed videos
            for i, (orig, recon) in enumerate(zip(val_videos, reconstructed)):
                # Select frames to visualize (first, middle, last)
                frames_to_show = [0, len(orig)//2, -1]
                
                # Create grid
                grid_orig = torchvision.utils.make_grid(
                    [orig[j] for j in frames_to_show], 
                    nrow=len(frames_to_show)
                )
                grid_recon = torchvision.utils.make_grid(
                    [recon[j] for j in frames_to_show], 
                    nrow=len(frames_to_show)
                )
                
                # Combine original and reconstruction
                grid = torch.cat([grid_orig, grid_recon], dim=1)
                
                # Save image
                save_path = os.path.join(
                    args.output_dir, 
                    f"validation_{global_step}_{i}.png"
                )
                torchvision.utils.save_image(grid, save_path)
                
                # Log to wandb if enabled
                if accelerator.is_main_process and args.report_to == "wandb":
                    wandb.log({
                        f"validation_{i}": wandb.Image(save_path),
                        "epoch": epoch,
                    })
    
    model.train()

def main():
    args = parse_args()
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # Initialize accelerator with proper device mapping
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    # Set device for current process
    local_rank = accelerator.local_process_index
    torch.cuda.set_device(local_rank)  # Set the GPU device for this process
    
    # Initialize wandb only on the main process
    if args.report_to == "wandb" and accelerator.is_main_process:
        wandb.init(
            project="temporal-ae",
            name=args.output_dir,
            config=args,
        )
    
    # Load config
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if args.seed is not None:
        set_seed(args.seed)

    # Load model with correct architecture
    if args.model_config_name_or_path is None and args.pretrained_model_name_or_path is None:
        model = AutoencoderKLTemporalDecoder(
            in_channels=3,
            out_channels=3,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"
            ],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            latent_channels=4,
            sample_size=args.resolution,  # Use resolution from args
            scaling_factor=0.18215,
            force_upcast=True,
        )
    elif args.pretrained_model_name_or_path is not None:
        # /data/42-julia-hpc-rz-cv/sig95vg/checkpoints/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/vae
        model = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", variant="fp16")
    else:
        config = AutoencoderKLTemporalDecoder.load_config(args.model_config_name_or_path)
        model = AutoencoderKLTemporalDecoder.from_config(config)

    # Enable flash attention if asked
    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    # Create dataset using config values
    train_dataset = DatasetFromLMDB(
        lmdb_path=cfg['lmdb_path'],
        pkl_path=cfg['pkl_path'],
        num_frames=cfg['num_frames'],
        frame_interval=cfg['frame_interval'][0],  # Take first value from list
        transform=get_transforms_video(
            resolution=cfg['gt_size'][0],  # Use height from gt_size
        ),
    )
    
    # Create dataloader with DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
    ) if accelerator.num_processes > 1 else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Create loss function
    loss_fn = VAELoss(kl_weight=args.kl_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * 0.05),  # 5% warmup
        num_training_steps=max_train_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Add EMA model initialization after model creation
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            model_cls=AutoencoderKLTemporalDecoder,
            model_config=model.module.config
        )

    # Modify the save/load hooks for EMA
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "temporal_ae_ema"))
                
                model = models[0]
                model.save_pretrained(os.path.join(output_dir, "temporal_ae"))
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "temporal_ae_ema"), AutoencoderKLTemporalDecoder)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            
            model = models.pop()
            load_model = AutoencoderKLTemporalDecoder.from_pretrained(input_dir, subfolder="temporal_ae")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Training loop
    global_step = 0
    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                # Get video frames from the dataset
                video = batch["video"]  # Shape: [B, C, T, H, W]
                
                # Reshape video for VAE input
                B, C, T, H, W = video.shape
                video = video.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                video_flat = video.reshape(B * T, C, H, W)  # [B*T, C, H, W]
                
                # First encode to get posterior
                posterior = model.module.encode(video_flat).latent_dist
                
                # Get latent sample
                if model.training:
                    latents = posterior.sample()
                else:
                    latents = posterior.mode()
                
                # Decode the latents
                decoder_output = model.module.decode(latents, num_frames=T)
                reconstructed = decoder_output.sample
                
                # Reshape back to video format for loss calculation
                reconstructed = reconstructed.reshape(B, T, C, H, W)  # [B, T, C, H, W]
                reconstructed = reconstructed.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                
                # Calculate losses directly using reconstructed output
                loss_dict = loss_fn(reconstructed, video.permute(0, 2, 1, 3, 4), posterior)
                loss = loss_dict["loss"]
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                global_step += 1
                progress_bar.update(1)
            
            # Log metrics
            if global_step % 100 == 0:
                logs = {
                    "loss": loss_dict["loss"].detach().item(),
                    "recon_loss": loss_dict["recon_loss"].detach().item(),
                    "kl_loss": loss_dict["kl_loss"].detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # Log to wandb
                if args.report_to == "wandb" and accelerator.is_main_process:
                    wandb.log(logs, step=global_step)
            
            # Modify validation to use EMA model when available
            if args.validation_videos and global_step % args.validation_steps == 0:
                if args.use_ema:
                    # Store the VAE parameters temporarily and load the EMA parameters to perform inference
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                
                log_validation(
                    model=accelerator.unwrap_model(model),
                    args=args,
                    validation_videos=args.validation_videos,
                    accelerator=accelerator,
                    global_step=global_step,
                    epoch=epoch,
                )
                
                if args.use_ema:
                    # Switch back to the original VAE parameters
                    ema_model.restore(model.parameters())
            
            # Save checkpoint
            if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # Save model checkpoint
                    pipeline = accelerator.unwrap_model(model)
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    pipeline.save_pretrained(checkpoint_dir)
                    
                    # Save optimizer and scheduler
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                    }, os.path.join(checkpoint_dir, "optimizer.pt"))
                    
                    # Log to wandb
                    if args.report_to == "wandb":
                        artifact = wandb.Artifact(
                            f"model-{wandb.run.id}", type="model",
                            description=f"Model checkpoint at step {global_step}"
                        )
                        artifact.add_dir(checkpoint_dir)
                        wandb.log_artifact(artifact)
                        
                    logger.info(f"Saved checkpoint at step {global_step}")

    # At the end of training, save the final EMA model
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained(os.path.join(args.output_dir, "temporal_ae_final"))

    # Finish wandb run
    if args.report_to == "wandb" and accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main() 