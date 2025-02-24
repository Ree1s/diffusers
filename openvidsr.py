import os
import csv
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from copy import deepcopy
import cv2
# mmcv-based transforms from basicsr
from openvid.datasets.mmcv_transforms import (
    RandomBlur, RandomResize, RandomNoise,
    RandomJPEGCompression, RandomVideoCompression,
    UnsharpMasking, Clip, RescaleToZeroOne
)
from basicsr.data.transforms import augment, single_random_crop_video
from basicsr.utils import img2tensor
from omegaconf import OmegaConf
import lmdb
import pickle

class RealVSRCSVVideoDataset(Dataset):
    """
    A CSV-based video dataset that mimics RealVSRRecurrentDataset for Openvid-1M.
    Reads (video_path, caption) from CSV, loads MP4 with torchvision,
    applies random sampling, random crop, and two-stage degrade pipeline
    from the final config.

    The dataset returns a dict:
      {
        'lqs': (T, C, H, W),
        'gts': (T, C, H, W),
        'text': <caption>,
        'video_path': <str>
      }
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (dict): The final config dictionary containing:
                - num_frame, gt_size, interval_list, random_reverse, use_hflip, use_rot, ...
                - degradation_1, degradation_2
                - transforms
            csv_path (str): CSV with columns [video_filename, caption]
            root (str): Directory containing the .mp4 files
        """
        super().__init__()
        self.cfg = OmegaConf.load(cfg)
        # ✅ Load keys from keys.pkl (FAST, avoids LMDB iteration)
        with open(self.cfg.pkl_path, 'rb') as f:
            self.keys = pickle.load(f)

        print(f"✅ Loaded {len(self.keys)} keys from {self.cfg.pkl_path}")

        self.env = None  # LMDB environment is initialized in `_init_lmdb()`
        # self.csv_path = self.cfg.csv_path
        # self.root = self.cfg.root
        # Read keys from LMDB
        # self.keys = []
        # env = lmdb.open(self.cfg.lmdb_path, readonly=True, lock=False, readahead=False)
        # with env.begin() as txn:
        #     cursor = txn.cursor()
        #     for key, _ in cursor.iternext():
        #         self.keys.append(key.decode())  # Decode LMDB keys into strings

        # self.env = None  # Do not open LMDB in __init__()
        # read dataset-level configs
        self.num_frames = self.cfg['num_frames']          # e.g. 5
        self.gt_size = list(self.cfg['gt_size'])              # e.g. 512
        self.interval_list = self.cfg['frame_interval']  # e.g. [1]
        self.random_reverse = self.cfg['random_reverse']  # false
        self.use_hflip = self.cfg['use_hflip']          # true
        self.use_rot = self.cfg['use_rot']              # false
        self.flip_sequence = self.cfg.get('flip_sequence', False)  # false

        # read CSV => self.samples
        # samples = []
        # with open(self.csv_path, "r") as f:
        #     reader = csv.reader(f)
        #     rows = list(reader)
        # for row in rows[1:]:
        #     vid_name = row[0]
        #     caption = row[1] if len(row) > 1 else ""
        #     fullpath = os.path.join(self.root, vid_name)
        #     if os.path.exists(fullpath):
        #         samples.append([fullpath, caption])
        # self.samples = samples

        # parse degrade_1
        d1 = self.cfg['degradation_1']
        self.random_blur_1 = RandomBlur(**d1.get('random_blur', {}))
        self.random_resize_1 = RandomResize(**d1.get('random_resize', {}))
        self.random_noise_1 = RandomNoise(**d1.get('random_noise', {}))
        self.random_jpeg_1 = RandomJPEGCompression(**d1.get('random_jpeg', {}))
        self.random_mpeg_1 = RandomVideoCompression(**d1.get('random_mpeg', {}))

        # parse degrade_2
        d2 = self.cfg['degradation_2']
        self.random_blur_2 = RandomBlur(**d2.get('random_blur', {}))
        self.random_resize_2 = RandomResize(**d2.get('random_resize', {}))
        self.random_noise_2 = RandomNoise(**d2.get('random_noise', {}))
        self.random_jpeg_2 = RandomJPEGCompression(**d2.get('random_jpeg', {}))
        self.random_mpeg_2 = RandomVideoCompression(**d2.get('random_mpeg', {}))

        self.resize_final = RandomResize(**d2.get('resize_final', {}))
        self.blur_final = RandomBlur(**d2.get('blur_final', {}))

        # parse transforms
        tcfg = self.cfg['transforms']
        self.usm = UnsharpMasking(**tcfg.get('usm', {}))
        self.clip = Clip(**tcfg.get('clip', {}))
        self.rescale = RescaleToZeroOne(**tcfg.get('rescale', {}))
    def _init_lmdb(self):
        """Ensures that each worker initializes its own LMDB environment."""
        if self.env is None:
            self.env = lmdb.open(self.cfg.lmdb_path, readonly=True, lock=False, readahead=True, max_readers=32)
        # if self.env is None:
        #     self.env = lmdb.open(self.cfg.lmdb_path, readonly=True, lock=False, readahead=False)
    def __len__(self):
        # return len(self.samples)
        return len(self.keys)
    # def __getitem__(self, index):
    #     # retry logic if video is broken
    #     for _ in range(5):
    #         try:
    #             return self._getitem_core(index)
    #         except Exception as e:
    #             print(f'[WARN] index={index}, error={e}, pick new index...')
    #             index = random.randint(0, len(self.samples)-1)
    #     raise RuntimeError("Too many repeated failures in dataset loading.")

    def __getitem__(self, index):
        # index = 0
        # path, caption = self.samples[index]
        # with self.env.begin() as txn:
        #   data = txn.get(str(index).encode())
        #    if data is None:
        #         raise IndexError(f"Index {index} not found in LMDB database.")
        self._init_lmdb()  # Ensure LMDB is opened per worker
        video_id = self.keys[index]

        with self.env.begin(buffers=True) as txn:
            data = txn.get(video_id.encode())
            if data is None:
                raise FileNotFoundError(f"❌ Missing LMDB Key: {video_id}")

        video_path, caption = pickle.loads(data)
        # 1) load entire video => (T, C, H, W)
        vframes, _, info = torchvision.io.read_video(
            filename=video_path, pts_unit="sec", output_format="TCHW"
        )
        total_frames = vframes.shape[0]
        # if total_frames < self.num_frames:
        #     print(f"Video {path} has only {total_frames} frames; interpolating to {self.num_frames}.")
        #     vframes = self.temporal_interpolate(vframes, self.num_frames)
        #     total_frames = self.num_frames

        # 2) pick an interval from interval_list => sample contiguous frames
        interval = random.choice(self.interval_list)  # e.g. [1] => just 1
        max_start = total_frames - self.num_frames * interval
        start_idx = random.randint(0, max_start) if max_start>0 else 0
        end_idx = start_idx + self.num_frames * interval
        frame_inds = np.arange(start_idx, end_idx, interval)

        # 3) slice out frames
        clip_np = vframes[frame_inds]  # (T, C, H, W)

        # 4) optionally random reverse (cfg says false, so we skip)
        if self.random_reverse and random.random() < 0.5:
            clip_np = clip_np.flip(dims=[0])

        # 5) TCHW => THWC for mmcv degrade
        clip_np = clip_np.permute(0,2,3,1).numpy()  # => (T,H,W,C)

        # 6) random crop GT
        frame_list = [clip_np[i] for i in range(clip_np.shape[0])]
        frame_list = single_random_crop_video(frame_list, self.gt_size, video_path)

        # 7) augment: horizontal flips or rotation if configured
        #    realvsr code calls `augment(imgs, use_hflip, use_rot)`
        #    flip_sequence is false => we do not flip entire sequence in time dimension
        frame_list = augment(frame_list, self.use_hflip, self.use_rot)

        # 8) gts + lqs
        img_gts = frame_list
        img_lqs = deepcopy(img_gts)
        out_dict = {'gts': img_gts, 'lqs': img_lqs}

        # 9) USM on gts
        out_dict = self.usm.transform(out_dict)

        # 10) First degrade
        out_dict = self.random_blur_1(out_dict)
        out_dict = self.random_resize_1(out_dict)
        out_dict = self.random_noise_1(out_dict)
        out_dict = self.random_jpeg_1(out_dict)
        out_dict = self.random_mpeg_1(out_dict)

        # 11) Second degrade
        out_dict = self.random_blur_2(out_dict)
        out_dict = self.random_resize_2(out_dict)
        out_dict = self.random_noise_2(out_dict)
        out_dict = self.random_jpeg_2(out_dict)
        out_dict = self.random_mpeg_2(out_dict)

        # 12) Final steps: resize + blur
        out_dict = self.resize_final(out_dict)
        out_dict = self.blur_final(out_dict)

        # 13) clip + rescale
        out_dict = self.clip(out_dict)
        out_dict = self.rescale.transform(out_dict)

        # 14) convert to TCHW torch tensor
        out_dict['gts'] = img2tensor(out_dict['gts'], bgr2rgb=False)  # => (T,C,H,W)
        out_dict['lqs'] = img2tensor(out_dict['lqs'], bgr2rgb=False)  # => (T,C,H,W)
        out_dict['gts'] = torch.stack(out_dict['gts'])
        out_dict['lqs'] = torch.stack(out_dict['lqs'])
        # Normalize images from [0, 1] to [-1, 1]
        mean = torch.tensor([0.5, 0.5, 0.5], device=out_dict['gts'].device).view(1, -1, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=out_dict['gts'].device).view(1, -1, 1, 1)
        out_dict['gts'] = (out_dict['gts'] - mean) / std
        out_dict['lqs'] = (out_dict['lqs'] - mean) / std
        return {
            'lqs': out_dict['lqs'],
            'gts': out_dict['gts'],
            'text': caption,
            'video_path': video_path
        }
    def temporal_interpolate(self, video: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        Interpolate the video tensor along the temporal dimension to have target_frames frames.
        video: Tensor of shape (T, C, H, W)
        Returns a Tensor of shape (target_frames, C, H, W)
        """
        video = video.float()  
        T, C, H, W = video.shape
        # Reshape so that the temporal dimension becomes the "length" dimension for interpolation.
        # First, permute to (C, H, W, T) then reshape to (1, C*H*W, T)
        video_reshaped = video.permute(1, 2, 3, 0).reshape(1, C * H * W, T)
        # Use linear interpolation (treating the sequence as 1D data) to get target_frames.
        video_interp = torch.nn.functional.interpolate(
            video_reshaped, size=target_frames, mode='linear', align_corners=False
        )
        # Reshape back to (target_frames, C, H, W)
        video_interp = video_interp.reshape(C, H, W, target_frames).permute(3, 0, 1, 2)
        return video_interp

cfg = {
    'num_frame': 32,
    'gt_size': [720, 1280],
    'interval_list': [1],
    'random_reverse': False,
    'use_hflip': True,
    'use_rot': False,
    'flip_sequence': False,

    'degradation_1': {
        'random_blur': {
            'params': {
                'kernel_size': [7, 9, 11, 13, 15, 17, 19, 21],
                'kernel_list': [
                    'iso','aniso','generalized_iso','generalized_aniso',
                    'plateau_iso','plateau_aniso','sinc'
                ],
                'kernel_prob': [0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
                'sigma_x': [0.2, 3],
                'sigma_y': [0.2, 3],
                'rotate_angle': [-3.1416, 3.1416],
                'beta_gaussian': [0.5, 4],
                'beta_plateau': [1, 2],
                'sigma_x_step': 0.02,
                'sigma_y_step': 0.02,
                'rotate_angle_step': 0.31416,
                'beta_gaussian_step': 0.05,
                'beta_plateau_step': 0.1,
                'omega_step': 0.0628
            },
            'keys': ['lqs']
        },
        'random_resize': {
            'params': {
                'resize_mode_prob': [0.2, 0.7, 0.1],  # up, down, keep
                'resize_scale': [0.15, 1.5],
                'resize_opt': ['bilinear', 'area', 'bicubic'],
                'resize_prob': [0.3333, 0.3333, 0.3334],
                'resize_step': 0.015,
                'is_size_even': True
            },
            'keys': ['lqs']
        },
        'random_noise': {
            'params': {
                'noise_type': ['gaussian','poisson'],
                'noise_prob': [0.5, 0.5],
                'gaussian_sigma': [1, 30],
                'gaussian_gray_noise_prob': 0.4,
                'poisson_scale': [0.05, 3],
                'poisson_gray_noise_prob': 0.4,
                'gaussian_sigma_step': 0.1,
                'poisson_scale_step': 0.005
            },
            'keys': ['lqs']
        },
        'random_jpeg': {
            'params': {
                'quality': [30, 95],
                'quality_step': 3
            },
            'keys': ['lqs']
        },
        'random_mpeg': {
            'params': {
                'codec': ['libx264','h264','mpeg4'],
                'codec_prob': [0.3333,0.3333,0.3334],
                'bitrate': [1e4, 1e5]
            },
            'keys': ['lqs']
        }
    },

    'degradation_2': {
        'random_blur': {
            'params': {
                'prob': 0.8,
                'kernel_size': [7, 9, 11, 13, 15, 17, 19, 21],
                'kernel_list': [
                    'iso','aniso','generalized_iso','generalized_aniso',
                    'plateau_iso','plateau_aniso','sinc'
                ],
                'kernel_prob': [0.405,0.225,0.108,0.027,0.108,0.027,0.1],
                'sigma_x': [0.2,1.5],
                'sigma_y': [0.2,1.5],
                'rotate_angle': [-3.1416,3.1416],
                'beta_gaussian': [0.5,4],
                'beta_plateau': [1,2],
                'sigma_x_step': 0.02,
                'sigma_y_step': 0.02,
                'rotate_angle_step': 0.31416,
                'beta_gaussian_step': 0.05,
                'beta_plateau_step': 0.1,
                'omega_step': 0.0628
            },
            'keys': ['lqs']
        },
        'random_resize': {
            'params': {
                'resize_mode_prob': [0.3, 0.4, 0.3],
                'resize_scale': [0.3, 1.2],
                'resize_opt': ['bilinear','area','bicubic'],
                'resize_prob': [0.3333,0.3333,0.3334],
                'resize_step': 0.03,
                'is_size_even': True
            },
            'keys': ['lqs']
        },
        'random_noise': {
            'params': {
                'noise_type': ['gaussian','poisson'],
                'noise_prob': [0.5, 0.5],
                'gaussian_sigma': [1,25],
                'gaussian_gray_noise_prob': 0.4,
                'poisson_scale': [0.05,2.5],
                'poisson_gray_noise_prob': 0.4,
                'gaussian_sigma_step': 0.1,
                'poisson_scale_step': 0.005
            },
            'keys': ['lqs']
        },
        'random_jpeg': {
            'params': {
                'quality': [30,95],
                'quality_step': 3
            },
            'keys': ['lqs']
        },
        'random_mpeg': {
            'params': {
                'codec': ['libx264','h264','mpeg4'],
                'codec_prob': [0.3333,0.3333,0.3334],
                'bitrate': [1e4,1e5]
            },
            'keys': ['lqs']
        },
        'resize_final': {
            'params': {
                'target_size': [180,320],
                'resize_opt': ['bilinear','area','bicubic'],
                'resize_prob': [0.3333,0.3333,0.3334]
            },
            'keys': ['lqs']
        },
        'blur_final': {
            'params': {
                'prob': 0.8,
                'kernel_size': [7,9,11,13,15,17,19,21],
                'kernel_list': ['sinc'],
                'kernel_prob': [1],
                'omega': [1.0472,3.1416],
                'omega_step': 0.0628
            },
            'keys': ['lqs']
        }
    },

    'transforms': {
        'usm': {
            'kernel_size': 51,
            'sigma': 0,
            'weight': 0.5,
            'threshold': 10,
            'keys': ['gts']
        },
        'clip': {
            'keys': ['lqs']
        },
        'rescale': {
            'keys': ['lqs','gts']
        }
    }
}


from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Suppose 'cfg' is defined above (the final config),
    # or you loaded it from a YAML file. Then:
    csv_path = "dataset/OpenVid-1M/data/train/OpenVid-1M_subset.csv"
    root_dir = "dataset/OpenVid-1M/video/"

    dataset = RealVSRCSVVideoDataset(cfg, csv_path, root_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )
    import matplotlib.pyplot as plt

    def show_multiple_images(lqs, gts, idx=0):
        """
        Visualizes multiple frames from the batch, side by side.
        Displays the first, middle, and last frame from the video sequence.
        """
        num_frames = len(lqs)  # Assuming the batch size is 1
        frame_indices = [0, num_frames // 2, num_frames - 1]  # First, middle, last frames

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        for i, idx in enumerate(frame_indices):
            lq_img = lqs[idx].cpu().numpy().squeeze(0).transpose(1, 2, 0)  # (C, H, W) => (H, W, C)
            gt_img = gts[idx].cpu().numpy().squeeze(0).transpose(1, 2, 0)
            # lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
            # gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            axes[i, 0].imshow(lq_img)
            axes[i, 0].set_title(f"LQ Frame {idx+1}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_img)
            axes[i, 1].set_title(f"GT Frame {idx+1}")
            axes[i, 1].axis('off')
            
        plt.savefig('temp_multi.png')

    # Example usage in your dataloader loop:
    for batch in dataloader:
        lqs, gts, text, video_path = batch['lqs'], batch['gts'], batch['text'], batch['video_path']
        show_multiple_images(lqs, gts, idx=0)  # Show multiple frames from the first video
        print(text)
        print(video_path)
        break


