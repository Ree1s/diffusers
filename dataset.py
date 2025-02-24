import csv
import os
import pickle
import lmdb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import video_transforms
from video_transforms import center_crop_arr

import json
import ipdb


def get_transforms_video(resolution=256, target_range=(-1, 1)):
    """
    Args:
        resolution (int): Target resolution
        target_range (tuple): Desired output range, default (-1, 1)
    """
    if target_range == (-1, 1):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif target_range == (0, 1):
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        raise ValueError(f"Unsupported target range: {target_range}")
        
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=mean, std=std, inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform


class DatasetFromLMDB(torch.utils.data.Dataset):
    """Load video according to the LMDB database.

    Args:
        lmdb_path (str): Path to the LMDB database.
        pkl_path (str): Path to the pickle file containing keys.
        num_frames (int): Number of frames to load.
        frame_interval (int): Interval between frames.
        transform (callable): Transform to apply to the video frames.
    """

    def __init__(
        self,
        lmdb_path,
        pkl_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
    ):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.pkl_path = pkl_path
        
        # Load keys from pickle file
        with open(pkl_path, 'rb') as f:
            self.keys = pickle.load(f)
        print(f"✅ Loaded {len(self.keys)} keys from {pkl_path}")

        self.env = None  # LMDB environment initialized in _init_lmdb
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)

    def _init_lmdb(self):
        """Initialize LMDB environment for each worker."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                max_readers=32
            )

    def getitem(self, index):
        self._init_lmdb()
        video_id = self.keys[index]

        # Get video path and caption from LMDB
        with self.env.begin(buffers=True) as txn:
            data = txn.get(video_id.encode())
            if data is None:
                raise FileNotFoundError(f"❌ Missing LMDB Key: {video_id}")
            
        video_path, text = pickle.loads(data)

        # Load video frames
        vframes, aframes, info = torchvision.io.read_video(
            filename=video_path,
            pts_unit="sec",
            output_format="TCHW"
        )
        total_frames = len(vframes)

        # Handle videos with insufficient frames
        loop_index = index
        while total_frames < self.num_frames:
            loop_index = (loop_index + 1) % len(self.keys)
            video_id = self.keys[loop_index]
            with self.env.begin(buffers=True) as txn:
                data = txn.get(video_id.encode())
                if data is None:
                    continue
            video_path, text = pickle.loads(data)
            vframes, aframes, info = torchvision.io.read_video(
                filename=video_path,
                pts_unit="sec",
                output_format="TCHW"
            )
            total_frames = len(vframes)

        # Sample frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.num_frames, \
            f"Video {video_path} has insufficient frames"
        
        frame_indices = np.linspace(
            start_frame_ind,
            end_frame_ind - 1,
            self.num_frames,
            dtype=int
        )
        video = vframes[frame_indices]

        # Apply transforms
        if self.transform is not None:
            video = self.transform(video)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):  # Retry logic
            try:
                return self.getitem(index)
            except Exception as e:
                print(f"Error loading index {index}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data samples")

    def __len__(self):
        return len(self.keys)


if __name__ == '__main__':
    # Example usage with OpenVid paths
    dataset = DatasetFromLMDB(
        lmdb_path="/data/42-julia-hpc-rz-cv/sig95vg/OpenVid-1M/dataset_index.lmdb",
        pkl_path="/data/42-julia-hpc-rz-cv/sig95vg/OpenVid-1M/keys.pkl",
        transform=get_transforms_video(resolution=256),
        num_frames=16,
        frame_interval=3,
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=1
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # Test the dataloader
    for video_data in loader:
        video = video_data["video"]
        text = video_data["text"]
        print(f"Video shape: {video.shape}")
        print(f"Text: {text}")
        
        # Check data range
        min_val = video.min().item()
        max_val = video.max().item()
        mean_val = video.mean().item()
        print(f"Data range: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}")
        break