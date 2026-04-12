import cv2
import numpy as np
import random
from tqdm import tqdm

def extract_patches_from_videos(gt_frames, lr_frames, scale, patch_size=33, num_patches=10000):
    """
    Extract LR-HR patch pairs from aligned video frames.
    gt_frames: list of HR frames (full size)
    lr_frames: list of LR frames (smaller size)
    scale: upscaling factor
    patch_size: size of HR patch (e.g., 33). LR patch size = patch_size // scale
    """
    hr_patches = []
    lr_patches = []
    hr_h, hr_w = gt_frames[0].shape[:2]
    lr_h, lr_w = lr_frames[0].shape[:2]
    # LR patch size in LR image space
    lr_patch_size = patch_size // scale
    
    for _ in tqdm(range(num_patches), desc="Extracting patches"):
        # Randomly select a frame
        frame_idx = random.randint(0, len(gt_frames)-1)
        gt = gt_frames[frame_idx]
        lr = lr_frames[frame_idx]
        # Randomly select top-left corner in HR space
        hr_x = random.randint(0, hr_w - patch_size)
        hr_y = random.randint(0, hr_h - patch_size)
        hr_patch = gt[hr_y:hr_y+patch_size, hr_x:hr_x+patch_size]
        # Corresponding LR patch location (scaled down)
        lr_x = hr_x // scale
        lr_y = hr_y // scale
        lr_patch = lr[lr_y:lr_y+lr_patch_size, lr_x:lr_x+lr_patch_size]
        hr_patches.append(hr_patch)
        lr_patches.append(lr_patch)
    return lr_patches, hr_patches