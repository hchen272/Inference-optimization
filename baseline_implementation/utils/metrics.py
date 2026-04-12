import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_psnr(img1, img2):
    """Compute PSNR between two images (both in BGR uint8)."""
    return psnr(img1, img2, data_range=255)

def compute_ssim(img1, img2):
    """Compute SSIM between two images (convert to grayscale to avoid channel issues)."""
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        gray2 = img2
    # Adjust win_size if image is too small
    min_dim = min(gray1.shape)
    win_size = min(7, min_dim) if min_dim % 2 == 1 else min(7, min_dim - 1)
    if win_size < 3:
        win_size = 3
    return ssim(gray1, gray2, data_range=255, win_size=win_size)

def compute_frame_metrics(gt_frame, sr_frame):
    """Return PSNR and SSIM for a single frame pair."""
    # Ensure same size
    if gt_frame.shape != sr_frame.shape:
        sr_frame = cv2.resize(sr_frame, (gt_frame.shape[1], gt_frame.shape[0]))
    psnr_val = compute_psnr(gt_frame, sr_frame)
    ssim_val = compute_ssim(gt_frame, sr_frame)
    return psnr_val, ssim_val