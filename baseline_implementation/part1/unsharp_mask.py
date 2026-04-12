import cv2
import numpy as np

def unsharp_mask(frame, sigma=1.0, strength=1.5):
    """
    Apply unsharp masking to enhance edges.
    frame: input image (uint8, BGR).
    sigma: standard deviation for Gaussian blur.
    strength: amount of edge enhancement (>=0).
    Returns: enhanced image (uint8).
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    # Compute high frequency component
    high_freq = cv2.subtract(frame.astype(np.float32), blurred.astype(np.float32))
    # Add back with strength
    enhanced = frame.astype(np.float32) + strength * high_freq
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def apply_unsharp_mask_to_video(frames, sigma=1.0, strength=1.5):
    """Apply unsharp masking to each frame in a list."""
    return [unsharp_mask(f, sigma, strength) for f in frames]