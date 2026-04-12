import cv2

def bicubic_upsample(frame, scale=2):
    """Bicubic interpolation upsampling."""
    if scale < 1:
        raise ValueError("Scale must be >= 1")
    h, w = frame.shape[:2]
    new_w, new_h = w * scale, h * scale
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def lanczos_upsample(frame, scale=2):
    """Lanczos interpolation upsampling."""
    h, w = frame.shape[:2]
    new_w, new_h = w * scale, h * scale
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)