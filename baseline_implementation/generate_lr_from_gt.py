import os
import cv2
from tqdm import tqdm

def generate_lr_videos(gt_dir="data/gt_videos", lr_dir="data/input_videos", scale=2, overwrite=False):
    """
    Generate low-resolution videos from ground truth videos by downscaling.
    
    Args:
        gt_dir: Directory containing ground truth (HR) videos.
        lr_dir: Directory to save low-resolution videos.
        scale: Downscaling factor (e.g., 2 for 1/2 size).
        overwrite: If True, overwrite existing LR videos; otherwise skip.
    """
    os.makedirs(lr_dir, exist_ok=True)
    
    # Find all video files in gt_dir
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(video_extensions)]
    
    if not gt_files:
        print(f"No video files found in {gt_dir}")
        return
    
    print(f"Found {len(gt_files)} GT videos. Scale factor = {scale}")
    
    for gt_file in tqdm(gt_files, desc="Generating LR videos"):
        gt_path = os.path.join(gt_dir, gt_file)
        lr_path = os.path.join(lr_dir, gt_file)  # keep same name
        
        if os.path.exists(lr_path) and not overwrite:
            print(f"  Skipping {gt_file} (already exists)")
            continue
        
        # Open GT video
        cap = cv2.VideoCapture(gt_path)
        if not cap.isOpened():
            print(f"  Cannot open {gt_path}, skipping")
            continue
        
        # Get original properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Compute LR size
        lr_width = width // scale
        lr_height = height // scale
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(lr_path, fourcc, fps, (lr_width, lr_height))
        
        # Read, downscale, and write each frame
        for _ in tqdm(range(total_frames), desc=f"  Processing {gt_file}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
            lr_frame = cv2.resize(frame, (lr_width, lr_height), interpolation=cv2.INTER_CUBIC)
            out.write(lr_frame)
        
        cap.release()
        out.release()
        print(f"  Saved: {lr_path} (size: {lr_width}x{lr_height})")
    
    print("All LR videos generated.")

if __name__ == "__main__":
    # You can change scale parameter here (2, 3, 4, etc.)
    generate_lr_videos(gt_dir="data/gt_videos", lr_dir="data/input_videos", scale=2, overwrite=False)