import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from utils.video_io import read_video_frames
from utils.metrics import compute_frame_metrics
import matplotlib.pyplot as plt

def evaluate_single_video(gt_frames, sr_frames, method_name):
    """Same as before, returns dict of metrics."""
    assert len(gt_frames) == len(sr_frames), "Frame count mismatch"
    psnr_list = []
    ssim_list = []
    for i, (gt, sr) in enumerate(tqdm(zip(gt_frames, sr_frames), total=len(gt_frames), desc=f"  {method_name}", leave=False)):
        psnr_val, ssim_val = compute_frame_metrics(gt, sr)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
    return {
        "method": method_name,
        "avg_psnr": float(np.mean(psnr_list)),
        "avg_ssim": float(np.mean(ssim_list)),
        "psnr_per_frame": psnr_list,
        "ssim_per_frame": ssim_list
    }

def save_comparison_figure(gt_frames, sr_dict, output_dir, frame_indices=None):
    """Same as before, saves a figure for one video."""
    if frame_indices is None:
        n_frames = len(gt_frames)
        indices = [0, n_frames // 2, n_frames - 1] if n_frames > 2 else [0]
    else:
        indices = frame_indices
    
    methods = list(sr_dict.keys())
    n_methods = len(methods)
    n_rows = n_methods + 1
    n_cols = len(indices)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1: axes = [axes]
    if n_cols == 1: axes = [[ax] for ax in axes]
    
    # GT row
    for col, idx in enumerate(indices):
        axes[0][col].imshow(cv2.cvtColor(gt_frames[idx], cv2.COLOR_BGR2RGB))
        axes[0][col].set_title(f"GT Frame {idx}")
        axes[0][col].axis('off')
    
    # Methods
    for row, method in enumerate(methods, start=1):
        for col, idx in enumerate(indices):
            sr_frame = sr_dict[method][idx]
            axes[row][col].imshow(cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB))
            axes[row][col].set_title(f"{method}\nFrame {idx}")
            axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_frames.png"), dpi=150)
    plt.close()

def plot_average_bar_chart(avg_results, output_dir):
    """Plot PSNR and SSIM bar chart from average results."""
    methods = [r['method'] for r in avg_results]
    psnr_vals = [r['avg_psnr'] for r in avg_results]
    ssim_vals = [r['avg_ssim'] for r in avg_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(methods))
    width = 0.6
    
    ax1.bar(x, psnr_vals, width, color='skyblue', edgecolor='black')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Average PSNR across videos')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    for i, v in enumerate(psnr_vals):
        ax1.text(i, v + 0.5, f'{v:.2f}', ha='center')
    
    ax2.bar(x, ssim_vals, width, color='lightgreen', edgecolor='black')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Average SSIM across videos')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    for i, v in enumerate(ssim_vals):
        ax2.text(i, v + 0.005, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_metrics_bar_chart.png"), dpi=150)
    plt.close()
    print(f"Average bar chart saved to {output_dir}/average_metrics_bar_chart.png")

def main():
    # Configuration
    gt_root = "data/gt_videos"          # folder containing GT videos (same names as input videos)
    output_root = "outputs"             # root where per-video output subfolders are
    scale = 2
    results_all = []   # accumulate per-video, per-method metrics
    
    # Iterate over each subfolder in output_root (each subfolder corresponds to one video)
    for video_name in os.listdir(output_root):
        video_output_dir = os.path.join(output_root, video_name)
        if not os.path.isdir(video_output_dir):
            continue
        
        # Find GT video
        gt_video_path = os.path.join(gt_root, f"{video_name}.mp4")
        if not os.path.exists(gt_video_path):
            print(f"GT video not found for {video_name}. Skipping evaluation for this video.")
            continue
        
        print(f"\nEvaluating video: {video_name}")
        gt_frames = read_video_frames(gt_video_path)
        
        # Find all output videos in this subfolder that match scale (e.g., contain "x2")
        output_videos = [f for f in os.listdir(video_output_dir) if f.endswith('.mp4') and f"x{scale}" in f]
        if not output_videos:
            print(f"  No output videos found for {video_name} with scale {scale}. Skipping.")
            continue
        
        sr_dict = {}
        video_results = []
        for video_file in output_videos:
            method_name = video_file.replace('.mp4', '')
            sr_frames = read_video_frames(os.path.join(video_output_dir, video_file))
            # Trim or pad to match GT length
            if len(sr_frames) > len(gt_frames):
                sr_frames = sr_frames[:len(gt_frames)]
            elif len(sr_frames) < len(gt_frames):
                sr_frames += [sr_frames[-1]] * (len(gt_frames) - len(sr_frames))
            metrics = evaluate_single_video(gt_frames, sr_frames, method_name)
            video_results.append(metrics)
            sr_dict[method_name] = sr_frames
        
        # Save per-video JSON
        json_path = os.path.join(video_output_dir, "metrics_results.json")
        with open(json_path, 'w') as f:
            json.dump(video_results, f, indent=4)
        print(f"  Metrics saved to {json_path}")
        
        # Generate comparison figure for this video
        save_comparison_figure(gt_frames, sr_dict, video_output_dir)
        
        # Append to overall results (with video name)
        for m in video_results:
            m['video'] = video_name
            results_all.append(m)
    
    # Save overall aggregated results (all videos combined)
    if results_all:
        overall_json_path = os.path.join(output_root, "overall_metrics.json")
        with open(overall_json_path, 'w') as f:
            json.dump(results_all, f, indent=4)
        print(f"\nOverall metrics saved to {overall_json_path}")
        
        # Optionally, compute and save average metrics per method across videos
        method_avg = {}
        for item in results_all:
            method = item['method']
            if method not in method_avg:
                method_avg[method] = {'psnr': [], 'ssim': []}
            method_avg[method]['psnr'].append(item['avg_psnr'])
            method_avg[method]['ssim'].append(item['avg_ssim'])
        
        avg_results = []
        for method, vals in method_avg.items():
            avg_results.append({
                'method': method,
                'avg_psnr': np.mean(vals['psnr']),
                'avg_ssim': np.mean(vals['ssim']),
                'std_psnr': np.std(vals['psnr']),
                'std_ssim': np.std(vals['ssim'])
            })
        avg_json_path = os.path.join(output_root, "average_metrics_across_videos.json")
        with open(avg_json_path, 'w') as f:
            json.dump(avg_results, f, indent=4)
        print(f"Average metrics across videos saved to {avg_json_path}")
        
        # Generate bar chart for average metrics
        plot_average_bar_chart(avg_results, output_root)
    else:
        print("No valid video results found.")

if __name__ == "__main__":
    main()