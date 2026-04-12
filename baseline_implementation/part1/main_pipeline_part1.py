#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main pipeline for Part 1: Baseline methods.
Processes all videos in data/input_videos/, saves outputs to outputs/<video_name>/,
and records inference-only timing results.
"""

import os
import sys
import time
import json
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video_io import read_video_frames, write_video_frames
from part1.spatial_upsample import bicubic_upsample, lanczos_upsample
from part1.temporal_average import apply_temporal_average_to_video
from part1.unsharp_mask import apply_unsharp_mask_to_video
from part1.srcnn_inference import load_srcnn_model, srcnn_upsample_frame


def process_single_video(input_video_path, output_dir, scale=2, use_srcnn=True):
    """
    Process a single video: generate all baseline outputs and record inference times.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read low-resolution frames
    print("Reading low-resolution video...")
    lr_frames = read_video_frames(input_video_path)
    if not lr_frames:
        print("  No frames read. Skipping.")
        return
    n_frames = len(lr_frames)
    timing_results = {}

    # ------------------------------------------------------------------
    # 1. Bicubic pipeline (core: bicubic_upsample on each frame)
    # ------------------------------------------------------------------
    print("\n--- Bicubic pipeline ---")
    start = time.time()
    bicubic_frames = [bicubic_upsample(f, scale) for f in lr_frames]
    elapsed = time.time() - start
    write_video_frames(bicubic_frames, os.path.join(output_dir, f"bicubic_x{scale}.mp4"))
    timing_results[f"bicubic_x{scale}"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    # Temporal averaging on bicubic (core: apply_temporal_average_to_video)
    start = time.time()
    bicubic_temporal = apply_temporal_average_to_video(bicubic_frames)
    elapsed = time.time() - start
    write_video_frames(bicubic_temporal, os.path.join(output_dir, f"bicubic_x{scale}_temporal_avg.mp4"))
    timing_results[f"bicubic_x{scale}_temporal_avg"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    # Unsharp masking on temporal averaged (core: apply_unsharp_mask_to_video)
    start = time.time()
    bicubic_temporal_unsharp = apply_unsharp_mask_to_video(bicubic_temporal)
    elapsed = time.time() - start
    write_video_frames(bicubic_temporal_unsharp, os.path.join(output_dir, f"bicubic_x{scale}_temporal_avg_unsharp.mp4"))
    timing_results[f"bicubic_x{scale}_temporal_avg_unsharp"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    # ------------------------------------------------------------------
    # 2. Lanczos pipeline
    # ------------------------------------------------------------------
    print("\n--- Lanczos pipeline ---")
    start = time.time()
    lanczos_frames = [lanczos_upsample(f, scale) for f in lr_frames]
    elapsed = time.time() - start
    write_video_frames(lanczos_frames, os.path.join(output_dir, f"lanczos_x{scale}.mp4"))
    timing_results[f"lanczos_x{scale}"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    start = time.time()
    lanczos_temporal = apply_temporal_average_to_video(lanczos_frames)
    elapsed = time.time() - start
    write_video_frames(lanczos_temporal, os.path.join(output_dir, f"lanczos_x{scale}_temporal_avg.mp4"))
    timing_results[f"lanczos_x{scale}_temporal_avg"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    start = time.time()
    lanczos_temporal_unsharp = apply_unsharp_mask_to_video(lanczos_temporal)
    elapsed = time.time() - start
    write_video_frames(lanczos_temporal_unsharp, os.path.join(output_dir, f"lanczos_x{scale}_temporal_avg_unsharp.mp4"))
    timing_results[f"lanczos_x{scale}_temporal_avg_unsharp"] = {
        "inference_time_s": elapsed,
        "inference_fps": n_frames / elapsed if elapsed > 0 else 0
    }

    # ------------------------------------------------------------------
    # 3. SRCNN pipeline (if enabled)
    # ------------------------------------------------------------------
    if use_srcnn:
        print("\n--- SRCNN pipeline ---")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weights_path = f"models/srcnn_x{scale}.pth"
        if os.path.exists(weights_path):
            model = load_srcnn_model(weights_path, device=device)
            start = time.time()
            srcnn_frames = []
            for frame in tqdm(lr_frames, desc="  SRCNN inference"):
                hr_frame = srcnn_upsample_frame(frame, model, device, scale)
                srcnn_frames.append(hr_frame)
            elapsed = time.time() - start
            write_video_frames(srcnn_frames, os.path.join(output_dir, f"srcnn_x{scale}.mp4"))
            timing_results[f"srcnn_x{scale}"] = {
                "inference_time_s": elapsed,
                "inference_fps": n_frames / elapsed if elapsed > 0 else 0
            }
        else:
            print(f"  SRCNN weights not found at {weights_path}. Skipping SRCNN.")

    # Save timing results
    timing_path = os.path.join(output_dir, "timing_results.json")
    with open(timing_path, 'w') as f:
        json.dump(timing_results, f, indent=4)
    print(f"Timing results saved to {timing_path}")

    print("\nAll output videos saved in", output_dir)


def main():
    # Configuration
    input_dir = "data/input_videos"          # folder containing all LR videos
    output_root = "outputs"                  # root output folder (subfolders per video)
    scale = 2
    use_srcnn = True

    # Find all video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]

    if not input_files:
        print(f"No video files found in {input_dir}")
        return

    for video_file in input_files:
        video_name = os.path.splitext(video_file)[0]
        input_path = os.path.join(input_dir, video_file)
        output_dir = os.path.join(output_root, video_name)
        print(f"\n===== Processing video: {video_name} =====")
        process_single_video(input_path, output_dir, scale, use_srcnn)

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()