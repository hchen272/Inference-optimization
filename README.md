# CVGP – Video Super-Resolution for AIAA 3201

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

> **Course Project – AIAA 3201 Introduction to Computer Vision (Spring 2026)**  
> Team members: [Honglinag Chen], [Boyong Hou]  

This repository contains the complete pipeline for video super‑resolution (VSR), from classic hand‑crafted baselines to state‑of‑the‑art AI‑driven methods. We evaluate on mandatory datasets (self‑captured low‑light video, REDS‑sample, Vimeo‑LR clips) and optional benchmarks (REDS, Vimeo‑90K). The project follows the requirements of the term project instruction (CVPR template, arXiv upload optional, code & demo mandatory).

---

## Overview

We address the task of **video super‑resolution** – reconstructing a high‑resolution (HR) video from a low‑resolution (LR) input while maintaining temporal consistency. The pipeline progresses from simple spatial interpolation (bicubic, Lanczos) and early CNNs (SRCNN) to bidirectional propagation with optical flow (BasicVSR++), perceptual enhancement (Real‑ESRGAN), and finally a novel uncertainty‑aware hybrid refinement (Part 3). Quantitative metrics (PSNR, SSIM, LPIPS, tLPIPS) and qualitative comparisons are reported.

**Key features**:
- Frame‑wise and temporal baselines
- State‑of‑the‑art alignment using deformable convolutions (BasicVSR++)
- GAN‑based perceptual loss for realistic texture synthesis
- Uncertainty‑guided mixture of BasicVSR++ and generative prior (ControlNet / Flow Matching)

---

## Methods Implemented

### Part 1: Baseline – Hand‑crafted

Spatial upsampling: Bicubic, Lanczos (OpenCV), and SRCNN [1] (3‑layer CNN).

Temporal baseline: Weighted average of neighboring frames (after bicubic upscaling) + optional unsharp masking.

Expected outcome: Blurry textures, SRCNN slightly better than interpolation, noticeable flickering.

### Part 2: SOTA Reproduction – AI‑driven Pipeline

BasicVSR++ [2]: Bidirectional propagation with second‑order grid propagation and deformable convolution alignment (using SpyNet for flow). We use the official implementation adapted to our data format.

## Results

Quantitative results on **REDS‑sample** (×2 upscaling, averaged over two video clips).  
*LPIPS, FID, tLPIPS are still under computation and will be added in the final report.*

| Method | PSNR ↑ | SSIM ↑ |
|--------|--------|--------|
| Bicubic | 32.69 | 0.9570 |
| Lanczos | 32.69 | 0.9566 |
| SRCNN | 30.26 | 0.9097 |
| Bicubic + Temporal Avg | 29.95 | 0.9046 |
| Bicubic + Temporal Avg + Unsharp | 29.89 | 0.9050 |
| Lanczos + Temporal Avg | 29.94 | 0.9043 |
| Lanczos + Temporal Avg + Unsharp | 29.85 | 0.9036 |
| BasicVSR++ (FP32) | 37.42 | 0.9617 |
| BasicVSR++ (FP16) | 37.38 | 0.9615 |
| Real‑ESRGAN | TBD | TBD |
| Our hybrid (Part 3) | TBD | TBD |

## References

[1] Dong, C., et al. Image super-resolution using deep convolutional networks. TPAMI, 2015. 

[2] Chan, K. C., et al. BasicVSR++: Improving video super-resolution with enhanced propagation and alignment. CVPR, 2022. 
