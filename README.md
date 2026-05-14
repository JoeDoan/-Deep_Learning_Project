# Video Object Tracking on MOT16

## Using Mask R-CNN and Siamese Re-Identification

**CS 5567 — Deep Learning | University of Missouri–Kansas City | April 2026**

**Team Members:** Joe Doan, Cory Stever, Ibrahim Alborno

---

## Overview

This project implements a complete **end-to-end multi-pedestrian tracking pipeline** on the MOT16 benchmark, combining three core components:

| Component | Architecture | Dataset | Key Result |
|-----------|-------------|---------|------------|
| **Detector** | Mask R-CNN (ResNet-50-FPN) | MOT16 | Val loss: **0.3783** |
| **ReID** | Siamese Network + BN Neck | Market-1501 | mAP: **73.93%**, Rank-1: **89.22%** |
| **Tracker** | DeepSORT-style (Hungarian) | MOT16-09 | **22 IDs** across **525 frames** |

```
Frame → Detect (Mask R-CNN) → Crop → ReID Embed (Siamese) → Hungarian Match → Track → Video
```

---

## Repository Structure

```
.
├── README.md                  
├── final submission/
│   └── Full_Pipeline.ipynb    # Complete pipeline notebook (all 3 parts)
├── notebooks/                 # Standalone development notebooks
│   ├── MaskRCNN_A100_Tracking-2.ipynb
│   ├── ReID_Improved.ipynb
│   ├── Tracking_Pipeline.ipynb
│   └── midterm_video_object_tracking_notebook-2.ipynb
├── figures/                   # All evaluation figures
│   ├── det_sample_data.png
│   ├── det_training_curves.png
│   ├── det_inference_sample.png
│   ├── reid_training_curves.png
│   ├── reid_cmc_curve.png
│   ├── reid_baseline_comparison.png
│   ├── reid_tsne.png
│   ├── reid_top5_matches.png
│   └── tracking_sample_frames.png
└── report/
    ├── Object_Tracking.pdf                     # Final technical project report
    └── comp_sci_5567_final_presentation.pptx   # Final presentation slides
```

---

## Quick Start

### 1. View Results (No Setup Required)

Open [`final submission/Full_Pipeline.ipynb`](final%20submission/Full_Pipeline.ipynb) directly on GitHub — all outputs (training logs, plots, metrics, tracking frames) are pre-rendered.

### 2. Run the Pipeline

The notebook is designed for **Google Colab** with GPU runtime.

#### Step 1: Download Model Checkpoints

Pre-trained checkpoints are available on Google Drive:

📥 **[Download Checkpoints](https://drive.google.com/drive/folders/187jDH2i2iyabvQlm3g950IwyL0xHLnLo?usp=sharing)**

Upload the checkpoints to your Google Drive at these paths:
```
MyDrive/
├── MaskRCNN_MOT16/
│   └── maskrcnn_mot16_checkpoint.pth    # Mask R-CNN detector weights
└── ReID_Checkpoints/
    └── reid_improved_v2.pth             # Siamese ReID model weights
```

#### Step 2: Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoeDoan/-Deep_Learning_Project/blob/main/final%20submission/Full_Pipeline.ipynb)

Or manually: upload `Full_Pipeline.ipynb` to Google Colab, select **GPU runtime** (Runtime → Change runtime type → NVIDIA A100 GPU), and hit **Run All**.

#### Step 3: What Happens

| Part | What Runs | Time |
|------|-----------|------|
| **Part 1** | Loads Mask R-CNN checkpoint, runs evaluation & visualization | ~2 min |
| **Part 2** | Loads ReID checkpoint, evaluates on Market-1501, generates plots | ~5 min |
| **Part 3** | Runs full tracking pipeline on MOT16-09 (525 frames) | ~10 min |

> **Note:** If checkpoints are found, training loops are skipped automatically. Without checkpoints, full training takes ~2 hours (Part 1) + ~1.5 hours (Part 2) on a NVIDIA A100 GPU.

---

## Results

### Person Detection (Mask R-CNN)

| Metric | Value |
|--------|-------|
| Best Validation Loss | **0.3783** |
| Training Epochs | 40 (early stop at ~35) |
| GPU | NVIDIA A100 with AMP |

### Person Re-Identification (Siamese Network)

| Metric | Value |
|--------|-------|
| mAP | **73.93%** |
| Rank-1 Accuracy | **89.22%** |
| Rank-5 Accuracy | 96.05% |
| Rank-10 Accuracy | 97.45% |

### Multi-Object Tracking (DeepSORT-Style)

| Metric | Value |
|--------|-------|
| Total Frames | 525 |
| Unique IDs Tracked | **22** |
| Total Track Entries | 4,480 |
| Cost Function | 50% IoU + 50% Cosine Distance |

---

## Technical Details

### Architecture

- **Detector:** Mask R-CNN with ResNet-50-FPN backbone, fine-tuned on MOT16 with AdamW, differential learning rates, linear warmup + cosine annealing
- **ReID:** Improved Siamese Network with ResNet-50 backbone, BN Neck (Luo et al., 2019), 256-d embeddings, Triplet + Cross-Entropy loss with label smoothing
- **Tracker:** Hungarian algorithm with composite cost matrix (IoU + cosine distance), EMA embedding updates (α=0.8), max missed frames = 90

### Key References

1. He et al. (2017) — *Mask R-CNN*, ICCV
2. Wojke et al. (2017) — *Simple Online and Realtime Tracking with a Deep Association Metric*, ICIP
3. Luo et al. (2019) — *Bag of Tricks and a Strong Baseline for Deep Person Re-identification*, CVPRW
4. Hermans et al. (2017) — *In Defense of the Triplet Loss for Person Re-Identification*
5. Milan et al. (2016) — *MOT16: A Benchmark for Multi-Object Tracking*

---

*CS 5567 Deep Learning — University of Missouri–Kansas City*
