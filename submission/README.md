# Video Object Tracking on MOT16 Using Mask R-CNN and Siamese Re-Identification

**CS 5542 — Deep Learning | University of Missouri–Kansas City | April 2026**

**Team Members:** Joe Doan, Cory Stever, Ibrahim Alborno

---

## Overview

This project implements a complete **end-to-end multi-pedestrian tracking pipeline** on the [MOT16](https://motchallenge.net/data/MOT16/) benchmark. The system integrates three core components:

1. **Mask R-CNN Detector** — Fine-tuned on MOT16 for person detection and instance segmentation
2. **Siamese ReID Network** — Trained on Market-1501 for appearance-based person re-identification
3. **DeepSORT-Style Tracker** — Hungarian algorithm with combined IoU + cosine distance cost matrix for cross-frame identity association

### Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| ReID | mAP (Market-1501) | **73.93%** |
| ReID | Rank-1 Accuracy | **89.22%** |
| ReID | Rank-5 Accuracy | 96.05% |
| ReID | Rank-10 Accuracy | 97.45% |
| Tracker | Sequence | MOT16-09 |
| Tracker | Total Frames | 525 |
| Tracker | Unique IDs | **22** |
| Tracker | Track Entries | 4,480 |

---

## Repository Structure

```
.
├── README.md                           # This file
├── Full_Pipeline.ipynb                 # ⭐ Complete pipeline (all 3 parts merged)
├── report/                             # Project report
│   └── Object_Tracking.pdf             # Final compiled report (PDF)
├── figures/                            # All figures used in the report
│   ├── det_sample_data.png             # MOT16 training data samples
│   ├── det_training_curves.png         # Mask R-CNN training/val loss curves
│   ├── det_inference_sample.png        # Detector inference results
│   ├── reid_training_curves.png        # ReID loss & LR schedule
│   ├── reid_cmc_curve.png              # CMC curve (Rank-1 = 89.22%)
│   ├── reid_baseline_comparison.png    # Baseline vs improved ReID comparison
│   ├── reid_tsne.png                   # t-SNE embedding visualization
│   ├── reid_top5_matches.png           # Top-5 gallery match examples
│   └── tracking_sample_frames.png      # Tracking output on MOT16-09
├── MaskRCNN_A100_Tracking-2.ipynb      # Part 1: Mask R-CNN detector
├── ReID_Improved.ipynb                 # Part 2: Siamese ReID network
└── Tracking_Pipeline.ipynb             # Part 3: Integrated tracking pipeline
```

> **📌 Start here:** [`Full_Pipeline.ipynb`](Full_Pipeline.ipynb) contains all three parts in a single notebook with all outputs preserved. The individual notebooks are also included for reference.

---

## Pipeline Architecture

```
Frame → Detect (Mask R-CNN) → Crop → ReID Embed (Siamese) → Hungarian Match → Track → Video
```

### 1. Person Detection — Mask R-CNN (ResNet-50-FPN)

- **Backbone:** ResNet-50-FPN, pretrained on COCO
- **Training:** 40 epochs on MOT16 with A100 GPU, AMP (mixed precision)
- **Optimizer:** AdamW with differential learning rates (Heads: 3e-3, Backbone: 3e-4)
- **LR Schedule:** Linear warmup (4 epochs) + Cosine annealing
- **Best validation loss:** 0.3783
- **Notebook:** `MaskRCNN_A100_Tracking-2.ipynb`

### 2. Person Re-Identification — Improved Siamese Network

- **Backbone:** ResNet-50 with BN Neck (Luo et al., 2019)
- **Embedding:** 256-dimensional L2-normalized features
- **Loss:** Triplet Loss + Cross-Entropy with label smoothing
- **Training:** 80 epochs on Market-1501 with warmup + cosine annealing
- **Performance:** 73.93% mAP, 89.22% Rank-1
- **Notebook:** `ReID_Improved.ipynb`

### 3. Multi-Object Tracking — DeepSORT-Style

- **Cost matrix:** 50% IoU + 50% cosine distance
- **Association:** Hungarian algorithm with cost gate = 0.7
- **EMA update:** α = 0.8 for embedding smoothing
- **Max missed frames:** 90 (before track deletion)
- **Notebook:** `Tracking_Pipeline.ipynb`

---

## How to Run

All notebooks are designed to run on **Google Colab** with GPU acceleration (A100 recommended).

### Prerequisites

- Google Colab with GPU runtime
- MOT16 dataset uploaded to Google Drive at `/content/drive/MyDrive/Colab Notebooks/MOT16/`
- Market-1501 dataset (for ReID training)

### Execution Order

1. **Train the detector:**
   Open and run `MaskRCNN_A100_Tracking-2.ipynb` — trains Mask R-CNN on MOT16

2. **Train the ReID model:**
   Open and run `ReID_Improved.ipynb` — trains the Siamese network on Market-1501

3. **Run the tracking pipeline:**
   Open and run `Tracking_Pipeline.ipynb` — integrates detector + ReID for end-to-end tracking on MOT16-09

---

## Report

The final project report is available as a pre-compiled PDF at [`report/Object_Tracking.pdf`](report/Object_Tracking.pdf).

---

## References

1. He, K., et al. (2017). *Mask R-CNN*. ICCV.
2. Wojke, N., et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric*. ICIP.
3. Luo, H., et al. (2019). *Bag of Tricks and a Strong Baseline for Deep Person Re-identification*. CVPRW.
4. Hermans, A., et al. (2017). *In Defense of the Triplet Loss for Person Re-Identification*.
5. Milan, A., et al. (2016). *MOT16: A Benchmark for Multi-Object Tracking*.
6. Zheng, L., et al. (2015). *Scalable Person Re-identification: A Benchmark*. ICCV.

---

## License

This project was developed as part of the CS 5542 Deep Learning course at UMKC.
