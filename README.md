# Multi-Object Tracking with Mask R-CNN & Siamese ReID

**CS 5567 — Deep Learning | University of Missouri–Kansas City | April 2026**

**Team Members:** 
- Joe Doan
- Cory Stever
- Ibrahim Alborno

---

## Project Overview
This project implements an end-to-end Multi-Object Tracking (MOT) pipeline evaluated on the MOT16 dataset. Instead of relying purely on simple IoU (Intersection over Union) bounding box overlaps, our system uses a DeepSORT-style approach. We combined a fine-tuned Mask R-CNN detector with a Siamese Re-Identification (ReID) network to maintain persistent IDs for people even through occlusions or when they cross paths.

## Architecture

Our pipeline consists of two main components:

### 1. Object Detection (Mask R-CNN)
- Replaced the baseline Faster R-CNN with a Mask R-CNN architecture.
- Trained to detect pedestrians (class 1) using the MOT16 dataset.
- **Optimizations:** A100 GPU optimized using Automatic Mixed Precision (AMP), batch size 8, and a Warmup + Cosine Annealing learning rate schedule.
- **Evaluation:** Evaluated using Precision-Recall (PR) AUC instead of traditional ROC AUC, as the background class is effectively infinite in object detection.

### 2. Person Re-Identification (Siamese Network)
- Built on a ResNet-50 backbone.
- Extracts a 256-dimensional embedding vector representing a person's appearance.
- **Improvements over baseline:**
  - Added a **BN Neck** to cleanly separate the Triplet loss (L2 normalized) and Cross-Entropy loss objectives.
  - Heavy data augmentation (Random Erasing, Color Jitter, Random Crop) to simulate occlusions and lighting changes.
  - Custom learning rate scheduler (10 epochs warmup -> cosine annealing).
  - Label smoothing applied to the Cross-Entropy loss.

### 3. Tracking Pipeline
The final pipeline extracts detections frame-by-frame, crops the detected pedestrians, and feeds them into the ReID model. The Hungarian algorithm is then used to match new detections to existing tracks based on cosine distance between embeddings.

---

## Repository Structure

- `report/`
  - `Object_Tracking.pdf` — The final technical report detailing our methodologies and results.
  - `comp_sci_5567_final_presentation.pptx` — The slides for our final presentation.
- `final submission/`
  - `Full_Pipeline.ipynb` — **(Main File)** The consolidated notebook containing the entire project pipeline (Mask R-CNN -> ReID -> Tracker).
- `notebooks/`
  - `MaskRCNN_A100_Tracking-2.ipynb` — Standalone notebook for training the Mask R-CNN detector.
  - `ReID_Improved.ipynb` — Standalone notebook for training and evaluating the Siamese ReID model.
  - `Tracking_Pipeline.ipynb` — The tracking logic that integrates the two models.
  - `midterm_video_object_tracking_notebook-2.ipynb` — Our baseline Phase 1 submission (Faster R-CNN + Simple IoU Tracker).

## How to Run

1. Open `final submission/Full_Pipeline.ipynb` in Google Colab (A100 GPU recommended).
2. Mount your Google Drive to access the MOT16 dataset.
3. Update the `DATA_ROOT` paths in the notebook to point to your MOT16 directory.
4. Run all cells sequentially to train the models and generate the final tracking video.
