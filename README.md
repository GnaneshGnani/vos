# VITA-TCOVIS: Temporally Consistent Video Object Segmentation

## Project Overview

This project implements a hybrid deep learning architecture for **Video Object Segmentation (VOS)**, designed to maintain consistent object identities across video frames. It synthesizes the object-token paradigm from **VITA** (Video Instance Segmentation via Token Association) with the temporal consistency mechanisms of **TCOVIS**.

Standard segmentation models often treat video frames independently, leading to "ID switching" where an object (e.g., a car) is correctly segmented but assigned a different ID in subsequent frames. This project addresses that challenge by introducing:

1.  **Object Tokens:** Learnable queries that detect objects within a frame.
2.  **Temporal Association:** A projection module that links tokens across time.
3.  **Global Tracking:** An inference-time memory bank that assigns consistent global IDs based on embedding similarity.

-----

## Codebase Structure & Detailed Explanation

The project is modularized into five core components handling data, modeling, training, loss calculation, and inference.

### 1\. `main.py` (The Orchestrator)

This script acts as the entry point for training and experimentation.

  * **Argument Parsing:** It manages hyperparameters (learning rates, loss weights, batch sizes) via `argparse`, allowing users to switch between local debugging and high-performance A100 runs without changing code.
  * **Experiment Tracking:** It integrates **ClearML** to log loss curves, visual masks, and system metrics remotely.
  * **Training Loop:** It iterates through epochs, triggering the forward pass, loss calculation, and backpropagation. It creates separate parameter groups for the backbone (lower learning rate) and the transformer heads (higher learning rate).
  * **Checkpointing:** It automatically saves model states (`model_epoch_X.pth`) and triggers the final evaluation protocol upon completion.

### 2\. `modules.py` (The Architecture)

This file defines the neural network architecture (`VITA_TCOVIS`).

  * **Backbone (ResNet-50):** The model uses a pre-trained ResNet-50 (layers 1-3) to extract spatial feature maps from video frames.
  * **Projection Layer:** A $1\times1$ Convolution reduces the high-dimensional backbone features (2048 channels) to the model's hidden dimension (256 channels).
  * **Transformer Decoder:** This is the core "brain" of the model. It takes a fixed set of learnable **Object Tokens** (queries) and attends to the image features to locate objects.
  * **Heads:**
      * **Mask Head:** Generates segmentation masks via a dot product between the processed tokens and the spatial feature map.
      * **Temporal Association Module:** A TCOVIS-inspired MLP (Linear $\to$ ReLU $\to$ LayerNorm) that projects the object tokens into a 128-dimensional embedding space used specifically for tracking.

### 3\. `utils.py` (Loss Functions & Matching)

This file contains the mathematical logic that guides the training process.

  * **Hungarian Matcher:** Deep learning models output a fixed number of predictions (e.g., 50), but an image may have only 2 objects. The matcher solves a bipartite matching problem to assign the best predicted token to each ground truth object based on a cost matrix (Classification Cost + Mask Cost + Dice Cost).
  * **TCOVISCriterion:** The joint loss function that optimizes the model:
      * **Mask Loss (BCE):** Ensures pixel-level accuracy (binary cross-entropy).
      * **Dice Loss:** Maximizes the Intersection-over-Union (IoU) between predicted and ground-truth shapes. This is critical for preventing the model from outputting empty or fuzzy masks.
      * **Supervised Matching Loss:** Forces the embeddings of the *same* object in Frame $t$ and Frame $t+1$ to be similar (high dot product).
      * **Contrastive Loss:** Pushes the embeddings of *different* objects apart, ensuring distinct representations in the vector space.

### 4\. `dataset.py` (Data Loading)

This script manages the YouTube-VOS dataset.

  * **Clip Sampling:** It loads short video clips (e.g., 5 consecutive frames) to teach the model temporal consistency.
  * **Heuristic ID Selection:** Since the model has a fixed number of tokens, the loader scans the clip and selects the top $N$ most frequent objects to train on. This ensures that the model focuses on the primary objects in the scene.
  * **Normalization:** Converts images to tensors and normalizes pixel values to the standard ImageNet range.

### 5\. `inference.py` (Global Tracking)

This file handles the deployment logic during testing.

  * **Memory Bank:** The `GlobalTracker` class maintains a dictionary of active tracks (`{TrackID: Embedding Vector}`).
  * **Online Assignment:** For every new frame, it matches the newly predicted objects to the Memory Bank using the Hungarian algorithm on embedding similarity.
  * **Momentum Update:** When a match is found, the memory bank entry is updated using a moving average ($\mu_{new} = (1-\beta)\mu_{old} + \beta e_{new}$). This allows the tracker to adapt to gradual changes in the object's appearance (e.g., rotation) without forgetting its identity.

-----

## Comparison: This Project vs. SOTA Papers

This implementation is a streamlined adaptation of state-of-the-art research. Below are the key differences and the rationale behind them (optimizing for 16GB VRAM and educational clarity).

| Feature | Original Papers (VITA / Mask2Former) | This Implementation | Reason for Difference |
| :--- | :--- | :--- | :--- |
| **Pixel Decoder** | Uses a **Multi-scale Deformable Attention** Pixel Decoder to upsample features. | Uses **Bilinear Interpolation** directly from low-res features. | **Hardware/Complexity:** Deformable Attention requires custom CUDA kernels and high VRAM. Interpolation is lightweight and stable. |
| **Backbone** | Often uses Swin Transformer (Large) or ResNet-101. | Uses **ResNet-50**. | **Speed:** ResNet-50 provides a better trade-off between speed and accuracy for single-GPU training. |
| **Loss Function** | Uses Dice Loss + Focal Loss. | Uses **Dice Loss + BCE Loss**. | **Stability:** BCE is standard and easier to tune for smaller datasets than Focal Loss. |
| **Contrastive Learning** | Uses a **Memory Bank** of negative samples from previous batches. | Uses **In-Batch Negatives** (other objects in the current frame). | **Memory:** Storing a global memory bank requires significant GPU memory management; in-batch negatives are sufficient for learning distinctness. |
| **Query Selection** | Uses 100 queries and Class-Balanced Sampling. | Uses configurable queries (10-50) and Frequency-Based Sampling. | **Simplicity:** Frequency-based sampling is a robust heuristic that avoids the complexity of writing a custom class-balanced sampler. |

-----

## Installation

Ensure you have a Python environment with PyTorch installed.

```bash
pip install torch torchvision numpy scipy matplotlib clearml opencv-python Pillow
```

## Dataset Setup

The project expects the **YouTube-VOS** dataset structure:

```text
./dataset/
  └── train/
      ├── JPEGImages/    # Folders of video frames (e.g., 00a2b4...)
      └── Annotations/   # Folders of palette-based PNG masks
```

-----

## Usage

### 1\. High-Performance Training (Recommended)

For GPUs like the A100 (16GB), use the following configuration. Setting `num_tokens` to 50 ensures the data loader captures all objects in a clip, bypassing heuristic limitations.

```bash
python main.py \
  --enable_clearml \
  --project_name "VITA_TCOVIS_Project" \
  --dataset_root "./dataset" \
  --epochs 20 \
  --batch_size 8 \
  --num_tokens 50 \
  --w_dice 5.0 \
  --w_match 2.0
```

### 2\. Debugging / Local Run

For smaller GPUs or quick testing:

```bash
python main.py --epochs 2 --batch_size 2 --num_tokens 10 --run_dir "debug_run"
```

### 3\. Key Arguments

  * `--enable_clearml`: Activates remote logging (Loss curves, Visual masks).
  * `--w_dice`: Weight for Dice Loss. Keep this high (5.0) for good mask shapes.
  * `--w_match`: Weight for Temporal Matching. Increase (e.g., 2.0) if the model struggles to track objects.
  * `--beta`: Momentum for the inference memory bank (Default 0.2). Higher values make the tracker update faster but potentially less stable.