# Lymphatic Vessel Segmentation with CTO-Net

This project provides a semi-supervised pipeline for segmenting lymphatic vessels in videos. The primary architecture is **CTO-Net**, a custom model featuring a **Res2Net-50** backbone. The pipeline employs the **Mean Teacher** method for semi-supervised learning. It also includes support for **UNet++** and an experimental **CTO Stitch-ViT** model.

## Table of Contents
- [Key Features](#key-features)
- [Models](#models)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training Pipeline](#training-pipeline)
  - [Running with Stitch-ViT](#running-with-stitch-vit)
- [Tools & Scripts](#tools--scripts)
- [GUI Application](#gui-application)
- [Results](#results)
- [Technical Details](#technical-details)

## Key Features

*   **Primary Architecture (CTO-Net):** A high-performance segmentation model with a **Res2Net-50** backbone designed for detailed feature extraction.
*   **Semi-Supervised Learning:** Leverages unlabeled video data via the **Mean Teacher** method to improve accuracy and reduce annotation effort.
*   **Boundary-Aware Loss:** Combines Dice Loss and Boundary Loss for sharp and accurate detection of vessel edges.
*   **2-Stage Training Pipeline:**
    1.  **Stage 1:** Train a baseline model on labeled data.
    2.  **Stage 2:** Refine the model using Mean Teacher with both labeled and unlabeled data.
*   **Flexible Configuration:** Easily tune parameters for each stage via dedicated JSON files.
*   **GUI for Analysis:** An interactive tool for visualizing predictions and measuring vessel diameters.

## Models

This project supports three models:

1.  **CTO-Net (Default):** The primary model of this project. It uses a **Res2Net-50** backbone and a custom architecture designed for vessel segmentation. The configuration for this model is `cto`.
2.  **CTO Stitch-ViT (Experimental):** An enhanced version of CTO-Net that incorporates Vision Transformer (ViT) blocks with a stitching mechanism. This model aims to capture more global contextual features. The configuration for this is `cto_stitchvit`.
3.  **UNet++:** A well-known architecture for medical image segmentation, available as an alternative. It can be configured by setting the model `name` to `unetpp` in the JSON config.

## Project Structure

```text
.
├── app.py                      # GUI Application entry point
├── config_stage1.json          # Config for Stage 1 (CTO-Net)
├── config_stage2.json          # Config for Stage 2 (CTO-Net)
├── config_stage1_stitchvit.json # Config for Stage 1 (Stitch-ViT)
├── config_stage2_stitchvit.json # Config for Stage 2 (Stitch-ViT)
├── requirements.txt
├── data/
│   ├── annotated/              # Labeled images and JSON annotations
│   │   ├── Human/
│   │   └── Rat/
│   ├── masks/                  # Generated binary masks
│   │   ├── Human/
│   │   └── Rat/
│   ├── frames/                 # Extracted frames from unlabeled videos
│   │   ├── Human/
│   │   └── Rat/
│   └── video/                  # Raw unlabeled videos
│       ├── Human/
│       └── Rat/
├── logs/                       # Training logs, curves, and prediction images
│   ├── Human/
│   └── Rat/
├── models/                     # Saved model checkpoints (.pth files)
│   └── checkpoints/
│       ├── Human/
│       └── Rat/
└── src/                        # Source code
    └── models/
        ├── cto/                # Source for the main CTO-Net
        └── cto_stitchvit/      # Source for the Stitch-ViT variant
```

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Data Type:**
    Open the configuration files you intend to use (e.g., `config_stage1.json`). Set the `"type"` field to either `"Human"` or `"Rat"` to specify your dataset. The framework will automatically use the corresponding subdirectories.

    **Example (`config_stage1.json`):**
    ```json
    {
        "type": "Human",
        ...
    }
    ```

## Usage

### Data Preparation

1.  **Labeled Data:** Place your images and `.json` annotation files in `data/annotated/<type>/` (e.g., `data/annotated/Human/`).
2.  **Convert Annotations to Masks:** Generate binary masks for training.
    ```bash
    python -m tools.scripts.convert_json_to_mask --input data/annotated --output data/masks
    ```
3.  **Unlabeled Data:** Place your raw videos in `data/video/<type>/`.
4.  **Extract Frames from Videos:** Prepare frames for semi-supervised learning.
    ```bash
    python -m tools.scripts.extract_frames --video_dir data/video --output_dir data/frames --fps 1
    ```

### Training Pipeline

The recommended way to train is to run the full pipeline, which executes Stage 1 and Stage 2 sequentially using the default **CTO-Net** model.

**Run the full pipeline (CTO-Net):**
```bash
python -m src.main all --visualize
```
This command uses `config_stage1.json` and `config_stage2.json` by default.

**Running stages individually (CTO-Net):**
You can also run each stage separately.

*   **Stage 1: Train Baseline**
    ```bash
    python -m src.main baseline --config config_stage1.json
    ```
*   **Stage 2: Train Final (Mean Teacher)**
    ```bash
    python -m src.main final --config config_stage2.json
    ```

### Running with Stitch-ViT

To use the **CTO Stitch-ViT** model, specify its configuration files using the `--config` flag.

**Full Pipeline (Stitch-ViT):**
```bash
# First, run Stage 1 with its config
python -m src.main baseline --config config_stage1_stitchvit.json --visualize

# Then, run Stage 2 with its config (it will use the weights from Stage 1)
python -m src.main final --config config_stage2_stitchvit.json --visualize
```

### Additional Flags

*   `--config <path>`: Use a custom configuration file.
*   `--small-test`: Run on a small subset for debugging.
*   `--visualize`: Generate and save prediction plots after training.
*   `--early-stop-patience <int>`: Override the early stopping patience from the config file.

## Tools & Scripts

Useful scripts are located in `tools/scripts/`.

*   **Compare Models:**
    Generates a visual comparison of predictions from two different models.
    ```bash
    python -m tools.scripts.compare_models --log-dir1 <path_to_model1_logs> --log-dir2 <path_to_model2_logs>
    ```

*   **Plot Training Curves:**
    Generates loss and metric curves from a training log directory.
    ```bash
    python -m tools.scripts.plot_training_curves --log-dir <path_to_log_directory>
    ```

*   **Visualize Predictions:**
    Loads a trained model to generate and save prediction images on a test set.
    ```bash
    python -m tools.scripts.visualize_predictions --log-dir <path_to_log_directory>
    ```

## GUI Application

The project includes a graphical user interface for interactive prediction and analysis.

**Launch the GUI:**
```bash
python app.py
```

## Results

After running a training pipeline, the outputs are saved in the `logs/` and `models/` directories, organized by the `type` and `experiment_name` from your config.

For an experiment named `"cto"` of type `"Human"`, you will find:
*   **Model Checkpoints:** `models/checkpoints/Human/cto/baseline.pth` and `models/checkpoints/Human/cto/final.pth`.
*   **Training Logs & Visuals:** `logs/Human/cto/`. This directory contains:
    *   `_detailed_curves.png`: Dice and loss curves.
    *   `_loss_curve.png`: Combined loss curve.
    *   `baseline_predictions.png`: Visual results from the baseline model.
    *   `final_predictions.png`: Visual results from the final Mean Teacher model.
    *   `training.log`: A text file with detailed logs.

## Technical Details

*   **Core Model:** CTO-Net with a Res2Net-50 backbone.
*   **Semi-Supervised Strategy:** Mean Teacher, where the Teacher model's weights are an Exponential Moving Average (EMA) of the Student's weights. The Student learns from both labeled data (supervised loss) and the Teacher's predictions on unlabeled data (consistency loss).
*   **Loss Function:** A combination of Binary Cross-Entropy (BCE), Dice Loss, and a custom Boundary Loss.
