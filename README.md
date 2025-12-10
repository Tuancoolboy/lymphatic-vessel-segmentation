# Lymphatic Vessel Segmentation with CTO-Net

This project enhances and extends a foundational supervised learning pipeline by introducing a **semi-supervised** approach for segmenting lymphatic vessels. The core of this project is the implementation of the **Mean Teacher** algorithm to leverage large amounts of unlabeled video data, alongside the evaluation of advanced architectures like **CTO-Net** and **CTO Stitch-ViT**.

**Presentation Video:** [Link to your 30-minute presentation video will be here]

## Table of Contents
- [Project Background](#project-background)
- [Project Team and Task Division](#project-team-and-task-division)
- [Key Features](#key-features)
- [Models](#models)
- [Dataset Details](#dataset-details)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Tools & Scripts](#tools--scripts)
- [GUI Application](#gui-application)
- [Experimental Results and Analysis](#experimental-results-and-analysis)
- [Project Analysis: Successes and Limitations](#project-analysis-successes-and-limitations)
- [Key Learnings](#key-learnings)

## Project Team and Task Division

This project was a collaborative effort between two members. The tasks were divided to ensure comprehensive coverage of all project aspects, from model development to data management and final analysis.

| Member Name              | Student ID | Primary Responsibilities                                                                                                                               |
| ------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Vũ Hải Tuấn**          | 2353280    | <ul><li>Led the implementation and training of the core model architectures: **CTO-Net** and the experimental **CTO Stitch-ViT**.</li><li>Developed the supervised training pipeline (Stage 1).</li></ul> |
| **Lê Hoàng Chí Vĩ**      | 2353336    | <ul><li>Led the implementation and refinement of the semi-supervised learning pipeline (Stage 2) using the **Mean Teacher** algorithm.</li><li>Developed data processing and results visualization scripts.</li></ul> |

**Joint Contributions (Lê Hoàng Chí Vĩ & Vũ Hải Tuấn):**
*   Design and tuning of the composite loss function (BCE, Dice, and Boundary Loss).
*   Analysis of experimental results to identify model strengths and weaknesses.
*   Preparation of the final report, documentation, and presentation.
 
## Project Background

This work is a direct continuation of the project **"Deep learning in Medical Researches: Lymphatic Vessel Segmentation"**. The original project established a fully supervised pipeline and introduced foundational models like UNet++ for this task.

We extend this work by integrating a semi-supervised learning paradigm to reduce the dependency on manually annotated data. The interactive GUI application (`app.py`) was originally developed by **Vũ Hoàng Tùng** as part of the foundational project, and we have adapted it for our new pipeline. We extend our sincere gratitude for this significant contribution.

## Key Features

*   **Primary Architecture (CTO-Net):** A high-performance segmentation model with a **Res2Net-50** backbone designed for detailed feature extraction.
*   **Semi-Supervised Learning:** Leverages unlabeled video data via the **Mean Teacher** method to improve accuracy and reduce annotation effort.
*   **Boundary-Aware Loss:** Combines Dice Loss and Boundary Loss for sharp and accurate detection of vessel edges.
*   **2-Stage Training Pipeline:**
    1.  **Stage 1:** Train a baseline model on a small set of labeled data.
    2.  **Stage 2:** Refine the model using Mean Teacher with both labeled and a large amount of unlabeled data.
*   **Flexible Configuration:** Easily tune parameters for each stage via dedicated JSON files.
*   **GUI for Analysis:** An interactive tool for visualizing predictions and measuring vessel diameters.

## Models

This project supports three models:

1.  **CTO-Net (Default):** The primary model of this project. It uses a **Res2Net-50** backbone and a custom architecture designed for vessel segmentation. The configuration for this model is `cto`.
2.  **CTO Stitch-ViT (Experimental):** An enhanced version of CTO-Net that incorporates Vision Transformer (ViT) blocks with a stitching mechanism. This model aims to capture more global contextual features. The configuration for this is `cto_stitchvit`.
3.  **UNet++:** A well-known architecture for medical image segmentation, available as an alternative which was used in previous project (root project). It can be configured by setting the model `name` to `unetpp` in the JSON config.
 
## Dataset Details

This project utilizes two main datasets for lymphatic vessel segmentation. All data was provided by the lab of researcher **Lê Quỳnh Trâm**, who also performed all manual annotations for the labeled data.

### Human Dataset
*   **Source:** Lab of researcher Lê Quỳnh Trâm.
*   **Content:** Video recordings of lymphatic vessels in human tissue.
*   **Statistics:**
    *   **Labeled Data:** 33 images with manual annotations.
    *   **Unlabeled Data:** 3 videos, from which frames are extracted.
*   **Processing:** Labeled data (`.json` annotations) are converted into binary masks. For unlabeled data, frames are extracted from the videos to be used in the semi-supervised training stage.

### Rat Dataset
*   **Source:** Lab of researcher Lê Quỳnh Trâm.
*   **Content:** Video recordings of lymphatic vessels in rat tissue.
*   **Statistics:**
    *   **Labeled Data:** 33 images with manual annotations.
    *   **Unlabeled Data:** 8 videos, from which frames are extracted.
*   **Processing:** Similar to the Human dataset, masks are generated from annotations, and frames are extracted from videos.


## Project Structure
```text
.
├── app.py                      # GUI Application entry point
├── config_stage1.json          # Config for Stage 1 (CTO-Net)
├── config_stage2.json          # Config for Stage 2 (CTO-Net)
├── requirements.txt
├── data/
│   ├── annotated/              # Labeled images and JSON annotations
│   ├── masks/                  # Generated binary masks
│   ├── frames/                 # Extracted frames from unlabeled videos
│   └── video/                  # Raw unlabeled videos
├── logs/                       # Training logs, curves, and prediction images
├── models/                     # Saved model checkpoints (.pth files)
└── src/                        # Source code
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
    Open the configuration files (e.g., `config_stage1.json`). Set the `"type"` field to either `"Human"` or `"Rat"` to specify the dataset.

    **Example (`config_stage1.json`):**
    ```json
    {
        "type": "Human",
        ...
    }
    ```

## Usage

### Data Preparation

1.  **Labeled Data:** Place images and `.json` files in `data/annotated/<type>/`.
2.  **Convert Annotations to Masks:**
    ```bash
    python -m tools.scripts.convert_json_to_mask --input data/annotated --output data/masks
    ```
3.  **Unlabeled Data:** Place videos in `data/video/<type>/`.
4.  **Extract Frames from Videos:**
    ```bash
    python -m tools.scripts.extract_frames --video_dir data/video --output_dir data/frames --fps 1
    ```

### Training Pipeline

Run the full pipeline (Stage 1 and Stage 2) for the default **CTO-Net** model:
```bash
python -m src.main all --visualize
```
This uses `config_stage1.json` and `config_stage2.json` by default.

To run with the experimental **Stitch-ViT** model, specify its configs:
```bash
# Stage 1
python -m src.main baseline --config config_stage1_stitchvit.json --visualize
# Stage 2
python -m src.main final --config config_stage2_stitchvit.json --visualize
```

## Tools & Scripts
Useful scripts are located in `tools/scripts/`.

*   **Compare Models:**
    ```bash
    python -m tools.scripts.compare_models --log-dir1 <path_to_model1_logs> --log-dir2 <path_to_model2_logs>
    ```
*   **Plot Training Curves:**
    ```bash
    python -m tools.scripts.plot_training_curves --log-dir <path_to_log_directory>
    ```
*   **Visualize Predictions:**
    ```bash
    python -m tools.scripts.visualize_predictions --log-dir <path_to_log_directory>
    ```

## GUI Application
The project includes a graphical user interface for interactive prediction and analysis.
**Launch the GUI:**
```bash
python app.py
```

## Experimental Results and Analysis

The model's performance was evaluated using Dice Score, Intersection over Union (IoU), Precision, and Recall. The training process involves two stages: baseline training on labeled data and semi-supervised refinement using the Mean Teacher method.

### Training Curves
The training curves below show the progression of the Dice score and loss for both the baseline (Stage 1) and final (Stage 2) models on the Human dataset. We can observe that the metrics stabilize, indicating successful convergence.

*(Note: These are example images from the project's output logs.)*

**Stage 1 - Baseline Training Curves (`cto` model):**
![Baseline Curves](logs/Human/cto/Human_20251122_234019_detailed_curves.png)

**Stage 2 - Final Model Training Curves (`cto` model):**
![Final Curves](logs/Human/cto/Human_20251123_000145_detailed_curves.png)

**Analysis:**
- The Stage 1 curves show the model rapidly learning from the 33 labeled images.
- The Stage 2 curves demonstrate further stabilization and slight improvements as the model learns from thousands of unlabeled frames, guided by the Mean Teacher. This highlights the effectiveness of the semi-supervised approach.

### Prediction Quality
Visual inspection shows a significant improvement from the baseline model to the final model. The final model produces much cleaner segmentations with fewer false positives.

**Baseline Model Predictions (Stage 1):**
![Baseline Predictions](logs/Human/cto/baseline_predictions.png)

**Final Model Predictions (Stage 2):**
![Final Predictions](logs/Human/cto/final_predictions.png)

**Analysis:**
- The baseline model correctly identifies the main vessel structures but suffers from noise and disconnected artifacts.
- The final model, after Mean Teacher training, produces significantly smoother and more coherent vessel segmentations. The consistency loss used in Stage 2 helps the model learn the underlying structure of the vessels, resulting in more robust predictions.

## Project Analysis: Successes and Limitations

### Successes
1.  **Effective Semi-Supervised Learning:** The Mean Teacher pipeline successfully leveraged a large amount of unlabeled video data to significantly improve segmentation quality, moving from noisy baseline predictions to clean, coherent final masks. This demonstrates the viability of this approach for reducing manual annotation costs.
2.  **High-Quality Segmentation:** The combination of the Res2Net backbone, custom CTO-Net architecture, and a boundary-aware loss function proved effective at producing sharp and accurate segmentations of lymphatic vessels.
3.  **End-to-End Pipeline:** The project provides a complete, usable pipeline, from raw video processing and mask generation to training, evaluation, and interactive analysis with a GUI.

### Limitations
1.  **Experimental Stitch-ViT:** The `cto_stitchvit` model is still experimental. While it shows promise in capturing global context, it requires more extensive tuning and evaluation to definitively prove its benefits over the standard CTO-Net.
2.  **Dependence on Pseudo-Labels:** The performance of the semi-supervised stage is highly dependent on the quality of the pseudo-labels generated by the teacher model. In cases of significant domain shift between labeled and unlabeled data, this could potentially degrade performance.
3.  **Computational Cost:** The two-stage training process, especially with a large number of unlabeled frames, is computationally intensive and time-consuming.

## Key Learnings

This project provided valuable hands-on experience in several key areas of deep learning and computer vision:

1.  **Semi-Supervised Learning:** Gained a deep, practical understanding of the Mean Teacher method, including the implementation of consistency loss and Exponential Moving Average (EMA) for weight updates.
2.  **Advanced Model Architecture:** Learned to implement and integrate complex architectures, including the multi-scale Res2Net backbone and experimental Vision Transformer blocks.
3.  **Loss Function Engineering:** Acquired experience in designing and balancing a composite loss function (BCE, Dice, Boundary) to optimize for specific segmentation characteristics like edge sharpness.
4.  **Building a Full ML Pipeline:** Developed skills in creating an end-to-end workflow, encompassing data preprocessing, model training, results visualization, and building an interactive application for real-world use.
5.  **Reproducibility and Configuration:** Learned the importance of structured code, clear documentation, and external configuration files (`.json`) for ensuring experiments are easy to reproduce and modify.