"""Main entry point for the semi-supervised lymphatic vessel segmentation pipeline.

This script orchestrates a 2-stage training process:
1.  **Baseline Training (Stage 1):** A model is trained on a small, fully-labeled dataset
    using k-fold cross-validation to create a robust supervised baseline.
    The resulting model is saved as `baseline.pth`.

2.  **Final Training (Stage 2):** The `baseline.pth` model is used to initialize a
    **Mean Teacher** setup. The "student" model is trained on both the labeled dataset
    and a larger unlabeled dataset, learning from the "teacher" model's predictions
    on unlabeled data. This semi-supervised approach improves generalization.

Example usage:
    # Run Stage 1 (train baseline model)
    python -m src.main baseline --config config_stage1.json

    # Run Stage 2 (train final model using Mean Teacher)
    python -m src.main final --config config_stage2.json

    # Run the complete two-stage pipeline sequentially
    python -m src.main all
"""
import os
import json
import argparse
import platform
from typing import Optional
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, random_split

from src.config import ExperimentConfig
from src.models.model_factory import get_model, save_checkpoint, load_checkpoint
from src.data.datasets import LabeledVesselDataset, VideoDataset
from src.training.trainer import train_kfold
from src.training.mean_teacher import MeanTeacherTrainer
from src.utils.logging import TrainingLogger
from src.utils.augment import (
    create_train_transform,
    create_val_transform,
    create_strong_transform,
    create_weak_transform
)
from src.visualization import visualize_predictions, visualize_evaluation_table

try:
    import psutil
except ImportError:
    psutil = None

def load_config(config_path: str) -> ExperimentConfig:
    """
    Loads and validates the experiment configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        ExperimentConfig: A validated configuration object.
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}. Using default configuration.")
        return ExperimentConfig()
        
    with open(config_path) as f:
        config_dict = json.load(f)
    return ExperimentConfig.from_dict(config_dict)

def train_baseline(config: ExperimentConfig, logger: TrainingLogger):
    """
    Executes Stage 1 of the pipeline: Training a baseline model.

    This function sets up the dataset and data augmentations, then uses k-fold
    cross-validation to train a model on the labeled data. The best-performing
    model across all folds is saved as 'baseline.pth', which serves as the
    starting point for Stage 2.

    Args:
        config (ExperimentConfig): The configuration for this training stage.
        logger (TrainingLogger): Logger for recording metrics and artifacts.

    Returns:
        The trained baseline model.
    """
    print("\n=== Stage 1: Training Baseline Model (Supervised) ===")
    
    # Setup data augmentation pipelines for training and validation
    train_transform = create_train_transform(p=config.augmentation.train_prob)
    val_transform = create_val_transform()
    
    # Create the dataset for labeled images
    dataset = LabeledVesselDataset(
        image_root=config.paths.labeled_dir,
        mask_dir=config.paths.labeled_masks_dir,
        transform=train_transform,
        target_size=config.data.image_size
    )
    
    if len(dataset) == 0:
        raise RuntimeError(f"No labeled data found in '{config.paths.labeled_dir}'. Please check the path.")
        
    # Train the model using k-fold cross-validation for robustness
    model = train_kfold(
        model_factory=lambda: get_model(config.model),
        dataset=dataset,
        model_config=config.model,
        config=config.training,
        logger=logger
    )
    
    # Save the final trained model
    save_path = os.path.join(config.paths.model_dir, "baseline.pth")
    save_checkpoint(model, save_path)
    if logger:
        logger.log_model_path(save_path)
    print(f"✓ Saved baseline model to {save_path}")
    
    return model

def train_final(config: ExperimentConfig, logger: TrainingLogger):
    """
    Executes Stage 2 of the pipeline: Training a final model using Mean Teacher.

    This function implements the semi-supervised learning phase. It loads the
    pre-trained 'baseline.pth' model to initialize both a "student" and a "teacher"
    model. The student is trained on labeled data (supervised loss) and unlabeled
    data (consistency loss against the teacher's predictions). The teacher's weights
    are an exponential moving average (EMA) of the student's weights, providing a more
    stable and accurate target.

    Args:
        config (ExperimentConfig): The configuration for this training stage.
        logger (TrainingLogger): Logger for recording metrics and artifacts.

    Returns:
        The trained final (teacher) model.
    """
    print("\n=== Stage 2: Training Final Model (Semi-Supervised with Mean Teacher) ===")

    # Define data augmentation pipelines for weak and strong augmentations,
    # crucial for consistency regularization in Mean Teacher
    train_transform = create_train_transform(p=config.augmentation.train_prob)
    val_transform = create_val_transform()
    weak_transform = create_weak_transform(p=0.3)
    strong_transform = create_strong_transform(p=0.5)

    # --- Labeled Dataset Setup ---
    # Create two versions of the labeled dataset: one for training (with augmentations)
    # and one for validation (without augmentations)
    train_labeled_ds = LabeledVesselDataset(
        image_root=config.paths.labeled_dir,
        mask_dir=config.paths.labeled_masks_dir,
        transform=train_transform,
        target_size=config.data.image_size,
    )
    if len(train_labeled_ds) == 0:
        raise RuntimeError(f"No labeled data found in '{config.paths.labeled_dir}'.")

    val_ds = LabeledVesselDataset(
        image_root=config.paths.labeled_dir,
        mask_dir=config.paths.labeled_masks_dir,
        transform=val_transform,
        target_size=config.data.image_size,
    )

    # Split labeled data into training and validation sets
    total_labeled = len(train_labeled_ds)
    val_len = max(1, int(0.1 * total_labeled))
    train_labeled_len = total_labeled - val_len
    train_indices, val_indices = random_split(range(total_labeled), [train_labeled_len, val_len])
    train_labeled_set = torch.utils.data.Subset(train_labeled_ds, train_indices)
    val_set = torch.utils.data.Subset(val_ds, val_indices)

    # --- Unlabeled Dataset Setup ---
    unlabeled_ds = None
    unlabeled_dirs = [config.paths.unlabeled_dir, config.paths.labeled_dir]
    for unlabeled_dir in unlabeled_dirs:
        if os.path.exists(unlabeled_dir) and os.path.isdir(unlabeled_dir):
            unlabeled_ds = VideoDataset(
                image_root=unlabeled_dir,
                transform=weak_transform,
                target_size=config.data.image_size,
            )
            if len(unlabeled_ds) > 0:
                print(f"✓ Found {len(unlabeled_ds)} unlabeled images in '{unlabeled_dir}' for Mean Teacher.")
                break
    
    if unlabeled_ds is None or len(unlabeled_ds) == 0:
        print("Warning: No unlabeled data found. Mean Teacher will proceed using only labeled data.")
        unlabeled_ds = None

    # --- DataLoaders Setup ---
    labeled_loader = DataLoader(
        train_labeled_set, batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers
    ) if unlabeled_ds is not None else None
    val_loader = DataLoader(
        val_set, batch_size=config.training.batch_size, shuffle=False, num_workers=config.data.num_workers
    )

    # --- Model and Trainer Initialization ---
    base_model = get_model(config.model)
    
    # Stage 2 requires the baseline model from Stage 1 as a starting point
    baseline_path = os.path.join(config.paths.model_dir, "baseline.pth")
    if not os.path.exists(baseline_path):
        raise RuntimeError(
            f"Baseline model not found at {baseline_path}.\n"
            "Run Stage 1 first to generate the baseline model."
        )
    print(f"Loading baseline model from {baseline_path}...")
    base_model = load_checkpoint(base_model, baseline_path, device=config.training.device)
    print("✓ Initialized student and teacher models with baseline weights.")
    
    # Create student and teacher models from the loaded baseline
    student_model = base_model
    teacher_model = deepcopy(base_model)
    
    # The teacher model is not trained via backpropagation, its parameters are updated
    # as an EMA of the student's parameters.
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    trainer = MeanTeacherTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        optimizer=optimizer,
        model_config=config.model,
        config=config.training,
        logger=logger,
        device=config.training.device
    )
    
    # Start the semi-supervised training process
    model = trainer.train(labeled_loader, unlabeled_loader, val_loader, scheduler)
    print("✓ Mean Teacher training completed. The final model is the teacher model.")

    # Save the final model (the teacher model is typically more accurate)
    os.makedirs(config.paths.model_dir, exist_ok=True)
    save_path = os.path.join(config.paths.model_dir, "final.pth")
    save_checkpoint(model, save_path)
    if logger:
        logger.log_model_path(save_path)
    print(f"Saved final model to {save_path}")

    return model

def log_system_info(logger, device):
    """
    Logs essential system and hardware information for reproducibility.

    Args:
        logger (TrainingLogger): The logger instance.
        device (str): The computation device ('cpu', 'cuda', 'mps').
    """
    logger.log_message("--- System Information ---")
    logger.log_message(f"Operating System: {platform.system()} {platform.release()}")
    if psutil:
        logger.log_message(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    if device == "cuda" and torch.cuda.is_available():
        logger.log_message(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.log_message("Device: Apple MPS")
    else:
        logger.log_message(f"Device: CPU ({platform.processor()})")
        if psutil:
            logger.log_message(f"CPU Cores: {psutil.cpu_count(logical=True)}")
    logger.log_message("--------------------------")

def run_visualize_eval(config: ExperimentConfig, specific_log_dir: Optional[str] = None):
    """
    Generates and saves a text-based evaluation summary table from training logs.

    This function locates the `metrics.csv` file in a specified or the most recent
    log directory and creates a formatted summary table.

    Args:
        config (ExperimentConfig): The experiment configuration.
        specific_log_dir (Optional[str]): Path to a specific log directory.
            If None, the most recent log directory will be used.
    """
    print(f"\n=== Generating Evaluation Table for type '{config.type}' ===")

    target_log_dir = specific_log_dir
    if target_log_dir is None:
        base_log_dir = config.paths.log_dir
        if not os.path.isdir(base_log_dir):
            print(f"Log directory not found: {base_log_dir}"); return
        try:
            # Find the most recently modified log directory
            target_log_dir = max(
                [os.path.join(base_log_dir, d) for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))],
                key=os.path.getmtime
            )
        except ValueError:
            print(f"No training logs found in {base_log_dir}"); return
    
    if not os.path.isdir(target_log_dir):
        print(f"Specified log directory does not exist: {target_log_dir}"); return

    metrics_file = os.path.join(target_log_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"metrics.csv not found in the specified log directory: {target_log_dir}"); return

    print(f"Using metrics from: {metrics_file}")
    output_path = os.path.join(config.paths.model_dir, f"evaluation_summary_{os.path.basename(target_log_dir)}.txt")
    
    visualize_evaluation_table(metrics_file, output_path)
    print(f"✓ Successfully generated evaluation table at: {output_path}")

def main():
    """
    Main entry point for the command-line interface.

    Parses arguments to determine which stage of the pipeline to run (`baseline`,
    `final`, `all`, or `visualize_eval`). It handles configuration loading,
    environment setup (e.g., small tests, device selection), and orchestrates
    the calls to the appropriate training or utility functions.
    """
    parser = argparse.ArgumentParser(description="Lymphatic Vessel Segmentation Training Pipeline")
    parser.add_argument(
        "stage",
        choices=["baseline", "final", "all", "visualize_eval"],
        help="The training stage to run: 'baseline' for Stage 1, 'final' for Stage 2, 'all' to run both, 'visualize_eval' to generate summary."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a custom config file. Overrides stage-specific defaults."
    )
    parser.add_argument(
        "--small-test", action="store_true",
        help="Run on a small data subset for quick debugging."
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Specify a log directory for the 'visualize_eval' stage."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate and save prediction visualizations after training."
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="Override the early stopping patience value from the config file."
    )
    args = parser.parse_args()
    
    # --- Configuration Loading ---
    # Load the appropriate config file based on the selected stage, unless a
    # specific config file is provided via the --config flag
    if args.config:
        config = load_config(args.config)
    elif args.stage == "baseline":
        config = load_config("config_stage1.json")
    elif args.stage in ["final", "visualize_eval"]:
        config = load_config("config_stage2.json")
    elif args.stage == "all":
        # For the 'all' stage, we start with Stage 1 config. Stage 2 config will be loaded later
        config = load_config("config_stage1.json")
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    # --- Environment & Parameter Overrides ---
    # On Windows, PyTorch multiprocessing can be problematic. Force single-worker data loading
    if platform.system() == "Windows" and config.data.num_workers > 0:
        print("Windows detected. Setting num_workers to 0 to prevent multiprocessing issues.")
        config.data.num_workers = 0

    if args.early_stop_patience is not None:
        config.training.early_stop_patience = args.early_stop_patience
    
    if args.small_test:
        # Override data paths to point to a smaller test sample for debugging
        candidate_annot = os.path.join(config.paths.labeled_dir, "test_sample")
        candidate_flat = os.path.join("data", "test_sample")
        labeled_test_dir = candidate_annot if os.path.isdir(candidate_annot) else (candidate_flat if os.path.isdir(candidate_flat) else config.paths.labeled_dir)
        unlabeled_test_dir = (os.path.join(config.paths.unlabeled_dir, "test_sample") if os.path.isdir(os.path.join(config.paths.unlabeled_dir, "test_sample")) else (os.path.join("data", "test_sample") if os.path.isdir(os.path.join("data", "test_sample")) else config.paths.unlabeled_dir))
        config.paths.labeled_dir = labeled_test_dir
        config.paths.unlabeled_dir = unlabeled_test_dir
        print(f"--- Running in SMALL TEST mode with data from: {config.paths.labeled_dir} ---")
        
    # --- Logger and Device Setup ---
    logger = None
    if args.stage in ["baseline", "final", "all"]:
        # Auto-detect best available device (GPU, MPS, or CPU)
        device = config.training.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
        config.training.device = device
        
        # Initialize logger for the first stage
        logger = TrainingLogger(config.paths.log_dir)
        log_system_info(logger, device)
        logger.log_config(config.to_dict())

    # --- Pipeline Stage Execution ---
    try:
        # Stage 1 Execution
        if args.stage in ["baseline", "all"]:
            model = train_baseline(config, logger)
            if args.visualize and model is not None:
                ds = LabeledVesselDataset(image_root=config.paths.labeled_dir, mask_dir=config.paths.labeled_masks_dir, transform=create_val_transform(), target_size=config.data.image_size)
                fig = visualize_predictions(model, ds, num_samples=4, device=config.training.device)
                fig_path = os.path.join(config.paths.model_dir, "baseline_predictions.png")
                fig.savefig(fig_path); print(f"Saved baseline visualizations to {fig_path}")
        
        # Stage 2 Execution
        if args.stage in ["final", "all"]:
            if args.stage == "all":
                # When running the 'all' pipeline, load the config for Stage 2
                config_path_stage2 = args.config if args.config else "config_stage2.json"
                config = load_config(config_path_stage2)
                # Re-apply CLI overrides
                if args.early_stop_patience is not None: config.training.early_stop_patience = args.early_stop_patience
                if platform.system() == "Windows": config.data.num_workers = 0

                # Set up a new logger and device for stage 2
                device = config.training.device
                if device == "auto": device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
                config.training.device = device
                logger = TrainingLogger(config.paths.log_dir)
                log_system_info(logger, device)
                logger.log_config(config.to_dict())

            model = train_final(config, logger)
            if args.visualize and model is not None:
                ds = LabeledVesselDataset(image_root=config.paths.labeled_dir, mask_dir=config.paths.labeled_masks_dir, transform=create_val_transform(), target_size=config.data.image_size)
                fig = visualize_predictions(model, ds, num_samples=4, device=config.training.device)
                fig_path = os.path.join(config.paths.model_dir, "final_predictions.png")
                fig.savefig(fig_path); print(f"Saved final visualizations to {fig_path}")

        # Utility Stage Execution
        if args.stage == "visualize_eval":
            run_visualize_eval(config, specific_log_dir=args.log_dir)
            
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred during pipeline execution: {str(e)}")
        if logger:
            logger.log_message(f"FATAL ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()