#!/usr/bin/env python3
"""
Optimized YOLO Training for High-End GPUs
Maximizes GPU utilization for RTX 4000 Ada (21GB VRAM)
"""

import argparse
import torch
from ultralytics import YOLO
from pathlib import Path
import random
import shutil
import os
import logging
from datetime import datetime

def optimize_gpu_settings():
    """Optimize GPU settings for maximum performance"""
    if not torch.cuda.is_available():
        return False
    
    # Set memory fraction to use most of GPU memory
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print(f"GPU Optimizations Enabled")
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"   CUDA Capability: {torch.cuda.get_device_capability()}")
    
    return True

def get_optimal_batch_size(vram_gb: float, model_size: str):
    """Calculate optimal batch size based on VRAM and model size"""
    # Base batch sizes for different models on 21GB VRAM
    base_batches = {
        'n': 128,  # Nano can handle very large batches
        's': 96,   # Small 
        'm': 64,   # Medium
        'l': 48,   # Large
        'x': 32    # Extra Large
    }
    
    # Scale based on actual VRAM
    scale_factor = vram_gb / 21.0
    optimal_batch = int(base_batches.get(model_size, 64) * scale_factor)
    
    # Ensure it's a multiple of 8 for optimal tensor core usage
    optimal_batch = (optimal_batch // 8) * 8
    
    return max(8, optimal_batch)

def create_subset_dataset(original_data_path: str, subset_ratio: float = 0.2, output_dir: str = "subset_dataset"):
    """Create a subset of the dataset using symbolic links"""
    original_path = Path(original_data_path).parent
    subset_path = original_path / output_dir
    
    print(f"📦 Creating {subset_ratio*100}% subset dataset...")
    print(f"   Original: {original_path}")
    print(f"   Subset: {subset_path}")
    
    # Create subset directory structure
    for split in ['train', 'val', 'test']:
        (subset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (subset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    total_original = 0
    total_subset = 0
    
    for split in ['train', 'val', 'test']:
        original_img_dir = original_path / 'images' / split
        original_label_dir = original_path / 'labels' / split
        subset_img_dir = subset_path / 'images' / split
        subset_label_dir = subset_path / 'labels' / split
        
        if not original_img_dir.exists():
            continue
            
        # Get all image files
        image_files = list(original_img_dir.glob('*.jpg'))
        total_original += len(image_files)
        
        # Calculate subset size
        subset_size = max(1, int(len(image_files) * subset_ratio))
        
        # Randomly sample files
        random.seed(42)  # For reproducibility
        subset_files = random.sample(image_files, subset_size)
        total_subset += len(subset_files)
        
        print(f"   {split}: {len(image_files)} -> {len(subset_files)} images")
        
        # Create symbolic links for images and labels
        for img_file in subset_files:
            # Image symlink
            img_link = subset_img_dir / img_file.name
            if img_link.exists():
                img_link.unlink()
            img_link.symlink_to(img_file.resolve())
            
            # Label symlink
            label_file = original_label_dir / f"{img_file.stem}.txt"
            label_link = subset_label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                if label_link.exists():
                    label_link.unlink()
                label_link.symlink_to(label_file.resolve())
    
    # Create new data.yaml
    subset_yaml = subset_path / 'data.yaml'
    yaml_content = f"""# VeRi Dataset Subset ({subset_ratio*100}%) for YOLO Car Detection
path: {subset_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names: ['Car']  # class names
"""
    
    with open(subset_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Subset dataset created!")
    print(f"   Total images: {total_original} -> {total_subset} ({total_subset/total_original*100:.1f}%)")
    print(f"   Config: {subset_yaml}")
    
    return str(subset_yaml)

def setup_logging_and_folders(output_dir: str, model_size: str, subset_ratio: float | None = None):
    """Setup logging and create organized folder structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create specific run directory
    subset_suffix = f"_subset{int(subset_ratio*100)}pct" if subset_ratio else "_full"
    run_name = f"yolo12m{model_size}{subset_suffix}_{timestamp}"
    run_dir = output_path / run_name
    run_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    weights_dir = run_dir / "weights"
    
    logs_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = logs_dir / f"training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created organized folder structure:")
    logger.info(f"   Run directory: {run_dir}")
    logger.info(f"   Logs: {logs_dir}")
    logger.info(f"   Plots: {plots_dir}")
    logger.info(f"   Weights: {weights_dir}")
    
    return run_dir, logs_dir, plots_dir, weights_dir, logger

def organize_training_outputs(run_dir: Path, logs_dir: Path, plots_dir: Path, weights_dir: Path, logger):
    """Organize training outputs into structured folders"""
    
    # The YOLO training creates a folder structure like: project/name/
    yolo_output_dir = run_dir
    
    # Move/copy plots if they exist
    plots_source = yolo_output_dir
    for plot_file in plots_source.glob("*.png"):
        if plot_file.name not in ["confusion_matrix.png", "confusion_matrix_normalized.png", 
                                  "F1_curve.png", "P_curve.png", "PR_curve.png", "R_curve.png",
                                  "results.png", "train_batch*.jpg", "val_batch*.jpg"]:
            continue
        dest_file = plots_dir / plot_file.name
        if plot_file.exists() and not dest_file.exists():
            shutil.copy2(plot_file, dest_file)
            logger.info(f"Copied plot: {plot_file.name} -> {dest_file}")
    
    # Copy weights to our organized weights folder
    weights_source = yolo_output_dir / "weights"
    if weights_source.exists():
        for weight_file in weights_source.glob("*.pt"):
            dest_file = weights_dir / weight_file.name
            if not dest_file.exists():
                shutil.copy2(weight_file, dest_file)
                logger.info(f"Copied weights: {weight_file.name} -> {dest_file}")
    
    # Create summary file
    summary_file = run_dir / "training_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"YOLO Training Summary\n")
        f.write(f"====================\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run Directory: {run_dir}\n")
        f.write(f"Logs: {logs_dir}\n")
        f.write(f"Plots: {plots_dir}\n")
        f.write(f"Weights: {weights_dir}\n")
        f.write(f"\nFiles Created:\n")
        f.write(f"- Logs: {len(list(logs_dir.glob('*.log')))} files\n")
        f.write(f"- Plots: {len(list(plots_dir.glob('*.png')))} files\n")
        f.write(f"- Weights: {len(list(weights_dir.glob('*.pt')))} files\n")
    
    logger.info(f"Created training summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Optimized YOLO training for high-end GPUs')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='Model size (n=nano recommended for speed)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--output', default='optimized_training', help='Output directory')
    parser.add_argument('--batch-size', type=int, help='Override auto batch size')
    parser.add_argument('--subset-ratio', type=float, default=None, 
                       help='Use subset of dataset (0.1 = 10%, 0.2 = 20%, etc.)')
    
    args = parser.parse_args()
    
    # GPU optimization
    gpu_available = optimize_gpu_settings()
    if not gpu_available:
        print("No GPU available - this script is optimized for GPU training")
        return
    
    # Create subset dataset if requested
    data_path = args.data
    if args.subset_ratio:
        if not (0 < args.subset_ratio <= 1.0):
            print("Subset ratio must be between 0 and 1.0")
            return
        print(f"Creating {args.subset_ratio*100}% subset of dataset for faster training...")
        data_path = create_subset_dataset(args.data, args.subset_ratio)
    else:
        print(f"Using full dataset: {args.data}")
    
    # Setup logging and organized folders
    run_dir, logs_dir, plots_dir, weights_dir, logger = setup_logging_and_folders(
        args.output, args.model_size, args.subset_ratio
    )
    
    # Calculate optimal batch size
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if args.batch_size:
        batch_size = args.batch_size
        print(f"Using manual batch size: {batch_size}")
    else:
        batch_size = get_optimal_batch_size(vram_gb, args.model_size)
        print(f"Auto-calculated optimal batch size: {batch_size}")
    
    # Load model
    model_name = f'yolo12{args.model_size}.pt'
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    # Optimized training parameters for speed
    train_args = {
        'data': data_path,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': batch_size,
        'device': 0,  # Use first GPU explicitly
        'workers': 16,  # Increase workers for faster data loading
        'project': str(run_dir.parent),
        'name': run_dir.name,
        'exist_ok': True,
        'cache': 'ram',  # Cache dataset on disk for faster access without overwhelming RAM
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'save_period': 10,
        'val': True,
        'plots': True,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'seed': 42,
        'single_cls': True,
        'freeze': ['backbone.']
    }
    
    logger.info(f"Starting optimized training...")
    logger.info(f"   Model: YOLO11{args.model_size.upper()}")
    logger.info(f"   Dataset: {data_path}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Image size: {args.imgsz}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Disk caching: Enabled")
    logger.info(f"   Workers: 16")
    if args.subset_ratio:
        logger.info(f"   Dataset subset: {args.subset_ratio*100}%")
    
    # Monitor GPU before training
    logger.info(f"GPU Status:")
    logger.info(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    logger.info(f"   Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
    # Start training
    try:
        results = model.train(**train_args)
        
        logger.info(f"Training completed!")
        
        # Organize training outputs
        organize_training_outputs(run_dir, logs_dir, plots_dir, weights_dir, logger)
        
        # Find best model
        best_model_path = run_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            logger.info(f"Best model saved: {best_model_path}")
            
            # Quick validation
            logger.info(f"Running validation...")
            val_results = model.val(data=data_path, device=0)
            logger.info(f"   mAP50: {val_results.box.map50:.4f}")
            logger.info(f"   mAP50-95: {val_results.box.map:.4f}")
            
            # Log final results
            logger.info(f"All outputs organized in: {run_dir}")
            logger.info(f"   Plots: {plots_dir}")
            logger.info(f"   Weights: {weights_dir}")
            logger.info(f"   Logs: {logs_dir}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"GPU Out of Memory!")
            logger.error(f"   Current batch size: {batch_size}")
            logger.error(f"   Try reducing batch size: --batch-size {batch_size // 2}")
        else:
            logger.error(f"Training error: {e}")
    
    except KeyboardInterrupt:
        logger.info(f"Training interrupted by user")
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        logger.info(f"GPU cache cleared")

if __name__ == "__main__":
    main() 