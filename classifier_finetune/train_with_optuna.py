import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler
import logging
from datetime import datetime
import json
from pathlib import Path
import sys

# Import our existing functions
from finetune_models import create_model, get_data_transforms, VehicleDataset, train_epoch, validate_epoch, evaluate_model

def setup_logging(model_type, n_trials):
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"optuna_tuning_{model_type}_{n_trials}trials_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def log_print(message):
    """Print and log a message"""
    print(message)
    logging.info(message)

def objective(trial, epochs, data_dir, model_type):
    """Optuna objective function for hyperparameter optimization"""
    
    # Define hyperparameter search space
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Optimizer choice
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    
    # Learning rate scheduler
    scheduler_name = trial.suggest_categorical('scheduler', ['step', 'cosine'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print trial info
    trial_msg = f"\nTrial {trial.number + 1} - LR: {lr:.6f}, Batch: {batch_size}, WD: {weight_decay:.6f}, " \
                f"Dropout: {dropout_rate:.2f}, Optim: {optimizer_name}, Sched: {scheduler_name}"
    log_print(trial_msg)
    
    try:
        # Create data transforms
        train_transform, val_transform = get_data_transforms()
        
        # Create datasets
        train_dataset = VehicleDataset(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = VehicleDataset(os.path.join(data_dir, 'val'), transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create model
        model = create_model(model_type, num_classes=len(train_dataset.classes), dropout_rate=dropout_rate, device=device)
        
        # Setup optimizer
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:  # adamw
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Setup scheduler
        if scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        else:  # cosine
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 8
        
        for epoch in range(epochs):  # Limit to 25 epochs for tuning
            # Training phase
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation phase
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step()
            
            # Print epoch progress (every epoch for better visibility)
            epoch_msg = f"  Epoch {epoch+1}/{epochs} - Train: {train_acc:.3f}, Val: {val_acc:.3f}"
            log_print(epoch_msg)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                log_print(f"    -> New best validation accuracy: {best_val_acc:.3f}")
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                log_print(f"  Early stopping at epoch {epoch+1}")
                break
            
            # Report intermediate value for pruning
            trial.report(val_acc, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                log_print(f"  Trial pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
        
        log_print(f"  Trial {trial.number + 1} completed - Best Val Acc: {best_val_acc:.3f}")
        return best_val_acc
        
    except Exception as e:
        log_print(f"  Trial {trial.number + 1} failed: {e}")
        return 0.0

def run_hyperparameter_tuning(data_dir, model_type, n_trials, epochs):
    """Run hyperparameter optimization using Optuna"""
    
    # Setup logging
    log_file = setup_logging(model_type, n_trials)
    
    log_print(f"Starting hyperparameter optimization for {model_type}")
    log_print(f"Number of trials: {n_trials}")
    log_print(f"Data directory: {data_dir}")
    log_print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    log_print(f"Log file: {log_file}")
    log_print("=" * 60)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    # Run optimization with progress callback
    def progress_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            log_print(f"Completed trial {trial.number + 1}/{n_trials} - Best so far: {study.best_value:.3f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            log_print(f"Pruned trial {trial.number + 1}/{n_trials}")
    
    study.optimize(
        lambda trial: objective(trial, epochs, data_dir, model_type), 
        n_trials=n_trials, 
        timeout=1800,  # 30 minutes timeout
        callbacks=[progress_callback]
    )
    
    # Save results and train final model
    final_test_acc = save_optimization_results(study, model_type, data_dir, epochs)
    
    return study, final_test_acc

def train_final_model_with_test(best_params, model_type, data_dir, epochs):
    """Train final model with best parameters and evaluate on test set"""
    log_print("\n" + "="*60)
    log_print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    log_print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = VehicleDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = VehicleDataset(os.path.join(data_dir, 'val'), transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=2)
    
    # Create model
    model = create_model(model_type, num_classes=len(train_dataset.classes), 
                        dropout_rate=best_params['dropout_rate'], device=device)
    
    # Setup optimizer
    if best_params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], 
                              weight_decay=best_params['weight_decay'])
    else:  # adamw
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], 
                               weight_decay=best_params['weight_decay'])
    
    # Setup scheduler
    if best_params['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:  # cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with more epochs for final model
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 15
    best_model_state = None
    
    log_print("Training final model...")
    
    for epoch in range(epochs*2):  # More epochs for final training
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_print(f"  Epoch {epoch+1}/50 - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            log_print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set if available
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        log_print("\n" + "="*50)
        log_print("FINAL TEST SET EVALUATION")
        log_print("="*50)
        
        test_dataset = VehicleDataset(test_dir, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], 
                                shuffle=False, num_workers=2)
        
        _, _, test_accuracy = evaluate_model(model, test_loader, train_dataset.classes, device, 'test')
        
        log_print(f"Final test accuracy: {test_accuracy:.4f}")
        return test_accuracy
    else:
        log_print("No test directory found - skipping test evaluation")
        return None

def save_optimization_results(study, model_type, data_dir, epochs):
    """Save optimization results to files and train final model with test evaluation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("hyperparameter_results")
    results_dir.mkdir(exist_ok=True)
    
    # Best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    log_print(f"\nOptimization completed!")
    log_print(f"Best validation accuracy: {best_value:.4f}")
    log_print(f"Best parameters: {best_params}")
    
    # Train final model with best parameters and evaluate on test set
    test_accuracy = train_final_model_with_test(best_params, model_type, data_dir, epochs)
    
    results = {
        'model_type': model_type,
        'best_val_accuracy': best_value,
        'test_accuracy': test_accuracy,
        'best_params': best_params,
        'n_trials': len(study.trials),
        'timestamp': timestamp
    }
    
    # Save to JSON
    results_file = results_dir / f"best_params_{model_type}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print(f"Results saved to {results_file}")
    
    # Create training script with best parameters
    create_training_script(best_params, best_value, test_accuracy, model_type, timestamp)
    
    return test_accuracy

def create_training_script(best_params, best_val_accuracy, test_accuracy, model_type, timestamp):
    """Create a training script with the best hyperparameters"""
    
    test_acc_str = f"{test_accuracy:.4f}" if test_accuracy is not None else "N/A"
    
    script_content = f'''# Auto-generated training script with best hyperparameters
# Best validation accuracy: {best_val_accuracy:.4f}
# Test accuracy: {test_acc_str}
# Generated on: {timestamp}

import torch
from finetune_models import train_model

# Best hyperparameters from Optuna optimization
best_params = {best_params}

# Train with best parameters
if __name__ == "__main__":
    train_model(
        model_type='{model_type}',
        data_dir='dataset',
        learning_rate={best_params['learning_rate']},
        batch_size={best_params['batch_size']},
        weight_decay={best_params['weight_decay']},
        dropout_rate={best_params['dropout_rate']},
        optimizer_name='{best_params['optimizer']}',
        scheduler_name='{best_params['scheduler']}',
        epochs=100,
        early_stopping_patience=15
    )
'''
    
    script_file = Path(f"train_best_{model_type}_{timestamp}.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    log_print(f"Training script created: {script_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--data-dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--model-type', type=str, default='resnet', 
                       choices=['efficientnetv2', 'resnet'], help='Model type to tune')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for final training')
    
    args = parser.parse_args()
    
    # Run optimization
    study, final_test_acc = run_hyperparameter_tuning(args.data_dir, args.model_type, args.n_trials, args.epochs)
    
    print(f"\nHyperparameter optimization completed!")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    if final_test_acc is not None:
        print(f"Final test accuracy: {final_test_acc:.4f}")
    else:
        print("Test accuracy: N/A (no test set found)")

if __name__ == "__main__":
    main() 