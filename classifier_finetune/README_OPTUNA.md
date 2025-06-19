# Optuna Hyperparameter Tuning for Vehicle Classification

This guide explains how to use Optuna for automated hyperparameter optimization of your vehicle classification models.

## Overview

Optuna is a hyperparameter optimization framework that automatically finds the best hyperparameters for your machine learning models. It uses advanced algorithms like Tree-structured Parzen Estimators (TPE) to efficiently search the hyperparameter space.

## Features

- **Automated Hyperparameter Search**: Automatically finds optimal learning rates, batch sizes, dropout rates, etc.
- **Multiple Optimizers**: Tests Adam, AdamW, and SGD optimizers
- **Learning Rate Schedulers**: Experiments with Step, Cosine, and Plateau schedulers
- **Early Pruning**: Automatically stops poorly performing trials to save time
- **Results Export**: Saves best parameters and generates training scripts
- **Support for Multiple Models**: Works with both EfficientNetV2 and ResNet

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Basic Hyperparameter Tuning

```bash
python run_optuna_tuning.py
```

This will run 20 trials each for EfficientNetV2 and ResNet models.

### 2. Run Custom Hyperparameter Tuning

```bash
python train_with_optuna.py --model_type efficientnetv2 --n_trials 50 --data_dir dataset
```

### 3. Command Line Options

```bash
python train_with_optuna.py --help
```

Available options:
- `--data_dir`: Path to dataset directory (default: 'dataset')
- `--model_type`: Model type to tune ('efficientnetv2' or 'resnet')
- `--n_trials`: Number of optimization trials (default: 30)

## Hyperparameter Search Space

The following hyperparameters are automatically optimized:

### Learning Rate
- **Range**: 1e-5 to 1e-2 (logarithmic scale)
- **Description**: Controls how much the model weights are updated

### Batch Size
- **Options**: [16, 32, 64]
- **Description**: Number of samples processed before updating model weights

### Weight Decay
- **Range**: 1e-6 to 1e-3 (logarithmic scale)
- **Description**: L2 regularization to prevent overfitting

### Dropout Rate
- **Range**: 0.1 to 0.5
- **Description**: Probability of dropping neurons during training

### Optimizer
- **Options**: ['adam', 'adamw']
- **Description**: Optimization algorithm

### Learning Rate Scheduler
- **Options**: ['step', 'cosine']
- **Description**: Strategy for adjusting learning rate during training

## Output Files

After running hyperparameter tuning, the following files are generated:

### 1. Results Directory: `hyperparameter_results/`

```
hyperparameter_results/
├── best_params_efficientnetv2_20241201_143022.json
├── all_trials_efficientnetv2_20241201_143022.json
├── best_params_resnet_20241201_143022.json
└── all_trials_resnet_20241201_143022.json
```

### 2. Best Parameters JSON

```json
{
  "model_type": "efficientnetv2",
  "best_accuracy": 85.67,
  "best_params": {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "weight_decay": 1e-5,
    "dropout_rate": 0.3,
    "optimizer": "adamw",
    "scheduler": "cosine"
  },
  "n_trials": 30,
  "timestamp": "20241201_143022"
}
```

### 3. Auto-Generated Training Script

A training script is automatically created with the best parameters:

```python
# train_best_efficientnetv2_20241201_143022.py
from finetune_models import train_model

# Best hyperparameters from Optuna optimization
best_params = {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "weight_decay": 1e-5,
    "dropout_rate": 0.3,
    "optimizer": "adamw",
    "scheduler": "cosine"
}

# Train with best parameters
if __name__ == "__main__":
    train_model(
        model_type='efficientnetv2',
        data_dir='dataset',
        learning_rate=0.0001,
        batch_size=32,
        weight_decay=1e-5,
        dropout_rate=0.3,
        optimizer_name='adamw',
        scheduler_name='cosine',
        epochs=100,
        early_stopping_patience=15
    )
```

## Advanced Usage

### 1. Custom Search Space

To modify the hyperparameter search space, edit the `objective` function in `train_with_optuna.py`:

```python
def objective(trial, data_dir, model_type='efficientnetv2'):
    # Customize these ranges
    lr = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)  # Wider range
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])  # More options
    # ... other parameters
```

### 2. Add New Hyperparameters

To add new hyperparameters to optimize:

```python
def objective(trial, data_dir, model_type='efficientnetv2'):
    # Existing parameters...
    
    # New parameter
    momentum = trial.suggest_float('momentum', 0.8, 0.99)  # For SGD
    
    # Use in training...
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
```

### 3. Custom Objective Function

You can modify the objective to optimize for different metrics:

```python
def objective(trial, data_dir, model_type='efficientnetv2'):
    # ... training code ...
    
    # Return F1 score instead of accuracy
    return f1_score  # or any other metric
```

## Optimization Strategies

### 1. Pruning

Optuna automatically prunes (stops) trials that are performing poorly:

- **Median Pruner**: Stops trials that are worse than the median of completed trials
- **Configurable**: Adjust `n_startup_trials` and `n_warmup_steps` in the study creation

### 2. Sampling Strategy

- **TPE Sampler**: Uses Tree-structured Parzen Estimators for efficient search
- **Seed**: Fixed seed (42) for reproducible results

### 3. Timeout

- **Default**: 30 minutes per model
- **Configurable**: Modify the `timeout` parameter in `study.optimize()`

## Tips for Better Results

### 1. Dataset Size
- Larger datasets benefit from more trials (50-100)
- Smaller datasets work well with 20-30 trials

### 2. Time Constraints
- Reduce `n_trials` for faster results
- Increase `timeout` for more thorough search

### 3. Model Selection
- Start with EfficientNetV2 (generally better performance)
- Test both models to compare

### 4. Resource Management
- Monitor GPU memory usage
- Adjust batch sizes if needed

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in search space
   - Use smaller models

2. **Slow Training**
   - Reduce number of trials
   - Use fewer epochs per trial

3. **Poor Results**
   - Increase number of trials
   - Widen hyperparameter search ranges
   - Check dataset quality

### Error Messages

- **"Trial failed"**: Usually due to memory issues or invalid parameters
- **"Study already exists"**: Delete existing study database or use different study name

## Example Workflow

1. **Prepare Dataset**
   ```bash
   python split_dataset.py
   ```

2. **Run Hyperparameter Tuning**
   ```bash
   python run_optuna_tuning.py
   ```

3. **Review Results**
   ```bash
   # Check best parameters
   cat hyperparameter_results/best_params_*.json
   ```

4. **Train with Best Parameters**
   ```bash
   python train_best_efficientnetv2_*.py
   ```

5. **Evaluate Results**
   ```bash
   python evaluate_model.py
   ```

## Performance Comparison

Typical improvements from hyperparameter tuning:

- **Accuracy**: +5-15% improvement
- **Training Time**: Optimized batch sizes reduce training time
- **Generalization**: Better regularization prevents overfitting

## Next Steps

After finding the best hyperparameters:

1. Train the final model with more epochs
2. Evaluate on test set
3. Deploy the model
4. Monitor performance in production

For more advanced hyperparameter tuning, consider:
- Multi-objective optimization (accuracy vs. speed)
- Neural Architecture Search (NAS)
- Transfer learning from similar tasks 