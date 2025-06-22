#!/usr/bin/env python3
"""
Auto-generated training script with best hyperparameters from Optuna
Best validation accuracy: 0.9563
Test accuracy: 0.9517
Generated on: 20250622_121912
"""

import sys
from finetune_models import main

# Best hyperparameters from Optuna optimization
best_params = {
    'learning_rate': 0.003967605077052989, 
    'batch_size': 64, 
    'weight_decay': 6.358358856676247e-05, 
    'dropout_rate': 0.3832290311184182, 
    'optimizer': 'adamw', 
    'scheduler': 'step'
}

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING WITH OPTUNA BEST HYPERPARAMETERS")
    print("=" * 60)
    print(f"Best params: {best_params}")
    print("=" * 60)
    
    # Override sys.argv to pass the best hyperparameters to the main function
    sys.argv = [
        'train_best_model.py',
        '--model', 'resnet',
        '--epochs', '100',
        '--batch-size', str(best_params['batch_size']),
        '--lr', str(best_params['learning_rate']),
        '--weight-decay', str(best_params['weight_decay']),
        '--dropout-rate', str(best_params['dropout_rate']),
        '--patience', '15',
        '--data-dir', 'dataset'
    ]
    
    # Call the main function from finetune_models.py
    main()
