#!/usr/bin/env python3
"""
Fixed Training Script for Vehicle Classification
Addresses the low validation accuracy issues with better configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import copy
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

cudnn.benchmark = True

def main():
    """Main training function"""
    
    print("Starting Fixed Vehicle Classification Training")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Improved data augmentation strategy
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Less aggressive cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Small rotation for realism
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    data_dir = 'split_dataset'
    print(f"Using dataset: {data_dir}")
    
    # Load datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    # Better batch size
    batch_size = 16
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], 
                                            batch_size=batch_size,
                                            shuffle=True, 
                                            num_workers=4,
                                            pin_memory=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], 
                                          batch_size=batch_size,
                                          shuffle=False, 
                                          num_workers=4,
                                          pin_memory=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], 
                                           batch_size=batch_size,
                                           shuffle=False, 
                                           num_workers=4,
                                           pin_memory=True)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Batch size: {batch_size}")
    
    # Create model - Use EfficientNet-V2-S with proper handling
    model_conv = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    
    # Freeze only some parameters for better learning
    for param in model_conv.features.parameters():
        param.requires_grad = False
    
    # Allow fine-tuning of the last few layers
    for param in model_conv.features[-3:].parameters():
        param.requires_grad = True
    
    # Get the correct number of input features
    # EfficientNet-V2-S has a different structure
    if hasattr(model_conv.classifier, '__len__') and len(model_conv.classifier) > 1:
        # It's a Sequential module
        for layer in model_conv.classifier:
            if isinstance(layer, nn.Linear):
                num_ftrs = layer.in_features
                break
    else:
        # It's a single Linear layer
        num_ftrs = model_conv.classifier.in_features
    
    print(f"Number of input features: {num_ftrs}")
    
    # Create improved classifier
    model_conv.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    
    model_conv = model_conv.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model_conv.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model_conv.parameters() if p.requires_grad):,}")
    
    # Improved loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Better optimizer with weight decay
    optimizer_conv = optim.AdamW(filter(lambda p: p.requires_grad, model_conv.parameters()), 
                                lr=0.001, weight_decay=0.01)
    
    # Better scheduler
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv, mode='max', 
                                                     factor=0.5, patience=5, verbose=True)
    
    # Training function with early stopping
    def train_model_with_early_stopping(model, criterion, optimizer, scheduler, num_epochs=50, patience=10):
        since = time.time()
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        epochs_no_improve = 0
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc.item())
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Deep copy the model
                if phase == 'val':
                    scheduler.step(epoch_acc)  # ReduceLROnPlateau needs the metric
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
            
            print()
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
        # Load best model weights
        model.load_state_dict(best_model_wts)
        return model, history, best_acc.item()
    
    # Train the model
    print("\nStarting training...")
    model_conv, history, best_val_acc = train_model_with_early_stopping(
        model_conv, criterion, optimizer_conv, exp_lr_scheduler, 
        num_epochs=50, patience=10
    )
    
    # Plot training history
    def plot_training_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Training Loss', color='blue')
        ax1.plot(history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Training Accuracy', color='blue')
        ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plot_training_history(history)
    
    # Evaluation function
    def evaluate_model(model, dataloader, class_names, phase='test'):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        
        print(f"\n{phase.upper()} RESULTS:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Average Precision: {np.mean(precision):.4f}")
        print(f"Average Recall: {np.mean(recall):.4f}")
        print(f"Average F1-Score: {np.mean(f1):.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
        
        return all_labels, all_preds, accuracy
    
    # Confusion matrix function
    def plot_confusion_matrix(labels, preds, class_names, title='Confusion Matrix'):
        cm = confusion_matrix(labels, preds)
        cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'{title} - Raw Counts')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # Plot percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title(f'{title} - Percentages')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm, cm_percent
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("VALIDATION EVALUATION")
    print("="*50)
    
    val_labels, val_preds, val_accuracy = evaluate_model(
        model_conv, dataloaders['val'], class_names, 'validation'
    )
    
    # Plot confusion matrix
    cm_val, cm_val_percent = plot_confusion_matrix(
        val_labels, val_preds, class_names, 'Validation Set'
    )
    
    # Test on test set if available
    if dataset_sizes['test'] > 0:
        print("\n" + "="*50)
        print("TEST EVALUATION")
        print("="*50)
        
        test_labels, test_preds, test_accuracy = evaluate_model(
            model_conv, dataloaders['test'], class_names, 'test'
        )
        
        # Plot test confusion matrix
        cm_test, cm_test_percent = plot_confusion_matrix(
            test_labels, test_preds, class_names, 'Test Set'
        )
    
    # Save model
    model_save_path = 'fixed_efficientnet_v2_s_vehicle_classifier.pth'
    torch.save({
        'model_state_dict': model_conv.state_dict(),
        'optimizer_state_dict': optimizer_conv.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'val_accuracy': val_accuracy,
        'training_history': history
    }, model_save_path)
    
    print(f"\nModel saved as '{model_save_path}'")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    if dataset_sizes['test'] > 0:
        print(f"Final test accuracy: {test_accuracy:.4f}")
    
    print("\nKey improvements made:")
    print("- Better data augmentation with more realistic transforms")
    print("- Improved model architecture with deeper classifier")
    print("- AdamW optimizer with weight decay")
    print("- Label smoothing for better generalization")
    print("- Early stopping with ReduceLROnPlateau scheduler")
    print("- Larger batch size and better data loading")
    
    improvement = val_accuracy - 0.494681  # Your original accuracy
    print(f"\nImprovement over original: {improvement:.4f} ({improvement*100:.2f}%)")
    
    if val_accuracy < 0.7:
        print("\n⚠️  Consider using the enhanced augmented dataset for even better results!")
        print("   Run: python3 traffic_cctv_augmentation_enhanced.py")
    else:
        print("\n✅ Great performance! Ready for deployment.")

if __name__ == "__main__":
    main() 