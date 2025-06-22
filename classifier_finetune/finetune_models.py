#!/usr/bin/env python3
"""
Vehicle Classification Fine-tuning Script
Supports both EfficientNetV2 and ResNet models
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
import logging
import argparse
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

cudnn.benchmark = True

class VehicleDataset(torch.utils.data.Dataset):
    """Custom dataset for vehicle classification"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get class names from directory structure
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
                for img_path in class_dir.glob('*.jpeg'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
                for img_path in class_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = pil_loader(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms():
    """Get data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(model_type, num_classes, dropout_rate=0.3, device=None, pretrained=True):
    """Unified model creation for EfficientNetV2 and ResNet."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type.lower() == 'efficientnetv2':
        model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1' if pretrained else None)
        for param in model.features[:-3].parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
    elif model_type.lower() == 'resnet':
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    return model.to(device)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with progress bar"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Add progress bar
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{running_corrects.double() / total_samples:.3f}'
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch with progress bar"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Add progress bar
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{running_corrects.double() / total_samples:.3f}'
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def setup_logging(log_dir, model_name):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"vehicle_classifier_{model_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def pil_loader(path):
    """Custom PIL loader that handles transparency"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        img = img.convert('RGB')
        return img

def create_data_transforms():
    """Create data transforms optimized for vehicle classification"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    return data_transforms

def train_model(model, 
                dataloaders, 
                dataset_sizes,
                learning_rate,
                batch_size,
                weight_decay,
                dropout_rate,
                optimizer_name,
                scheduler_name,
                criterion, 
                optimizer, 
                scheduler, 
                device, 
                num_epochs=25, 
                patience=7, 
                save_every=5, 
                model_name="model"):
    """Train the model with early stopping and periodic model saving"""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Create models directory for periodic saves
    model_save_dir = Path('models')
    model_save_dir.mkdir(exist_ok=True)
    
    logging.info(f"Starting training for {num_epochs} epochs with patience {patience}")
    logging.info(f"Models will be saved every {save_every} epochs")
    
    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data with progress bar
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}"):
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
            
            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Update scheduler and check for best model
            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    logging.info(f'New best model with validation accuracy: {best_acc:.4f}')
                else:
                    epochs_no_improve += 1
        
        # Save model every save_every epochs
        if (epoch + 1) % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = model_save_dir / f'vehicle_classifier_{model_name}_epoch_{epoch+1}_{timestamp}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history,
                'model_architecture': model_name,
            }, checkpoint_path)
            logging.info(f'Model checkpoint saved at epoch {epoch+1}: {checkpoint_path}')
            print(f'Model checkpoint saved at epoch {epoch+1}: {checkpoint_path}')
        
        print()
        
        # Early stopping
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Acc: {best_acc:4f}')
    
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history, best_acc.item()

def plot_training_history(history, save_path):
    """Plot and save training history"""
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, dataloader, class_names, device, phase='test'):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {phase}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division='warn')
    
    logging.info(f"\n{phase.upper()} RESULTS:")
    logging.info(f"Overall Accuracy: {accuracy:.4f}")
    logging.info(f"Average Precision: {np.mean(precision):.4f}")
    logging.info(f"Average Recall: {np.mean(recall):.4f}")
    logging.info(f"Average F1-Score: {np.mean(f1):.4f}")
    
    print(f"\n{phase.upper()} RESULTS:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Average Precision: {np.mean(precision):.4f}")
    print(f"Average Recall: {np.mean(recall):.4f}")
    print(f"Average F1-Score: {np.mean(f1):.4f}")
    
    print("\nDetailed Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division='warn')
    print(report)
    logging.info(f"\nClassification Report:\n{report}")
    
    return all_labels, all_preds, accuracy

def plot_confusion_matrix(labels, preds, class_names, save_path, title='Confusion Matrix'):
    """Plot and save confusion matrix"""
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm, cm_percent

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Vehicle Classification Fine-tuning')
    parser.add_argument('--model', type=str, choices=['efficientnetv2', 'resnet'], 
                       default='efficientnetv2', help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--data-dir', type=str, default='dataset', help='Dataset directory')
    
    args = parser.parse_args()
    
    print(f"Vehicle Classification Fine-tuning - {args.model.upper()}")
    print("=" * 60)
    
    # Setup logging
    log_file = setup_logging('logs', args.model)
    logging.info(f"Starting vehicle classifier training with {args.model}")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    print(f"Using device: {device}")
    
    # Data directory
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory {data_dir} not found.")
    
    # Create data transforms
    data_transforms = create_data_transforms()
    
    # Load datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x],
                                              loader=pil_loader)
                      for x in ['train', 'val'] if os.path.exists(os.path.join(data_dir, x))}
    
    # Check if train/val directories exist
    if len(image_datasets) == 0:
        raise FileNotFoundError(f"No train/val directories found in {data_dir}. Please run split_dataset.py first.")
    
    if 'train' not in image_datasets:
        raise FileNotFoundError(f"Training directory not found in {data_dir}")
    
    if 'val' not in image_datasets:
        raise FileNotFoundError(f"Validation directory not found in {data_dir}")
    
    # Create data loaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], 
                                            batch_size=args.batch_size,
                                            shuffle=True, 
                                            num_workers=8,
                                            pin_memory=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], 
                                          batch_size=args.batch_size,
                                          shuffle=False, 
                                          num_workers=4,
                                          pin_memory=True)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets.keys()}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Class names: {class_names}")
    logging.info(f"Dataset sizes: {dataset_sizes}")
    logging.info(f"Batch size: {args.batch_size}")
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Batch size: {args.batch_size}")
    
    # Create model
    model = create_model(args.model, num_classes, dropout_rate=0.3, device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                              factor=0.5, patience=3)
    
    # Train model
    print("\nStarting training...")
    logging.info("Starting model training")
    
    model, history, best_val_acc = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=0.01,
        dropout_rate=0.3,
        optimizer_name='AdamW',
        scheduler_name='ReduceLROnPlateau',
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        save_every=5,
        model_name=args.model
    )
    
    # Save training history plot
    plot_save_dir = Path('plots')
    plot_save_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_plot_path = plot_save_dir / f'vehicle_classifier_{args.model}_training_history_{timestamp}.png'
    plot_training_history(history, history_plot_path)
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("VALIDATION EVALUATION")
    print("="*50)
    
    val_labels, val_preds, val_accuracy = evaluate_model(
        model, dataloaders['val'], class_names, device, 'validation'
    )
    
    # Plot validation confusion matrix
    val_cm_path = plot_save_dir / f'vehicle_classifier_{args.model}_val_confusion_matrix_{timestamp}.png'
    plot_confusion_matrix(val_labels, val_preds, class_names, val_cm_path, 'Validation Set')
    
    # Evaluate on test set if available
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        print("\n" + "="*50)
        print("TEST SET EVALUATION")
        print("="*50)
        
        # Create test dataset and dataloader
        test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'], loader=pil_loader)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                     batch_size=args.batch_size,
                                                     shuffle=False, 
                                                     num_workers=4,
                                                     pin_memory=True)
        
        test_labels, test_preds, test_accuracy = evaluate_model(
            model, test_dataloader, class_names, device, 'test'
        )
        
        # Plot test confusion matrix
        test_cm_path = plot_save_dir / f'vehicle_classifier_{args.model}_test_confusion_matrix_{timestamp}.png'
        plot_confusion_matrix(test_labels, test_preds, class_names, test_cm_path, 'Test Set')
    else:
        print("No test directory found - skipping test evaluation")
        test_accuracy = None
    
    # Save model
    model_save_dir = Path('models')
    model_save_dir.mkdir(exist_ok=True)
    
    model_save_path = model_save_dir / f'vehicle_classifier_{args.model}_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'training_history': history,
        'model_architecture': args.model,
        'dataset': 'vehicle_classification'
    }, model_save_path)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    summary_info = f"""
    Model: {args.model.upper()}
    Dataset: Vehicle Classification
    Classes: {num_classes} ({', '.join(class_names)})
    
    Final validation accuracy: {val_accuracy:.4f}
    Final test accuracy: {f'{test_accuracy:.4f}' if test_accuracy is not None else 'N/A'}
    
    Total parameters: {total_params:,}
    Trainable parameters: {trainable_params:,}
    
    Model saved: {model_save_path}
    Log file: {log_file}
    Training plots: {plot_save_dir}
    """
    
    print(summary_info)
    logging.info(summary_info)
    
    print("Training completed successfully!")
    logging.info("Training completed successfully")

if __name__ == "__main__":
    main()