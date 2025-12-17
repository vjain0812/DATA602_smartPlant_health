"""
Train ResNet model for plant health classification
Uses transfer learning with pretrained ResNet50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class PlantHealthClassifier:
    def __init__(self, num_classes, model_name='resnet50', pretrained=True):
        """
        Initialize ResNet classifier
        
        Args:
            num_classes: Number of disease classes
            model_name: Model architecture (resnet50, resnet101, efficientnet_b0)
            pretrained: Use pretrained weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        if model_name.startswith('resnet'):
            self.model = models.__dict__[model_name](pretrained=pretrained)
            # Replace final layer
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif model_name.startswith('efficientnet'):
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.num_classes = num_classes
        self.class_names = None
        
    def prepare_data(self, train_dir, val_dir, batch_size=32, num_workers=4):
        """Prepare data loaders with augmentation"""
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        # Store class names
        self.class_names = train_dataset.classes
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\nDataset prepared:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Number of classes: {len(self.class_names)}")
        print(f"  Batch size: {batch_size}")
        
        return self.train_loader, self.val_loader
    
    def train(self, epochs=20, lr=0.001, save_path='models/resnet/best_model.pth'):
        """Train the model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        best_val_acc = 0.0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("TRAINING STARTED")
        print(f"{'='*70}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc='Training')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 
                                 'acc': 100 * train_correct / train_total})
            
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = 100 * val_correct / val_total
            
            # Update history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            # Print epoch results
            print(f"\nResults:")
            print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
            print(f"  Val Loss: {epoch_val_loss:.4f}   | Val Acc: {epoch_val_acc:.2f}%")
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'class_names': self.class_names
                }, save_path)
                print(f"  ✓ Best model saved! (Val Acc: {best_val_acc:.2f}%)")
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        
        return history
    
    def evaluate(self, save_dir='models/resnet'):
        """Evaluate model and generate metrics"""
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds, 
            target_names=self.class_names,
            digits=3
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - ResNet Plant Disease Classification', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        save_path = Path(save_dir) / 'confusion_matrix.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Confusion matrix saved to {save_path}")
        plt.close()
        
        # Calculate accuracy
        accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        
        return accuracy, cm
    
    def plot_training_history(self, history, save_path='models/resnet/training_history.png'):
        """Plot training curves"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(history['train_acc'], label='Train', linewidth=2)
        ax1.plot(history['val_acc'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history['train_loss'], label='Train', linewidth=2)
        ax2.plot(history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to {save_path}")
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet classifier')
    parser.add_argument('--train-dir', type=str, default='dataset_subset/train',
                       help='Training data directory')
    parser.add_argument('--val-dir', type=str, default='dataset_subset/valid',
                       help='Validation data directory')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'efficientnet_b0'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models/resnet',
                       help='Directory to save model')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*15 + "RESNET PLANT DISEASE CLASSIFIER")
    print("="*70)
    
    # Check data exists
    if not Path(args.train_dir).exists():
        print(f"\n❌ Training directory not found: {args.train_dir}")
        print("   Please run: python create_subset.py first")
        return
    
    # Count classes
    num_classes = len(list(Path(args.train_dir).iterdir()))
    
    # Initialize classifier
    classifier = PlantHealthClassifier(
        num_classes=num_classes,
        model_name=args.model,
        pretrained=True
    )
    
    # Prepare data
    classifier.prepare_data(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size
    )
    
    # Train
    history = classifier.train(
        epochs=args.epochs,
        lr=args.lr,
        save_path=f'{args.save_dir}/best_model.pth'
    )
    
    # Plot training history
    classifier.plot_training_history(
        history,
        save_path=f'{args.save_dir}/training_history.png'
    )
    
    # Evaluate
    accuracy, cm = classifier.evaluate(save_dir=args.save_dir)
    
    # Save metadata
    metadata = {
        'model': args.model,
        'num_classes': num_classes,
        'class_names': classifier.class_names,
        'final_accuracy': float(accuracy),
        'epochs_trained': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }
    
    with open(f'{args.save_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model and metadata saved to {args.save_dir}/")
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()