"""
Train LSTM model for watering needs prediction
Predicts if plant needs watering in next 24 hours based on sensor data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

class SensorDataset(Dataset):
    """Dataset for sensor time-series data"""
    
    def __init__(self, data, sequence_length=24, features=None):
        """
        Args:
            data: DataFrame with sensor readings
            sequence_length: Length of input sequence (hours)
            features: List of feature columns to use
        """
        self.sequence_length = sequence_length
        
        if features is None:
            self.features = ['temperature', 'humidity', 'soil_moisture', 
                           'light_intensity', 'npk_nitrogen', 'npk_phosphorus', 
                           'npk_potassium', 'health_score']
        else:
            self.features = features
        
        # Extract features and target
        self.X = data[self.features].values
        self.y = data['watering_needed'].values
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence
        X_seq = self.X[idx:idx + self.sequence_length]
        y_val = self.y[idx + self.sequence_length]
        
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_val])

class LSTMWateringPredictor(nn.Module):
    """LSTM model for watering prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMWateringPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

class WateringTrainer:
    """Trainer for LSTM watering predictor"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = LSTMWateringPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        self.scaler = None
        
    def prepare_data(self, csv_path, sequence_length=24, train_split=0.8, batch_size=32):
        """Load and prepare data"""
        
        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"  Total records: {len(df):,}")
        print(f"  Features: {df.columns.tolist()}")
        
        # Split by plant_id (each plant is separate)
        plant_ids = df['plant_id'].unique()
        n_train = int(len(plant_ids) * train_split)
        
        train_plants = plant_ids[:n_train]
        val_plants = plant_ids[n_train:]
        
        train_df = df[df['plant_id'].isin(train_plants)].reset_index(drop=True)
        val_df = df[df['plant_id'].isin(val_plants)].reset_index(drop=True)
        
        print(f"\n  Training plants: {len(train_plants)}")
        print(f"  Validation plants: {len(val_plants)}")
        print(f"  Training samples: {len(train_df):,}")
        print(f"  Validation samples: {len(val_df):,}")
        
        # Create datasets
        train_dataset = SensorDataset(train_df, sequence_length=sequence_length)
        val_dataset = SensorDataset(val_df, sequence_length=sequence_length)
        
        # Store scaler for later use
        self.scaler = train_dataset.scaler
        self.features = train_dataset.features
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        print(f"\n  Sequence length: {sequence_length} hours")
        print(f"  Batch size: {batch_size}")
        print(f"  Input features: {len(train_dataset.features)}")
        
        return self.train_loader, self.val_loader
    
    def train(self, epochs=50, lr=0.001, save_path='models/lstm/best_model.pth'):
        """Train the LSTM model"""
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_acc = 0.0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("LSTM TRAINING STARTED")
        print(f"{'='*70}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Store predictions
                preds = (outputs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(y_batch.cpu().numpy())
            
            train_loss /= len(self.train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())
            
            val_loss /= len(self.val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'scaler': self.scaler,
                    'features': self.features
                }, save_path)
                
                if (epoch + 1) % 5 == 0:
                    print(f"  ✓ Best model saved! (Val Acc: {best_val_acc:.4f})")
        
        print(f"\n{'='*70}")
        print("LSTM TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        
        return history
    
    def evaluate(self, save_dir='models/lstm'):
        """Evaluate model"""
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n" + "="*70)
        print("LSTM EVALUATION")
        print("="*70)
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                
                outputs = self.model(X_batch)
                preds = (outputs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Water', 'Need Water'],
                   yticklabels=['No Water', 'Need Water'])
        plt.title('LSTM Watering Prediction - Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        save_path = Path(save_dir) / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Confusion matrix saved to {save_path}")
        plt.close()
        
        return accuracy, precision, recall, f1
    
    def plot_training_history(self, history, save_path='models/lstm/training_history.png'):
        """Plot training curves"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(history['train_acc'], label='Train', linewidth=2)
        ax1.plot(history['val_acc'], label='Validation', linewidth=2)
        ax1.set_title('LSTM Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history['train_loss'], label='Train', linewidth=2)
        ax2.plot(history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('LSTM Loss', fontsize=14, fontweight='bold')
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
    
    parser = argparse.ArgumentParser(description='Train LSTM watering predictor')
    parser.add_argument('--data', type=str, default='data/timeseries/sensor_data.csv',
                       help='Path to sensor data CSV')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--sequence-length', type=int, default=24,
                       help='Input sequence length (hours)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--hidden-size', type=int, default=64,
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*15 + "LSTM WATERING PREDICTOR")
    print("="*70)
    
    # Check data file
    if not Path(args.data).exists():
        print(f"\n❌ Data file not found: {args.data}")
        print("   Run: python generate_sensor_data.py")
        return
    
    # Initialize trainer
    trainer = WateringTrainer(
        input_size=8,  # Number of sensor features
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    
    # Prepare data
    trainer.prepare_data(
        args.data,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    
    # Train
    history = trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        save_path='models/lstm/best_model.pth'
    )
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Evaluate
    accuracy, precision, recall, f1 = trainer.evaluate()
    
    # Save metadata
    metadata = {
        'sequence_length': args.sequence_length,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'epochs': args.epochs,
        'final_accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    with open('models/lstm/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Model and metadata saved to models/lstm/")
    print("\n🎉 LSTM training complete!")
    print("\nNext steps:")
    print("  1. Create pipeline: python pipeline.py")
    print("  2. Build API: python api/main.py")
    print("  3. Build frontend: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()