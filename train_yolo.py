"""
Train YOLOv5 model for plant disease detection and localization
Uses Ultralytics YOLOv5 for object detection
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import shutil
import matplotlib.pyplot as plt
import json

class YOLOPlantDetector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLO detector
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
                       nano is fastest, large is most accurate
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load pretrained YOLOv8 model (newer than v5)
        model_name = f'yolov8{model_size}.pt'
        self.model = YOLO(model_name)
        
        print(f"Loaded YOLOv8-{model_size} model")
    
    def train(self, data_yaml, epochs=10, img_size=416, batch_size=8, save_dir='models/yolo'):
        """
        Train YOLO model
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Batch size
            save_dir: Directory to save results
        """
        
        print("\n" + "="*70)
        print("STARTING YOLO TRAINING")
        print("="*70)
        
        # Check if data.yaml exists
        if not Path(data_yaml).exists():
            print(f"\n❌ Error: {data_yaml} not found!")
            print("   Please create YOLO format dataset first:")
            print("   python create_subset.py --create-yolo")
            return None
        
        # Train the model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=save_dir,
            device=self.device,
            patience=5,
            save=True,
            plots=True,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("YOLO TRAINING COMPLETED")
        print("="*70)
        
        return results
    
    def evaluate(self, data_yaml):
        """Evaluate model on validation set"""
        
        print("\n" + "="*70)
        print("EVALUATING YOLO MODEL")
        print("="*70)
        
        # Validate the model
        metrics = self.model.val(data=data_yaml)
        
        # Print metrics
        print(f"\nResults:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def predict_image(self, image_path, conf_threshold=0.25, save_path='results/yolo_prediction.jpg'):
        """
        Predict on a single image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detections
            save_path: Where to save annotated image
        """
        
        # Run inference
        results = self.model.predict(
            image_path,
            conf=conf_threshold,
            save=True,
            show_labels=True,
            show_conf=True
        )
        
        # Get detections
        for result in results:
            boxes = result.boxes
            
            print(f"\nDetections: {len(boxes)}")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                
                print(f"  Class: {result.names[cls]}")
                print(f"  Confidence: {conf:.2%}")
                print(f"  Box: {coords}")
        
        return results
    
    def export_model(self, format='onnx', save_path='models/yolo/best.onnx'):
        """Export model to different formats"""
        
        print(f"\nExporting model to {format}...")
        self.model.export(format=format)
        print(f"✅ Model exported")

def create_quick_yolo_data():
    """
    Create a quick YOLO dataset from the mini subset
    For rapid testing
    """
    print("\n" + "="*70)
    print("CREATING QUICK YOLO DATASET")
    print("="*70)
    
    # Check if mini subset exists
    if not Path('dataset_mini/train').exists():
        print("❌ Mini subset not found. Run: python create_mini_subset.py")
        return None
    
    # Create YOLO structure
    yolo_path = Path('dataset_yolo_mini')
    yolo_path.mkdir(exist_ok=True)
    
    for split in ['train', 'val']:
        (yolo_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get class names
    source_split = 'train'
    source_dir = Path('dataset_mini') / source_split
    classes = sorted([f.name for f in source_dir.iterdir() if f.is_dir()])
    
    print(f"\nFound {len(classes)} classes")
    
    # Copy images and create labels
    import shutil
    
    for split_name, source_name in [('train', 'train'), ('val', 'valid')]:
        source_path = Path('dataset_mini') / source_name
        
        if not source_path.exists():
            continue
        
        total = 0
        for class_idx, class_name in enumerate(classes):
            class_dir = source_path / class_name
            
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG'))
            
            for img_path in images:
                # Copy image
                dest_img = yolo_path / 'images' / split_name / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Create label (full image bounding box)
                label_file = yolo_path / 'labels' / split_name / f"{img_path.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                
                total += 1
        
        print(f"  {split_name}: {total} images")
    
    # Create data.yaml
    yaml_content = f"""# Mini Plant Disease Dataset - YOLO Format
path: {yolo_path.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = yolo_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Quick YOLO dataset created!")
    print(f"   Location: {yolo_path}")
    print(f"   Config: {yaml_path}")
    
    return str(yaml_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO disease detector')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data.yaml file')
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l'],
                       help='Model size (n=nano, s=small, m=medium, l=large)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--img-size', type=int, default=416,
                       help='Image size')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--quick', action='store_true',
                       help='Use quick mini dataset for testing')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*15 + "YOLO PLANT DISEASE DETECTOR")
    print("="*70)
    
    # Create quick dataset if requested
    if args.quick or args.data is None:
        print("\nCreating quick YOLO dataset from mini subset...")
        args.data = create_quick_yolo_data()
        
        if args.data is None:
            return
    
    # Check data file
    if not Path(args.data).exists():
        print(f"\n❌ Data file not found: {args.data}")
        print("   Run: python create_subset.py --create-yolo")
        return
    
    # Initialize detector
    detector = YOLOPlantDetector(model_size=args.model_size)
    
    # Train
    print(f"\nTraining configuration:")
    print(f"  Model: YOLOv8-{args.model_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.img_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {detector.device}")
    
    results = detector.train(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        save_dir='yolo_training'
    )
    
    if results is None:
        return
    
    # Evaluate
    print("\n" + "="*70)
    detector.evaluate(args.data)
    
    # Save model info
    model_info = {
        'model_size': args.model_size,
        'epochs': args.epochs,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'data_yaml': args.data
    }
    
    save_path = Path('models/yolo')
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(save_path / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✅ Model saved to {save_path}")
    print("\n🎉 YOLO training complete!")
    print("\nNext steps:")
    print("  1. Check results in 'runs/detect/yolo_training/'")
    print("  2. Train LSTM: python train_lstm.py")
    print("  3. Create pipeline: python pipeline.py")

if __name__ == "__main__":
    main()