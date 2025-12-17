"""
Create a manageable subset of the plant disease dataset
Reduces 87K images to ~7K images for faster training
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def create_subset(source_dir, target_dir, samples_per_class=150):
    """
    Create a smaller subset of the dataset
    
    Args:
        source_dir: Original dataset directory (train or valid)
        target_dir: New subset directory
        samples_per_class: Number of images per class
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    total_copied = 0
    class_stats = []
    
    class_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    print(f"\nProcessing {len(class_folders)} classes...")
    
    for class_folder in tqdm(class_folders, desc="Creating subset"):
        # Get all images
        images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.JPG'))
        
        if len(images) == 0:
            print(f"⚠️  {class_folder.name}: No images found")
            continue
        
        # Random sample
        num_samples = min(samples_per_class, len(images))
        selected = random.sample(images, num_samples)
        
        # Create class folder in target
        target_class = target_path / class_folder.name
        target_class.mkdir(exist_ok=True)
        
        # Copy images
        for img in selected:
            shutil.copy2(img, target_class / img.name)
        
        total_copied += len(selected)
        class_stats.append({
            'class': class_folder.name,
            'original': len(images),
            'selected': len(selected)
        })
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUBSET CREATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total images copied: {total_copied:,}")
    print(f"Total classes: {len(class_stats)}")
    print(f"\nPer-class breakdown:")
    print(f"{'Class':<45} {'Original':<10} {'Selected':<10}")
    print(f"{'-'*70}")
    
    for stat in class_stats[:10]:  # Show first 10
        print(f"{stat['class']:<45} {stat['original']:<10} {stat['selected']:<10}")
    
    if len(class_stats) > 10:
        print(f"... and {len(class_stats) - 10} more classes")
    
    return total_copied

def create_yolo_format(subset_dir, output_dir):
    """
    Convert subset to YOLO format for object detection
    Creates bounding box annotations for the entire image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create images and labels directories
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Create class mapping
    subset_path = Path(subset_dir)
    classes = sorted([f.name for f in (subset_path / 'train').iterdir() if f.is_dir()])
    
    # Save class names
    with open(output_path / 'classes.txt', 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    print(f"\n{'='*70}")
    print(f"CONVERTING TO YOLO FORMAT")
    print(f"{'='*70}")
    print(f"Number of classes: {len(classes)}")
    
    for split in ['train', 'val']:
        source_split = 'train' if split == 'train' else 'valid'
        source_dir = subset_path / source_split
        
        if not source_dir.exists():
            continue
        
        total_images = 0
        
        for class_idx, class_name in enumerate(tqdm(classes, desc=f"Processing {split}")):
            class_dir = source_dir / class_name
            
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG'))
            
            for img_path in images:
                # Copy image
                dest_img = output_path / 'images' / split / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Create label file (full image bounding box)
                label_file = output_path / 'labels' / split / f"{img_path.stem}.txt"
                with open(label_file, 'w') as f:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    # For full image: center at 0.5, 0.5, width and height 1.0
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                
                total_images += 1
        
        print(f"  {split}: {total_images} images")
    
    # Create data.yaml for YOLOv5
    yaml_content = f"""# Plant Disease Dataset - YOLO Format
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(classes)}  # number of classes
names: {classes}
"""
    
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ YOLO format dataset created at: {output_path}")
    print(f"   Use 'data.yaml' for training")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create dataset subset')
    parser.add_argument('--train-samples', type=int, default=50, 
                       help='Images per class for training (default: 50)')
    parser.add_argument('--val-samples', type=int, default=20,
                       help='Images per class for validation (default: 20)')
    parser.add_argument('--create-yolo', action='store_true',
                       help='Also create YOLO format dataset')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*20 + "DATASET SUBSET CREATOR")
    print("="*70)
    
    # Check if dataset exists
    if not Path('dataset/train').exists():
        print("\n❌ Error: 'dataset/train' not found!")
        print("   Please download the dataset first:")
        print("   https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        exit(1)
    
    # Create subset
    print("\n📁 Creating training subset...")
    train_count = create_subset(
        'dataset/train', 
        'dataset_subset/train', 
        samples_per_class=args.train_samples
    )
    
    print("\n📁 Creating validation subset...")
    val_count = create_subset(
        'dataset/valid', 
        'dataset_subset/valid', 
        samples_per_class=args.val_samples
    )
    
    print(f"\n{'='*70}")
    print(f"✅ SUBSET CREATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Training images: {train_count:,}")
    print(f"Validation images: {val_count:,}")
    print(f"Total: {train_count + val_count:,}")
    print(f"\nLocation: dataset_subset/")

    if args.create_yolo:
        print("\n" + "="*70)
        create_yolo_format('dataset_subset', 'dataset_yolo')
    
    print("\n🎉 Ready for training!")
    print("\nNext steps:")
    print("  1. Train ResNet: python src/train_resnet.py")
    print("  2. Train YOLO: python src/train_yolo.py")
    print("  3. Train LSTM: python src/train_lstm.py")