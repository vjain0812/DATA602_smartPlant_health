"""
Create ultra-tiny subset for RAPID testing
Only 10 classes, 20 images each = 200 training images total
"""

import shutil
from pathlib import Path
import random
from tqdm import tqdm

def create_mini_subset(source_dir, target_dir, num_classes=10, samples_per_class=20):
    """
    Create minimal subset for quick testing
    
    Args:
        source_dir: Original dataset directory
        target_dir: Mini subset directory
        num_classes: Number of classes to include (default: 10)
        samples_per_class: Images per class (default: 20)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    if not source_path.exists():
        print(f"❌ Source not found: {source_dir}")
        return

    all_classes = [f for f in source_path.iterdir() if f.is_dir()]
    selected_classes = random.sample(all_classes, min(num_classes, len(all_classes)))
    
    print(f"\nSelected {len(selected_classes)} classes:")
    for cls in selected_classes:
        print(f"  • {cls.name}")
    
    total_copied = 0
    
    print(f"\nCopying images...")
    for class_folder in tqdm(selected_classes):

        images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.JPG'))
        
        if len(images) == 0:
            continue

        num_samples = min(samples_per_class, len(images))
        selected = random.sample(images, num_samples)

        target_class = target_path / class_folder.name
        target_class.mkdir(exist_ok=True)

        for img in selected:
            shutil.copy2(img, target_class / img.name)
        
        total_copied += len(selected)
    
    return total_copied, len(selected_classes)

if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "MINI SUBSET CREATOR")
    print("="*70)
    print("\nCreating ultra-tiny subset for RAPID testing...")
    print("Perfect for quick experiments and debugging!")

    print("\n📁 Creating mini training set...")
    train_count, train_classes = create_mini_subset(
        'dataset/train',
        'dataset_mini/train',
        num_classes=10,
        samples_per_class=20
    )

    print("\n📁 Creating mini validation set...")
    val_count, val_classes = create_mini_subset(
        'dataset/valid',
        'dataset_mini/valid',
        num_classes=10,
        samples_per_class=10
    )
    
    print(f"\n{'='*70}")
    print("✅ MINI SUBSET CREATED!")
    print(f"{'='*70}")
    print(f"Training: {train_count} images, {train_classes} classes")
    print(f"Validation: {val_count} images, {val_classes} classes")
    print(f"Total: {train_count + val_count} images")
    print(f"\nLocation: dataset_mini/")
    
    print(f"\n⚡ SUPER FAST TRAINING:")
    print("  python train_resnet.py --train-dir dataset_mini/train --val-dir dataset_mini/valid --epochs 5 --batch-size 16")
    print("\n  Expected time: 2-3 minutes! 🚀")