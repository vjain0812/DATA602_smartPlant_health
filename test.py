python -c "
from pathlib import Path
import os

print('Checking for trained models...\n')

# Check ResNet
resnet_paths = [
    'models/resnet/best_model.pth',
    'models/resnet_mini/best_model.pth'
]
for p in resnet_paths:
    if Path(p).exists():
        print(f'✓ Found ResNet: {p}')
        print(f'  Size: {Path(p).stat().st_size / 1024 / 1024:.1f} MB')
    else:
        print(f'✗ Not found: {p}')

print()

# Check YOLO
yolo_paths = [
    'runs/detect/yolo_training/weights/best.pt',
    'runs/detect/train/weights/best.pt',
    'yolo_training/weights/best.pt'
]
for p in yolo_paths:
    if Path(p).exists():
        print(f'✓ Found YOLO: {p}')
        print(f'  Size: {Path(p).stat().st_size / 1024 / 1024:.1f} MB')
    else:
        print(f'✗ Not found: {p}')

print()

# Check LSTM
lstm_path = 'models/lstm/best_model.pth'
if Path(lstm_path).exists():
    print(f'✓ Found LSTM: {lstm_path}')
    print(f'  Size: {Path(lstm_path).stat().st_size / 1024 / 1024:.1f} MB')
else:
    print(f'✗ Not found: {lstm_path}')

print('\nSearching for any model files...')
for ext in ['*.pth', '*.pt']:
    for f in Path('.').rglob(ext):
        print(f'  Found: {f}')
"