"""
Hybrid Plant Health Monitoring Pipeline - Fixed Watering Prediction
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

class HybridPlantMonitor:
    def __init__(self, 
             resnet_path='models/resnet/best_model.pth',
             yolo_path='runs/detect/yolo_training/weights/best.pt',
             lstm_path='models/lstm/best_model.pth'):
        
        """Initialize with automatic path detection"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing Hybrid Plant Monitor...")
        print(f"   Device: {self.device}")
        
        # Auto-detect ResNet
        if resnet_path is None:
            resnet_paths = [
                'models/resnet_mini/best_model.pth',
                'models/resnet/best_model.pth',
                'best_model.pth'
            ]
            for p in resnet_paths:
                if Path(p).exists():
                    resnet_path = p
                    break
        
        # Auto-detect YOLO
        if yolo_path is None:
            yolo_paths = [
                'runs/detect/yolo_training/weights/best.pt',
                'runs/detect/train/weights/best.pt',
                'yolo_training/weights/best.pt',
                'best.pt'
            ]
            for p in yolo_paths:
                if Path(p).exists():
                    yolo_path = p
                    break
        
        # Auto-detect LSTM
        if lstm_path is None:
            lstm_paths = [
                'models/lstm/best_model.pth',
                'lstm_model.pth'
            ]
            for p in lstm_paths:
                if Path(p).exists():
                    lstm_path = p
                    break
        
        # Load models
        self.resnet_model = None
        self.resnet_classes = None
        if resnet_path and Path(resnet_path).exists():
            try:
                self._load_resnet(resnet_path)
                print(f"   ✓ ResNet loaded from {resnet_path}")
            except Exception as e:
                print(f"   ⚠ ResNet load failed: {e}")
        else:
            print(f"   ⚠ ResNet not found")
        
        self.yolo_model = None
        if yolo_path and Path(yolo_path).exists():
            try:
                self._load_yolo(yolo_path)
                print(f"   ✓ YOLO loaded from {yolo_path}")
            except Exception as e:
                print(f"   ⚠ YOLO load failed: {e}")
        else:
            print(f"   ⚠ YOLO not found")
        
        self.lstm_model = None
        self.lstm_scaler = None
        self.lstm_features = None
        if lstm_path and Path(lstm_path).exists():
            try:
                self._load_lstm(lstm_path)
                print(f"   ✓ LSTM loaded from {lstm_path}")
                print(f"   LSTM features: {self.lstm_features}")
            except Exception as e:
                print(f"   ⚠ LSTM load failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠ LSTM not found at {lstm_path}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Check if at least one model loaded
        if not any([self.resnet_model, self.yolo_model, self.lstm_model]):
            raise Exception("No models could be loaded!")
        
        print("✅ Pipeline initialized!")
    
    def _load_resnet(self, model_path):
        """Load ResNet classification model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.resnet_classes = checkpoint.get('class_names', [])
        num_classes = len(self.resnet_classes)
        
        self.resnet_model = models.resnet50(weights=None)
        num_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(num_features, num_classes)
        
        self.resnet_model.load_state_dict(checkpoint['model_state_dict'])
        self.resnet_model = self.resnet_model.to(self.device)
        self.resnet_model.eval()
    
    def _load_yolo(self, model_path):
        """Load YOLO detection model"""
        self.yolo_model = YOLO(model_path)
    
    def _load_lstm(self, model_path):
        """Load LSTM forecasting model"""
        # Load with weights_only=False to allow sklearn scaler
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.lstm_scaler = checkpoint.get('scaler')
        self.lstm_features = checkpoint.get('features', [])
        
        print(f"   Loading LSTM checkpoint...")
        print(f"   Scaler present: {self.lstm_scaler is not None}")
        print(f"   Features: {self.lstm_features}")
        
        # Simple LSTM model definition
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc1 = nn.Linear(hidden_size, 32)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(32, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc1(lstm_out[:, -1, :])
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                return self.sigmoid(out)
        
        input_size = len(self.lstm_features) if self.lstm_features else 8
        self.lstm_model = SimpleLSTM(input_size, 64, 2)
        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model = self.lstm_model.to(self.device)
        self.lstm_model.eval()
    
    def predict_disease(self, image_path, top_k=5):
        """ResNet: Classify plant disease"""
        if self.resnet_model is None:
            return {'error': 'ResNet model not loaded'}
        
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.resnet_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'disease': self.resnet_classes[idx],
                'confidence': float(prob),
                'percentage': float(prob * 100)
            })
        
        return {
            'primary_disease': predictions[0]['disease'],
            'confidence': predictions[0]['confidence'],
            'top_predictions': predictions
        }
    
    def detect_disease_location(self, image_path, conf_threshold=0.25):
        """YOLO: Detect disease regions"""
        if self.yolo_model is None:
            return {'error': 'YOLO model not loaded', 'num_detections': 0, 'detections': [], 'diseased_area_total': 0}
        
        results = self.yolo_model.predict(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                
                detections.append({
                    'disease': result.names[cls_id],
                    'confidence': conf,
                    'bbox': {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]},
                    'area_percentage': self._calculate_bbox_area(coords, result.orig_shape)
                })
        
        return {
            'num_detections': len(detections),
            'detections': detections,
            'diseased_area_total': sum(d['area_percentage'] for d in detections)
        }
    
    def _calculate_bbox_area(self, coords, img_shape):
        """Calculate bbox area percentage"""
        bbox_area = (coords[2] - coords[0]) * (coords[3] - coords[1])
        img_area = img_shape[0] * img_shape[1]
        return (bbox_area / img_area) * 100
    
    def predict_watering(self, sensor_data_sequence):
        """LSTM: Predict watering needs"""
        print(f"\n💧 predict_watering called")
        print(f"   LSTM model loaded: {self.lstm_model is not None}")
        print(f"   LSTM scaler loaded: {self.lstm_scaler is not None}")
        print(f"   LSTM features: {self.lstm_features}")
        print(f"   Sensor data length: {len(sensor_data_sequence) if sensor_data_sequence else 0}")
        
        if self.lstm_model is None:
            print(f"   ❌ LSTM model not loaded")
            return {'error': 'LSTM model not loaded', 'watering_needed': False, 'probability': 0.5}
        
        if self.lstm_scaler is None:
            print(f"   ❌ LSTM scaler not loaded")
            return {'error': 'LSTM scaler not loaded', 'watering_needed': False, 'probability': 0.5}
        
        if not sensor_data_sequence:
            print(f"   ❌ No sensor data provided")
            return {'error': 'No sensor data provided', 'watering_needed': False, 'probability': 0.5}
        
        try:
            print(f"   Building feature array...")
            features_array = []
            for i, reading in enumerate(sensor_data_sequence):
                try:
                    row = [reading[feat] for feat in self.lstm_features]
                    features_array.append(row)
                except KeyError as e:
                    print(f"   ❌ Missing feature {e} in reading {i}")
                    print(f"      Reading keys: {reading.keys()}")
                    print(f"      Expected features: {self.lstm_features}")
                    raise
            
            print(f"   ✓ Feature array built: {len(features_array)} x {len(features_array[0])}")
            
            X = np.array(features_array)
            print(f"   ✓ NumPy array shape: {X.shape}")
            
            X_scaled = self.lstm_scaler.transform(X)
            print(f"   ✓ Data scaled")
            
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
            print(f"   ✓ Tensor created: {X_tensor.shape}")
            
            with torch.no_grad():
                output = self.lstm_model(X_tensor)
                probability = float(output[0][0])
            
            print(f"   ✓ Prediction complete: probability = {probability:.4f}")
            
            watering_needed = probability > 0.5
            hours_until = int((1 - probability) * 48) if watering_needed else int((1 - probability) * 72)
            
            result = {
                'watering_needed': watering_needed,
                'probability': probability,
                'confidence': abs(probability - 0.5) * 2,
                'hours_until_watering': hours_until,
                'recommendation': self._get_watering_recommendation(probability)
            }
            
            print(f"   ✓ Result: {result}")
            return result
            
        except Exception as e:
            print(f"   ❌ LSTM prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': f'LSTM prediction failed: {str(e)}', 'watering_needed': False, 'probability': 0.5}
    
    def _get_watering_recommendation(self, probability):
        """Generate recommendation"""
        if probability > 0.8:
            return "⚠️ Water immediately - soil moisture critical"
        elif probability > 0.6:
            return "💧 Water within next 6 hours"
        elif probability > 0.4:
            return "🕐 Water within 24 hours"
        else:
            return "✅ Soil moisture adequate - no watering needed"
    
    def comprehensive_analysis(self, image_path, sensor_data_sequence=None):
        """Complete analysis"""
        print(f"\n🔍 comprehensive_analysis called")
        print(f"   Image path: {image_path}")
        print(f"   Sensor data provided: {sensor_data_sequence is not None}")
        if sensor_data_sequence:
            print(f"   Sensor data length: {len(sensor_data_sequence)}")
            print(f"   First reading: {sensor_data_sequence[0]}")
        
        results = {'image_path': str(image_path)}
        
        # Disease classification
        if self.resnet_model:
            print(f"   Running ResNet classification...")
            results['disease_classification'] = self.predict_disease(image_path)
            print(f"   ✓ ResNet complete")
        
        # Disease localization
        if self.yolo_model:
            print(f"   Running YOLO detection...")
            results['disease_localization'] = self.detect_disease_location(image_path)
            print(f"   ✓ YOLO complete")
        
        # Watering prediction
        print(f"   Checking watering prediction...")
        if self.lstm_model and sensor_data_sequence:
            print(f"   ✓ LSTM model available and sensor data provided - running prediction...")
            results['watering_prediction'] = self.predict_watering(sensor_data_sequence)
            print(f"   ✓ Watering prediction complete")
        else:
            msg = []
            if not self.lstm_model:
                msg.append("LSTM model not loaded")
            if not sensor_data_sequence:
                msg.append("No sensor data provided")
            results['watering_prediction'] = {'message': ' | '.join(msg)}
            print(f"   ⚠️ Skipped watering prediction: {results['watering_prediction']['message']}")
        
        # Health score
        print(f"   Calculating health score...")
        results['overall_health_score'] = self._calculate_health_score(results)
        print(f"   ✓ Health score: {results['overall_health_score']}")
        
        return results
    
    def _calculate_health_score(self, results):
        """Calculate health score"""
        score = 100
        
        disease_data = results.get('disease_classification', {})
        if disease_data and 'healthy' not in disease_data.get('primary_disease', '').lower():
            confidence = disease_data.get('confidence', 0)
            score -= (confidence * 30)
        
        location_data = results.get('disease_localization', {})
        diseased_area = location_data.get('diseased_area_total', 0)
        score -= min(diseased_area, 20)
        
        watering_data = results.get('watering_prediction', {})
        if watering_data.get('watering_needed', False):
            probability = watering_data.get('probability', 0)
            score -= (probability * 20)
        
        return max(0, min(100, round(score)))