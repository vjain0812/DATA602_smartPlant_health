"""
FastAPI Backend for Smart Plant Health Monitor
Provides REST API endpoints for disease detection and watering prediction
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path
import uvicorn
import tempfile
import shutil

# Add parent directory to path to import pipeline
sys.path.append(str(Path(__file__).parent.parent))
from pipeline import HybridPlantMonitor

# Initialize FastAPI app
app = FastAPI(
    title="Smart Plant Health Monitor API",
    description="AI-powered plant disease detection and health monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

# Pydantic models
class SensorReading(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float
    light_intensity: float
    npk_nitrogen: Optional[float] = 75.0
    npk_phosphorus: Optional[float] = 70.0
    npk_potassium: Optional[float] = 80.0
    health_score: Optional[float] = 85.0

class WateringRequest(BaseModel):
    sensor_readings: List[SensorReading]

class HealthResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    try:
        print("🚀 Initializing Smart Plant Health Monitor...")
        pipeline = HybridPlantMonitor()
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        pipeline = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Plant Health Monitor API",
        "version": "1.0.0",
        "status": "running" if pipeline is not None else "error",
        "endpoints": {
            "disease_classification": "/api/v1/predict/disease",
            "disease_detection": "/api/v1/predict/yolo",
            "watering_prediction": "/api/v1/predict/watering",
            "comprehensive": "/api/v1/predict/comprehensive",
            "health_check": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Pipeline not initialized"}
        )
    
    return {
        "status": "healthy",
        "models": {
            "resnet": pipeline.resnet_model is not None,
            "yolo": pipeline.yolo_model is not None,
            "lstm": pipeline.lstm_model is not None
        }
    }

@app.post("/api/v1/predict/disease")
async def predict_disease(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    ResNet: Classify plant disease from image
    
    Args:
        file: Plant leaf image
        top_k: Number of top predictions to return
    
    Returns:
        Disease classification results
    """
    if pipeline is None or pipeline.resnet_model is None:
        raise HTTPException(
            status_code=503,
            detail="ResNet model not available"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Run prediction
        results = pipeline.predict_disease(tmp_path, top_k=top_k)
        
        # Clean up
        Path(tmp_path).unlink()
        
        return {
            "status": "success",
            "data": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/yolo")
async def predict_yolo(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    """
    YOLO: Detect and localize diseases on leaf
    
    Args:
        file: Plant leaf image
        confidence: Confidence threshold (0-1)
    
    Returns:
        Bounding boxes and detected diseases
    """
    if pipeline is None or pipeline.yolo_model is None:
        raise HTTPException(
            status_code=503,
            detail="YOLO model not available"
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Run detection
        results = pipeline.detect_disease_location(tmp_path, conf_threshold=confidence)
        
        # Clean up
        Path(tmp_path).unlink()
        
        return {
            "status": "success",
            "data": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/watering")
async def predict_watering(request: WateringRequest):
    """
    LSTM: Predict watering needs from sensor data
    
    Args:
        request: Recent sensor readings (last 24 hours)
    
    Returns:
        Watering prediction and recommendation
    """
    if pipeline is None or pipeline.lstm_model is None:
        raise HTTPException(
            status_code=503,
            detail="LSTM model not available"
        )
    
    try:
        # Convert sensor readings to list of dicts
        sensor_sequence = [reading.dict() for reading in request.sensor_readings]
        
        # Run prediction
        results = pipeline.predict_watering(sensor_sequence)
        
        return {
            "status": "success",
            "data": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/comprehensive")
async def comprehensive_analysis(
    file: UploadFile = File(...),
    temperature: Optional[float] = Form(None),
    humidity: Optional[float] = Form(None),
    soil_moisture: Optional[float] = Form(None),
    light_intensity: Optional[float] = Form(None),
    npk_nitrogen: Optional[float] = Form(75.0),
    npk_phosphorus: Optional[float] = Form(70.0),
    npk_potassium: Optional[float] = Form(80.0)
):
    """
    Complete analysis: ResNet + YOLO + LSTM
    
    Args:
        file: Plant leaf image
        temperature: Current temperature (°C)
        humidity: Current humidity (%)
        soil_moisture: Current soil moisture (%)
        light_intensity: Current light intensity (lux)
        npk_nitrogen: Nitrogen level
        npk_phosphorus: Phosphorus level
        npk_potassium: Potassium level
    
    Returns:
        Comprehensive plant health assessment with NPK data
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    
    try:
        # Save image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Prepare sensor data if provided
        sensor_data = None
        if all(v is not None for v in [temperature, humidity, soil_moisture, light_intensity]):
            print(f"📊 Creating sensor sequence with: Temp={temperature}, Humidity={humidity}, Soil={soil_moisture}, Light={light_intensity}")
            
            # Create 24 hours of mock data with gradual soil moisture decrease
            sensor_data = []
            for i in range(24):
                reading = {
                    'temperature': float(temperature),
                    'humidity': float(humidity),
                    'soil_moisture': float(soil_moisture - (i * 0.5)),  # Gradual decrease
                    'light_intensity': float(light_intensity),
                    'npk_nitrogen': float(npk_nitrogen),
                    'npk_phosphorus': float(npk_phosphorus),
                    'npk_potassium': float(npk_potassium),
                    'health_score': 85.0
                }
                sensor_data.append(reading)
            
            print(f"✅ Created sensor sequence with {len(sensor_data)} readings")
            print(f"   First reading: {sensor_data[0]}")
            print(f"   Last reading: {sensor_data[-1]}")
        else:
            print(f"⚠️ Incomplete sensor data - LSTM will be skipped")
            print(f"   temperature: {temperature}")
            print(f"   humidity: {humidity}")
            print(f"   soil_moisture: {soil_moisture}")
            print(f"   light_intensity: {light_intensity}")
        
        # Run comprehensive analysis
        print(f"🔄 Running comprehensive analysis...")
        results = pipeline.comprehensive_analysis(tmp_path, sensor_data)
        print(f"✅ Analysis complete!")
        
        # Add NPK data to results
        results['npk_nutrients'] = {
            'nitrogen': npk_nitrogen,
            'phosphorus': npk_phosphorus,
            'potassium': npk_potassium,
            'status': {
                'nitrogen': 'optimal' if 70 <= npk_nitrogen <= 80 else 'adjust' if 60 <= npk_nitrogen <= 90 else 'critical',
                'phosphorus': 'optimal' if 65 <= npk_phosphorus <= 75 else 'adjust' if 55 <= npk_phosphorus <= 85 else 'critical',
                'potassium': 'optimal' if 75 <= npk_potassium <= 85 else 'adjust' if 65 <= npk_potassium <= 95 else 'critical'
            }
        }
        
        # Debug watering prediction
        if 'watering_prediction' in results:
            print(f"💧 Watering prediction: {results['watering_prediction']}")
        else:
            print(f"⚠️ No watering prediction in results")
        
        # Clean up
        Path(tmp_path).unlink()
        
        return {
            "status": "success",
            "data": results
        }
        
    except Exception as e:
        print(f"❌ Error in comprehensive_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/info")
async def get_models_info():
    """Get information about loaded models"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "status": "success",
        "models": {
            "resnet": {
                "loaded": pipeline.resnet_model is not None,
                "num_classes": len(pipeline.resnet_classes) if pipeline.resnet_classes else 0,
                "classes": pipeline.resnet_classes[:10] if pipeline.resnet_classes else []
            },
            "yolo": {
                "loaded": pipeline.yolo_model is not None,
            },
            "lstm": {
                "loaded": pipeline.lstm_model is not None,
                "features": pipeline.lstm_features if pipeline.lstm_features else []
            }
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )

if __name__ == "__main__":
    print("="*70)
    print(" "*15 + "SMART PLANT HEALTH MONITOR API")
    print("="*70)
    print("\n🚀 Starting FastAPI server...")
    print("   API docs: http://localhost:8000/docs")
    print("   Health check: http://localhost:8000/health")
    print("\n   Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )