Here is a **clean, student-level `README.md`** written in **Markdown code**, ready to paste directly into your project.

```md
# Smart Plant Health Monitor

A machine learning based system for plant disease detection and watering prediction using image data and sensor readings.

---

## Project Objective

The objective of this project is to monitor plant health by:
- Identifying plant diseases from leaf images
- Detecting infected regions on leaves
- Predicting whether a plant needs watering using sensor data

---

## Models Used

| Task | Model |
|----|----|
| Disease classification | ResNet50 |
| Disease detection | YOLOv8 |
| Watering prediction | LSTM |

---

## Dataset

### Plant Disease Dataset
- Source: Kaggle – New Plant Diseases Dataset
- 38 disease and healthy classes
- RGB leaf images

### Sensor Dataset
- Temperature
- Humidity
- Soil moisture
- Light intensity
- NPK values
- Health score  
(Sensor data is synthetically generated)

---

## Project Structure

```

SPH/
├── dataset_subset/        # Image dataset for training
├── dataset_yolo_mini/     # YOLO formatted dataset
├── data/timeseries/       # Sensor data
├── models/                # Saved models
├── train_resnet.py        # ResNet training
├── train_yolo.py          # YOLO training
├── train_lstm.py          # LSTM training
├── pipeline.py            # Combined prediction pipeline
├── main.py                # FastAPI backend
└── streamlit_app.py       # Web interface

````

---

## Installation

### Requirements
- Python 3.8+
- CUDA GPU (optional but recommended)

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

---

## Dataset Preparation

```bash
python create_subset.py
python create_mini_subset.py
```

---

## Model Training

### Train ResNet

```bash
python train_resnet.py
```

### Train YOLO

```bash
python train_yolo.py
```

### Train LSTM

```bash
python generate_sensor_data.py
python train_lstm.py
```

---

## Running the Application

### Start Backend

```bash
python main.py
```

### Start Frontend

```bash
streamlit run streamlit_app.py
```

Open browser at:

```
http://localhost:8501
```

---

## Working of the System

1. User uploads a leaf image
2. ResNet classifies the disease
3. YOLO detects infected regions
4. Sensor data is passed to LSTM
5. System predicts watering requirement
6. Results are displayed on the web interface

---

## Results

* Disease classification accuracy: ~89%
* Disease detection mAP: ~85%
* Watering prediction accuracy: ~92%

---

## Tools & Technologies

* Python
* PyTorch
* YOLOv8
* FastAPI
* Streamlit
* OpenCV
* NumPy, Pandas

---

## Conclusion

This project shows how multiple machine learning models can be combined to build an intelligent plant health monitoring system. It helps in early disease detection and efficient watering management.

```
