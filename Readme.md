# Smart Plant Health Monitor

A machine learning based system for plant disease detection and watering prediction using image data and sensor readings.

---

## Project Objective

The objective of this project is to monitor plant health by:

* Identifying plant diseases from leaf images
* Detecting infected regions on leaves
* Predicting whether a plant needs watering using sensor data

---

## Models Used

| Task | Model |
| --- | --- |
| Disease classification | ResNet50 |
| Disease detection | YOLOv8 |
| Watering prediction | LSTM |

---

## Dataset

### Plant Disease Dataset

* Source: Kaggle – New Plant Diseases Dataset
* 38 disease and healthy classes
* RGB leaf images

### Sensor Dataset

* Includes Temperature, Humidity, Soil moisture, Light intensity, NPK values, and Health score
* Sensor data is synthetically generated for time-series analysis

---

## Project Structure

```text
SPH/
├── dataset_subset/      # Image dataset for training
├── dataset_yolo_mini/   # YOLO formatted dataset
├── data/timeseries/     # Sensor data
├── models/              # Saved models
├── train_resnet.py      # ResNet training
├── train_yolo.py        # YOLO training
├── train_lstm.py        # LSTM training
├── pipeline.py          # Combined prediction pipeline
├── main.py              # FastAPI backend
└── streamlit_app.py     # Web interface

```

---

## Installation

### Requirements

* Python 3.8+
* CUDA GPU (optional but recommended)

### Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate

```

2. Install dependencies:

```bash
pip install -r requirements.txt

```

---

## Execution Guide

### Model Training

```bash
# Train ResNet
python train_resnet.py

# Train YOLO
python train_yolo.py

# Train LSTM
python generate_sensor_data.py
python train_lstm.py

```

### Running the Application

1. Start the Backend:

```bash
python main.py

```

2. Start the Frontend:

```bash
streamlit run streamlit_app.py

```

Open the browser at: `http://localhost:8501`

---

## How it Works

1. User uploads a leaf image.
2. ResNet classifies the specific disease.
3. YOLO detects and boxes infected regions.
4. Sensor data is processed by the LSTM model.
5. System predicts watering requirements and displays all results on the interface.

---

## Results

* Disease classification accuracy: ~89%
* Disease detection mAP: ~85%
* Watering prediction accuracy: ~92%

---

## Tools and Technologies

* Python
* PyTorch
* YOLOv8
* FastAPI
* Streamlit
* OpenCV
* NumPy, Pandas

---

## Conclusion

This project demonstrates the integration of multiple machine learning architectures to build a comprehensive plant health monitoring system. It assists in early disease identification and precision irrigation management.
