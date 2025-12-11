# Smart Plant Health Monitor

An AI-powered system combining YOLOv5, ResNet, and LSTMs to classify plant health with 90% accuracy and predict nutritional needs.

## Team Members
- Raj Prasad Ambavane (UID: 121998644)
- Dhruv Dubey (UID: 122129887)
- Vansh Pradeep Jain (UID: 122082892)
- Shantanu Ramavat (UID: 121997708)

## Project Overview
- **Computer Vision**: YOLOv5 for leaf detection + ResNet for disease classification
- **Time-Series Prediction**: LSTM for nutritional needs forecasting
- **Deployment**: FastAPI microservice with Streamlit UI, containerized in Docker
- **Target Accuracy**: 90%+

## Dataset
- **Source**: [New Plant Diseases Dataset](http://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Size**: ~87,000 RGB images
- **Classes**: 38 classes (healthy + diseased leaves)

## Project Structure
\\\
plant_monitor/
├── data/              # Dataset storage
├── models/            # Trained model weights
├── src/               # Source code
├── notebooks/         # Jupyter notebooks for experimentation
├── config/            # Configuration files
├── docker/            # Docker configurations
└── tests/             # Unit tests
\\\

## Setup Instructions
\\\ash
# Clone repository
git clone https://github.com/vjain0812/DATA602_smartPlant_health.git
cd plant_monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
\\\

## Usage
(To be updated as development progresses)

## Development Phases
- [x] Phase 1: Environment Setup
- [ ] Phase 2: Data Preparation
- [ ] Phase 3: YOLOv5 Development
- [ ] Phase 4: ResNet Classification
- [ ] Phase 5: LSTM Time-Series
- [ ] Phase 6: FastAPI Backend
- [ ] Phase 7: Streamlit UI
- [ ] Phase 8: Docker Containerization
- [ ] Phase 9: HPC Deployment

## License
Academic Project - DATA602
