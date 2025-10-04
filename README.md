# 🌱 Smart Plant Health Monitoring

## 📌 Project Overview
Plant diseases are a major threat to global food security, reducing crop yields and farmer income. Early and accurate detection of plant diseases can help prevent large-scale losses and improve agricultural productivity.  
This project leverages **deep learning** and **computer vision** to build an automated **plant health monitoring system** that can identify plant diseases from leaf images with high accuracy.

Using the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download), we aim to train **Convolutional Neural Networks (CNNs)** and explore **transfer learning models** such as ResNet, EfficientNet, and DenseNet to detect plant diseases across multiple crops.

---

## 📂 Dataset
- **Source:** [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download)  
- **Size:** ~87,000 images  
- **Categories:** 38 different plant disease classes + healthy leaves  
- **Image Quality:** High-resolution, well-structured folder hierarchy  

### ✅ Why This Dataset?
- **Large-scale, labeled images** → Enables training of deep neural networks.  
- **Diversity of classes** → Covers multiple crops and disease types for generalization.  
- **Benchmark availability** → Publicly benchmarked on Kaggle, allowing fair comparison.  
- **Relevance** → Aligns directly with the goal of automated plant disease detection.  

---

## 🎯 Objectives
1. Build a robust deep learning model for **multi-class plant disease classification**.  
2. Compare **CNN architectures** and **transfer learning** approaches.  
3. Optimize model performance with techniques like **data augmentation**, **learning rate scheduling**, and **regularization**.  
4. Deploy the trained model into a simple **web or mobile application** for real-time usage by farmers.  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Deep Learning Frameworks:** TensorFlow / PyTorch  
- **Model Architectures:** CNNs, ResNet, EfficientNet, DenseNet  
- **Tools & Libraries:** OpenCV, scikit-learn, Matplotlib, NumPy, Pandas  
- **Deployment (future scope):** Streamlit / Flask / FastAPI, possibly mobile integration  

---

## 🚀 Project Workflow
1. **Data Preprocessing**  
   - Train-validation-test split  
   - Image resizing & normalization  
   - Data augmentation (rotation, zoom, flip, brightness)  

2. **Model Development**  
   - Baseline CNN  
   - Transfer Learning with pre-trained models  
   - Hyperparameter tuning  

3. **Evaluation**  
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion matrix & class-wise performance  

4. **Deployment (Future Work)**  
   - Web-based demo using Streamlit/Flask  
   - Mobile-friendly model integration for field usage  

---

## 📊 Expected Outcomes
- High accuracy in plant disease classification.  
- A scalable and generalizable solution for **smart agriculture**.  
- A deployed prototype to showcase real-time disease detection.  

---

## 👥 Team & Contributions
- **Data Selection & Preprocessing** → Understanding dataset and preparing it for training.  
- **Model Training & Evaluation** → Designing CNN models and applying transfer learning.  
- **Deployment & Presentation** → Building user-friendly interface and project documentation.  

---

## 🔮 Future Scope
- Integration with **IoT devices** for continuous monitoring.  
- Building a **mobile app** for farmers to upload leaf images.  
- Adding **disease severity estimation** and **treatment suggestions**.  
- Expanding dataset with region-specific crops for better applicability.  

---

## 📜 References
- [Kaggle Dataset - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download)  
- Research papers on deep learning in agriculture and plant pathology.  

---

💡 *This project aims to bridge the gap between AI research and real-world agriculture, empowering farmers with a practical, accessible, and automated disease detection tool.*
