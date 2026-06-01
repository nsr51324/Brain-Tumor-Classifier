# Brain Tumor Classification using Deep Learning

## Overview
This project is a deep learning-based image classification system that detects and classifies brain tumors from MRI images into four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary

The model is built using a Convolutional Neural Network (CNN) trained on MRI scans.

## Features
- Image classification using CNN
- Data preprocessing using ImageDataGenerator
- Train/Validation/Test split
- Confusion Matrix visualization
- Classification Report (Precision, Recall, F1-score)
- Streamlit web app for real-time prediction

##  Model Architecture
The model is a deep CNN consisting of:
- Multiple Conv2D layers (64 → 512 filters)
- MaxPooling layers for feature reduction
- Fully connected Dense layers
- Softmax output layer for 4 classes

Optimizer: Adamax  
Loss Function: Categorical Crossentropy  
Metric: Accuracy  

## Dataset
MRI brain tumor dataset containing:
- Training images
- Testing images

Each image is labeled into one of the four classes.

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Results
- Training Accuracy: ~98%
- Validation Accuracy: ~95%
- Test Accuracy: ~93–94%

## Installation

```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
pip install -r requirements.txt
