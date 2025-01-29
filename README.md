Project: Disease Detection in Maize Using Deep Learning 🌿

## Overview
This project focuses on differentiating and predicting **Leaf Spot, Rust, and Puccinia sorghi** in maize crops using an **EfficientNet** model.

## Dataset(5600) 📂
- Images of **Leaf Spot, Rust, Puccinia sorghi, and Healthy Leaves**.
- Collected from maize fields in Sri Lanka.
- Preprocessed and augmented for deep learning.

## Model Architecture 🧠
Using **Densenet 121**, trained with **TensorFlow & Keras**.

## Screenshots 📸
### **1. Sample Data**
![Dataset Samples](images/dataset_samples.png)
![Model Training Graph](F:\1.Reaserch\balanced\data\Blight\Corn_Blight (42).jpg)
![Model Training Graph](images/training_loss.png)


### **2. Model Training Progress**
![Training Loss Graph](images/training_loss.png)

## Installation & Usage 🛠️
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
python train.py
