Project: Disease Detection in Maize Using Deep Learning 🌿
Machine Learning Project - Image Classification Model (DenseNet121): This project involves the development of an image classification model using DenseNet121, a deep learning architecture known for its efficient feature reuse and high performance. The model is trained to classify images into predefined categories, leveraging DenseNet's dense connectivity pattern to enhance feature propagation and reduce overfitting. The model's accuracy and efficiency are validated on a labeled dataset, aiming to deliver reliable image classification results.

Web Development - Flask, HTML, CSS: The web development aspect of the project utilizes Flask, a lightweight Python web framework, to build a user-friendly interface for interacting with the image classification model. The frontend is designed using HTML and CSS, ensuring a responsive and visually appealing design. This allows users to upload images, interact with the model, and view classification results in real time through an intuitive web interface.

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
