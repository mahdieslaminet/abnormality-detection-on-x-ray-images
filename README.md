# Abnormality Detection on X-ray Images Using AI
University Semester Project

Project Overview
This project focuses on automatic abnormality detection in musculoskeletal X-ray images using Artificial Intelligence (Deep Learning).

The system is developed and evaluated using the MURA (Musculoskeletal Radiographs) dataset, which is one of the largest publicly available datasets for medical X-ray abnormality detection.

The goal of this project is to design, train, and evaluate a deep learning–based model capable of distinguishing normal and abnormal X-ray studies, supporting computer-aided diagnosis (CAD) in medical imaging.

Google Drive Resources
All project-related documents, reports, and supplementary materials are available in the following Google Drive folder:

🔗 Google Drive Link:

https://drive.google.com/drive/folders/1Cvb4J8bvPEnrZQaycc52xJwt7zLzxJwi

GitHub Repository

The GitHub repository contains the full implementation of the project, including code, notebooks, and configuration files required to reproduce the experiments.

Dataset
MURA Dataset
The project uses the MURA (Musculoskeletal Radiographs) dataset, released by the Stanford ML Group.

Key characteristics:

Over 40,000 X-ray images
7 upper-extremity body parts:
Wrist
Elbow
Shoulder
Finger
Forearm
Humerus
Hand
Binary labels:
Normal
Abnormal
Study-level labels based on expert radiologist annotations

📌 Official Dataset Page:

https://stanfordmlgroup.github.io/competitions/mura/

Method
The abnormality detection task is formulated as a binary image classification problem.

Pipeline Overview
Data Loading

X-ray images are loaded and resized
Labels are extracted at the study level
Preprocessing

Image normalization
Resizing to fixed input dimensions
Optional data augmentation (rotation, flipping, contrast adjustment)
Model Architecture

Convolutional Neural Network (CNN)
Transfer learning using a pretrained backbone (e.g., ResNet, DenseNet)
Final fully connected layers adapted for binary classification
Loss Function

Binary Cross-Entropy Loss
Optimization

Adam optimizer
Learning rate scheduling
Training
Framework: Python with Deep Learning libraries (TensorFlow/Keras or PyTorch)
Batch-based training
Validation performed on a held-out validation set
Class imbalance handled using weighted loss or sampling strategies
Training is conducted using Jupyter Notebooks, enabling step-by-step experimentation and visualization.

Results

The trained model is evaluated using standard medical imaging metrics, including:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Results demonstrate the capability of deep learning models to learn meaningful representations from musculoskeletal X-ray images and distinguish abnormal cases with reasonable performance for a university-level project.

Running the Project

Using Jupyter Notebook (Local)
Clone the repository:
bash
git clone https://github.com/mahdieslaminet/abnormality-detection-on-x-ray-images.git
cd abnormality-detection-on-x-ray-images
Install dependencies:
bash
pip install -r requirements.txt
Launch Jupyter Notebook:
bash
jupyter notebook
Open the provided notebooks and run cells sequentially.

Running on Google Colab
Although the project was developed using Jupyter Notebook, it can also be run on Google Colab:

Upload the notebook files to Google Colab
Upload or mount the MURA dataset using Google Drive
Install required dependencies inside Colab:
python
!pip install -r requirements.txt
Run the notebook cells as usual
Using Colab allows access to free GPU acceleration, which significantly reduces training time.

What I Added

End-to-end implementation of an abnormality detection pipeline
Preprocessing and dataset handling for MURA
Training and evaluation workflow
Experimental analysis and result interpretation
Academic report and presentation materials
Clear structure suitable for a university semester project
Source Credit
Primary Dataset & Paper

Rajpurkar et al.,

MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs,

Stanford ML Group

Deep Learning Concepts & Models

ResNet
DenseNet
CNN-based medical image analysis literature
Referenced GitHub Repositories

Open-source implementations and tutorials related to medical image classification and transfer learning (used for learning and inspiration only)
All external sources are used strictly for educational and research purposes.

Disclaimer

This project is developed solely for educational purposes as part of a university semester course.

It is not intended for clinical or diagnostic use.
