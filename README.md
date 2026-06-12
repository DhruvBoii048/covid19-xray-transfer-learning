# COVID-19 Chest X-Ray Classification using Transfer Learning

## Overview

This project is an independent implementation of the methodology proposed in the research paper:

**Facilitating COVID Recognition from X-rays with Computer Vision Models and Transfer Learning**
Varde et al., Multimedia Tools and Applications, 2024.

The original paper did not provide publicly available source code. Therefore, the complete classification pipeline was implemented independently based on the methodology, architecture descriptions, and experimental procedures described in the publication.

The objective is to classify chest X-ray images into three categories:

* COVID-19
* Pneumonia
* Normal (Healthy)

The project reproduces the transfer learning strategy presented in the paper and compares the performance of:

* VGG16
* VGG19
* ResNet101

---

## Research Paper

Reference:

Varde, A. S., et al.

"Facilitating COVID Recognition from X-rays with Computer Vision Models and Transfer Learning."

Published in Multimedia Tools and Applications, 2024.

This repository is intended solely as an educational and research implementation of the methodology proposed in the paper.

---

## Methodology

The implementation follows the workflow described in the paper:

1. Chest X-ray image acquisition
2. Image preprocessing and normalization
3. Data augmentation
4. Transfer learning using pretrained ImageNet models
5. Training of classifier heads
6. Evaluation using accuracy, loss curves, and confusion matrices

The project follows Transfer Learning Strategy 1 described in the paper:

* Load pretrained ImageNet weights
* Freeze convolutional feature extraction layers
* Train only the final classification layers on chest X-ray images

---

## Data Augmentation

To improve generalization and reduce overfitting, the following augmentation techniques are implemented:

* Horizontal Flipping
* Rotation
* Translation (Shift)
* Zooming
* Gaussian Noise Injection

Augmentation is implemented manually using NumPy and PIL operations.

---

## Models Implemented

### VGG16

Transfer-learning based VGG16 model with pretrained ImageNet weights and custom classifier head.

### VGG19

Transfer-learning based VGG19 model with pretrained ImageNet weights and custom classifier head.

### ResNet101

Implementation of ResNet101 using bottleneck residual blocks and skip connections with pretrained ImageNet weights.

---

## Dataset

Chest X-ray dataset:

https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

Classes:

* COVID
* Pneumonia
* Normal

Expected folder structure:

data/
├── train/
│   ├── covid/
│   ├── pneumonia/
│   └── normal/
│
└── val/
├── covid/
├── pneumonia/
└── normal/

---

## Project Structure

.
├── dataset.py
├── main.py
├── resnet_model.py
├── train_eval.py
├── vgg_models.py
├── README.md
└── data/
├── train/
└── val/

---

## Installation

Clone the repository:

git clone https://github.com/USERNAME/REPOSITORY_NAME.git

Move into the project directory:

cd REPOSITORY_NAME

Install dependencies:

pip install torch torchvision numpy pillow matplotlib scikit-learn

---

## Running Experiments

Run all models:

python main.py --model all --epochs 20

Run VGG16 only:

python main.py --model vgg16 --epochs 20

Run VGG19 only:

python main.py --model vgg19 --epochs 20

Run ResNet101 only:

python main.py --model resnet101 --epochs 20

---

## Evaluation Metrics

The implementation evaluates models using:

* Classification Accuracy
* Cross-Entropy Loss
* Confusion Matrix
* Training Curves

Generated outputs include:

* Accuracy plots
* Loss plots
* Confusion matrices
* Model checkpoints

---

## Technologies Used

* Python
* PyTorch
* NumPy
* PIL
* Matplotlib
* Scikit-Learn

---

## Key Learning Outcomes

* Research paper analysis and reproduction
* Transfer learning workflows
* Medical image classification
* CNN architectures
* VGG and ResNet implementations
* Data augmentation techniques
* Deep learning model evaluation

---

## Disclaimer

This project is intended for educational and research purposes only.

It should not be used as a clinical diagnostic tool. Medical diagnosis must always be performed by qualified healthcare professionals.
