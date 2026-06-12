# COVID-19 X-Ray Classification using Transfer Learning

This project is an independent implementation of the methodology proposed in the paper. The original publication did not provide source code, so the complete training and evaluation pipeline was developed based on the paper's descriptions and experimental setup.

Implementation of the paper:
**"Facilitating COVID recognition from X-rays with computer vision models and transfer learning"**
Varde et al., Multimedia Tools and Applications, 2023.

Paper: https://link.springer.com/article/10.1007/s11042-023-15744-9

---

## What this does

Classifies chest X-ray images into 3 classes:
- **COVID-positive**
- **Pneumonia-positive**
- **Normal / Healthy**

Uses Transfer Learning (Strategy 1 from the paper): pretrained ImageNet weights are
loaded into VGG16 / VGG19 / ResNet101, all convolutional layers are frozen, and only
the final classifier head is trained on chest X-ray data.

---

## Key Features

- Independent implementation of a published research paper
- Transfer learning with VGG16, VGG19, and ResNet101
- Manual image augmentation using NumPy and PIL
- Chest X-ray classification into COVID, Pneumonia, and Normal classes
- Training curve and confusion matrix visualization
- Model performance comparison across architectures

---

## Files

| File | What it does |
|------|-------------|
| `dataset.py` | Loads images from folders; all augmentation (flip, rotate, shift, zoom, noise) done manually with NumPy/PIL |
| `vgg_models.py` | VGG16 and VGG19 architectures defined layer by layer; pretrained weight remapping |
| `resnet_model.py` | ResNet101 with bottleneck blocks and skip connections; pretrained weights |
| `train_eval.py` | Training loop, evaluation, accuracy/loss curves, confusion matrix |
| `main.py` | Entry point — runs one or all models and prints comparison |

---

## Requirements

```
pip install torch torchvision pillow numpy matplotlib scikit-learn
```
Note: `torchvision` is needed only for `load_state_dict_from_url` utility.
The actual model architectures and data pipeline are all custom.

---

## Dataset Setup

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia
Note: The dataset is not included in this repository due to size constraints. Please download it from the original source and organize it as shown below.

Organize as:
```
data/
  train/
    covid/
    pneumonia/
    normal/
  val/
    covid/
    pneumonia/
    normal/
```

---

## Running

Run all 3 models (as per paper):
```bash
python main.py --train_dir data/train --val_dir data/val --epochs 20
```

Run a single model:
```bash
python main.py --model vgg16 --epochs 25 --batch_size 16 --lr 0.001
```

Run with fewer samples (to replicate paper's Table 1):
Just put fewer images in the train folder — e.g. 50 per class, 100 per class, etc.

---

## Reported Results from the Original Paper

| Model     | Samples | Accuracy |
|-----------|---------|----------|
| VGG-16    | 50      | 97%      |
| VGG-19    | 100     | 99%      |
| ResNet101 | 250     | 75%      |

VGG models outperform ResNet101 on small datasets because their architecture
transfers better to medical X-ray features with limited fine-tuning data.

---

## Disclaimer

This project is intended for educational and research purposes only and should not be used for medical diagnosis.
