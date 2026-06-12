import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random


# 0 = COVID, 1 = Pneumonia, 2 = Normal
CLASS_MAP = {'covid': 0, 'pneumonia': 1, 'normal': 2}


def load_image(path):
    """open image and resize to 224x224 as required by VGG and ResNet"""
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


#  Manual augmentation functions 

def random_horizontal_flip(img):
    if random.random() > 0.5:
        return img[:, ::-1, :].copy()
    return img


def random_rotation(img, max_angle=15):
    """rotated image by a random angle using PIL under the hood
    only PIL is used here, not any transform library"""
    angle = random.uniform(-max_angle, max_angle)
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return np.array(pil_img, dtype=np.float32)


def random_shift(img, max_shift=0.1):
    """shift the image horizontally/vertically by a small fraction"""
    h, w = img.shape[:2]
    shift_x = int(random.uniform(-max_shift, max_shift) * w)
    shift_y = int(random.uniform(-max_shift, max_shift) * h)

    result = np.zeros_like(img)
    # figure out source and destination slices
    src_x_start = max(0, -shift_x)
    src_x_end   = min(w, w - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end   = min(w, w + shift_x)

    src_y_start = max(0, -shift_y)
    src_y_end   = min(h, h - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end   = min(h, h + shift_y)

    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        img[src_y_start:src_y_end, src_x_start:src_x_end]
    return result


def random_zoom(img, zoom_range=(0.9, 1.1)):
    """zoom in or out, then crop/pad back to original size"""
    factor = random.uniform(*zoom_range)
    h, w = img.shape[:2]
    new_h = int(h * factor)
    new_w = int(w * factor)

    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    zoomed = np.array(pil_img, dtype=np.float32)

    # crop center if zoomed in, or pad with zeros if zoomed out
    result = np.zeros((h, w, 3), dtype=np.float32)
    y_off = abs(new_h - h) // 2
    x_off = abs(new_w - w) // 2

    if factor >= 1.0:
        # zoomed in - crop center part
        result = zoomed[y_off:y_off + h, x_off:x_off + w]
    else:
        # zoomed out - paste into zero canvas
        result[y_off:y_off + new_h, x_off:x_off + new_w] = zoomed

    return result


def add_gaussian_noise(img, mean=0, std=5.0):
    """add a bit of gaussian noise to the image"""
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img + noise
    return np.clip(noisy, 0, 255)


def normalize(img):
    """normalize using ImageNet mean/std so pre-trained weights work correctly"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
    img = (img - mean) / (std + 1e-7)
    return img


def augment_image(img):
    """apply all augmentation steps to one training image"""
    img = random_horizontal_flip(img)
    img = random_rotation(img, max_angle=15)
    img = random_shift(img, max_shift=0.08)
    img = random_zoom(img, zoom_range=(0.92, 1.08))
    img = add_gaussian_noise(img, std=3.0)
    return img


#  Dataset class

class XRayDataset(Dataset):

    def __init__(self, root_dir, apply_augmentation=False):
        self.root_dir = root_dir
        self.apply_augmentation = apply_augmentation
        self.samples = []   # list of (filepath, label)

        for class_name, label in CLASS_MAP.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                print(f"Warning: folder not found -> {class_folder}")
                continue
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(class_folder, fname)
                    self.samples.append((full_path, label))

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = load_image(img_path)

        if self.apply_augmentation:
            img = augment_image(img)

        img = normalize(img)
        # convert HWC -> CHW for PyTorch
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img)
        return img_tensor, label


def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=2):
    train_dataset = XRayDataset(train_dir, apply_augmentation=True)
    val_dataset   = XRayDataset(val_dir,   apply_augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
