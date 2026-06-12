import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ['COVID', 'Pneumonia', 'Normal']


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}]  loss: {loss.item():.4f}")

    epoch_loss = total_loss / total
    epoch_acc  = correct / total * 100
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / total
    val_acc  = correct / total * 100
    return val_loss, val_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, model_name,
                num_epochs=20, learning_rate=0.001, save_dir='checkpoints'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    model = model.to(device)

    # cross entropy loss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer - only update parameters that require gradients
    # (conv layers are frozen, so only fc layers will be updated)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    # reduce LR by 0.1 if val loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.1, patience=5)

    os.makedirs(save_dir, exist_ok=True)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   []
    }

    best_val_acc  = 0.0
    best_epoch    = 0

    print(f"\n{'='*55}")
    print(f"  Starting training: {model_name}")
    print(f"  Epochs: {num_epochs}   LR: {learning_rate}")
    print(f"{'='*55}\n")

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                 criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch}/{num_epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
              f"({elapsed:.1f}s)")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            save_path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best! Saved to {save_path}")

    print(f"\nBest val acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    return model, history


def plot_training_curves(history, model_name, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # accuracy plot
    axes[0].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4)
    axes[0].plot(epochs, history['val_acc'],   'r-o', label='Val Acc',   markersize=4)
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # loss plot
    axes[1].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    axes[1].plot(epochs, history['val_loss'],   'r-o', label='Val Loss',   markersize=4)
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves -> {save_path}")


def plot_confusion_matrix(preds, labels, model_name, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True Label')
    ax.set_title(f'{model_name} - Confusion Matrix')

    # write numbers inside each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {save_path}")
