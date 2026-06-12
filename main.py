"""
COVID-19 X-Ray Classification using Transfer Learning
======================================================
Implementation of the paper:
  "Facilitating COVID recognition from X-rays with computer vision
   models and transfer learning" (Varde et al., 2023)

Models:  VGG16, VGG19, ResNet101
Dataset: 3 classes -> COVID (0), Pneumonia (1), Normal (2)
Strategy: Transfer learning Strategy 1 - freeze
 pretrained conv
          layers, only train custom classifier head on X-ray data

You can get the dataset from:
  https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia
"""

import torch
import argparse
from dataset     import get_dataloaders
from vgg_models  import build_vgg16, build_vgg19
from resnet_model import build_resnet101
from train_eval  import train_model, evaluate, plot_training_curves, plot_confusion_matrix
import torch.nn as nn


def run_experiment(model_name, train_dir, val_dir,
                   epochs, batch_size, lr, num_classes=3):

    print(f"\n{'#'*60}")
    print(f"#  Running experiment with {model_name}")
    print(f"{'#'*60}")

    # build the model
    if model_name == 'vgg16':
        model = build_vgg16(num_classes=num_classes, pretrained=True, freeze_conv=True)
    elif model_name == 'vgg19':
        model = build_vgg19(num_classes=num_classes, pretrained=True, freeze_conv=True)
    elif model_name == 'resnet101':
        model = build_resnet101(num_classes=num_classes, pretrained=True, freeze_backbone=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}  ({trainable/total*100:.1f}%)")

    # data loaders
    train_loader, val_loader = get_dataloaders(
        train_dir, val_dir,
        batch_size=batch_size,
        num_workers=0   # set to 2+ if you have enough CPU cores
    )

    # train
    trained_model, history = train_model(
        model, train_loader, val_loader,
        model_name  = model_name,
        num_epochs  = epochs,
        learning_rate = lr
    )

    # final evaluation on val set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, preds, labels = evaluate(trained_model, val_loader, criterion, device)
    print(f"\nFinal validation accuracy ({model_name}): {val_acc:.2f}%")

    # plots
    plot_training_curves(history, model_name)
    plot_confusion_matrix(preds, labels, model_name)

    return val_acc


def main():
    parser = argparse.ArgumentParser(description='COVID X-Ray Classification')
    parser.add_argument('--train_dir',  type=str, default='data/train',
                        help='Path to training data folder')
    parser.add_argument('--val_dir',    type=str, default='data/val',
                        help='Path to validation data folder')
    parser.add_argument('--model',      type=str, default='all',
                        choices=['vgg16', 'vgg19', 'resnet101', 'all'],
                        help='Which model to run (default: all)')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=0.001)
    args = parser.parse_args()

    models_to_run = ['vgg16', 'vgg19', 'resnet101'] if args.model == 'all' else [args.model]

    results = {}
    for model_name in models_to_run:
        acc = run_experiment(
            model_name  = model_name,
            train_dir   = args.train_dir,
            val_dir     = args.val_dir,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr
        )
        results[model_name] = acc

    print("\n" + "="*40)
    print("  Final Results Summary")
    print("="*40)
    for name, acc in results.items():
        print(f"  {name:<12}  {acc:.2f}%")
    best = max(results, key=results.get)
    print(f"\n  Best model: {best} ({results[best]:.2f}%)")


if __name__ == '__main__':
    main()
