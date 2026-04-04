import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from utils.dataset import CustomDataset
from models.model import build_model, load_config
from train import get_transforms, compute_iou


def load_checkpoint(model, checkpoint_path: str, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    total_iou = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        total_iou += compute_iou(preds, masks)
    return total_iou / len(loader)


def visualize(model, dataset, device, n=5, save_dir=None):
    threshold = 0.5 # make it an arg?
    indices = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    axes[0][0].set_title("Image")
    axes[0][1].set_title("Ground Truth")
    axes[0][2].set_title("Prediction")

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
        pred_bin = (pred.squeeze().cpu().detach().numpy() > threshold).astype(np.uint8)

        img_np  = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()

        axes[row][0].imshow(img_np)
        axes[row][1].imshow(mask_np, cmap="gray")
        axes[row][2].imshow(pred_bin, cmap="gray")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / "predictions.png")
        print(f"Saved to {save_dir}/predictions.png")
    else:
        plt.show()


def eval(config_path="./config.yaml", checkpoint="./checkpoints/best.pth", split="test"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    model = load_checkpoint(model, checkpoint, device)

    root    = Path(config["data"]["root"])
    size    = config["data"]["input_size"]
    dataset = CustomDataset(root, split, transforms=get_transforms(size, train=False))
    loader  = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    iou = evaluate(model, loader, device)
    print(f"[{split}] IoU: {iou:.4f}")

    visualize(model, dataset, device, n=5, save_dir="./eval_output")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--config",     type=str, default="./config.yaml")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pth")
    parser.add_argument("--split",      type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n",          type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--save-dir",   type=str, default="./eval_output")
    parser.add_argument("--no-vis",     action="store_true", help="Skip visualization")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    model = load_checkpoint(model, args.checkpoint, device)

    root    = Path(config["data"]["root"])
    size    = config["data"]["input_size"]
    dataset = CustomDataset(root, args.split, transforms=get_transforms(size, train=False))
    loader  = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    iou = evaluate(model, loader, device)
    print(f"[{args.split}] IoU: {iou:.4f}")

    if not args.no_vis:
        visualize(model, dataset, device, n=args.n, save_dir=args.save_dir)
