import torch
from torch.utils.data import DataLoader
from pathlib import Path
import albumentations as A
from tqdm import tqdm

from utils.dataset import CustomDataset
from models.model import build_model, load_config
##speed


def get_transforms(input_size: int, train: bool):
    # maybe more transformes later
    if train:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=30, p=0.4),
        ])
    return A.Compose([A.Resize(input_size, input_size)])


def get_loaders(config: dict):
    root       = Path(config["data"]["root"])
    input_size = config["data"]["input_size"]
    bs         = config["training"]["batch_size"]
    nw         = config["training"]["num_workers"]

    train_ds = CustomDataset(root, "train", transforms=get_transforms(input_size, train=True))
    test_ds = CustomDataset(root, "test", transforms=get_transforms(input_size, train=False))
    val_ds   = CustomDataset(root, "val",   transforms=get_transforms(input_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, test_loader, val_loader

def get_optimizer(model, config: dict):
    lr = config["training"]["learning_rate"]
    if config["training"]["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def get_scheduler(optimizer, config: dict):
    name = config["training"]["scheduler"]
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


def compute_loss(pred, mask, config: dict):
    # dice + bce, yaml has the mix coeffs
    bce  = torch.nn.functional.binary_cross_entropy_with_logits(pred, mask)
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * mask).sum()
    dice = 1 - (2 * intersection + 1) / (pred_sigmoid.sum() + mask.sum() + 1)
    bw = config["loss"]["bce_weight"]
    dw = config["loss"]["dice_weight"]
    return bw * bce + dw * dice


def compute_iou(pred, mask, threshold=0.5):
    pred_bin     = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * mask).sum()
    union        = (pred_bin + mask).clamp(0, 1).sum()
    return (intersection / (union + 1e-6)).item()


def train_one_epoch(model, loader, optimizer, config, device):
    model.train()
    total_loss, total_iou = 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss  = compute_loss(preds, masks, config)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou  += compute_iou(preds, masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def val_one_epoch(model, loader, config, device):
    model.eval()
    total_loss, total_iou = 0, 0
    pbar = tqdm(loader, desc="Val  ", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss  = compute_loss(preds, masks, config)
        total_loss += loss.item()
        total_iou  += compute_iou(preds, masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    n = len(loader)
    return total_loss / n, total_iou / n


def train(config_path="./config.yaml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    train_loader, val_loader, test_loader = get_loaders(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    save_dir = Path(config["checkpoints"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    monitor   = config["checkpoints"]["monitor"]
    best      = float("inf") if monitor == "val_loss" else 0.0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, config, device)
        val_loss,   val_iou   = val_one_epoch(model, val_loader, config, device)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch:03d} | "
              f"train_loss: {train_loss:.4f}  train_iou: {train_iou:.4f} | "
              f"val_loss: {val_loss:.4f}  val_iou: {val_iou:.4f}")

        metric   = val_loss if monitor == "val_loss" else val_iou
        improved = metric < best if monitor == "val_loss" else metric > best
        if improved:
            best = metric
            best_path = save_dir / "best.pth"
            if best_path.exists():
                best_path.unlink()
            torch.save(model.state_dict(), best_path)
            tqdm.write(f"  -> Epoch {epoch:03d} saved best (val_iou={val_iou:.4f})")

 
    # final evaluation on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(save_dir / "best.pth", map_location=device))
    test_loss, test_iou = val_one_epoch(model, test_loader, config, device)
    print(f"[test] loss: {test_loss:.4f}  iou: {test_iou:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    train(args.config)

    
