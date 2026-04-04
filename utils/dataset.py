import sys
import os
import argparse
import shutil
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split


def download_kaggle_dataset(dataset = "weitianqi/plantseg", dest = './data'):
    # dataset name is the URL, NOT the downloaded dataset path
    import kagglehub
    
    ## if already downloaded, kagglehub handles it
    path = kagglehub.dataset_download(dataset,output_dir=dest)
    print("Path to dataset files:", path)

#DELETE
def copy_dummy_to_raw():
    shutil.rmtree('./data/raw',ignore_errors=True)
    os.mkdir('./data/raw')
    shutil.copytree('./data/raw_DUMMY','./data/raw',dirs_exist_ok=True)
#######

class DatasetInfo:
    def __init__(self, root: Path):
        self.root = Path(root)
    
    def summary(self):
        print("="*30)
        print(f"  Dataset: {self.root.name}")
        print("="*30)

        for split in ["train", "val", "test"]:
            img_dir  = self.root / "images"      / split
            mask_dir = self.root / "annotations" / split

            if not img_dir.exists():
                continue

            images = list(img_dir.glob("*.*"))
            masks  = list(mask_dir.glob("*.png")) if mask_dir.exists() else []
            paired = sum(1 for i in images if (mask_dir / i.with_suffix('.png').name).exists())

            print(f"  [{split}]")
            print(f"    Images      : {len(images)}")
            print(f"    Masks       : {len(masks)}")
            print(f"    Paired      : {paired}")
            if len(images) != paired:
                print(f"    Missing masks: {len(images) - paired}")
        print("="*30)

class CustomDataset(Dataset):
    def __init__(self, root: Path, split: str, transforms=None):
        self.img_dir  = root / "images"      / split
        self.mask_dir = root / "annotations" / split
        self.transforms = transforms

        self.samples = [
            f.stem for f in self.img_dir.glob("*.jpg")
            if (self.mask_dir / f.with_suffix('.png').name).exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(str(self.img_dir  / f"{name}.jpg"))
        mask  = cv2.imread(str(self.mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = (mask > 0).astype(np.uint8)  # binary: 0 or 1

        if self.transforms:
            aug   = self.transforms(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask  = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and dataset formating")
    parser.add_argument("--dataset", type=Path, default='weitianqi/plantseg')
    parser.add_argument("--download-dir", type=Path, default="./data")
    args = parser.parse_args()

    # download_kaggle_dataset()
    # copy_dummy_to_raw()
    dataset_stats = DatasetInfo(Path(args.download_dir/ 'plantsegv2')).summary()