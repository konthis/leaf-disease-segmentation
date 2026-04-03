import sys
import os
import argparse
import shutil
import json
from pathlib import Path

import random
from sklearn.model_selection import train_test_split


def download_kaggle_dataset(dataset = "weitianqi/plantseg", dest = './data/raw'):
    # dataset name is the URL, NOT the downloaded dataset path
    import kagglehub
    
    ## if already downloaded, kagglehub handles it
    path = kagglehub.dataset_download(dataset,output_dir=dest)
    print("Path to dataset files:", path)

def get_samples(dir = "./data/YOLO_format/plantsegv2"):
    # Iterates through label txt (YOLO format), checks if theres image with the same name. Returns a list of valid samples (only the names)
    dir = Path(dir)
    samples = []
    for f in Path(dir / 'labels').glob("*.txt"):
        filename = str(f).split("/")[-1].split(".")[0]
        for ext in [".jpg"]: ## maybe add extra ext. later
            if Path(dir / "images" / (filename + ext)).exists():
                print("yo")
    return samples

def convert_to_YOLO(raw_dir= './data/raw/plantsegv2', out_dir = './data/YOLO_format/plantsegv2'):
    from ultralytics.data.converter import convert_coco
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    convert_coco(
        labels_dir=raw_dir,
        save_dir=out_dir,
        use_keypoints=False,
        use_segments=True,
        cls91to80=False,
    )
    ## MOVE LABELS IF IN A FOLDER
    # if /labels/ann_folder/*.txt -> i want /labels/*.txt
    ann_folder = next(os.walk(out_dir / "labels"))[1]
    if ann_folder:
        for f in Path(out_dir / "labels" /ann_folder[0]).glob("*.txt"):
            shutil.move(str(f), str(out_dir / "labels" / f.name))
        os.rmdir(out_dir / "labels" / ann_folder[0])
    

    # MOVE images
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for split in ["train","test","val"]:
        if not (out_dir / "images" / split).exists():
            shutil.move(raw_dir / "images" / split, out_dir / "images")
    








    # samples = get_samples(out_path)
    # seed = 42
    # random.seed(seed)
    # indices = list(range(len(samples)))
    # train_idx, test_idx = train_test_split(indices, test_size=test_split, random_state=seed)
    # train_idx, val_idx  = train_test_split(train_idx, test_size=val_split / (1 - test_split), random_state=seed)

    # for split in ['train','test','val']:
    #     (out_path / "images" / split).mkdir(parents=True, exist_ok=True)

    #     (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and dataset formating")
    parser.add_argument("--dataset", type=Path, default='weitianqi/plantseg')
    parser.add_argument("--download-dst", type=Path, default="./data/raw")
    parser.add_argument("--format-dst", type=Path, default="./data/YOLO_format")
    args = parser.parse_args()

    # download_kaggle_dataset()
    convert_to_YOLO()
    # get_samples()

    