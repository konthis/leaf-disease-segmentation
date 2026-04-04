import segmentation_models_pytorch as smp
import yaml
from pathlib import Path


def build_model(config: dict):
    model = smp.create_model(
        arch=config["model"]["architecture"],
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["num_classes"],
    )
    return model


def load_config(config_path: str = "./config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    model = build_model(config)
    print(model)
