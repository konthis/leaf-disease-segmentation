import torch
import cv2
import numpy as np
import gradio as gr
from pathlib import Path

from models.model import build_model, load_config
from train import get_transforms


def load_model(config_path="./config.yaml", checkpoint="./checkpoints/best.pth"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model, device, config


def predict(image, model, device, config, threshold=0.5, alpha=0.5):
    input_size = config["data"]["input_size"]
    transforms = get_transforms(input_size, train=False)

    aug = transforms(image=image)
    inp = torch.from_numpy(aug["image"]).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        pred = torch.sigmoid(model(inp.unsqueeze(0).to(device)))

    pred_bin = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)

    # resize mask back to original image size
    orig_h, orig_w = image.shape[:2]
    pred_bin = cv2.resize(pred_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # red overlay on original image
    overlay = image.copy()
    overlay[pred_bin == 1] = (
        overlay[pred_bin == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    ).astype(np.uint8)

    mask_vis = (pred_bin * 255).astype(np.uint8)
    return overlay, mask_vis


def run_app():
    ### absolutly basic app for segmenting an uploaded image
    model, device, config = load_model()

    def inference(image):
        overlay, mask = predict(image, model, device, config)
        return overlay, mask

    with gr.Blocks(title="Leaf Disease Segmentation") as demo:
        gr.Markdown("## Leaf Disease Segmentation")
        gr.Markdown("Upload a leaf image to segment diseased regions.")

        with gr.Row():
            input_image  = gr.Image(type="numpy", label="Input Image")

        with gr.Row():
            overlay_out  = gr.Image(label="Overlay")
            mask_out     = gr.Image(label="Predicted Mask", image_mode="L")

        btn = gr.Button("Segment")
        btn.click(fn=inference, inputs=input_image, outputs=[overlay_out, mask_out])

    demo.launch(server_name="0.0.0.0") # local-dynamic port





if __name__ == "__main__":
    run_app()
