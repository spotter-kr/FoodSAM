# inference_utils.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Modeel load
def load_sam_model(checkpoint_path="ckpts/sam_vit_h_4b8939.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(device)
    generator = SamAutomaticMaskGenerator(sam)
    return generator

# Inference & Mask plot
def run_inference(image: Image.Image, generator) -> np.ndarray:
    image_np = np.array(image.convert("RGB"))
    masks = generator.generate(image_np)

    if not masks:
        raise ValueError("No masks generated")

    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    mask = masks[0]['segmentation'].astype(np.uint8) * 255

    alpha = 0.4
    mask_colored = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=2)
    overlay = (image_np * (1 - alpha) + mask_colored * alpha).astype(np.uint8)
    return image_np, overlay


def show_results(original: np.ndarray, overlay: np.ndarray):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("SAM Predicted Mask")
    plt.axis('off')

    plt.show()
