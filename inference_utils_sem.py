import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from FoodSAM.FoodSAM_tools.predict_semantic_mask import semantic_predict


def load_sam_model(checkpoint_path="ckpts/sam_vit_h_4b8939.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(device)
    generator = SamAutomaticMaskGenerator(sam)
    return generator


def load_category_names(path="FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt"):
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    id2name = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        cid = int(parts[0].lstrip('\ufeff'))
        name = parts[1].strip()
        id2name[cid] = name
    return id2name


def run_inference_with_semantic(image: Image.Image,
                                generator,
                                id2name,
                                tmp_img_path="temp_input.jpg",
                                output_dir="temp_output",
                                sem_ckpt="ckpts/SETR_MLA/iter_80000.pth",
                                sem_cfg="configs/SETR_MLA_768x768_80k_base.py"):
    from mmseg.apis import inference_segmentor, init_segmentor

    image.save(tmp_img_path)

    model = init_segmentor(sem_cfg, sem_ckpt, device='cuda' if torch.cuda.is_available() else 'cpu')
    result = inference_segmentor(model, tmp_img_path)
    pred = result[0]  # (H, W) ndarray

    image_np = np.array(image.convert("RGB"))
    overlay = image_np.copy()

    colormap = (plt.cm.tab20(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    for cls_id in np.unique(pred):
        if cls_id == 0:
            continue

        mask = (pred == cls_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        color = colormap[cls_id % 256].tolist()  # 색상 선택
        color_mask = np.zeros_like(image_np)
        color_mask[mask > 0] = color

        alpha = 0.4
        overlay = np.where(mask[:, :, None].astype(bool),
                           (overlay * (1 - alpha) + color_mask * alpha).astype(np.uint8),
                           overlay)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = id2name.get(cls_id, f"cls{cls_id}")
        cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    return image_np, overlay

def show_results(original: np.ndarray, overlay: np.ndarray):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Semantic Prediction + Labels")
    plt.axis('off')

    plt.show()
