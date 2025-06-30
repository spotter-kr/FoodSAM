import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def generate_pred_instances(result_root, num_class=104):
    folders = sorted([d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))])
    for folder in tqdm(folders, desc="Generating pred_instances.json"):
        folder_path = os.path.join(result_root, folder)
        enhanced_mask_path = os.path.join(folder_path, "enhance_mask.png")
        softmax_path = os.path.join(folder_path, "softmax_probs.npy")
        sam_mask_dir = os.path.join(folder_path, "sam_mask")

        # 존재 여부 확인 로그
        if not os.path.exists(enhanced_mask_path):
            print(f"[SKIP] {folder}: enhanced_mask.png not found")
            continue
        if not os.path.exists(softmax_path):
            print(f"[SKIP] {folder}: softmax_probs.npy not found")
            continue
        if not os.path.exists(sam_mask_dir):
            print(f"[SKIP] {folder}: sam_mask/ directory not found")
            continue

        enhanced_mask = cv2.imread(enhanced_mask_path, cv2.IMREAD_GRAYSCALE)
        softmax_probs = np.load(softmax_path)  # shape: (C, H, W)
        pred_instances = []

        mask_files = [f for f in os.listdir(sam_mask_dir) if f.endswith(".png")]
        if len(mask_files) == 0:
            print(f"[SKIP] {folder}: No .png files in sam_mask/")
            continue

        for mask_file in mask_files:
            mask_id = os.path.splitext(mask_file)[0]
            mask_path = os.path.join(sam_mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            binary_mask = mask > 128
            if binary_mask.sum() == 0:
                continue

            cls_ids, counts = np.unique(enhanced_mask[binary_mask], return_counts=True)
            majority_idx = np.argmax(counts)
            category_id = int(cls_ids[majority_idx])
            if category_id == 0 or category_id >= num_class:
                continue

            conf = float(softmax_probs[category_id][binary_mask].mean())

            pred_instances.append({
                "mask_id": mask_id,
                "category_id": category_id,
                "confidence": conf
            })

        if len(pred_instances) == 0:
            print(f"[EMPTY] {folder}: No valid instances generated")
        else:
            print(f"[OK] {folder}: {len(pred_instances)} instances saved")

        json_path = os.path.join(folder_path, "pred_instances.json")
        with open(json_path, 'w') as f:
            json.dump(pred_instances, f, indent=4)
