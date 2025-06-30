import os
import json
import numpy as np
import cv2
from tqdm import tqdm


def load_gt_mask(gt_path):
    return cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)


def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union


def voc_ap(rec, prec):
    return np.mean([prec[rec >= t].max() if np.any(rec >= t) else 0
                    for t in np.linspace(0, 1, 11)])


def coco_ap(rec, prec):
    # Integration over all points (101-point interpolation)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def merge_masks(preds, image_shape):
    class_to_mask = {}
    for pred in preds:
        cls = pred['category_id']
        mask_path = os.path.join(os.path.dirname(pred['json_path']), 'sam_mask', f"{pred['mask_id']}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128
        if cls not in class_to_mask:
            class_to_mask[cls] = np.zeros(image_shape, dtype=bool)
        class_to_mask[cls] |= mask
    return class_to_mask


def evaluate_ap_merged(data_root, gt_root, category_txt_path):
    with open(category_txt_path, 'r', encoding='utf-8-sig') as f:
        categories = [x.strip().split(':') for x in f.readlines() if ':' in x]
        id2name = {int(k): v for k, v in categories}

    voc_aps = []
    coco_aps = []

    if os.path.isdir(data_root) and os.path.exists(os.path.join(data_root, 'pred_instances.json')):
        folders = [data_root]  # 단일 이미지
    else:
        folders = [os.path.join(data_root, x) for x in os.listdir(data_root)
                   if os.path.exists(os.path.join(data_root, x, 'pred_instances.json'))]

    for folder in tqdm(folders):
        json_path = os.path.join(folder, 'pred_instances.json')
        with open(json_path, 'r') as f:
            preds = json.load(f)
        for p in preds:
            p['json_path'] = json_path

        img_basename = os.path.basename(folder)
        gt_path = os.path.join(gt_root, img_basename + '.png') if os.path.isdir(gt_root) else gt_root
        if not os.path.exists(gt_path):
            continue

        gt_mask = load_gt_mask(gt_path)
        h, w = gt_mask.shape

        merged_preds = merge_masks(preds, (h, w))

        gt_classes = np.unique(gt_mask)
        y_true, y_score = [], []
        for cls, pred_mask in merged_preds.items():
            if cls not in gt_classes:
                y_true.append(0)
            else:
                gt_bin = gt_mask == cls
                y_true.append(iou(pred_mask, gt_bin) > 0.5)
            confs = [p['confidence'] for p in preds if p['category_id'] == cls]
            y_score.append(np.mean(confs))

        if not y_true:
            continue
        y_true, y_score = np.array(y_true), np.array(y_score)
        sorted_idx = np.argsort(-y_score)
        y_true = y_true[sorted_idx]
        tp = np.cumsum(y_true)
        fp = np.cumsum(1 - y_true)
        rec = tp / (np.sum(y_true) + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        voc_aps.append(voc_ap(rec, prec))
        coco_aps.append(coco_ap(rec, prec))

    print(f"\n✅ VOC mAP@0.50: {np.mean(voc_aps):.4f}")
    print(f"✅ COCO mAP@0.50: {np.mean(coco_aps):.4f}")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True, help='Path to Output/Panoramic_Results')
parser.add_argument('--gt_root', type=str, required=True, help='Path to GT masks (pngs)')
parser.add_argument('--category_txt', type=str, required=True, help='Path to foodseg103_category_id.txt')
args = parser.parse_args()

evaluate_ap_merged(args.data_root, args.gt_root, args.category_txt)
