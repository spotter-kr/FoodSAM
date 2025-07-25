#!/bin/bash

# FoodSeg103 테스트셋 이미지 경로 및 저장 폴더 지정
DATA_ROOT="dataset/FoodSeg103/Images"
GT_ROOT="dataset/FoodSeg103/Images/ann_dir/test"
CATEGORY_TXT="FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt"

# Ensure the output directory is clean for a fresh run
echo "Cleaning up output directory: $OUTPUT_ROOT"
rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# 1. Semantic segmentation 실행
python FoodSAM/semantic_ap.py --data_root "$DATA_ROOT" --output "$OUTPUT_ROOT" --eval "$@"

# 2. 클래스 단위 AP 평가 (VOC, COCO, All Point)
python FoodSAM/evaluate_class_ap.py --data_root "$OUTPUT_ROOT" --gt_root "$GT_ROOT" --category_txt "$CATEGORY_TXT"