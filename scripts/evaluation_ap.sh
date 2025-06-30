#!/bin/bash

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate FoodSAM

# 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=4

# FoodSeg103 테스트셋 이미지 경로 및 저장 폴더 지정
DATA_ROOT="/home/jeongin/FoodSAM/dataset/FoodSeg103/Images"
GT_ROOT="/home/jeongin/FoodSAM/dataset/FoodSeg103/Images/ann_dir/test"
OUTPUT_ROOT="/home/jeongin/FoodSAM/Output_AP"
CATEGORY_TXT="/home/jeongin/FoodSAM/FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt"

# 1. Semantic segmentation 실행
python FoodSAM/semantic_ap.py \
  --data_root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT" \
  --eval

# 2. 클래스 단위 AP 평가 (VOC, COCO, All Point)
python FoodSAM/evaluate_class_ap.py \
  --data_root "$OUTPUT_ROOT" \
  --gt_root "$GT_ROOT" \
  --category_txt "$CATEGORY_TXT"
