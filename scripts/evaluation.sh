#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate FoodSAM

# 사용할 GPU 디바이스 번호 지정
export CUDA_VISIBLE_DEVICES=4
# FoodSeg103 Test set 이미지 경로 입력
python FoodSAM/semantic.py \
  --data_root dataset/FoodSeg103/Images \
  --output Output/Semantic_Results \
  --eval