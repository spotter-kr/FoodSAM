#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate FoodSAM

# 사용할 GPU 디바이스 번호 지정
export CUDA_VISIBLE_DEVICES=4
# 이미지가 들어있는 폴더 경로와 출력 생성 경로
python FoodSAM/semantic.py \
  --data_root ./image_subset \
  --output inference