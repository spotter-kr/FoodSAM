#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate FoodSAM

# 사용할 GPU 디바이스 번호 지정
export CUDA_VISIBLE_DEVICES=4

# 단일 테스트 이미지 경로와 결과 출력 폴더명 지정
python FoodSAM/semantic.py \
  --img_path dataset/FoodSeg103/Images/img_dir/test/00001977.jpg \
  --output single_inference