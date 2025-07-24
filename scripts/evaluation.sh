#!/bin/bash
# FoodSeg103 Test set 이미지 경로 입력
python FoodSAM/semantic.py --data_root dataset/FoodSeg103/Images --output Output/Semantic_Results --eval