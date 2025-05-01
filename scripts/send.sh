#!/bin/bash

# 確保 conda 被正確初始化
source /home/pika/archiconda3/etc/profile.d/conda.sh

# 激活 Conda 環境
conda activate jetson

# 進入目標目錄
cd /opencv_zoo/models/pose_estimation_mediapipe

# 執行 Python 腳本
python send.py
