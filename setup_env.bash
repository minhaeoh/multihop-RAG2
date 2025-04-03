#!/bin/bash

# Create new conda environment
ENV_NAME="multihop"

# 기존 환경이 있다면 제거
conda deactivate
conda env remove -n $ENV_NAME

# 새로운 환경 생성
echo "Creating new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# 환경 활성화
source activate $ENV_NAME

# 기본 패키지 설치
pip install torch
pip install numpy==1.24.3
pip install faiss-gpu
pip install datasets
pip install dotenv
pip install transformers

conda env export > environment.yml
echo "Environment setup complete. Please activate the environment with 'source activate $ENV_NAME'"

