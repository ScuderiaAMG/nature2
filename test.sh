#!/bin/bash
# setup.sh

echo "Creating conda environment..."
conda create -n nature2 python=3.10 -y
conda activate nature2

echo "Installing PyTorch with CUDA 12.1 (compatible with 12.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing Atari environment and other dependencies..."
pip install gymnasium[atari,accept-rom-license]==0.28.1
pip install ale-py==0.8.1
pip install opencv-python numpy tqdm tensorboard

echo "Environment setup complete! Run 'conda activate dqn_nature2015' to start."
