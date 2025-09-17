#!/usr/bin/env bash

# Script to bootstrap a conda environment with GPU-ready dependencies for Text2UI
# Targets NVIDIA V100 GPUs via CUDA 11.8 builds of PyTorch.

set -euo pipefail

ENV_NAME=${1:-text2ui}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda is not available on PATH. Install Miniconda or Anaconda first." >&2
  exit 1
fi

# Enable `conda activate` inside this non-interactive shell.
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Conda environment '$ENV_NAME' already exists. Skipping creation."
else
  echo "[INFO] Creating conda environment '$ENV_NAME' with Python ${PYTHON_VERSION}..."
  conda create -y -n "$ENV_NAME" python="${PYTHON_VERSION}"
fi

echo "[INFO] Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Install PyTorch with CUDA 11.8 builds suitable for V100 GPUs.
echo "[INFO] Installing PyTorch with CUDA 11.8 support (validated on NVIDIA V100)..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install repository dependencies (editable install for development convenience).
echo "[INFO] Installing Text2UI python dependencies..."
pip install --upgrade pip
pip install -e .

cat <<EOM

[SETUP COMPLETE]
Activate the environment in new shells with:
  conda activate ${ENV_NAME}

Run 'accelerate config' once inside the environment to tailor distributed settings for your V100 setup.
EOM
