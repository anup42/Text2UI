#!/usr/bin/env bash

# Script to bootstrap a conda environment with GPU-ready dependencies for Text2UI
# Targets NVIDIA V100 GPUs via CUDA 11.8 builds of PyTorch.

set -euo pipefail

ENV_NAME=${1:-text2ui}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

# Prefer solvers that dramatically reduce SAT solving time while keeping compatibility
if [[ -n "${CONDA_SOLVER:-}" ]]; then
  SOLVER=$CONDA_SOLVER
elif command -v micromamba >/dev/null 2>&1; then
  SOLVER=micromamba
elif command -v mamba >/dev/null 2>&1; then
  SOLVER=mamba
else
  SOLVER=conda
fi

if [[ "$SOLVER" == "micromamba" ]]; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
fi

if ! command -v "$SOLVER" >/dev/null 2>&1; then
  echo "[ERROR] Unable to locate solver '$SOLVER'. Install conda, mamba, or micromamba and ensure it is on PATH." >&2
  exit 1
fi

if ! "$SOLVER" env list >/dev/null 2>&1; then
  echo "[ERROR] Solver '$SOLVER' does not support 'env list'. Ensure you are using a conda-compatible solver." >&2
  exit 1
fi

if "$SOLVER" env list | awk 'NF && $1 !~ /^#/ {gsub(/\*$/, "", $1); print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Conda environment '$ENV_NAME' already exists. Skipping creation."
else
  echo "[INFO] Creating conda environment '$ENV_NAME' with Python ${PYTHON_VERSION}..."
  "$SOLVER" create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

# Install PyTorch (CUDA builds on Linux, CPU fallback elsewhere).
UNAME=$(uname -s 2>/dev/null || echo "")
DEFAULT_GPU_TARGET=0
if [[ "$UNAME" == "Linux" ]]; then
  DEFAULT_GPU_TARGET=1
fi
if [[ "${FORCE_GPU_PACKAGES:-}" == "1" ]]; then
  DEFAULT_GPU_TARGET=1
elif [[ "${FORCE_GPU_PACKAGES:-}" == "0" ]]; then
  DEFAULT_GPU_TARGET=0
fi

if (( DEFAULT_GPU_TARGET )); then
  echo "[INFO] Installing PyTorch with CUDA 11.8 support (validated on NVIDIA V100)..."
  "$SOLVER" install -y -n "$ENV_NAME" pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
else
  echo "[INFO] Installing PyTorch (CPU build) via conda-forge..."
  "$SOLVER" install -y -n "$ENV_NAME" pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
fi

# Install repository dependencies (editable install for development convenience).
echo "[INFO] Installing Text2UI python dependencies..."
"$SOLVER" run -n "$ENV_NAME" python -m pip install --upgrade pip
"$SOLVER" run -n "$ENV_NAME" python -m pip install -e .

if [[ "$SOLVER" == "micromamba" ]]; then
  ACTIVATE_HINT="micromamba activate ${ENV_NAME}"
else
  ACTIVATE_HINT="conda activate ${ENV_NAME}"
fi

cat <<EOM

[SETUP COMPLETE]
Activate the environment in new shells with:
  ${ACTIVATE_HINT}

Run 'accelerate config' once inside the environment to tailor distributed settings for your V100 setup.
EOM
