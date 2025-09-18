#!/usr/bin/env bash

# Script to bootstrap a conda environment with GPU-ready dependencies for Text2UI
# Targets NVIDIA V100 GPUs via CUDA 11.8 builds of PyTorch.

set -euo pipefail

ENV_NAME=${1:-text2ui}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
SKIP_DEPS=${TEXT2UI_ENV_SETUP_SKIP_DEPS:-0}

UNAME=$(uname -s 2>/dev/null || echo "")

SOLVER_BIN="${CONDA_SOLVER:-}"
if [[ -n "$SOLVER_BIN" ]]; then
  if [[ ! -x "$SOLVER_BIN" ]]; then
    if command -v "$SOLVER_BIN" >/dev/null 2>&1; then
      SOLVER_BIN=$(command -v "$SOLVER_BIN")
    else
      echo "[ERROR] Unable to execute solver '$SOLVER_BIN'." >&2
      exit 1
    fi
  fi
else
  if command -v conda >/dev/null 2>&1; then
    SOLVER_BIN=$(command -v conda)
  else
    echo "[ERROR] Unable to locate 'conda'. Please install Anaconda or Miniconda and ensure 'conda' is on your PATH." >&2
    exit 1
  fi
fi

SOLVER_NAME=$(basename "$SOLVER_BIN")

if [[ "$SOLVER_NAME" != "conda" ]]; then
  echo "[ERROR] Only the 'conda' solver is supported by this script. Detected solver: '$SOLVER_NAME'." >&2
  echo "        Set CONDA_SOLVER to the path of the 'conda' executable if needed." >&2
  exit 1
fi

if ! "$SOLVER_BIN" env list >/dev/null 2>&1; then
  echo "[ERROR] Solver '$SOLVER_NAME' does not support 'env list'. Ensure you are using a valid conda installation." >&2
  exit 1
fi

CREATE_CHANNELS=(-c conda-forge)
GPU_CHANNELS=(-c pytorch -c nvidia -c conda-forge)
CPU_CHANNELS=(-c pytorch -c conda-forge)

if "$SOLVER_BIN" env list | awk 'NF && $1 !~ /^#/ {gsub(/\*$/, "", $1); print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Conda environment '$ENV_NAME' already exists. Skipping creation."
else
  echo "[INFO] Creating conda environment '$ENV_NAME' with Python ${PYTHON_VERSION}..."
  "$SOLVER_BIN" create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" "${CREATE_CHANNELS[@]}"
fi

if (( SKIP_DEPS )); then
  echo "[INFO] Skipping dependency installation due to TEXT2UI_ENV_SETUP_SKIP_DEPS=${SKIP_DEPS}."
else
  # Ensure modern build tooling for packages that require native extensions.
  echo "[INFO] Installing build tooling (cmake >= 3.25)..."
  "$SOLVER_BIN" install -y -n "$ENV_NAME" "cmake>=3.25" "${CREATE_CHANNELS[@]}"

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
    "$SOLVER_BIN" install -y -n "$ENV_NAME" pytorch torchvision torchaudio pytorch-cuda=11.8 "${GPU_CHANNELS[@]}"
  else
    echo "[INFO] Installing PyTorch (CPU build) via conda-forge..."
    "$SOLVER_BIN" install -y -n "$ENV_NAME" pytorch torchvision torchaudio cpuonly "${CPU_CHANNELS[@]}"
  fi

  # Install repository dependencies (editable install for development convenience).
  echo "[INFO] Installing Text2UI python dependencies..."
  "$SOLVER_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip
  "$SOLVER_BIN" run -n "$ENV_NAME" python -m pip install -e .
fi

ACTIVATE_HINT="conda activate ${ENV_NAME}"

cat <<EOM

[SETUP COMPLETE]
Activate the environment in new shells with:
  ${ACTIVATE_HINT}

Run 'accelerate config' once inside the environment to tailor distributed settings for your V100 setup.
EOM
