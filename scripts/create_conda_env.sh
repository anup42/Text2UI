#!/usr/bin/env bash

# Script to bootstrap a conda environment with GPU-ready dependencies for Text2UI
# Targets NVIDIA V100 GPUs via CUDA 11.8 builds of PyTorch.

set -euo pipefail

ENV_NAME=${1:-text2ui}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

UNAME=$(uname -s 2>/dev/null || echo "")

cleanup_paths=()
cleanup() {
  if ((${#cleanup_paths[@]})); then
    rm -rf "${cleanup_paths[@]}"
  fi
}
trap cleanup EXIT

bootstrap_micromamba() {
  local tmp_dir archive url
  tmp_dir=$(mktemp -d)
  cleanup_paths+=("$tmp_dir")
  archive="$tmp_dir/micromamba.tar.bz2"
  url="https://micro.mamba.pm/api/micromamba/linux-64/latest"
  echo "[INFO] Downloading a temporary micromamba binary for faster solves..."
  if command -v curl >/dev/null 2>&1; then
    curl -Ls "$url" -o "$archive"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$archive" "$url"
  else
    echo "[ERROR] curl or wget is required to bootstrap micromamba automatically." >&2
    exit 1
  fi
  tar -xjf "$archive" -C "$tmp_dir" bin/micromamba
  echo "$tmp_dir/bin/micromamba"
}

SOLVER_BIN=""
if [[ -n "${CONDA_SOLVER:-}" ]]; then
  SOLVER_BIN=$CONDA_SOLVER
elif command -v micromamba >/dev/null 2>&1; then
  SOLVER_BIN=$(command -v micromamba)
elif [[ "$UNAME" == "Linux" ]]; then
  SOLVER_BIN=$(bootstrap_micromamba)
elif command -v mamba >/dev/null 2>&1; then
  SOLVER_BIN=$(command -v mamba)
elif command -v conda >/dev/null 2>&1; then
  SOLVER_BIN=$(command -v conda)
else
  echo "[ERROR] Unable to locate a conda-compatible solver. Install conda, mamba, or micromamba." >&2
  exit 1
fi

if [[ -z "$SOLVER_BIN" ]]; then
  echo "[ERROR] Failed to resolve solver binary." >&2
  exit 1
fi

if [[ ! -x "$SOLVER_BIN" ]]; then
  if command -v "$SOLVER_BIN" >/dev/null 2>&1; then
    SOLVER_BIN=$(command -v "$SOLVER_BIN")
  else
    echo "[ERROR] Unable to execute solver '$SOLVER_BIN'." >&2
    exit 1
  fi
fi

SOLVER_NAME=$(basename "$SOLVER_BIN")

if [[ "$SOLVER_NAME" == "micromamba" ]]; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
fi

if ! "$SOLVER_BIN" env list >/dev/null 2>&1; then
  echo "[ERROR] Solver '$SOLVER_NAME' does not support 'env list'. Ensure you are using a conda-compatible solver." >&2
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

if [[ "$SOLVER_NAME" == "micromamba" ]]; then
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
