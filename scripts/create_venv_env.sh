#!/usr/bin/env bash

# Create a Python virtual environment for Text2UI and install project dependencies.
# Usage:
#   scripts/create_venv_env.sh [env_dir] [--with-dev]
# Example:
#   scripts/create_venv_env.sh .venv --with-dev

set -euo pipefail

ENV_DIR=".venv"
INSTALL_DEV=0
PYTHON_BIN=${PYTHON_BIN:-}

POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --with-dev)
      INSTALL_DEV=1
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done

if ((${#POSITIONAL[@]} > 0)); then
  ENV_DIR="${POSITIONAL[0]}"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] Neither 'python3' nor 'python' is available on PATH. Install Python 3.10+ first." >&2
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Specified PYTHON_BIN '$PYTHON_BIN' is not on PATH." >&2
  exit 1
fi

if [[ -d "$ENV_DIR" ]]; then
  echo "[INFO] Reusing existing virtual environment directory '$ENV_DIR'."
else
  echo "[INFO] Creating virtual environment in '$ENV_DIR' using '$PYTHON_BIN'..."
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

if [[ -x "$ENV_DIR/bin/python" ]]; then
  VENV_PYTHON="$ENV_DIR/bin/python"
elif [[ -x "$ENV_DIR/Scripts/python.exe" ]]; then
  VENV_PYTHON="$ENV_DIR/Scripts/python.exe"
elif [[ -x "$ENV_DIR/Scripts/python" ]]; then
  VENV_PYTHON="$ENV_DIR/Scripts/python"
else
  echo "[ERROR] Unable to locate the Python executable inside '$ENV_DIR'." >&2
  exit 1
fi

echo "[INFO] Upgrading pip inside the virtual environment..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "[INFO] Installing Text2UI package in editable mode..."
"$VENV_PYTHON" -m pip install -e .

if (( INSTALL_DEV )); then
  echo "[INFO] Installing development extras..."
  "$VENV_PYTHON" -m pip install .[dev]
fi

POSIX_ACTIVATE="source ${ENV_DIR}/bin/activate"
POWERSHELL_ACTIVATE=".\\${ENV_DIR}\\Scripts\\Activate.ps1"
CMD_ACTIVATE=".\\${ENV_DIR}\\Scripts\\activate.bat"

cat <<EOM

[SETUP COMPLETE]
Activate the environment with one of the following commands:
  # bash/zsh
  ${POSIX_ACTIVATE}

  # PowerShell
  ${POWERSHELL_ACTIVATE}

  # Command Prompt
  ${CMD_ACTIVATE}

Run 'accelerate config' inside the environment before large Qwen runs.
EOM
