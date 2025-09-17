#!/usr/bin/env bash

# Create a Python virtual environment for Text2UI and install project dependencies.
# Usage:
#   scripts/create_venv_env.sh [env_dir] [--with-dev] [--python PATH]
# Examples:
#   scripts/create_venv_env.sh .venv
#   scripts/create_venv_env.sh .venv --with-dev --python python3.11

set -euo pipefail

ENV_DIR=".venv"
INSTALL_DEV=0
PYTHON_BIN=${PYTHON_BIN:-}
EXPLICIT_PYTHON=0
[[ -n "$PYTHON_BIN" ]] && EXPLICIT_PYTHON=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-dev)
      INSTALL_DEV=1
      shift
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --python flag requires an interpreter path or name." >&2
        exit 1
      fi
      PYTHON_BIN=$2
      EXPLICIT_PYTHON=1
      shift 2
      ;;
    --python=*)
      PYTHON_BIN=${1#--python=}
      EXPLICIT_PYTHON=1
      shift
      ;;
    --help|-h)
      cat <<'EOM'
Usage: scripts/create_venv_env.sh [env_dir] [--with-dev] [--python PATH]

Flags:
  --with-dev       install optional dev dependencies defined in pyproject.toml
  --python PATH    interpreter to use for venv creation (defaults to best available >=3.10)

Environment variables:
  PYTHON_BIN       acts like --python when set
EOM
      exit 0
      ;;
    -* )
      echo "[ERROR] Unknown option '$1'." >&2
      exit 1
      ;;
    * )
      if [[ -n ${ENV_DIR_SET:-} ]]; then
        echo "[ERROR] Unexpected extra argument '$1'." >&2
        exit 1
      fi
      ENV_DIR=$1
      ENV_DIR_SET=1
      shift
      ;;
  esac
done

check_python_version() {
  local exe=$1
  "$exe" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' >/dev/null 2>&1
}

resolve_python_path() {
  local candidate=$1
  if command -v "$candidate" >/dev/null 2>&1; then
    command -v "$candidate"
  elif [[ -x "$candidate" ]]; then
    echo "$candidate"
  else
    return 1
  fi
}

if [[ -n "$PYTHON_BIN" ]]; then
  if ! PYTHON_BIN=$(resolve_python_path "$PYTHON_BIN"); then
    echo "[ERROR] Specified Python interpreter '$PYTHON_BIN' is not executable." >&2
    exit 1
  fi
  if ! check_python_version "$PYTHON_BIN"; then
    echo "[ERROR] Interpreter '$PYTHON_BIN' is older than Python 3.10. Provide a newer Python via --python or PYTHON_BIN." >&2
    exit 1
  fi
else
  CANDIDATES=(python3.12 python3.11 python3.10 python3 python)
  for candidate in "${CANDIDATES[@]}"; do
    if PY_CANDIDATE=$(resolve_python_path "$candidate") && check_python_version "$PY_CANDIDATE"; then
      PYTHON_BIN=$PY_CANDIDATE
      break
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "[ERROR] Unable to find Python 3.10+ on PATH. Install a newer Python or pass --python." >&2
    exit 1
  fi
fi

PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

if [[ -d "$ENV_DIR" ]]; then
  echo "[INFO] Reusing existing virtual environment directory '$ENV_DIR'."
else
  echo "[INFO] Creating virtual environment in '$ENV_DIR' using interpreter at '$PYTHON_BIN' (Python ${PY_VERSION})..."
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

if ! check_python_version "$VENV_PYTHON"; then
  echo "[ERROR] Virtual environment at '$ENV_DIR' uses Python < 3.10. Delete it or recreate with --python pointing to a newer interpreter." >&2
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
