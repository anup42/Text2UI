"""Tests for the environment setup helper scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"


def _run_script(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Execute a shell script and capture its output."""
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_create_venv_env_skip_pip(tmp_path):
    """The venv script should allow skipping pip-heavy operations for tests."""
    env_dir = tmp_path / "test-venv"
    script = SCRIPTS_DIR / "create_venv_env.sh"

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "TEXT2UI_ENV_SETUP_SKIP_PIP": "1",
        }
    )

    result = _run_script(["bash", str(script), str(env_dir)], env=env)

    assert env_dir.exists(), "Virtual environment directory should be created."
    assert "Skipping pip-related installation steps" in result.stdout
    assert "[SETUP COMPLETE]" in result.stdout


def _write_fake_solver(path: Path) -> None:
    script_body = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail

        LOG=${FAKE_SOLVER_LOG:?}
        STATE=${FAKE_SOLVER_STATE:?}

        printf "%s\n" "$*" >>"$LOG"

        cmd=${1:-}
        shift || true

        case "$cmd" in
          env)
            if [[ ${1:-} == "list" ]]; then
              if [[ -f "$STATE" ]]; then
                cat "$STATE"
              else
                printf "#\n"
              fi
            fi
            ;;
          create)
            env_name=""
            while [[ $# -gt 0 ]]; do
              case "$1" in
                -n)
                  env_name=$2
                  shift 2
                  ;;
                *)
                  shift
                  ;;
              esac
            done
            printf "%s\n" "$env_name" >"$STATE"
            ;;
          install|run)
            :
            ;;
        esac
        """
    )
    path.write_text(script_body)
    path.chmod(0o755)


def test_create_conda_env_with_fake_solver(tmp_path):
    """The conda script can be exercised with a fake solver implementation."""
    env_name = "text2ui-test"
    script = SCRIPTS_DIR / "create_conda_env.sh"
    fake_solver = tmp_path / "fake_solver.sh"
    log_path = tmp_path / "solver.log"
    state_path = tmp_path / "solver_state.txt"

    log_path.write_text("")
    _write_fake_solver(fake_solver)

    env = os.environ.copy()
    env.update(
        {
            "CONDA_SOLVER": str(fake_solver),
            "FAKE_SOLVER_LOG": str(log_path),
            "FAKE_SOLVER_STATE": str(state_path),
            "TEXT2UI_ENV_SETUP_SKIP_DEPS": "1",
            "PYTHON_VERSION": "3.10",
        }
    )

    result = _run_script(["bash", str(script), env_name], env=env)

    assert state_path.read_text().strip() == env_name
    log_contents = log_path.read_text().splitlines()
    assert any("env list" in entry for entry in log_contents)
    assert any("create" in entry and env_name in entry for entry in log_contents)
    assert "Skipping dependency installation" in result.stdout
    assert "[SETUP COMPLETE]" in result.stdout
