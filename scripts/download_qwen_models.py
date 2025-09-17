#!/usr/bin/env python
"""Download the Qwen models referenced in the Text2UI pipelines.

The script uses the Hugging Face Hub API to materialize local copies of the
large Qwen checkpoints leveraged by the voice and UI generation stages. By
default it downloads both models into ``models/qwen`` within the repository, but
you can pass ``--output-dir`` to select an alternative location or ``--model``
multiple times to customize the set of models.

Examples
--------
Download the default pair of models (voice + UI) to ``models/qwen``::

    python scripts/download_qwen_models.py

Download into a custom directory (one subdirectory per model)::

    python scripts/download_qwen_models.py --output-dir /mnt/models

Fetch an alternate list of models::

    python scripts/download_qwen_models.py \
        --model Qwen/Qwen1.5-110B-Chat \
        --model Qwen/Qwen1.5-Coder-32B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Default repositories referenced in the README and pipeline configs.
DEFAULT_MODELS = [
    "Qwen/Qwen2-72B-Instruct",       # voice generation model
    "Qwen/Qwen2.5-Coder-32B-Instruct",  # HTML/CSS generation model
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen models used by the Text2UI pipelines."
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "Hugging Face model repository to download."
            " Repeat to fetch multiple repositories. Defaults to the"
            " Qwen models referenced in the README."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/qwen"),
        help=(
            "Directory where the model folders will be stored. A subdirectory"
            " named after the repository will be created for each download."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Optional model revision (branch, tag, or commit SHA) to download.",
    )
    return parser.parse_args(argv)


def download_model(repo_id: str, destination_root: Path, revision: str | None = None) -> Path:
    """Download ``repo_id`` into ``destination_root`` using ``snapshot_download``."""
    destination_root.mkdir(parents=True, exist_ok=True)

    repo_name = repo_id.split("/")[-1]
    local_dir = destination_root / repo_name
    print(f"\n=== Downloading {repo_id} -> {local_dir} ===")

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        tqdm_class=None,  # rely on default console progress output
    )

    return local_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = args.models if args.models else DEFAULT_MODELS

    if not models:
        print("No models specified. Use --model to provide at least one repository.")
        return 1

    for repo_id in models:
        download_model(repo_id, args.output_dir, args.revision)

    print("\nAll requested models are available locally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
