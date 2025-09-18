"""Upload all files in a folder using the ``odc-sftp`` CLI.

This script walks a source directory recursively and uploads every file it finds to
the provided destination folder using the command::

    ~/odc-sftp storage upload file <source> <destination> --request-timeout 300

By default it adds ``--num_threads=16`` to each upload call. The destination paths
mirror the directory layout of the source folder.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload all files from a directory using the odc-sftp storage upload CLI."
        )
    )
    parser.add_argument(
        "src",
        type=Path,
        help="Path to the local source directory whose files will be uploaded.",
    )
    parser.add_argument(
        "dst",
        type=str,
        help=(
            "Destination directory path to use on the remote side. Relative file"
            " paths from the source directory are preserved."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="Number of threads to pass to odc-sftp via --num_threads (default: 16).",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Timeout value (in seconds) forwarded to odc-sftp (default: 300).",
    )
    return parser.parse_args()


def build_destination_path(dst_root: str, relative_file: Path) -> str:
    """Build the remote destination path for ``relative_file``.

    ``dst_root`` is treated as a POSIX-style path because odc-sftp targets a
    remote storage system.
    """

    posix_root = PurePosixPath(dst_root)
    posix_relative = PurePosixPath("/".join(relative_file.parts))
    return str(posix_root.joinpath(posix_relative))


def upload_file(
    executable: Path,
    src_file: Path,
    dst_file: str,
    request_timeout: int,
    num_threads: int,
) -> None:
    command = [
        str(executable),
        "storage",
        "upload",
        "file",
        str(src_file),
        dst_file,
        "--request-timeout",
        str(request_timeout),
        f"--num_threads={num_threads}",
    ]

    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Upload failed for {src_file} -> {dst_file}")


def main() -> None:
    args = parse_args()

    if not args.src.is_dir():
        raise ValueError(f"Source path '{args.src}' is not a directory")

    odc_sftp = Path("~/odc-sftp").expanduser()
    if not odc_sftp.exists():
        raise FileNotFoundError(
            f"Could not find odc-sftp executable at '{odc_sftp}'."
        )

    for file_path in sorted(args.src.rglob("*")):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(args.src)
        destination = build_destination_path(args.dst, relative)
        upload_file(
            executable=odc_sftp,
            src_file=file_path,
            dst_file=destination,
            request_timeout=args.request_timeout,
            num_threads=args.num_threads,
        )


if __name__ == "__main__":
    main()
