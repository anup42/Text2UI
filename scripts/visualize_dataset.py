"""Utilities for rendering dataset HTML samples to JPEG screenshots.

This script converts the first *n* entries from a JSON Lines (JSONL)
dataset into rendered JPEG screenshots.  Each dataset record is expected
to contain an ``output`` field with an HTML document that references one
or more CSS stylesheets (for example ``agent.css`` or ``agent2.css``).

The script performs the following steps:

1. Read a JSONL file containing dataset entries.
2. Inline the referenced CSS stylesheets so that the HTML can be rendered
   without external dependencies.
3. Use Playwright to render the HTML and capture it as a JPEG image.

Example usage::

    python scripts/visualize_dataset.py \
        data/samples/text2ui-3/generated_dataset-1.cache.jsonl \
        --count 3 --output-dir tmp_outputs/screenshots

The script depends on Playwright.  Install it (and the Chromium browser)
with::

    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional


if TYPE_CHECKING:  # pragma: no cover - typing only
    from playwright.async_api import Browser


try:
    from playwright.async_api import async_playwright  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    async_playwright = None  # type: ignore[assignment]


CSS_LINK_PATTERN = re.compile(r"<link[^>]+href=['\"]([^'\"]+)['\"][^>]*>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the first N HTML outputs from a JSONL dataset to JPEG"
            " screenshots."
        )
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to the JSONL dataset file containing HTML outputs.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of records to visualize (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp_outputs/dataset_visualizations"),
        help="Directory where JPEG files will be saved.",
    )
    parser.add_argument(
        "--css-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Additional directory to search for CSS files referenced by "
            "the HTML.  Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Viewport width used for rendering (default: 1024).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="JPEG quality (default: 90).",
    )
    return parser.parse_args()


def load_jsonl(path: Path, limit: int) -> List[dict]:
    records: List[dict] = []
    if limit <= 0:
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - debug aid
                raise ValueError(f"Failed to parse line {line_number}: {exc}") from exc
            records.append(payload)
            if len(records) >= limit:
                break
    return records


def build_css_search_dirs(jsonl_path: Path, extra_dirs: Optional[Iterable[Path]]) -> List[Path]:
    candidate_dirs: List[Path] = []
    seen = set()

    def add_dir(directory: Path) -> None:
        resolved = directory.resolve()
        if resolved not in seen and resolved.exists():
            candidate_dirs.append(resolved)
            seen.add(resolved)

    add_dir(jsonl_path.parent)
    add_dir(jsonl_path.parent.parent)
    repo_samples_dir = Path("data/samples")
    add_dir(repo_samples_dir)

    if extra_dirs:
        for extra in extra_dirs:
            add_dir(extra)

    return candidate_dirs


def inline_css(html: str, css_dirs: Iterable[Path]) -> str:
    """Replace linked stylesheets in ``html`` with inline ``<style>`` tags."""

    def replace(match: re.Match[str]) -> str:
        href = match.group(1)
        css_path = find_css_file(href, css_dirs)
        if css_path is None:
            print(f"[warning] CSS file '{href}' not found; keeping original link tag.")
            return match.group(0)
        css_content = css_path.read_text(encoding="utf-8")
        return f"<style>\n{css_content}\n</style>"

    return CSS_LINK_PATTERN.sub(replace, html)


def find_css_file(href: str, css_dirs: Iterable[Path]) -> Optional[Path]:
    href_path = Path(href)
    for directory in css_dirs:
        candidate = directory / href_path
        if candidate.exists():
            return candidate
    return None


async def render_html_to_jpeg(
    browser: "Browser",
    html: str,
    output_path: Path,
    css_dirs: Iterable[Path],
    *,
    width: int,
    quality: int,
) -> None:
    html_with_css = inline_css(html, css_dirs)

    page = await browser.new_page(viewport={"width": width, "height": 720})
    await page.set_content(html_with_css, wait_until="networkidle")

    # Ensure that web fonts (and therefore text glyphs) are fully rendered
    # before we capture the screenshot.  Without this, Chromium occasionally
    # grabs the frame before font loading completes, resulting in images with
    # missing text.  ``document.fonts.ready`` resolves when all fonts are
    # loaded (or have failed), so awaiting it gives the layout a chance to
    # stabilize.  Older browsers may not expose ``document.fonts``, so we fall
    # back to a short delay in that case.
    try:
        await page.evaluate("document.fonts.ready")
    except Exception:
        await page.wait_for_timeout(200)
    else:
        await page.wait_for_timeout(50)
    await page.screenshot(
        path=str(output_path),
        type="jpeg",
        full_page=True,
        quality=quality,
    )
    await page.close()


async def process_dataset(args: argparse.Namespace) -> List[Path]:
    if async_playwright is None:  # pragma: no cover - runtime check
        raise RuntimeError(
            "Playwright is required for rendering. Install it with 'pip install "
            "playwright' and run 'playwright install chromium'."
        )

    records = load_jsonl(args.jsonl_path, args.count)
    css_dirs = build_css_search_dirs(args.jsonl_path, args.css_dir or [])

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()

        for index, record in enumerate(records):
            html = record.get("output")
            if not isinstance(html, str):
                print(
                    f"[warning] Record {index} does not contain an 'output' HTML string; skipping."
                )
                continue

            output_path = output_dir / f"record_{index:04d}.jpg"
            print(f"Rendering record {index} -> {output_path}")
            await render_html_to_jpeg(
                browser,
                html,
                output_path,
                css_dirs,
                width=args.width,
                quality=args.quality,
            )
            saved_paths.append(output_path)

        await browser.close()

    return saved_paths


def main() -> None:
    args = parse_args()
    if not args.jsonl_path.exists():
        raise SystemExit(f"JSONL file not found: {args.jsonl_path}")
    try:
        saved_paths = asyncio.run(process_dataset(args))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not saved_paths:
        print("No records were rendered.")
    else:
        print("Saved JPEG screenshots:")
        for path in saved_paths:
            print(f" - {path}")


if __name__ == "__main__":
    main()
