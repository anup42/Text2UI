import argparse
from pathlib import Path

from text2ui.config import UIGenerationConfig, load_ui_config
from text2ui.ui_pipeline import run_ui_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert voice assistant outputs into HTML/CSS UI components.")
    parser.add_argument("--config", type=Path, default=Path("configs/ui_pipeline.yaml"), help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--output", type=Path, help="Override output file path")
    parser.add_argument("--input", type=Path, help="Override input voice dataset path")
    parser.add_argument("--use-stub", action="store_true", help="Use deterministic stub generator")
    args = parser.parse_args()

    config = load_ui_config(args.config)
    if args.model_name:
        config.model_name = args.model_name
    if args.output:
        config.output_file = args.output
    if args.input:
        config.input_file = args.input
    if args.use_stub:
        config.use_stub = True

    results = run_ui_pipeline(config)
    print(f"Generated {len(results)} UI samples -> {config.output_file}")


if __name__ == "__main__":
    main()
