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
    parser.add_argument("--batch-size", type=int, help="Override batch size for generation")
    parser.add_argument("--max-new-tokens", type=int, help="Override max new tokens per sample")
    parser.add_argument("--num-samples", type=int, help="Limit number of input records processed")
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
    if args.batch_size:
        config.batch_size = max(1, args.batch_size)
    if args.max_new_tokens:
        config.max_new_tokens = args.max_new_tokens

    limit = args.num_samples

    kwargs = {"batch_size": config.batch_size}
    if limit is not None:
        kwargs["max_samples"] = limit

    try:
        results = run_ui_pipeline(config, **kwargs)
    except TypeError as exc:
        exc_msg = str(exc)
        if "batch_size" in exc_msg or "max_samples" in exc_msg:
            results = run_ui_pipeline(config)
            if limit is not None:
                results = results[: max(0, limit)]
        else:
            raise

    print(f"Generated {len(results)} UI samples -> {config.output_file}")


if __name__ == "__main__":
    main()

