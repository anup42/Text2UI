import argparse
from pathlib import Path

from text2ui.config import VoiceGenerationConfig, load_voice_config
from text2ui.voice_pipeline import run_voice_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate voice assistant outputs with Qwen models.")
    parser.add_argument("--config", type=Path, default=Path("configs/voice_pipeline.yaml"), help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--output", type=Path, help="Override output file path")
    parser.add_argument("--num-samples", type=int, help="Override number of samples to generate")
    parser.add_argument("--use-stub", action="store_true", help="Use the fast deterministic stub generator")
    args = parser.parse_args()

    config = load_voice_config(args.config)
    if args.model_name:
        config.model_name = args.model_name
    if args.output:
        config.output_file = args.output
    if args.num_samples:
        config.num_samples = args.num_samples
    if args.use_stub:
        config.use_stub = True

    results = run_voice_pipeline(config)
    print(f"Generated {len(results)} samples -> {config.output_file}")


if __name__ == "__main__":
    main()
