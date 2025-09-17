import argparse
from pathlib import Path

from text2ui.config import load_ui_config, load_voice_config
from text2ui.ui_pipeline import run_ui_pipeline
from text2ui.voice_pipeline import run_voice_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both voice and UI generation pipelines in sequence.")
    parser.add_argument("--voice-config", type=Path, default=Path("configs/voice_pipeline.yaml"), help="Voice generation config path")
    parser.add_argument("--ui-config", type=Path, default=Path("configs/ui_pipeline.yaml"), help="UI conversion config path")
    parser.add_argument("--use-stub", action="store_true", help="Use stub generators for both stages")
    args = parser.parse_args()

    voice_config = load_voice_config(args.voice_config)
    ui_config = load_ui_config(args.ui_config)

    if args.use_stub:
        voice_config.use_stub = True
        ui_config.use_stub = True

    voice_results = run_voice_pipeline(voice_config)
    if ui_config.input_file != voice_config.output_file:
        ui_config.input_file = voice_config.output_file
    ui_results = run_ui_pipeline(ui_config)

    print(
        f"Voice samples: {len(voice_results)} -> {voice_config.output_file}\n"
        f"UI samples: {len(ui_results)} -> {ui_config.output_file}"
    )


if __name__ == "__main__":
    main()
