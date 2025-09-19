import argparse
import sys
import time
from pathlib import Path

try:  # pragma: no cover - accelerate only required for multi-GPU runs
    from accelerate import PartialState  # type: ignore
except ImportError:  # pragma: no cover - fallback when accelerate is absent
    PartialState = None  # type: ignore

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from text2ui.config import VoiceGenerationConfig, load_voice_config
from text2ui.voice_pipeline import run_voice_pipeline


def _resolve_cli_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _prepare_writer():
    writer_cls = SummaryWriter
    if writer_cls is None:
        try:  # pragma: no cover - optional dependency
            from tensorboardX import SummaryWriter as TensorboardXWriter  # type: ignore
        except ImportError:
            return None, None
        writer_cls = TensorboardXWriter
    root = Path("/tensorboard")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"voice_{int(time.time())}"
    writer = writer_cls(log_dir=str(run_dir))
    step = 0

    def add_batch(records):
        nonlocal step
        for record in records:
            writer.add_text("assistant_output", str(record.get("assistant_output", "")), step)
            step += 1

    return writer, add_batch


def _run_pipeline(config: VoiceGenerationConfig, batch_callback, debug: bool):
    try:
        run_voice_pipeline(config, batch_callback=batch_callback, debug=debug)
        return True
    except TypeError as exc:
        message = str(exc)
        if "batch_callback" in message or "debug" in message:
            run_voice_pipeline(config)
            return False
        raise


def _debug_log(results, already_logged: bool) -> None:
    if already_logged:
        return
    for record in results:
        print(f"[DEBUG] assistant_output: {record.get('assistant_output', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate voice assistant outputs with Qwen models.")
    parser.add_argument("--config", type=Path, default=Path("configs/voice_pipeline.yaml"), help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--output", type=Path, help="Override output file path")
    parser.add_argument("--num-samples", type=int, help="Override number of samples to generate")
    parser.add_argument("--batch-size", type=int, help="Override batch size for inference")
    parser.add_argument("--use-stub", action="store_true", help="Use the fast deterministic stub generator")
    parser.add_argument("--mlp", action="store_true", help="Write generated outputs to TensorBoard /tensorboard")
    parser.add_argument("--debug", action="store_true", help="Print assistant outputs as they are emitted")
    args = parser.parse_args()

    config = load_voice_config(_resolve_cli_path(args.config))
    if args.model_name:
        config.model_name = args.model_name
    if args.output:
        config.output_file = _resolve_cli_path(args.output)
    if args.num_samples:
        config.num_samples = args.num_samples
    if args.batch_size:
        config.batch_size = max(1, args.batch_size)
    if args.use_stub:
        config.use_stub = True

    writer = None
    batch_callback = None
    if args.mlp:
        writer, batch_callback = _prepare_writer()
        if writer is None or batch_callback is None:
            print(
                "[WARN] TensorBoard logging requested but not available. Install torch.utils.tensorboard or tensorboardX.",
                file=sys.stderr,
            )

    state = PartialState() if PartialState is not None else None
    is_main = bool(state is None or state.is_main_process)

    try:
        results = run_voice_pipeline(
            config,
            batch_callback=batch_callback if batch_callback else None,
            debug=args.debug,
        )
        already_logged = True
    except TypeError as exc:
        message = str(exc)
        if "batch_callback" in message or "debug" in message:
            results = run_voice_pipeline(config)
            already_logged = False
        else:
            raise

    try:
        if is_main:
            print(f"Generated {len(results)} samples -> {config.output_file}")
            if args.debug:
                _debug_log(results, already_logged)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if state is not None:
            state.wait_for_everyone()


if __name__ == "__main__":
    main()
