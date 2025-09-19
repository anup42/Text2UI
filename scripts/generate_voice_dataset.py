import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    dist = None  # type: ignore

try:
    from text2ui.voice_pipeline import DistributedContext as _DistributedContext, run_voice_pipeline
except (ImportError, AttributeError):
    from text2ui.voice_pipeline import run_voice_pipeline  # type: ignore

    @dataclass
    class _DistributedContext:  # type: ignore
        rank: int
        world_size: int
        local_rank: int

DistributedContext = _DistributedContext

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from text2ui.config import VoiceGenerationConfig, load_voice_config


def _resolve_cli_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _setup_distributed() -> tuple[Optional[DistributedContext], bool]:
    if torch is None or dist is None or not dist.is_available():
        return None, False
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None, False
    initialized = False
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        initialized = True
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank), initialized


def _run_pipeline(config: VoiceGenerationConfig, dist_ctx: Optional[DistributedContext]):
    try:
        return run_voice_pipeline(config, dist_ctx=dist_ctx)
    except TypeError as exc:
        if "unexpected keyword argument 'dist_ctx'" in str(exc):
            return run_voice_pipeline(config)
        raise


def _log_tensorboard(results, log_flag: bool) -> None:
    if not log_flag:
        return
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard logging requested but torch.utils.tensorboard is unavailable.")
    tb_root = Path("/tensorboard")
    tb_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_root))
    try:
        for idx, sample in enumerate(results):
            assistant_output = sample.get("assistant_output", "")
            text_value = str(assistant_output)
            writer.add_text("assistant_output", text_value, global_step=idx)
    finally:
        writer.flush()
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate voice assistant outputs with Qwen models.")
    parser.add_argument("--config", type=Path, default=Path("configs/voice_pipeline.yaml"), help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--output", type=Path, help="Override output file path")
    parser.add_argument("--num-samples", type=int, help="Override number of samples to generate")
    parser.add_argument("--use-stub", action="store_true", help="Use the fast deterministic stub generator")
    parser.add_argument("--mlp", action="store_true", help="Write generated outputs to TensorBoard /tensorboard")
    args = parser.parse_args()

    config = load_voice_config(_resolve_cli_path(args.config))
    if args.model_name:
        config.model_name = args.model_name
    if args.output:
        config.output_file = _resolve_cli_path(args.output)
    if args.num_samples:
        config.num_samples = args.num_samples
    if args.use_stub:
        config.use_stub = True

    dist_ctx, initialized = _setup_distributed()
    try:
        results = _run_pipeline(config, dist_ctx)
        if dist_ctx and dist_ctx.rank != 0:
            return
        _log_tensorboard(results, args.mlp)
        print(f"Generated {len(results)} samples -> {config.output_file}")
    finally:
        if initialized and dist is not None and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
