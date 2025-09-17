# Text2UI Pipelines

End-to-end data generation pipelines that (1) synthesize diverse voice-assistant responses with a Qwen chat model and (2) translate those responses into HTML/CSS UI cards using a Qwen code model. The repository ships with a stubbed dataset for quick inspection plus configurable scripts for full-scale runs on 8x NVIDIA V100 GPUs.

## Project Layout

- `configs/`: YAML configs for both stages (voice, UI)
- `data/prompts/`: optional seed prompts to augment automatic sampling
- `data/samples/`: generated datasets (stubbed defaults, can be replaced by real runs)
- `scripts/`: CLI entry points for each pipeline stage and a helper to chain them
- `src/text2ui/`: reusable pipeline logic (prompt sampling, LLM driver, stubs, I/O utils)

## Environments and Dependencies

The project targets Python 3.10+. Install runtime dependencies with either `pip` or `uv`:

```bash
pip install -e .
```

For large Qwen models you will additionally need:

- CUDA-capable GPUs (tested target: 8x V100 32 GB)
- PyTorch with CUDA (2.1 or newer recommended)
- `accelerate` properly configured for tensor parallelism
- Hugging Face access to the chosen Qwen models (`huggingface-cli login`)

## Running the Pipelines

### 1. Voice Assistant Output Generation

Produces ~1000 conversational responses covering weather, calendar, music, reminders, knowledge, smart home, navigation, and productivity domains.

```bash
# Configure your environment first
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate config  # choose multi-GPU / DeepSpeed ZeRO-3 if needed

python scripts/generate_voice_dataset.py \
  --config configs/voice_pipeline.yaml \
  --model-name Qwen/Qwen2-72B-Instruct
```

Key config knobs (see `configs/voice_pipeline.yaml`):

- `num_samples`: total conversations to synthesize (default 1000)
- `temperature` / `top_p`: diversity controls
- `max_new_tokens`: response budget per sample
- `use_stub`: switch to the fast deterministic generator (no GPU required)

Outputs land in `data/samples/voice_assistant_outputs.jsonl` (JSON Lines).

### 2. Voice Output to HTML/CSS Conversion

Transforms each voice sample into a responsive HTML document (single card UI) using a Qwen code model.

```bash
python scripts/generate_ui_dataset.py \
  --config configs/ui_pipeline.yaml \
  --model-name Qwen/Qwen2.5-Coder-32B-Instruct
```

Important settings (see `configs/ui_pipeline.yaml`):

- `input_file`: voice dataset path (defaults to stage 1 output)
- `max_new_tokens`: HTML/CSS length budget
- `temperature`: typically kept low for determinism
- `use_stub`: template-driven fallback for CPU-only preview runs

Results are stored at `data/samples/voice_to_ui_components.jsonl`.

### 3. Full Pipeline Orchestration

Run both stages in sequence (makes the UI stage reuse the latest voice dataset automatically):

```bash
python scripts/run_full_pipeline.py --use-stub
# or for production inference
python scripts/run_full_pipeline.py \
  --voice-config configs/voice_pipeline.yaml \
  --ui-config configs/ui_pipeline.yaml
```

## Included Stub Dataset

The repository already includes 1,000 generated voice samples and their corresponding UI renders created with the stub generators (`use_stub = true`). They demonstrate expected schema and let you iterate on downstream tooling without GPUs. Replace them with real Qwen outputs by running the scripts without the `--use-stub` flag.

## Extending the Pipelines

- Add more seed prompts in `data/prompts/voice_prompts_seed.jsonl` to force coverage of niche scenarios
- Tune generation parameters in the YAML configs for different diversity/quality trade-offs
- Swap in alternative Qwen variants (e.g., `Qwen2-72B-Instruct` vs `Qwen1.5-110B-Chat`) by editing the configs or passing `--model-name`
- Integrate retrieval/datasets by editing `src/text2ui/prompt_bank.py`

## Data Schema

### Voice Dataset (`voice_assistant_outputs.jsonl`)

```json
{
  "category": "weather",
  "persona": "sailing enthusiast",
  "locale": "Lisbon, Portugal",
  "user_prompt": "Give me the wind conditions for the Tagus this afternoon.",
  "assistant_output": "38 degC, Overcast. High: 38 degC, Low: 35 degC. Humidity: 31%, Wind: 12 km/h.",
  "model": "stub"  // replace with the actual Qwen model when you run inference,
  "generation": {"max_new_tokens": 320, "temperature": 0.85, "top_p": 0.9},
  "system_prompt": "..."
}
```

### UI Dataset (`voice_to_ui_components.jsonl`)

```json
{
  "category": "weather",
  "html": "<!DOCTYPE html>...",  // full HTML document
  "model": "stub"  // replace with the actual Qwen code model in production runs,
  "generation": {"max_new_tokens": 1024, "temperature": 0.25, "top_p": 0.9},
  "system_prompt": "...",
  "voice_sample": { ... }  // original voice record for traceability
}
```

## Tips for Multi-GPU Qwen Runs

- Prefer `accelerate launch` or `torchrun` to distribute inference across the 8 GPUs
- Enable `torch.backends.cuda.matmul.allow_tf32 = True` for faster matmuls on V100
- Consider `bitsandbytes` or `vLLM` if memory becomes a bottleneck
- Cache tokenizers/models locally using `huggingface-cli download` to avoid repeated downloads

## License

MIT License (inherit from parent repository policy)
