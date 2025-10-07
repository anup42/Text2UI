# Text2UI Pipelines

End-to-end data generation pipelines that (1) synthesize diverse voice-assistant responses with a Qwen chat model and (2) translate those responses into HTML/CSS UI cards using a Qwen code model. The repository ships with a stubbed dataset for quick inspection plus configurable scripts for full-scale runs on 8x NVIDIA V100 GPUs.

## Project Layout

- `configs/`: YAML configs for both stages (voice, UI)
- `data/prompts/`: optional seed prompts to augment automatic sampling
- `data/samples/`: generated datasets (stubbed defaults, can be replaced by real runs)
- `scripts/`: CLI entry points for each pipeline stage and a helper to chain them
- `src/text2ui/`: reusable pipeline logic (prompt sampling, LLM driver, stubs, I/O utils)

## Environments and Dependencies

### Quickstart (Conda + NVIDIA V100 GPUs)

Use the provided helper to create a CUDA-enabled conda environment that has been validated on V100 hardware:

```bash
# create the environment (defaults to name "text2ui")
bash scripts/create_conda_env.sh

# re-activate it in new shells
conda activate text2ui

# authenticate with Hugging Face once per machine
huggingface-cli login

# configure accelerate for 8x V100 tensor-parallel inference
accelerate config
```

The script installs Python 3.10, CUDA 11.8 builds of PyTorch, `cmake>=3.25`, and the editable `text2ui` package (including all runtime dependencies from `pyproject.toml`).

For environments without conda, you can still install dependencies manually:

```bash
pip install -e .
```

### Quickstart (Python virtualenv)

Use the lightweight helper to bootstrap a virtual environment with editable installs and optional dev tools:

```bash
# create the environment in .venv (auto-selects Python >=3.10)
bash scripts/create_venv_env.sh .venv

# specify a custom interpreter and include dev extras
bash scripts/create_venv_env.sh .venv --python python3.11 --with-dev

# show available flags
bash scripts/create_venv_env.sh --help

# activate it (PowerShell example)
.\.venv\Scripts\Activate.ps1
```

The script validates the interpreter version (requires Python 3.10+), ensures `cmake>=3.25` is installed, upgrades pip inside the venv, installs `text2ui` in editable mode, and prints activation hints for Bash, PowerShell, and Command Prompt.

Additional requirements for large Qwen models:

- CUDA-capable GPUs (tested target: 8x V100 32 GB)
- Properly configured `accelerate` for tensor parallelism
- Hugging Face access to the chosen Qwen models (`huggingface-cli login`)

### Downloading the Qwen Checkpoints Locally

Use the helper script to pre-download the models referenced in the pipelines.
This is optional but avoids first-run latency when you launch either stage.

```bash
python scripts/download_qwen_models.py
```

By default the weights are stored in `models/qwen/<model-name>`. Override the
location or the set of models as needed:

```bash
python scripts/download_qwen_models.py \
  --output-dir /mnt/qwen \
  --model Qwen/Qwen2-72B-Instruct \
  --model Qwen/Qwen2.5-Coder-32B-Instruct
```

For the fastest results, increase `--max-workers` to saturate your bandwidth and
ensure the optional [`hf_transfer`](https://github.com/huggingface/hf-transfer)
package is installed. The script will enable the accelerated transfer backend
automatically when available, or you can force it on with `--hf-transfer`.

Ensure you have already authenticated with Hugging Face (`huggingface-cli
login`) before running the script.

## Running the Pipelines

### 1. Voice Assistant Output Generation

Produces ~1000 conversational responses covering weather, calendar, music, reminders, knowledge, smart home, navigation, and productivity domains.

```bash
# 1. Ensure the conda environment is active
conda activate text2ui

# 2. Select the GPUs to use (example: all 8 V100s)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 3. (First run only) Configure accelerate for distributed inference
accelerate config

# 4. Launch the voice generation stage
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
# Continue in the activated environment
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
# Run the stubbed CPU-friendly version (quick sanity check)
python scripts/run_full_pipeline.py --use-stub

# Or execute the full GPU pipeline after voice+UI configs are tuned
python scripts/run_full_pipeline.py \
  --voice-config configs/voice_pipeline.yaml \
  --ui-config configs/ui_pipeline.yaml
```

### End-to-End Command Reference

For a clean V100 setup, run the following commands in order (customize paths/models as needed):

```bash
bash scripts/create_conda_env.sh text2ui
conda activate text2ui
huggingface-cli login
accelerate config
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python scripts/generate_voice_dataset.py \
  --config configs/voice_pipeline.yaml \
  --model-name Qwen/Qwen2-72B-Instruct
python scripts/generate_ui_dataset.py \
  --config configs/ui_pipeline.yaml \
  --model-name Qwen/Qwen2.5-Coder-32B-Instruct
```

Use the final JSONL artifacts in `data/samples/` as inputs to your downstream tooling.

## Included Stub Dataset

The repository already includes 1,000 generated voice samples and their corresponding UI renders created with the stub generators (`use_stub = true`). They demonstrate expected schema and let you iterate on downstream tooling without GPUs. Replace them with real Qwen outputs by running the scripts without the `--use-stub` flag.

## Dataset Visualization Helper

Quickly preview HTML outputs stored in a JSONL dataset with the `visualize_dataset.py` utility. The script inlines the referenced CSS and captures JPEG screenshots using Playwright.

1. Install the optional dependency and its browser runtime:

   ```bash
   pip install playwright
   playwright install chromium
   ```

2. Render the first few samples to `tmp_outputs/dataset_visualizations/`:

   ```bash
   python scripts/visualize_dataset.py \
     data/samples/text2ui-3/generated_dataset-1.cache.jsonl \
     --count 3
   ```

Use `--output-dir` to change the destination folder and `--css-dir` to supply additional lookup paths for stylesheet files such as `agent.css` or `agent2.css`.

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
