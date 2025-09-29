#!/usr/bin/env python3
"""Generate assistant to HTML training samples with Gemini Pro or ChatGPT.

This script batches scenario prompts, asks either Gemini or ChatGPT for HTML UIs that
leverage `agent2.css` (backward compatible with agent.css), and writes the aggregated dataset to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

try:  # Optional dependency, required only when provider == "gemini"
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

try:  # Optional dependency, required only when provider == "gemini"
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    google_exceptions = None

try:  # Optional dependency, required only when provider == "chatgpt"
    from openai import OpenAI  # type: ignore
    import openai as openai_module  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore
    openai_module = None  # type: ignore

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_CACHE_SUFFIX = ".cache.jsonl"
DEFAULT_SECRETS_FILE = Path("configs/api_keys2.json")


def load_secrets_dict(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("Failed to read secrets file %s: %s", path, exc)
        return {}
    if isinstance(data, dict):
        return data
    logging.warning("Secrets file %s must contain a JSON object.", path)
    return {}

CSS_CLASSES = [
    "agent-screen",
    "agent-header",
    "agent-eyebrow",
    "agent-summary",
    "agent-subtext",
    "agent-section",
    "agent-section-header",
    "agent-section-title",
    "agent-list",
    "agent-list-spaced",
    "agent-inline",
    "agent-inline-scroll",
    "agent-actions",
    "agent-toolbar",
    "agent-toolbar-title",
    "agent-app-bar",
    "agent-card",
    "agent-floating-card",
    "agent-card-title",
    "agent-card-body",
    "agent-callout",
    "agent-callout-title",
    "agent-callout-body",
    "agent-tag",
    "agent-badge",
    "agent-pill-badge",
    "agent-button",
    "agent-footer",
    "agent-note",
    "agent-breadcrumbs",
    "agent-breadcrumb",
    "agent-divider",
    "agent-kpis",
    "agent-kpi",
    "agent-kpi-label",
    "agent-kpi-value",
    "agent-stat-grid",
    "agent-stat",
    "agent-stat-label",
    "agent-stat-value",
    "agent-meta",
    "agent-metadata",
    "agent-metadata-row",
    "agent-progress",
    "agent-progress-label",
    "agent-progress-fill",
    "agent-progress-circle",
    "agent-progress-track",
    "agent-progress-indicator",
    "agent-progress-value",
    "agent-pill-group",
    "agent-pill",
    "agent-status",
    "agent-timer",
    "agent-avatar",
    "agent-avatar-stack",
    "agent-grid",
    "agent-grid-two",
    "agent-grid-three",
    "agent-grid-responsive",
    "agent-tab-bar",
    "agent-tab",
    "agent-bottom-nav",
    "agent-bottom-nav-item",
    "agent-bottom-sheet",
    "agent-toast",
    "agent-toast-info",
    "agent-toast-warning",
    "agent-toast-danger",
    "agent-empty-state",
    "agent-empty-title",
    "agent-empty-action",
    "agent-chart",
    "agent-chart-bar",
    "agent-chart-line",
    "agent-timeline",
    "agent-timeline-item",
    "agent-timeline-dot",
    "agent-form",
    "agent-field",
    "agent-field-label",
    "agent-field-control",
    "agent-toggle",
]

RATE_LIMIT_ERRORS: Tuple[type[BaseException], ...] = tuple()
if google_exceptions:  # pragma: no cover - depends on optional dependency
    RATE_LIMIT_ERRORS = (
        google_exceptions.ResourceExhausted,
        google_exceptions.DeadlineExceeded,
        google_exceptions.ServiceUnavailable,
        google_exceptions.InternalServerError,
        google_exceptions.Aborted,
    )

OPENAI_RETRY_ERRORS: Tuple[type[BaseException], ...] = tuple()
if openai_module is not None:  # pragma: no cover - depends on optional dependency
    candidates: List[type[BaseException]] = []
    for name in (
        "RateLimitError",
        "APIError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    ):
        candidate = getattr(openai_module, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            candidates.append(candidate)
    OPENAI_RETRY_ERRORS = tuple(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate assistant to HTML training samples with Gemini Pro or ChatGPT."
        )
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "chatgpt"),
        default="gemini",
        help="Which LLM provider to use for generation (default: gemini).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the selected provider. Falls back to GEMINI_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=Path("data/samples/scenario2.txt"),
        help="Path to the scenario list (one name per line).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples/text2ui/generated_dataset.json"),
        help="Path to write the final JSON array.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        help="Optional JSONL cache file to support resume. Defaults to output + %s" % DEFAULT_CACHE_SUFFIX,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name to use when provider=gemini.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model name to use when provider=chatgpt.",
    )
    parser.add_argument(
        "--secrets-file",
        type=Path,
        default=DEFAULT_SECRETS_FILE,
        help="Path to a JSON file containing provider keys (defaults to configs/api_keys.json).",
    )
    parser.add_argument(
        "--samples-per-call",
        type=int,
        default=10,
        help="Number of samples to request per API call.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=20000,
        help="Total number of samples to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=12000,
        help="Maximum tokens to request from the model.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=12.5,
        help="Minimum seconds between requests (default matches Gemini free-tier 5 RPM).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts per batch on recoverable errors.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=4.0,
        help="Base seconds for exponential backoff when retrying.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for scenario shuffling.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N successful batches.",
    )
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    explicit = (args.api_key or "").strip()
    if explicit:
        return explicit
    env_name = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"
    env_key = os.getenv(env_name, "").strip()
    if env_key:
        return env_key
    file_value = load_secret_from_file(args.secrets_file, args.provider)
    if file_value:
        return file_value
    return "YOUR_GEMINI_API_KEY" if args.provider == "gemini" else "YOUR_OPENAI_API_KEY"


def load_scenarios(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    scenarios = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not scenarios:
        raise ValueError(f"Scenario file {path} is empty.")
    return scenarios


def load_cache(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Skipping malformed cache line: %s", line[:80])
            continue
        if isinstance(obj, dict) and "input" in obj and "output" in obj:
            records.append({"input": str(obj["input"]), "output": str(obj["output"])})
    return records


def append_cache(path: Path, samples: Sequence[Dict[str, str]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False))
            handle.write("\n")





def load_secret_from_file(path: Path, provider: str) -> str:
    data = load_secrets_dict(path)
    entry = data.get("gemini") if provider == "gemini" else data.get("chatgpt")
    if isinstance(entry, dict):
        if provider == "gemini":
            api_keys = entry.get("api_keys")
            if isinstance(api_keys, list):
                for key in api_keys:
                    key_str = str(key).strip()
                    if key_str:
                        return key_str
            primary = entry.get("api_key")
            if isinstance(primary, str) and primary.strip():
                return primary.strip()
        else:
            token = entry.get("api_key")
            if isinstance(token, str) and token.strip():
                return token.strip()
        return ""
    if isinstance(entry, list):
        for item in entry:
            item_str = str(item).strip()
            if item_str:
                return item_str
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return ""


def gather_gemini_credentials(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    data = load_secrets_dict(args.secrets_file)
    entry = data.get("gemini")
    keys: list[str] = []
    models: list[str] = []

    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    cli_key = (args.api_key or "").strip()

    def append_unique(collection: list[str], value: str) -> None:
        if value and value not in collection:
            collection.append(value)

    if isinstance(entry, dict):
        api_keys = entry.get("api_keys")
        if isinstance(api_keys, list):
            for key in api_keys:
                append_unique(keys, str(key).strip())
        primary = entry.get("api_key")
        if isinstance(primary, str):
            append_unique(keys, primary.strip())
        model_list = entry.get("models")
        if isinstance(model_list, list):
            for name in model_list:
                append_unique(models, str(name).strip())
    elif isinstance(entry, list):
        for key in entry:
            append_unique(keys, str(key).strip())
    elif isinstance(entry, str):
        append_unique(keys, entry.strip())

    append_unique(keys, env_key)
    append_unique(keys, cli_key)

    keys = [k for k in keys if k]
    if not keys:
        raise ValueError("No Gemini API keys configured; provide --api-key or update the secrets file.")

    if args.model:
        models = [args.model]
    elif not models:
        models = [DEFAULT_GEMINI_MODEL]

    models = [m for m in models if m]
    return keys, models

def scenario_batches(scenarios: Sequence[str], batch_size: int, rng: random.Random) -> Iterator[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pool = list(scenarios)
    buffer: List[str] = []
    while True:
        while len(buffer) < batch_size:
            rng.shuffle(pool)
            buffer.extend(pool)
        batch = buffer[:batch_size]
        buffer = buffer[batch_size:]
        yield batch


PROMPT_TEMPLATE1 = """You craft UI training data for an assistant that renders responses as HTML.
Return JSON array with exactly {count} objects ({count} JSON objects per request). No commentary, no markdown fences.
Each object must contain:
  "input": Natural language assistant response (single paragraph, scenario-aligned, <= 5 sentences, ASCII only).
  "output": Complete HTML5 document that uses <link rel=\"stylesheet\" href=\"agent2.css\" /> and the provided CSS class set.
Guidelines:
- Wrap content in <main class=\"agent-screen\" data-scenario=\"SCENARIO\"> where data-scenario matches the scenario string exactly.
- Use only these CSS utility classes (append modifiers like secondary/subtle after agent-button when needed): {classes}.
- Include 2-4 sections with headers, summaries, and context-rich data tied to the scenario.
- Provide actionable controls using <button type=\"button\" class=\"agent-button ...\" data-action=\"...\">.
- Keep markup accessible (use headings, lists, aria labels where useful), ASCII characters only, and design for touch-friendly mobile interactions.
- Do not emit the literal sequence /n; use actual line breaks or spaces in the HTML output.
- Do not embed custom CSS or scripts; rely on agent2.css utility classes.
Scenarios to cover:
{scenario_lines}
"""


PROMPT_TEMPLATE = """You craft UI training data for an assistant that renders agent or voice assistant responses (like bixby, alexa, siri, ok google) as HTML.
Return JSON array with exactly {count} objects ({count} JSON objects per request). No commentary, no markdown fences.
Each object must contain:
  "input": Natural language assistant response output (single paragraph, scenario-aligned, <= 4 sentences, ASCII only).
  "output": Complete HTML5 document that uses <link rel=\"stylesheet\" href=\"agent2.css\" /> and the provided CSS class set.
Guidelines:
- Wrap content in <main class=\"agent-screen\" data-scenario=\"SCENARIO\"> where data-scenario matches the scenario string exactly.
- Use only these CSS utility classes (append modifiers like secondary/subtle after agent-button when needed): {classes}.
- Include 2-4 sections with headers, summaries, and context-rich data tied to the scenario.
- Provide actionable controls using <button type=\"button\" class=\"agent-button ...\" data-action=\"...\">.
- Keep markup accessible (use headings, lists, aria labels where useful), ASCII characters only, and design for touch-friendly mobile interactions.
- Do not emit the literal sequence \\n; use actual line breaks or spaces in the HTML output.
- Do not embed custom CSS or scripts; rely on agent2.css utility classes.
Scenarios to cover:
{scenario_lines}
"""



def build_prompt(batch: Sequence[str]) -> str:
    scenario_lines = "\n".join(f"{idx + 1}. {name}" for idx, name in enumerate(batch))
    classes = ", ".join(CSS_CLASSES)
    return PROMPT_TEMPLATE.format(
        count=len(batch),
        classes=classes,
        scenario_lines=scenario_lines,
    )


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def ensure_stylesheet(html: str) -> str:
    if "agent2.css" in html or "agent.css" in html:
        return html
    insertion = '<link rel="stylesheet" href="agent2.css" />'
    if "<head" in html:
        return html.replace("<head>", f"<head>\n  {insertion}", 1)
    return insertion + "\n" + html

def extract_samples(raw_text: str) -> List[Dict[str, str]]:
    cleaned = strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as err:
        raise ValueError(f"Model response is not valid JSON: {err}") from err
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            data = data["samples"]
        else:
            raise ValueError("Expected a JSON array of samples.")
    if not isinstance(data, list):
        raise ValueError("Model response must be a JSON array.")
    samples: List[Dict[str, str]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        input_text = str(obj.get("input", "")).strip()
        output_html = ensure_stylesheet(str(obj.get("output", "")).strip())
        if input_text and output_html:
            samples.append({"input": input_text, "output": output_html})
    return samples


def write_final_dataset(path: Path, samples: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(samples), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = resolve_api_key(args)

    if args.provider == "gemini":
        gemini_keys, gemini_models = gather_gemini_credentials(args)
        key_index = 0
        model_index = 0

        def configure_model() -> object:
            genai.configure(api_key=gemini_keys[key_index])  # type: ignore[arg-type]
            return genai.GenerativeModel(gemini_models[model_index])  # type: ignore[attr-defined]

        model = configure_model()
        max_rotations = len(gemini_keys) * max(1, len(gemini_models))

        def rotate_credentials(reason: object) -> None:
            nonlocal key_index, model_index, model
            switched_key = False
            if len(gemini_keys) > 1:
                key_index = (key_index + 1) % len(gemini_keys)
                switched_key = True
            if not switched_key and len(gemini_models) > 1:
                model_index = (model_index + 1) % len(gemini_models)
            elif switched_key and key_index == 0 and len(gemini_models) > 1:
                model_index = (model_index + 1) % len(gemini_models)
            logging.info(
                "Switching Gemini credentials to key %d/%d and model %s after %s.",
                key_index + 1,
                len(gemini_keys),
                gemini_models[model_index],
                reason,
            )
            model = configure_model()

        def extract_response_text(response: object) -> str:
            response_text = getattr(response, "text", None)
            if response_text:
                return str(response_text).strip()
            chunks: List[str] = []
            if hasattr(response, "candidates"):
                for candidate in getattr(response, "candidates", []):
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", []) if content is not None else []
                    for part in parts:
                        text_part = getattr(part, "text", None)
                        if text_part:
                            chunks.append(text_part)
            combined = "".join(chunks).strip()
            if not combined:
                raise ValueError("Received empty response text from Gemini.")
            return combined

        def call_model(prompt: str) -> str:
            nonlocal model
            rotation_attempts = 0
            while True:
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_output_tokens": args.max_output_tokens,
                            "response_mime_type": "application/json",
                        },
                        request_options={"timeout": 180},
                    )
                    return extract_response_text(response)
                except RATE_LIMIT_ERRORS as rate_err:  # type: ignore[arg-type]
                    rotation_attempts += 1
                    if rotation_attempts >= max_rotations:
                        raise
                    rotate_credentials(rate_err)
                    continue

        recoverable_types = RATE_LIMIT_ERRORS
    else:
        client = OpenAI(api_key=api_key)  # type: ignore[call-arg]

        def call_model(prompt: str) -> str:
            response = client.responses.create(
                model=args.openai_model,
                input=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_output_tokens=args.max_output_tokens,
                timeout=180,
            )
            response_text = getattr(response, "output_text", None)
            if response_text:
                return response_text.strip()
            chunks: List[str] = []
            for item in getattr(response, "output", []):
                content = getattr(item, "content", None)
                if isinstance(content, str):
                    chunks.append(content)
                    continue
                if isinstance(content, list):
                    for part in content:
                        text_part = getattr(part, "text", None)
                        if isinstance(text_part, str):
                            chunks.append(text_part)
            combined = "".join(chunks).strip()
            if not combined:
                raise ValueError("Received empty response text from ChatGPT.")
            return combined

        recoverable_types = OPENAI_RETRY_ERRORS


    scenarios = load_scenarios(args.scenario_file)
    cache_path = args.cache_file or args.output.with_suffix(DEFAULT_CACHE_SUFFIX)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_cache(cache_path)
    records: List[Dict[str, str]] = list(existing)
    seen_inputs = {sample["input"] for sample in records}

    if records:
        logging.info("Loaded %d cached samples from %s", len(records), cache_path)

    if len(records) >= args.target_samples:
        logging.info(
            "Cache already holds %d samples (target %d). Exporting and exiting.",
            len(records),
            args.target_samples,
        )
        write_final_dataset(args.output, records[: args.target_samples])
        logging.info("Wrote dataset to %s", args.output)
        return

    random_generator = random.Random(args.seed)
    batch_iter = scenario_batches(scenarios, args.samples_per_call, random_generator)

    last_request_ts = 0.0
    total_target = args.target_samples
    success_batches = 0

    while len(records) < total_target:
        remaining = total_target - len(records)
        batch_size = min(args.samples_per_call, remaining)
        batch = next(batch_iter)[:batch_size]
        prompt = build_prompt(batch)

        attempt = 0
        response_text = None
        while attempt < args.max_retries:
            attempt += 1
            wait_for = args.min_interval - (time.time() - last_request_ts)
            if wait_for > 0:
                time.sleep(wait_for)
            try:
                response_text = call_model(prompt)
                last_request_ts = time.time()
                break
            except Exception as err:  # noqa: BLE001 - we inspect recoverable types below
                should_retry = recoverable_types and isinstance(err, recoverable_types)
                backoff = args.retry_backoff * attempt
                logging.warning(
                    "%s call failed (%s). Sleeping %.1f s before retry %d/%d.",
                    args.provider.capitalize(),
                    err,
                    backoff,
                    attempt,
                    args.max_retries,
                )
                if attempt >= args.max_retries or not should_retry:
                    if not should_retry:
                        logging.error("Non-recoverable error from provider: %s", err)
                        raise
                time.sleep(backoff)

        if not response_text:
            raise RuntimeError("Failed to obtain a valid response after retries.")

        try:
            #print(response_text)
            samples = extract_samples(response_text)
        except ValueError as err:
            logging.warning("Could not parse model response: %s", err)
            continue

        if len(samples) > batch_size:
            samples = samples[:batch_size]
        elif len(samples) < batch_size:
            logging.warning(
                "Expected %d samples but received %d. Continuing with parsed items.",
                batch_size,
                len(samples),
            )

        fresh: List[Dict[str, str]] = []
        for sample in samples:
            if sample["input"] in seen_inputs:
                continue
            fresh.append(sample)
            seen_inputs.add(sample["input"])

        if not fresh:
            logging.warning("No new samples extracted from batch; retrying next batch.")
            continue

        records.extend(fresh)
        append_cache(cache_path, fresh)
        success_batches += 1

        if success_batches % args.log_every == 0 or len(records) >= total_target:
            logging.info(
                "Progress: %d/%d samples (%.1f%%)",
                len(records),
                total_target,
                (len(records) / total_target) * 100.0,
            )

    write_final_dataset(args.output, records[: total_target])
    logging.info("Generation complete. Saved %d samples to %s", total_target, args.output)
    logging.info("Cache retained at %s for resume or auditing.", cache_path)


if __name__ == "__main__":
    main()
