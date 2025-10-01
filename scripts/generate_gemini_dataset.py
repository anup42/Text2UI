#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

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
DEFAULT_PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "prompts"
    / "gemini_dataset_prompt.txt"
)


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

CSS_OPTIONAL_MODIFIERS = [
    "secondary",
    "subtle",
    "danger",
    "warning",
    "success",
    "info",
    "good",
    "warn",
    "active",
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

KEY_FAILURE_ERRORS: Tuple[type[BaseException], ...] = tuple()
if google_exceptions:  # pragma: no cover - depends on optional dependency
    failure_candidates: List[type[BaseException]] = []
    for name in ("PermissionDenied", "Unauthenticated"):
        candidate = getattr(google_exceptions, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            failure_candidates.append(candidate)
    KEY_FAILURE_ERRORS = tuple(failure_candidates)

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
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE_PATH,
        help="Path to the prompt template file used for generation.",
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
        "--threads",
        type=int,
        default=5,
        help=(
            "Number of parallel Gemini workers to run. Each worker uses a separate API key."
        ),
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


class GeminiKeyManager:
    """Assign unique Gemini API keys to workers and rotate failing keys."""

    def __init__(self, keys: Sequence[str], thread_count: int):
        all_keys = list(keys)
        if thread_count > len(all_keys):
            raise ValueError("thread_count cannot exceed the number of available keys")
        self._lock = Lock()
        self._labels: Dict[str, int] = {
            key: index + 1 for index, key in enumerate(all_keys)
        }
        self._initial = list(all_keys[:thread_count])
        self._available: List[str] = list(all_keys[thread_count:])
        self._assignments: Dict[int, str] = {}

    def get_key(self, worker_id: int) -> str:
        with self._lock:
            if worker_id in self._assignments:
                return self._assignments[worker_id]
            if worker_id < len(self._initial):
                key = self._initial[worker_id]
            elif self._available:
                key = self._available.pop(0)
            else:  # pragma: no cover - defensive branch
                raise RuntimeError("No Gemini API key available for worker %d" % worker_id)
            self._assignments[worker_id] = key
            return key

    def replace_key(self, worker_id: int) -> str | None:
        with self._lock:
            self._assignments.pop(worker_id, None)
            if self._available:
                key = self._available.pop(0)
                self._assignments[worker_id] = key
                return key
            return None

    def label_for(self, api_key: str) -> int | None:
        return self._labels.get(api_key)


def is_key_auth_failure(error: BaseException) -> bool:
    if KEY_FAILURE_ERRORS and isinstance(error, KEY_FAILURE_ERRORS):
        return True
    message = str(error).lower()
    for needle in ("api key", "permission_denied", "unauthenticated", "invalid key"):
        if needle in message:
            return True
    return False

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


_PROMPT_TEMPLATE_PATH = DEFAULT_PROMPT_TEMPLATE_PATH


@lru_cache(maxsize=1)
def get_prompt_template() -> str:
    try:
        return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - depends on external file
        raise FileNotFoundError(
            f"Prompt template file not found at {_PROMPT_TEMPLATE_PATH}"
        ) from exc


def set_prompt_template_path(path: Path) -> None:
    global _PROMPT_TEMPLATE_PATH
    _PROMPT_TEMPLATE_PATH = path
    get_prompt_template.cache_clear()


class _PromptTemplateDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - simple fallback
        if "-" in key:
            alias = key.replace("-", "_")
            if alias in self:
                return self[alias]
        return ""


_PROMPT_PLACEHOLDER_PATTERN = re.compile(r"{([a-zA-Z0-9_-]+)}")


def _render_prompt_template(template: str, replacements: Mapping[str, object]) -> str:
    """Render the prompt template safely.

    The previous implementation relied on :meth:`str.format_map`, which chokes when
    scenario strings contain curly braces. That meant legitimate scenario text such
    as ``{"amount":1580,"currency":"INR"}`` triggered ``ValueError`` because
    ``str.format`` interprets the text after ``:`` as a format specifier. By using a
    simple placeholder substitution we only replace known template fields (like
    ``{count}``) and leave everything else untouched, preserving literal braces in
    scenarios.
    """

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            value = replacements[key]
        except KeyError:
            return ""
        if not isinstance(value, str):
            return str(value)
        return value

    return _PROMPT_PLACEHOLDER_PATTERN.sub(replace, template)


def build_prompt(batch: Sequence[str]) -> str:
    scenario_lines = "\n".join(f"{idx + 1}. {name}" for idx, name in enumerate(batch))
    classes = ", ".join(CSS_CLASSES)
    optional_modifiers = ", ".join(CSS_OPTIONAL_MODIFIERS)
    template = get_prompt_template()
    replacements: Mapping[str, object] = _PromptTemplateDict(
        count=len(batch),
        classes=classes,
        scenario_lines=scenario_lines,
        optional_modifiers=optional_modifiers,
    )
    return _render_prompt_template(template, replacements)


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
    set_prompt_template_path(args.prompt_template)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = resolve_api_key(args)


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

    total_target = args.target_samples
    success_batches = 0
    start_time = time.time()

    def process_response(response_text: str, batch_size: int) -> None:
        nonlocal success_batches
        if len(records) >= total_target:
            return
        try:
            samples = extract_samples(response_text)
        except ValueError as err:
            logging.warning("Could not parse model response: %s", err)
            return

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
            return

        records.extend(fresh)
        append_cache(cache_path, fresh)
        success_batches += 1

        if success_batches % args.log_every == 0 or len(records) >= total_target:
            elapsed = time.time() - start_time
            samples_per_second = len(records) / elapsed if elapsed > 0 else 0.0
            logging.info(
                "Progress: %d/%d samples (%.1f%%, %.2f samples/s)",
                len(records),
                total_target,
                (len(records) / total_target) * 100.0,
                samples_per_second,
            )

    if args.provider == "gemini":
        gemini_keys, gemini_models = gather_gemini_credentials(args)
        if not gemini_keys:
            raise RuntimeError("No Gemini API keys available. Provide at least one key in the secrets file or environment.")

        thread_count = max(1, min(args.threads, len(gemini_keys)))
        if thread_count < args.threads:
            logging.warning(
                "Requested %d threads but only %d Gemini API keys available; reducing thread count.",
                args.threads,
                thread_count,
            )

        recoverable_types = RATE_LIMIT_ERRORS

        stop_token = object()
        task_queue: Queue[tuple[int, int, str]] = Queue()
        result_queue: Queue[tuple[int, int, str | None, BaseException | None]] = Queue()
        configure_lock = Lock()
        key_manager = GeminiKeyManager(gemini_keys, thread_count)

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

        def worker_loop(worker_id: int, model_name: str) -> None:
            api_key = key_manager.get_key(worker_id)
            key_label = key_manager.label_for(api_key)

            def format_label(label: int | None) -> str:
                return f" (API key #{label})" if label is not None else ""

            with configure_lock:
                genai.configure(api_key=api_key)  # type: ignore[arg-type]
                model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
            last_request_ts = 0.0
            while True:
                item = task_queue.get()
                if item is stop_token:
                    task_queue.task_done()
                    break
                task_id, batch_size, prompt = item
                attempt = 0
                response_text: str | None = None
                last_error: BaseException | None = None
                while attempt < args.max_retries:
                    attempt += 1
                    wait_for = args.min_interval - (time.time() - last_request_ts)
                    if wait_for > 0:
                        time.sleep(wait_for)
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
                        response_text = extract_response_text(response)
                        last_request_ts = time.time()
                        last_error = None
                        break
                    except Exception as err:  # noqa: BLE001 - inspect for retryable types
                        last_error = err
                        label_hint = format_label(key_label)
                        if is_key_auth_failure(err):
                            replacement_key = key_manager.replace_key(worker_id)
                            if not replacement_key:
                                logging.error(
                                    "Worker %d Gemini API key%s failed and no replacements remain: %s",
                                    worker_id + 1,
                                    label_hint,
                                    err,
                                )
                                break
                            logging.warning(
                                "Worker %d Gemini API key%s failed (%s); switching to a new key.",
                                worker_id + 1,
                                label_hint,
                                err,
                            )
                            api_key = replacement_key
                            key_label = key_manager.label_for(api_key)
                            label_hint = format_label(key_label)
                            with configure_lock:
                                genai.configure(api_key=api_key)  # type: ignore[arg-type]
                                model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
                            last_request_ts = 0.0
                            attempt = 0
                            continue
                        should_retry = recoverable_types and isinstance(err, recoverable_types)
                        backoff = args.retry_backoff * attempt
                        normalized_error = str(err).lower()
                        is_quota_issue = any(
                            needle in normalized_error
                            for needle in ("quota", "rate limit", "resource exhausted", "429")
                        )
                        quota_hint = label_hint if is_quota_issue else ""
                        logging.warning(
                            "Worker %d Gemini call failed%s (%s). Sleeping %.1f s before retry %d/%d.",
                            worker_id + 1,
                            quota_hint,
                            err,
                            max(args.min_interval, backoff),
                            attempt,
                            args.max_retries,
                        )
                        if attempt >= args.max_retries or not should_retry:
                            break
                        time.sleep(max(args.min_interval, backoff))
                result_queue.put((task_id, batch_size, response_text, last_error))
                task_queue.task_done()

        threads: List[Thread] = []
        for idx in range(thread_count):
            model_name = gemini_models[idx % len(gemini_models)] if gemini_models else args.model
            worker = Thread(
                target=worker_loop,
                args=(idx, model_name),
                daemon=True,
            )
            worker.start()
            threads.append(worker)

        logging.info("Spawning %d Gemini worker threads.", thread_count)

        inflight: Dict[int, int] = {}
        next_task_id = 0

        try:
            while len(records) < total_target or inflight:
                if len(records) < total_target:
                    while (
                        len(inflight) < thread_count
                        and len(records) + len(inflight) * args.samples_per_call < total_target
                    ):
                        remaining = total_target - len(records)
                        batch_size = min(args.samples_per_call, remaining)
                        batch = next(batch_iter)[:batch_size]
                        prompt = build_prompt(batch)
                        task_id = next_task_id
                        next_task_id += 1
                        inflight[task_id] = batch_size
                        task_queue.put((task_id, batch_size, prompt))
                try:
                    task_id, batch_size, response_text, error = result_queue.get(timeout=0.5)
                except Empty:
                    continue
                inflight.pop(task_id, None)
                if not response_text:
                    if error and not (
                        recoverable_types and isinstance(error, recoverable_types)
                    ):
                        logging.error("Non-recoverable error from Gemini worker: %s", error)
                        raise error
                    raise RuntimeError("Failed to obtain a valid response after retries.") from error
                process_response(response_text, batch_size)
        finally:
            task_queue.join()
            for _ in threads:
                task_queue.put(stop_token)
            for worker in threads:
                worker.join()
    else:
        client = OpenAI(api_key=api_key)  # type: ignore[call-arg]
        recoverable_types = OPENAI_RETRY_ERRORS
        last_request_ts = 0.0

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
                        response_text = response_text.strip()
                    else:
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
                        if combined:
                            response_text = combined
                        else:
                            raise ValueError("Received empty response text from ChatGPT.")
                    last_request_ts = time.time()
                    break
                except Exception as err:  # noqa: BLE001 - inspect recoverable types
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

            process_response(response_text, batch_size)

    write_final_dataset(args.output, records[: total_target])
    total_elapsed = time.time() - start_time
    avg_speed = total_target / total_elapsed if total_elapsed > 0 else 0.0
    logging.info(
        "Generation complete. Saved %d samples to %s in %.2f s (%.2f samples/s)",
        total_target,
        args.output,
        total_elapsed,
        avg_speed,
    )
    logging.info("Cache retained at %s for resume or auditing.", cache_path)


if __name__ == "__main__":
    main()
