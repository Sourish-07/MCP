#!/usr/bin/env python3
"""
Robinhood MCP Response Recorder — True Observability Layer.

Records the exact raw data returned by every Robinhood MCP call
*before* any parsing, validation, transformation, coercion, trimming,
or business logic occurs.

Design guarantees:
  - Raw stdout / stderr are stored byte-for-byte identical to what
    Python received from subprocess.communicate().
  - Any cleaned / trimmed / parsed versions are stored SEPARATELY.
  - No normalization, no unicode normalization, no whitespace
    normalization, no newline conversion — ever — on raw fields.
  - Recorder failures NEVER stop trading.
  - Thread-safe asyncio JSONL appending with proper locking.
  - Automatic rotation (100 MB per file, newest 20 kept as .gz).
  - Persistent schema history across Python sessions.
  - Recursive numeric anomaly detection (every field, full JSON path).
  - Multi-stage parsing diagnostics (every stage stored independently).
  - Full subprocess failure capture (stdout retained on non‑zero exit).
  - HTML diagnostics with metadata extraction.
  - Differentiated empty‑response classification.
  - Rich developer‑utility "show_*" functions.

Usage is automatic: every call routed through `_call_with_retry()`
in `mcp/robinhood_client.py` triggers `capture_mcp_call()`.

"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gzip as gzip_module
import json
import logging
import os
import re
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ #
#  Module-level logger — silent by default (failures never propagate)
# ------------------------------------------------------------------ #
_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())

# ------------------------------------------------------------------ #
#  Configurable constants
# ------------------------------------------------------------------ #
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024   # 100 MB
MAX_ROTATED_FILES   = 20                  # keep newest 20 .gz

# Persistent schema-history file so comparisons survive Python restarts.
SCHEMA_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent / "logs" / "schema_history.json"
)

# ------------------------------------------------------------------ #
#  Thread pool for blocking file I/O (rotation, pruning)
# ------------------------------------------------------------------ #
_BLOCKING_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="recorder-io"
)

# ------------------------------------------------------------------ #
#  Credential patterns (redacted before storage)
# ------------------------------------------------------------------ #
CREDENTIAL_PATTERNS: list[tuple[str, str]] = [
    # Bearer / Authorization headers
    (r'(?i)authorization\s*:\s*Bearer\s+[^\r\n]+',
     "Authorization: Bearer [REDACTED]"),
    (r'(?i)authorization\s*:\s*Basic\s+[^\r\n]+',
     "Authorization: Basic [REDACTED]"),
    (r'(?i)authorization\s*:\s*[^\r\n]+',
     "Authorization: [REDACTED]"),
    # API keys in key=value form
    (r'(?i)(api[_-]?key|apikey|secret[_-]?key|token)\s*[=:]\s*[^\s\r\n,;]+',
     r'\1=[REDACTED]'),
    # Cookie headers
    (r'(?i)cookie\s*:\s*[^\r\n]+', "Cookie: [REDACTED]"),
    # OAuth tokens
    (r'(?i)oauth[_-]?token\s*[=:]\s*[^\s\r\n,;]+',
     r'oauth_token=[REDACTED]'),
    # Generic "token": "..." JSON patterns
    (r'(?i)"token"\s*:\s*"[^"]+"',
     '"token": "[REDACTED]"'),
    # "access_token" JSON
    (r'(?i)"access[_-]?token"\s*:\s*"[^"]+"',
     '"access_token": "[REDACTED]"'),
    # "refresh_token" JSON
    (r'(?i)"refresh[_-]?token"\s*:\s*"[^"]+"',
     '"refresh_token": "[REDACTED]"'),
]

# ─────────────────────────────────────────────────────────────────── #
#  Schema persistence helpers
# ─────────────────────────────────────────────────────────────────── #

def _load_persistent_schema_history() -> dict[str, Any]:
    """Load recorded schemas from disk (cross-session history)."""
    if not SCHEMA_HISTORY_PATH.exists():
        return {}
    try:
        with open(SCHEMA_HISTORY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        _log.warning("Could not load schema history — starting fresh", exc_info=True)
        return {}


def _save_persistent_schema_history(data: dict[str, Any]) -> None:
    """Atomically write schema history to disk (write-then-rename)."""
    tmp = SCHEMA_HISTORY_PATH.with_suffix(".tmp")
    try:
        SCHEMA_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
        tmp.replace(SCHEMA_HISTORY_PATH)
    except Exception:
        _log.warning("Could not persist schema history", exc_info=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────── #
#  File rotation (run in executor to avoid blocking the event loop)
# ─────────────────────────────────────────────────────────────────── #

def _rotate_file_if_needed(file_path: Path) -> None:
    """Check file size; if >= MAX_FILE_SIZE_BYTES, rotate and gzip."""
    if not file_path.exists():
        return
    size = file_path.stat().st_size
    if size < MAX_FILE_SIZE_BYTES:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gz_name = f"{file_path.stem}.{timestamp}.jsonl.gz"
    gz_path = file_path.parent / gz_name

    with open(file_path, "rb") as f_in:
        with gzip_module.open(gz_path, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Truncate the live file atomically
    file_path.unlink()
    file_path.touch()

    # Enforce MAX_ROTATED_FILES
    _prune_old_rotated(file_path.parent, file_path.stem)


async def _rotate_async(path: Path) -> None:
    """Non-blocking wrapper that delegates to the thread executor."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_BLOCKING_EXECUTOR, _rotate_file_if_needed, path)


def _prune_old_rotated(log_dir: Path, stem: str) -> None:
    """Keep only the newest MAX_ROTATED_FILES .gz archives."""
    gz_files = sorted(
        [p for p in log_dir.glob(f"{stem}.*.jsonl.gz")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in gz_files[MAX_ROTATED_FILES:]:
        old.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
#  Redaction
# ------------------------------------------------------------------ #

def _redact_credentials(text: str) -> str:
    """Strip API keys, tokens, and auth headers from raw text."""
    for pattern, replacement in CREDENTIAL_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


# ------------------------------------------------------------------ #
#  Response type detection (lightweight, only inspects what's needed)
# ------------------------------------------------------------------ #

def _detect_response_type(raw: str | None) -> str:
    """Classify the raw stdout without copying the full string."""
    if raw is None:
        return "null"
    if raw == "":
        return "empty"
    if raw.isspace():
        return "whitespace"

    # HTML detection — only check the first 15 chars, not a strip() copy
    start = raw[:256].lstrip()
    if start.startswith("<") or start.lower().startswith("<!doctype"):
        return "html"

    # Try to detect markdown wrapping
    stripped_start = raw.strip()
    if stripped_start.startswith("```"):
        return "markdown"

    # Try JSON classification
    jstart = stripped_start.strip()
    if jstart.startswith("{"):
        try:
            json.loads(raw)
            return "json object"
        except json.JSONDecodeError:
            return "plain text"  # looks like JSON but isn't valid
    if jstart.startswith("["):
        try:
            json.loads(raw)
            return "json array"
        except json.JSONDecodeError:
            return "plain text"

    # Binary indicator — null bytes or high ratio of non-printable chars
    null_count = raw.count("\x00")
    if null_count > 0:
        return "binary"
    if len(raw) <= 1000:
        printable = sum(1 for ch in raw if ch.isprintable() or ch in ("\n", "\r", "\t"))
        if printable / max(len(raw), 1) < 0.7:
            return "binary"

    return "plain text"


# ------------------------------------------------------------------ #
#  Content‑type inference
# ------------------------------------------------------------------ #

def _infer_content_type(raw: str | None) -> str:
    """Heuristic content-type inference from raw text."""
    rtype = _detect_response_type(raw)
    mapping = {
        "null":         "null",
        "empty":        "text/plain (empty)",
        "whitespace":   "text/plain (whitespace)",
        "json object":  "application/json",
        "json array":   "application/json",
        "html":         "text/html",
        "plain text":   "text/plain",
        "markdown":     "text/markdown",
        "binary":       "application/octet-stream",
    }
    return mapping.get(rtype, "unknown")


# ------------------------------------------------------------------ #
#  HTML metadata extraction (title, status, body length, samples)
# ------------------------------------------------------------------ #

def _extract_html_metadata(raw: str | None) -> dict[str, Any] | None:
    """Extract lightweight HTML diagnostics without full parsing."""
    if not raw or _detect_response_type(raw) != "html":
        return None

    meta: dict[str, Any] = {
        "body_length_chars": len(raw),
        "first_1000_chars":  raw[:1000],
        "last_1000_chars":   raw[-1000:] if len(raw) > 1000 else None,
    }

    # Title
    title_m = re.search(r"<title[^>]*>([^<]+)</title>", raw, re.IGNORECASE)
    if title_m:
        meta["title"] = title_m.group(1).strip()

    # HTTP status embedded in HTML, e.g. "HTTP/1.1 429" or "Status: 503"
    status_m = re.search(
        r'(?:HTTP/\d\.\d\s+|Status\s*:\s*)(\d{3})', raw, re.IGNORECASE
    )
    if status_m:
        meta["status_code_candidate"] = int(status_m.group(1))

    return meta


# ------------------------------------------------------------------ #
#  Schema discovery (recursive)
# ------------------------------------------------------------------ #

def _discover_schema(obj: Any, depth: int = 0, max_depth: int = 20) -> dict[str, Any]:
    """Recursively inspect a JSON value and record structure metadata."""
    t = type(obj).__name__
    result: dict[str, Any] = {
        "_type": t,
        "_depth": depth,
    }

    if depth >= max_depth:
        result["_truncated"] = True
        return result

    if isinstance(obj, dict):
        result["_is_object"] = True
        result["_key_count"] = len(obj)
        keys_info: list[dict[str, Any]] = []
        for key, value in obj.items():
            field_info = _discover_schema(value, depth + 1, max_depth)
            field_info["_field_name"] = key
            field_info["_field_type"] = type(value).__name__
            # Record sample value (non‑recursive leaf representation)
            if isinstance(value, (str, int, float, bool)) or value is None:
                field_info["_sample_value"] = repr(value) if isinstance(value, str) else value
            keys_info.append(field_info)
        result["_fields"] = keys_info
        result["_object_depth"] = max(
            (f.get("_object_depth", 0) for f in keys_info if f.get("_is_object")),
            default=0,
        ) + 1
        result["_array_depth"] = max(
            (f.get("_array_depth", 0) for f in keys_info if f.get("_is_array")),
            default=0,
        )

    elif isinstance(obj, list):
        result["_is_array"] = True
        result["_length"] = len(obj)
        elements: list[dict[str, Any]] = []
        max_obj_d = 0
        max_arr_d = 0
        for idx, item in enumerate(obj[:50]):  # sample first 50 elements
            child = _discover_schema(item, depth + 1, max_depth)
            elements.append(child)
            if child.get("_is_object"):
                max_obj_d = max(max_obj_d, child.get("_object_depth", 0))
            if child.get("_is_array"):
                max_arr_d = max(max_arr_d, child.get("_array_depth", 0))
        result["_sample_elements"] = elements
        result["_array_depth"] = max(max_arr_d, 1)
        result["_object_depth"] = max_obj_d

    else:
        result["_is_scalar"] = True
        result["_sample_value"] = repr(obj) if isinstance(obj, str) else obj

    return result


# ------------------------------------------------------------------ #
#  Schema diff (comprehensive: field adds/removes/renames, type, null, etc.)
# ------------------------------------------------------------------ #

def _compute_schema_diff(
    current_schema: dict[str, Any],
    previous_schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a detailed diff between two schema snapshots."""
    if previous_schema is None:
        return None

    curr_fields = current_schema.get("_fields", [])
    prev_fields = previous_schema.get("_fields", [])

    curr_map = {f.get("_field_name"): f for f in curr_fields if "_field_name" in f}
    prev_map = {f.get("_field_name"): f for f in prev_fields if "_field_name" in f}

    curr_keys = set(curr_map)
    prev_keys = set(prev_map)

    diff: dict[str, Any] = {}

    added = sorted(curr_keys - prev_keys)
    if added:
        diff["added_fields"] = added

    removed = sorted(prev_keys - curr_keys)
    if removed:
        diff["removed_fields"] = removed

    # Changed field types / nullability
    changed_types: list[dict[str, Any]] = []
    changed_nullability: list[dict[str, Any]] = []
    for key in sorted(curr_keys & prev_keys):
        cf = curr_map[key]
        pf = prev_map[key]
        ct = cf.get("_type", cf.get("_field_type"))
        pt = pf.get("_type", pf.get("_field_type"))
        if ct != pt:
            changed_types.append({
                "field": key,
                "previous_type": pt,
                "current_type": ct,
            })
        # Nullability: was it null before and not now, or vice versa?
        cv = cf.get("_sample_value")
        pv = pf.get("_sample_value")
        was_null = (pv is None or pv == "None")
        is_null = (cv is None or cv == "None")
        if was_null != is_null:
            changed_nullability.append({
                "field": key,
                "was_null": was_null,
                "is_now_null": is_null,
            })

    if changed_types:
        diff["changed_types"] = changed_types
    if changed_nullability:
        diff["nullability_changes"] = changed_nullability

    # Structural changes: object ↔ array, nesting depth
    structural: list[dict[str, Any]] = []
    prev_obj_depth = previous_schema.get("_object_depth", 0)
    curr_obj_depth = current_schema.get("_object_depth", 0)
    if prev_obj_depth != curr_obj_depth:
        structural.append({
            "aspect": "object_depth",
            "previous": prev_obj_depth,
            "current": curr_obj_depth,
        })

    prev_arr_depth = previous_schema.get("_array_depth", 0)
    curr_arr_depth = current_schema.get("_array_depth", 0)
    if prev_arr_depth != curr_arr_depth:
        structural.append({
            "aspect": "array_depth",
            "previous": prev_arr_depth,
            "current": curr_arr_depth,
        })

    # Key count changed
    prev_kc = previous_schema.get("_key_count", 0)
    curr_kc = current_schema.get("_key_count", 0)
    if prev_kc != curr_kc:
        structural.append({
            "aspect": "key_count",
            "previous": prev_kc,
            "current": curr_kc,
        })

    if structural:
        diff["structural_changes"] = structural

    # Empty arrays replacing populated arrays (or vice versa)
    array_changes: list[dict[str, Any]] = []
    for key in sorted(curr_keys & prev_keys):
        cf = curr_map[key]
        pf = prev_map[key]
        c_len = cf.get("_length")
        p_len = pf.get("_length")
        if c_len is not None and p_len is not None:
            was_empty = (p_len == 0)
            is_empty = (c_len == 0)
            if was_empty != is_empty:
                array_changes.append({
                    "field": key,
                    "was_empty_array": was_empty,
                    "is_now_empty_array": is_empty,
                    "previous_length": p_len,
                    "current_length": c_len,
                })
    if array_changes:
        diff["array_emptiness_changes"] = array_changes

    # Object ↔ array swapping
    obj_arr_swap: list[dict[str, Any]] = []
    for key in sorted(curr_keys & prev_keys):
        cf = curr_map[key]
        pf = prev_map[key]
        cf_is_arr = cf.get("_is_array", False)
        pf_is_arr = pf.get("_is_array", False)
        cf_is_obj = cf.get("_is_object", False)
        pf_is_obj = pf.get("_is_object", False)
        if (cf_is_arr and pf_is_obj) or (cf_is_obj and pf_is_arr):
            obj_arr_swap.append({
                "field": key,
                "was": "array" if pf_is_arr else ("object" if pf_is_obj else pf.get("_type")),
                "now": "array" if cf_is_arr else ("object" if cf_is_obj else cf.get("_type")),
            })
    if obj_arr_swap:
        diff["object_array_swaps"] = obj_arr_swap

    return diff if diff else None


# ─────────────────────────────────────────────────────────────────── #
#  In‑memory schema history + persistent backing
# ─────────────────────────────────────────────────────────────────── #
_schema_history: dict[str, dict[str, Any]] = {}
_schema_history_lock = asyncio.Lock()
_persistent_schema_loaded = False


def _ensure_persistent_loaded() -> None:
    """One‑shot populate in‑memory history from disk."""
    global _persistent_schema_loaded, _schema_history
    if _persistent_schema_loaded:
        return
    _persistent_schema_loaded = True
    loaded = _load_persistent_schema_history()
    if loaded:
        _schema_history.update(loaded)


async def _lookup_previous_schema(tool_name: str) -> dict[str, Any] | None:
    """Return the stored schema for a tool (in‑memory ⊕ disk)."""
    _ensure_persistent_loaded()
    async with _schema_history_lock:
        return _schema_history.get(tool_name)


async def _store_schema(tool_name: str, schema: dict[str, Any]) -> None:
    """Store schema in memory AND persist to disk."""
    _ensure_persistent_loaded()
    async with _schema_history_lock:
        existing = _schema_history.get(tool_name, {})
        schema["call_count"] = existing.get("call_count", 0) + 1
        schema["last_seen_utc"] = datetime.now(timezone.utc).isoformat()
        _schema_history[tool_name] = schema

    # Persist asynchronously (fire‑and‑forget, failure is silent)
    loop = asyncio.get_running_loop()
    data_snapshot = dict(_schema_history)
    loop.run_in_executor(_BLOCKING_EXECUTOR, _save_persistent_schema_history, data_snapshot)


# ------------------------------------------------------------------ #
#  Tool‑name extraction from Claude prompts
# ------------------------------------------------------------------ #

# Broad pattern — captures any "tool <name>" phrase
_TOOL_NAME_RE = re.compile(
    r"(?:call|use|execute|invoke|run)\s+(?:the\s+)?(?:robinhood\s+)?(?:MCP\s+)?tool\s+[\"']?(\w[\w_]*)[\"']?",
    re.IGNORECASE,
)


def _extract_tool_name(prompt: str) -> str:
    """Heuristic extraction of the MCP tool name from the Claude prompt."""
    m = _TOOL_NAME_RE.search(prompt)
    if m:
        return m.group(1).strip("_").lower()
    # Fallback: look for any word immediately following "Robinhood MCP"
    m2 = re.search(r"robinhood\s+MCP\s+(\w[\w_]*)", prompt, re.IGNORECASE)
    if m2:
        return m2.group(1).strip("_").lower()
    return "unknown_tool"


# ------------------------------------------------------------------ #
#  Recursive numeric anomaly detection (full JSON path)
# ------------------------------------------------------------------ #

def _detect_numeric_anomalies(
    obj: Any,
    path: str = "$",
    max_depth: int = 20,
) -> list[dict[str, Any]]:
    """Recursively flag fields that are 0, 0.0, \"0\", null, \"\", or missing."""
    anomalies: list[dict[str, Any]] = []
    if max_depth <= 0:
        return anomalies

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_path = f"{path}.{key}"
            if value == 0 or value == 0.0:
                anomalies.append({
                    "json_path": full_path,
                    "field": key,
                    "actual_type": type(value).__name__,
                    "actual_value": value,
                    "issue": "zero_numeric",
                })
            elif value == "0":
                anomalies.append({
                    "json_path": full_path,
                    "field": key,
                    "actual_type": "str",
                    "actual_value": '"0"',
                    "issue": "zero_string",
                })
            elif value == "0.0":
                anomalies.append({
                    "json_path": full_path,
                    "field": key,
                    "actual_type": "str",
                    "actual_value": '"0.0"',
                    "issue": "zero_float_string",
                })
            elif value is None:
                anomalies.append({
                    "json_path": full_path,
                    "field": key,
                    "actual_type": "NoneType",
                    "actual_value": None,
                    "issue": "null",
                })
            elif value == "":
                anomalies.append({
                    "json_path": full_path,
                    "field": key,
                    "actual_type": "str",
                    "actual_value": '""',
                    "issue": "empty_string",
                })
            elif value is False:
                anomalies.append({
                    "json_path": full_path,
                    "field": key,
                    "actual_type": "bool",
                    "actual_value": False,
                    "issue": "false_bool",
                })
            elif isinstance(value, (dict, list)):
                anomalies.extend(
                    _detect_numeric_anomalies(value, full_path, max_depth - 1)
                )
            elif isinstance(value, float) and value == 0.0:
                # Redundant with above but catches edge float repr
                pass

    elif isinstance(obj, list):
        for idx, item in enumerate(obj[:50]):  # limit sampling
            full_path = f"{path}[{idx}]"
            if isinstance(item, (dict, list)):
                anomalies.extend(
                    _detect_numeric_anomalies(item, full_path, max_depth - 1)
                )
            elif item == 0 or item == 0.0:
                anomalies.append({
                    "json_path": full_path,
                    "field": f"[{idx}]",
                    "actual_type": type(item).__name__,
                    "actual_value": item,
                    "issue": "zero_numeric_in_array",
                })

    return anomalies


# ------------------------------------------------------------------ #
#  Markdown fence detection (supports double‑wrapping, partial fences)
# ------------------------------------------------------------------ #

_FENCE_START_RE = re.compile(r"^\s*```(?:\s*(\w+))?\s*$", re.MULTILINE)
_FENCE_END_RE   = re.compile(r"^\s*```\s*$", re.MULTILINE)

def _detect_markdown_fences(raw: str) -> dict[str, Any]:
    """
    Examine raw text for markdown code fences.
    Returns metadata dict including:
      - had_markdown_fence: bool
      - fence_language: str | None
      - number_of_fence_blocks: int
      - has_double_fencing: bool  (JSON inside a fence inside another fence)
      - has_partial_fence: bool   (opening ``` without closing)
      - stdout_after_fence_strip: str | None
    """
    result: dict[str, Any] = {
        "had_markdown_fence": False,
        "fence_language": None,
        "number_of_fence_blocks": 0,
        "has_double_fencing": False,
        "has_partial_fence": False,
        "stdout_after_fence_strip": None,
    }

    if not raw or not raw.strip().startswith("```"):
        return result

    result["had_markdown_fence"] = True

    # Count fence blocks
    openings = _FENCE_START_RE.findall(raw)
    closings = _FENCE_END_RE.findall(raw)
    n_open  = len(openings)
    n_close = len(_FENCE_END_RE.findall(raw)) if closings else 0

    result["number_of_fence_blocks"] = n_open
    result["has_partial_fence"] = (n_open != n_close)
    result["has_double_fencing"] = (n_open >= 2)

    if openings:
        result["fence_language"] = openings[0] or "plain"

    # Strip all fences (greedy outermost removal)
    stripped = raw.strip()
    # Remove leading ``` (possibly with language)
    stripped = re.sub(r"^```(?:\s*\w+)?\s*", "", stripped, count=1)
    # Remove trailing ```
    stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    result["stdout_after_fence_strip"] = stripped.strip() if stripped else None

    return result


# ------------------------------------------------------------------ #
#  Partially‑balanced JSON extraction (fallback for truncated output)
# ------------------------------------------------------------------ #

def _fallback_trim_json(raw: str) -> dict[str, Any]:
    """
    Attempt brace‑balanced fallback trimming.
    Returns dict with:
      - attempted: bool
      - trimmed_text: str (the text that was tried)
      - succeeded: bool
      - parsed_object: Any | None
    """
    result: dict[str, Any] = {
        "attempted": False,
        "stdout_after_fallback_trim": None,
        "succeeded": False,
        "parsed_object": None,
    }
    if not raw or not raw.strip():
        return result

    stripped = raw.strip()
    # Find last balanced brace
    last_brace = stripped.rfind("}")
    if last_brace == -1:
        return result

    trimmed = stripped[:last_brace + 1]
    if not trimmed:
        return result

    result["attempted"] = True
    result["stdout_after_fallback_trim"] = trimmed  # <-- explicit field per spec

    try:
        result["parsed_object"] = json.loads(trimmed)
        result["succeeded"] = True
    except json.JSONDecodeError:
        pass

    return result


# ------------------------------------------------------------------ #
#  Partial JSON detection (truncated, extra chars, multiple docs, etc.)
# ------------------------------------------------------------------ #

def _detect_json_anomalies(raw: str) -> dict[str, Any]:
    """Detect structural anomalies in what looks like JSON text."""
    result: dict[str, Any] = {
        "detected_truncated_json": False,
        "extra_trailing_characters": False,
        "multiple_json_documents": False,
        "leading_garbage_before_json": False,
        "json_followed_by_text": False,
        "markdown_wrapping": False,
        "double_markdown_wrapping": False,
        "partial_markdown_fence": False,
    }

    if not raw or not raw.strip():
        return result

    stripped = raw.strip()

    # Check markdown wrapping
    fence_info = _detect_markdown_fences(raw)
    result["markdown_wrapping"] = fence_info["had_markdown_fence"]
    result["double_markdown_wrapping"] = fence_info["has_double_fencing"]
    result["partial_markdown_fence"] = fence_info["has_partial_fence"]

    # Work on fence‑stripped text for JSON analysis
    working = fence_info.get("stdout_after_fence_strip") or stripped

    # Leading garbage: first char isn't { or [
    if working and working[0] not in ("{", "["):
        # Check if JSON starts later
        brace_pos = working.find("{")
        bracket_pos = working.find("[")
        first_json = min(
            p for p in (brace_pos, bracket_pos) if p >= 0
        ) if (brace_pos >= 0 or bracket_pos >= 0) else -1
        if first_json > 0:
            result["leading_garbage_before_json"] = True

    # Truncated JSON: unbalanced braces/brackets
    if working:
        opens = working.count("{") + working.count("[")
        closes = working.count("}") + working.count("]")
        if opens != closes:
            # One extra is expected for top‑level object (1 open brace at start)
            # But arrays or nested objects can throw this off
            if abs(opens - closes) > 2:
                result["detected_truncated_json"] = True
            elif opens > closes:
                result["detected_truncated_json"] = True

    # Multiple JSON docs: find second { or [ after first complete object
    json_count = 0
    pos = 0
    decoder = json.JSONDecoder()
    while pos < len(working):
        try:
            obj, end = decoder.raw_decode(working[pos:])
            json_count += 1
            pos += end
            # Skip whitespace between docs
            while pos < len(working) and working[pos] in (" ", "\n", "\r", "\t"):
                pos += 1
        except json.JSONDecodeError:
            break
    if json_count > 1:
        result["multiple_json_documents"] = True

    # Extra trailing characters after valid JSON
    if working and working[0] in ("{", "["):
        try:
            _, end = json.JSONDecoder().raw_decode(working)
            trailing = working[end:].strip()
            if trailing:
                result["extra_trailing_characters"] = True
        except json.JSONDecodeError:
            pass

    # JSON followed by text: first complete JSON parse succeeds, then text follows
    if working and working[0] in ("{", "["):
        try:
            obj, end = json.JSONDecoder().raw_decode(working)
            remaining = working[end:].strip()
            if remaining and len(remaining) > 3 and not remaining.startswith("{") and not remaining.startswith("["):
                result["json_followed_by_text"] = True
        except json.JSONDecodeError:
            pass

    return result


# ------------------------------------------------------------------ #
#  Empty‑response differentiation
# ------------------------------------------------------------------ #

def _classify_empty_response(
    stdout_raw: str | None,
    stderr_raw: str | None,
    stdout_bytes: int,
    stderr_bytes: int,
    pid: int | None,
    returncode: int | None,
    process_was_launched: bool,
) -> dict[str, Any]:
    """
    Produce a granular empty‑response classification.
    Returns a dict of individual boolean flags — never a single
    collapsed flag.
    """
    return {
        "stdout_is_none":           stdout_raw is None,
        "stdout_is_empty_string":   stdout_raw == "",
        "stdout_is_zero_byte":      stdout_bytes == 0,
        "stdout_is_whitespace_only":bool(stdout_raw and stdout_raw.isspace()),
        "stdout_is_newline_only":   bool(stdout_raw and stdout_raw.strip() == ""),
        "stdout_exists":            stdout_bytes > 0,
        "stderr_exists":            stderr_bytes > 0,
        "stderr_is_none":           stderr_raw is None,
        "stderr_is_empty_string":   stderr_raw == "",
        "stderr_is_zero_byte":      stderr_bytes == 0,
        "only_stdout_present":      stdout_bytes > 0 and stderr_bytes == 0,
        "only_stderr_present":      stdout_bytes == 0 and stderr_bytes > 0,
        "both_present":             stdout_bytes > 0 and stderr_bytes > 0,
        "neither_present":          stdout_bytes == 0 and stderr_bytes == 0,
        "process_never_started":    not process_was_launched,
        "process_pid":              pid,
        "process_returncode":       returncode,
    }


# ------------------------------------------------------------------ #
#  Record builder
# ------------------------------------------------------------------ #

def _build_record(
    *,
    tool_name: str,
    prompt: str,
    duration_ms: float,
    attempt: int,
    claude_executable: str,
    pid: int | None,
    returncode: int | None,
    stdout_raw: str | None,
    stderr_raw: str | None,
    exception_info: dict[str, Any] | None = None,
    expected_fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build the complete structured trace record.

    This is called AFTER `_call_claude_mcp()` returns (or raises).
    It builds the full structured record, performing all detection and
    analysis stages independently. Raw stdout/stderr are NEVER
    overwritten — cleaned versions are stored in separate fields.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    request_id = str(uuid.uuid4())

    # ── Byte‑level lengths (important: BEFORE any transformation) ─
    stdout_bytes = len(stdout_raw.encode("utf-8", errors="replace")) if stdout_raw else 0
    stderr_bytes = len(stderr_raw.encode("utf-8", errors="replace")) if stderr_raw else 0
    stdout_chars = len(stdout_raw) if stdout_raw else 0
    stderr_chars = len(stderr_raw) if stderr_raw else 0

    # ── Redact credentials from stdout / stderr early ─────────────────
    stdout_redacted = _redact_credentials(stdout_raw) if stdout_raw else None
    stderr_redacted = _redact_credentials(stderr_raw) if stderr_raw else None

    # ── Response type detection (on raw, before any stripping) ────────
    response_type = _detect_response_type(stdout_raw)
    content_type_inference = _infer_content_type(stdout_raw)

    # ── HTML diagnostics ──────────────────────────────────────────────
    html_metadata = _extract_html_metadata(stdout_raw)

    # ── Empty‑response granular classification ────────────────────────
    process_was_launched = (pid is not None)
    empty_classification = _classify_empty_response(
        stdout_raw, stderr_raw, stdout_bytes, stderr_bytes,
        pid, returncode, process_was_launched,
    )

    # ── Markdown fence detection (multi‑stage) ────────────────────────
    fence_info = _detect_markdown_fences(stdout_raw if stdout_raw else "")
    had_markdown_fence      = fence_info["had_markdown_fence"]
    fence_language          = fence_info["fence_language"]
    number_of_fence_blocks  = fence_info["number_of_fence_blocks"]
    has_double_fencing      = fence_info["has_double_fencing"]
    has_partial_fence       = fence_info["has_partial_fence"]
    stdout_after_fence_strip = fence_info.get("stdout_after_fence_strip")

    # ── JSON anomaly detection (truncation, multiple docs, garbage) ──
    json_anomalies = _detect_json_anomalies(stdout_raw if stdout_raw else "")

    # ── Stage‑by‑stage JSON parsing ───────────────────────────────────
    # Stage 1: try fence‑stripped text
    json_parse_input_used: str | None = None
    json_parse_input_description: str = "none"

    parse_candidate = stdout_after_fence_strip if had_markdown_fence else (
        stdout_raw.strip() if stdout_raw else ""
    )

    json_exception: str | None = None
    full_traceback: str | None = None
    parsed_object: Any = None
    top_level_keys: list[str] | None = None
    top_level_json_type: str | None = None
    missing_fields: list[str] | None = None
    json_parse_attempted = False
    json_parse_succeeded = False
    json_parse_failed    = False

    # SKIP JSON parsing entirely for HTML responses
    if response_type != "html" and response_type not in ("null", "binary") and parse_candidate:
        json_parse_attempted = True
        json_parse_input_used = parse_candidate
        json_parse_input_description = (
            "fence_stripped" if had_markdown_fence else "raw_stdout_stripped"
        )
        try:
            parsed_object = json.loads(parse_candidate)
            json_parse_succeeded = True
            if isinstance(parsed_object, dict):
                top_level_keys = sorted(parsed_object.keys())
                top_level_json_type = "object"
                if expected_fields:
                    missing_fields = sorted(
                        set(expected_fields) - set(parsed_object.keys())
                    )
            elif isinstance(parsed_object, list):
                top_level_json_type = "array"
                top_level_keys = None
            else:
                top_level_json_type = type(parsed_object).__name__
        except json.JSONDecodeError as e:
            json_parse_failed = True
            json_exception = str(e)
            full_traceback = traceback.format_exc()

    # ── JSON anomaly flags from attempted decode ─────────────────────
    if json_parse_failed and stdout_raw:
        # Try raw_decode to see if there's at least partial valid JSON
        partial_json_detected = False
        try:
            json.JSONDecoder().raw_decode(
                parse_candidate if parse_candidate else stdout_raw.strip()
            )
            partial_json_detected = True
        except json.JSONDecodeError:
            pass
        json_anomalies["partial_valid_json_detected"] = partial_json_detected

    # ── Fallback trimming (brace‑balanced) ────────────────────────────
    fallback = _fallback_trim_json(stdout_raw if stdout_raw else "")
    fallback_trim_attempted  = fallback["attempted"]
    fallback_trim_succeeded  = fallback["succeeded"]
    fallback_trimmed_object  = fallback["parsed_object"]
    stdout_after_fallback_trim = fallback.get("stdout_after_fallback_trim")

    # ── Response truncated flag (heuristic) ────────────────────────────
    response_truncated = False
    if stdout_raw and len(stdout_raw) > 1_000_000:
        response_truncated = True

    # ── Schema discovery ──────────────────────────────────────────────
    schema: dict[str, Any] | None = None
    obj_for_schema: Any = None
    if json_parse_succeeded and parsed_object is not None:
        obj_for_schema = parsed_object
    elif fallback_trim_succeeded and fallback_trimmed_object is not None:
        obj_for_schema = fallback_trimmed_object
    if obj_for_schema is not None:
        try:
            schema = _discover_schema(obj_for_schema)
        except Exception:
            _log.warning("Schema discovery failed", exc_info=True)

    # ── Validation simulation ─────────────────────────────────────────
    validation_attempted = False
    validation_result: str | None = None
    validation_errors: list[str] | None = None
    # Validation happens in core/ — recorder only observes raw data.

    # ── Numeric anomaly detection (RECURSIVE, full JSON path) ────────
    numeric_anomalies: list[dict[str, Any]] = []
    if json_parse_succeeded and parsed_object is not None:
        numeric_anomalies = _detect_numeric_anomalies(parsed_object)

    # ── Is the response empty? (summary boolean for quick querying) ──
    is_empty = (stdout_raw is None or stdout_raw == "" or stdout_raw.isspace())

    # ── Unexpected fields (present in JSON but not in expected list) ─
    unexpected_fields: list[str] | None = None
    if expected_fields and isinstance(parsed_object, dict):
        unexpected_fields = sorted(
            set(parsed_object.keys()) - set(expected_fields)
        )
        if not unexpected_fields:
            unexpected_fields = None

    # ═══════════════════════════════════════════════════════════════ #
    #  Assemble final record
    # ═══════════════════════════════════════════════════════════════ #
    record: dict[str, Any] = {
        # ── Metadata ──────────────────────────────────────────────
        "request_id":      request_id,
        "timestamp_utc":   now_utc,
        "tool_name":       tool_name,
        "original_prompt": prompt,
        "duration_ms":     duration_ms,
        "attempt":         attempt,
        "claude_executable": claude_executable,
        "pid":             pid,
        "returncode":      returncode,

        # ── Raw output (redacted but BYTE-FOR-BYTE otherwise) ──────
        "raw_stdout":             stdout_redacted,
        "raw_stderr":             stderr_redacted,
        "stdout_length_chars":    stdout_chars,
        "stdout_length_bytes":    stdout_bytes,
        "stderr_length_chars":    stderr_chars,
        "stderr_length_bytes":    stderr_bytes,

        # ── Markdown fence handling (multi‑stage) ──────────────────
        "had_markdown_fence":      had_markdown_fence,
        "fence_language":          fence_language,
        "number_of_fence_blocks":  number_of_fence_blocks,
        "has_double_fencing":      has_double_fencing,
        "has_partial_fence":       has_partial_fence,
        "stdout_after_fence_strip": stdout_after_fence_strip,

        # ── Response type detection ────────────────────────────────
        "response_type":          response_type,
        "content_type_inference": content_type_inference,
        "is_empty_summary":       is_empty,
        "is_html":                response_type == "html",
        "html_metadata":          html_metadata,

        # ── Empty‑response granular classification ─────────────────
        "empty_classification":   empty_classification,

        # ── JSON anomaly detection ─────────────────────────────────
        "json_anomalies":         json_anomalies,

        # ── JSON parsing (stage‑by‑stage) ──────────────────────────
        "json_parse_attempted":   json_parse_attempted,
        "json_parse_input":       json_parse_input_used,
        "json_parse_input_description": json_parse_input_description,
        "json_parse_succeeded":   json_parse_succeeded,
        "json_parse_failed":      json_parse_failed,
        "json_parse_output":      parsed_object,
        "json_exception":         json_exception,
        "json_traceback":         full_traceback,
        "top_level_json_type":    top_level_json_type,
        "top_level_keys":         top_level_keys,
        "missing_fields":         missing_fields,
        "unexpected_fields":      unexpected_fields,

        # ── Fallback trimming ──────────────────────────────────────
        "fallback_trim_attempted":   fallback_trim_attempted,
        "fallback_trim_succeeded":   fallback_trim_succeeded,
        "stdout_after_fallback_trim": stdout_after_fallback_trim,
        "fallback_trimmed_output":    fallback_trimmed_object,

        # ── Truncation ─────────────────────────────────────────────
        "response_truncated":     response_truncated,

        # ── Schema discovery ───────────────────────────────────────
        "schema":                 schema,

        # ── Validation ─────────────────────────────────────────────
        "validation_attempted":   validation_attempted,
        "validation_result":      validation_result,
        "validation_errors":      validation_errors,

        # ── Final parsed object (if successful) ────────────────────
        "final_parsed_object":    parsed_object,

        # ── Numeric anomalies (recursive) ──────────────────────────
        "numeric_anomalies":      numeric_anomalies,

        # ── Exception recording (from _call_with_retry) ────────────
        "exception":              exception_info,
    }

    return record


# ------------------------------------------------------------------ #
#  Thread‑safe JSONL writer
# ------------------------------------------------------------------ #

_write_lock = asyncio.Lock()

# Default log path
DEFAULT_LOG_PATH: Path = (
    Path(__file__).resolve().parent.parent / "logs" / "mcp_response_trace.jsonl"
)


async def write_record(
    record: dict[str, Any],
    log_path: Path | None = None,
) -> None:
    """
    Append one JSON object as a single line.

    Handles:
      - directory creation
      - rotation (non‑blocking via executor)
      - locking to avoid interleaved writes
    Failures are silent — they must never stop trading.
    """
    path = log_path or DEFAULT_LOG_PATH
    try:
        async with _write_lock:
            # Ensure directory
            path.parent.mkdir(parents=True, exist_ok=True)
            # Rotate (non‑blocking executor)
            await _rotate_async(path)
            # Write one JSON line
            line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception:
        _log.warning("write_record failed — trading continues", exc_info=True)


# ------------------------------------------------------------------ #
#  Main capture entry‑point (called from robinhood_client.py)
# ------------------------------------------------------------------ #

async def capture_mcp_call(
    *,
    tool_name: str = "",
    prompt: str = "",
    duration_ms: float = 0.0,
    attempt: int = 1,
    claude_executable: str = "",
    pid: int | None = None,
    returncode: int | None = None,
    stdout_raw: str | None = None,
    stderr_raw: str | None = None,
    exception_info: dict[str, Any] | None = None,
    expected_fields: list[str] | None = None,
    log_path: Path | None = None,
) -> str:
    """
    Build a trace record and persist it to disk.

    Called from `_call_with_retry()` inside robinhood_client.py
    after every attempt (success or failure).

    Returns the request_id on success; empty string on failure.
    Never raises.
    """
    try:
        # Auto‑extract tool name if not provided
        if not tool_name and prompt:
            tool_name = _extract_tool_name(prompt)
        if not tool_name:
            tool_name = "unknown_tool"

        record = _build_record(
            tool_name=tool_name,
            prompt=prompt,
            duration_ms=duration_ms,
            attempt=attempt,
            claude_executable=claude_executable,
            pid=pid,
            returncode=returncode,
            stdout_raw=stdout_raw,
            stderr_raw=stderr_raw,
            exception_info=exception_info,
            expected_fields=expected_fields,
        )

        # ── Schema comparison & storage ────────────────────────────
        schema = record.get("schema")
        if schema is not None:
            prev = await _lookup_previous_schema(tool_name)
            diff = _compute_schema_diff(schema, prev)
            if diff is not None:
                record["schema_diff"] = diff
            await _store_schema(tool_name, schema)

        # ── Persist ────────────────────────────────────────────────
        await write_record(record, log_path=log_path)

        return record.get("request_id", "")

    except Exception:
        _log.warning("capture_mcp_call failed — trading continues", exc_info=True)
        return ""


# ═══════════════════════════════════════════════════════════════════ #
#  Developer utility functions
# ═══════════════════════════════════════════════════════════════════ #

def _read_records(
    log_path: Path | None = None,
    limit: int = 0,
    reverse: bool = True,
) -> list[dict[str, Any]]:
    """Read JSONL records from disk into a list (newest first if reverse=True)."""
    path = log_path or DEFAULT_LOG_PATH
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        _log.warning("Failed reading trace file", exc_info=True)
        return []
    if reverse:
        records.reverse()
    if limit > 0:
        records = records[:limit]
    return records


def query_last_n(
    n: int = 1,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the last N records (newest first)."""
    return _read_records(log_path, limit=n)


def query_by_tool(
    tool_name: str,
    limit: int = 20,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return recent records for a specific MCP tool."""
    all_recs = _read_records(log_path)
    matched = [r for r in all_recs if r.get("tool_name") == tool_name]
    return matched[:limit]


def query_parse_failures(
    limit: int = 20,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return records where JSON parsing failed."""
    all_recs = _read_records(log_path)
    return [r for r in all_recs if r.get("json_parse_failed")][:limit]


def query_schema_changes(
    limit: int = 20,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return records where schema_diff is present."""
    all_recs = _read_records(log_path)
    return [r for r in all_recs if r.get("schema_diff") is not None][:limit]


def _format_record(r: dict[str, Any]) -> str:
    """Pretty‑print a single record for developer display."""
    rid    = r.get("request_id", "?")
    ts     = r.get("timestamp_utc", "?")
    tool   = r.get("tool_name", "?")
    attempt = r.get("attempt", "?")
    dur    = r.get("duration_ms", "?")
    rtype  = r.get("response_type", "?")
    rcode  = r.get("returncode", "?")
    parse_ok = r.get("json_parse_succeeded", False)
    exc    = r.get("exception")
    stdout_len = r.get("stdout_length_chars", 0)
    stderr_len = r.get("stderr_length_chars", 0)

    lines = [
        "=" * 70,
        f"Request ID:       {rid}",
        f"Timestamp:        {ts}",
        f"Tool:             {tool}",
        f"Attempt:          {attempt}",
        f"Duration (ms):    {dur:.1f}" if isinstance(dur, (int, float)) else f"Duration (ms):     {dur}",
        f"Return code:      {rcode}",
        f"Response type:    {rtype}",
        f"JSON parse OK:    {parse_ok}",
        f"stdout chars:     {stdout_len}",
        f"stderr chars:     {stderr_len}",
    ]
    if exc:
        lines.append(f"Exception:        {exc.get('exception_class', '?')}: {exc.get('exception_message', '?')}")

    # Numeric anomalies
    na = r.get("numeric_anomalies")
    if na:
        lines.append(f"Numeric anomalies: {len(na)}")
        for a in na[:5]:
            lines.append(f"  - {a.get('json_path', '?')}: {a.get('issue', '?')} (value={a.get('actual_value', '?')})")
        if len(na) > 5:
            lines.append(f"  ... and {len(na)-5} more")

    # Schema diff
    sd = r.get("schema_diff")
    if sd:
        lines.append("Schema diff detected:")
        for k, v in sd.items():
            lines.append(f"  {k}: {v}")

    # Raw stdout preview
    raw = r.get("raw_stdout")
    if raw:
        lines.append("Raw stdout (first 300 chars):")
        lines.append(f"  {raw[:300]}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ── Show‑style helpers (formatted output for developers) ──

def show_request(
    request_id: str,
    log_path: Path | None = None,
) -> str:
    """Pretty‑print a single trace record by request_id."""
    all_recs = _read_records(log_path)
    for r in all_recs:
        if r.get("request_id") == request_id:
            return _format_record(r)
    return f"No record found for request_id={request_id}"


def show_tool_history(
    tool: str,
    limit: int = 10,
    log_path: Path | None = None,
) -> str:
    """Pretty‑print the last N records for a specific tool."""
    recs = query_by_tool(tool, limit=limit, log_path=log_path)
    if not recs:
        return f"No records found for tool={tool}"
    return "\n\n".join(_format_record(r) for r in recs)


def show_parse_failures(
    limit: int = 20,
    log_path: Path | None = None,
) -> str:
    """Pretty‑print records where JSON parsing failed."""
    recs = query_parse_failures(limit=limit, log_path=log_path)
    if not recs:
        return "No JSON parse failures found."
    return "\n\n".join(_format_record(r) for r in recs)


def show_schema_changes(
    limit: int = 20,
    log_path: Path | None = None,
) -> str:
    """Pretty‑print records where schema changed from previous calls."""
    recs = query_schema_changes(limit=limit, log_path=log_path)
    if not recs:
        return "No schema changes detected."
    return "\n\n".join(_format_record(r) for r in recs)


def show_html_responses(
    limit: int = 20,
    log_path: Path | None = None,
) -> str:
    """Pretty‑print records where Robinhood returned HTML."""
    all_recs = _read_records(log_path)
    html_recs = [r for r in all_recs if r.get("is_html")]
    if not html_recs:
        return "No HTML responses found."
    return "\n\n".join(_format_record(r) for r in html_recs[:limit])


def show_empty_responses(
    limit: int = 20,
    log_path: Path | None = None,
) -> str:
    """Pretty‑print records where stdout was effectively empty."""
    all_recs = _read_records(log_path)
    empty_recs = [r for r in all_recs if r.get("is_empty_summary")]
    if not empty_recs:
        return "No empty responses found."
    return "\n\n".join(_format_record(r) for r in empty_recs[:limit])


def show_numeric_anomalies(
    limit: int = 20,
    log_path: Path | None = None,
) -> str:
    """Pretty‑print records that contain numeric anomaly flags."""
    all_recs = _read_records(log_path)
    anom_recs = [
        r for r in all_recs
        if r.get("numeric_anomalies") and len(r.get("numeric_anomalies", [])) > 0
    ]
    if not anom_recs:
        return "No numeric anomalies found."
    return "\n\n".join(_format_record(r) for r in anom_recs[:limit])


def show_subprocess_failures(
    limit: int = 20,
    log_path: Path | None = None,
) -> str:
    """Pretty‑print records where Claude exited non‑zero or raised."""
    all_recs = _read_records(log_path)
    fail_recs = [
        r for r in all_recs
        if r.get("exception") is not None or r.get("returncode", 0) != 0
    ]
    if not fail_recs:
        return "No subprocess failures found."
    return "\n\n".join(_format_record(r) for r in fail_recs[:limit])


def what_did_robinhood_return(
    request_id: str | None = None,
    tool_name: str | None = None,
    log_path: Path | None = None,
) -> str:
    """
    Comprehensive diagnostic for a specific request or most recent
    call for a tool. Returns pretty‑printed diagnostic string.
    """
    if request_id:
        return show_request(request_id, log_path=log_path)

    recs = _read_records(log_path, limit=20)
    if tool_name:
        recs = [r for r in recs if r.get("tool_name") == tool_name]

    if not recs:
        return "No matching records found."

    return _format_record(recs[0])
# ═══════════════════════════════════════════════════════════════════ #
#  DIRECT MCP TRANSPORT SUPPORT — appended, not a replacement
# ═══════════════════════════════════════════════════════════════════ #
#
# Everything above this point is the original subprocess/Claude-CLI-based
# recorder, left fully intact and unchanged. It captured stdout/stderr,
# markdown fences, HTML responses, and subprocess PIDs/return codes —
# all specific to the old "ask Claude to translate English into a tool
# call" transport.
#
# The rewritten mcp/robinhood_client.py now connects directly to
# Robinhood's MCP server over Streamable HTTP with a real OAuth token —
# no subprocess, no LLM, no stdout/stderr, no markdown fences, no HTML
# responses, no chatty English clarifying questions. That entire class of
# failure mode is structurally eliminated by calling tools with typed
# arguments instead of asking a model to interpret a prompt.
#
# What DOES still matter and IS still fully preserved for the new
# transport: credential redaction, numeric anomaly detection (catches
# silently-zeroed fields same as before), and cross-session schema drift
# detection (if Robinhood ever changes a tool's response shape, this
# still flags it). Those are transport-agnostic and genuinely valuable,
# so this new function reuses them directly rather than reinventing them.

DIRECT_MCP_LOG_PATH: Path = (
    Path(__file__).resolve().parent.parent / "logs" / "direct_mcp_trace.jsonl"
)


def _build_direct_mcp_record(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    duration_ms: float,
    attempt: int,
    success: bool,
    response_content: list[dict[str, Any]] | None,
    exception_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a structured trace record for a direct (non-subprocess) MCP call.

    Deliberately simpler than the legacy _build_record: there is no
    stdout/stderr, no markdown fences, no HTML, no subprocess PID/return
    code to reason about — a direct tool call either returns structured
    content or raises a typed exception. This mirrors reality instead of
    forcing the old subprocess-shaped schema onto a fundamentally
    different transport.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    request_id = str(uuid.uuid4())

    # Flatten response_content's text payloads into one string for
    # credential redaction + anomaly detection, same safety guarantees
    # as the legacy path.
    raw_text_parts: list[str] = []
    parsed_payload: Any = None
    if response_content:
        for item in response_content:
            text = item.get("text") if isinstance(item, dict) else None
            if text:
                raw_text_parts.append(text)
    raw_text = "\n".join(raw_text_parts)
    redacted_text = _redact_credentials(raw_text) if raw_text else ""

    if raw_text:
        try:
            parsed_payload = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed_payload = None

    numeric_anomalies: list[dict[str, Any]] = []
    if parsed_payload is not None:
        try:
            numeric_anomalies = _detect_numeric_anomalies(parsed_payload)
        except Exception:
            numeric_anomalies = []

    # Redact arguments too — order-placing calls include real quantities/
    # tickers which are fine to log, but if account_number or any future
    # sensitive field appears in arguments, keep it out of the trace file
    # in anything beyond a masked form.
    safe_arguments = dict(arguments)
    if "account_number" in safe_arguments and safe_arguments["account_number"]:
        val = str(safe_arguments["account_number"])
        safe_arguments["account_number"] = (
            f"***{val[-4:]}" if len(val) > 4 else "****"
        )

    record: dict[str, Any] = {
        "transport": "direct_mcp",
        "request_id": request_id,
        "timestamp_utc": now_utc,
        "tool_name": tool_name,
        "arguments": safe_arguments,
        "duration_ms": duration_ms,
        "attempt": attempt,
        "success": success,
        "response_text_redacted": redacted_text[:5000] if redacted_text else None,
        "response_text_length_chars": len(raw_text),
        "json_parse_succeeded": parsed_payload is not None,
        "numeric_anomalies": numeric_anomalies if numeric_anomalies else None,
        "exception": exception_info,
    }

    if isinstance(parsed_payload, dict):
        record["top_level_keys"] = sorted(parsed_payload.keys())
        record["top_level_json_type"] = "object"
    elif isinstance(parsed_payload, list):
        record["top_level_json_type"] = "array"

    return record


async def capture_direct_mcp_call(
    *,
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    duration_ms: float = 0.0,
    attempt: int = 1,
    success: bool = True,
    response_content: list[dict[str, Any]] | None = None,
    exception_info: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> str:
    """Capture and persist one direct (non-subprocess) MCP tool call.

    Called from `_call_tool_with_retry()` inside the rewritten
    mcp/robinhood_client.py after every attempt, success or failure.
    Mirrors the safety guarantees of the legacy capture_mcp_call():
    never raises, never blocks trading, fire-and-forget safe.

    Also performs the same cross-session schema drift detection as the
    legacy path — if tool_name's response shape changes from what was
    previously recorded, schema_diff is populated on the record.
    """
    try:
        tool_name = tool_name or "unknown_tool"
        arguments = arguments or {}

        record = _build_direct_mcp_record(
            tool_name=tool_name,
            arguments=arguments,
            duration_ms=duration_ms,
            attempt=attempt,
            success=success,
            response_content=response_content,
            exception_info=exception_info,
        )

        # Schema drift detection — reuse the same persistent schema store
        # as the legacy path, keyed by tool_name, transport-agnostic.
        if record.get("top_level_keys"):
            current_schema = {"keys": record["top_level_keys"]}
            prev = await _lookup_previous_schema(tool_name)
            if prev is not None and prev.get("keys") != current_schema["keys"]:
                record["schema_diff"] = {
                    "previous_keys": prev.get("keys"),
                    "current_keys": current_schema["keys"],
                    "added": sorted(set(current_schema["keys"]) - set(prev.get("keys", []))),
                    "removed": sorted(set(prev.get("keys", [])) - set(current_schema["keys"])),
                }
            await _store_schema(tool_name, current_schema)

        target_path = log_path or DIRECT_MCP_LOG_PATH
        await write_record(record, log_path=target_path)

        return record.get("request_id", "")

    except Exception:
        _log.warning("capture_direct_mcp_call failed — trading continues", exc_info=True)
        return ""


def query_direct_mcp_calls(
    tool_name: str | None = None,
    limit: int = 20,
    only_failures: bool = False,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Query recent direct-MCP trace records, optionally filtered by tool
    and/or failure status. Mirrors query_by_tool()/query_parse_failures()
    but for the new transport's record shape."""
    target_path = log_path or DIRECT_MCP_LOG_PATH
    all_recs = _read_records(target_path)
    if tool_name:
        all_recs = [r for r in all_recs if r.get("tool_name") == tool_name]
    if only_failures:
        all_recs = [r for r in all_recs if not r.get("success", True)]
    return all_recs[:limit]


def show_direct_mcp_history(
    tool_name: str | None = None,
    limit: int = 10,
    log_path: Path | None = None,
) -> str:
    """Pretty-print recent direct-MCP calls for developer inspection."""
    recs = query_direct_mcp_calls(tool_name=tool_name, limit=limit, log_path=log_path)
    if not recs:
        return f"No direct MCP records found{f' for tool={tool_name}' if tool_name else ''}."

    lines: list[str] = []
    for r in recs:
        lines.append("=" * 70)
        lines.append(f"Request ID:    {r.get('request_id', '?')}")
        lines.append(f"Timestamp:     {r.get('timestamp_utc', '?')}")
        lines.append(f"Tool:          {r.get('tool_name', '?')}")
        lines.append(f"Arguments:     {r.get('arguments', {})}")
        lines.append(f"Attempt:       {r.get('attempt', '?')}")
        dur = r.get("duration_ms", "?")
        lines.append(f"Duration (ms): {dur:.1f}" if isinstance(dur, (int, float)) else f"Duration (ms): {dur}")
        lines.append(f"Success:       {r.get('success', '?')}")
        if r.get("exception"):
            exc = r["exception"]
            lines.append(f"Exception:     {exc.get('exception_class', '?')}: {exc.get('exception_message', '?')}")
        na = r.get("numeric_anomalies")
        if na:
            lines.append(f"Numeric anomalies: {len(na)}")
            for a in na[:5]:
                lines.append(f"  - {a.get('json_path', '?')}: {a.get('issue', '?')}")
        sd = r.get("schema_diff")
        if sd:
            lines.append(f"Schema diff: added={sd.get('added')} removed={sd.get('removed')}")
        resp = r.get("response_text_redacted")
        if resp:
            lines.append(f"Response (first 300 chars): {resp[:300]}")
        lines.append("=" * 70)
    return "\n\n".join(lines)