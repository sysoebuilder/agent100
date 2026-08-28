import json
import logging
import re
import time
from pathlib import Path

from pydantic import ValidationError

from ai_agent_learning.paths import get_data_dir
from ai_agent_learning.schema import Context

SESSION_DIR = get_data_dir() / "contexts"

def save_context(context: Context) -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SESSION_DIR / f"{context.context_id}.json"

    try:
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                context.model_dump(mode="json"),
                file,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        raise

    return file_path

def load_context() -> list[Context]:
    started_at = time.perf_counter()
    contexts: list[Context] = []

    context_files = sorted(
        SESSION_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for file_path in context_files:
        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            context = Context.model_validate(data)
            contexts.append(context)

        except (OSError, json.JSONDecodeError, ValidationError):
            raise

    return contexts

def select_context(context_id: str) -> Context:
    contexts = load_context()
    for context in contexts:
        if context.context_id == context_id:
            return context

    return Context(context_id=context_id)
