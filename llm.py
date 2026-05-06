import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

_LOG_PATH = Path("llm_calls.jsonl")
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["GROQ_API_KEY"]
        base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def call_llm(
    stage: str,
    prompt: str,
    *,
    risk_tier=None,
    message_id=None,
    input_artifacts: list[str],
    output_artifact: str,
    few_shot_examples_included: bool = False,
    response_format=None,
) -> str:
    model = os.environ["GROQ_MODEL"]
    client = _get_client()

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content or ""

    record = {
        "stage": stage,
        "risk_tier": risk_tier,
        "message_id": message_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "groq",
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
        "few_shot_examples_included": few_shot_examples_included,
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return content
