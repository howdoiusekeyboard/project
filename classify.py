import json

from llm import call_llm
from vocab import CATEGORIES

CONFIDENCE_THRESHOLD = 0.7


def build_prompt(messages: list[dict], *, few_shot_block: str | None = None) -> str:
    categories_block = "\n".join(f"- {c}" for c in sorted(CATEGORIES))
    schema_example = (
        '{\n'
        '  "classifications": [\n'
        '    {"id": 1, "category": "payments", "confidence": 0.86, '
        '"needs_human_review": false, "reason": "short justification"}\n'
        '  ]\n'
        '}'
    )
    examples_section = ""
    if few_shot_block:
        examples_section = (
            "Operator-validated examples (use these to calibrate borderline cases — "
            "imitate the labels shown for similar messages):\n"
            f"{few_shot_block}\n\n"
        )
    return (
        "You are a support-message classifier for a regulated online-trading platform.\n"
        "Classify every message in the input list into EXACTLY ONE of the allowed categories below.\n\n"
        "Allowed categories:\n"
        f"{categories_block}\n\n"
        f"{examples_section}"
        "For each message, return:\n"
        "- id: integer matching the input id\n"
        "- category: one value drawn from the allowed-categories vocabulary above\n"
        "- confidence: a float between 0.0 and 1.0 representing your confidence in the chosen category\n"
        "- needs_human_review: boolean. Set true when confidence is below 0.7 OR the message is genuinely ambiguous between two or more categories.\n"
        "- reason: a brief one-sentence justification\n\n"
        "Output a single JSON object with the following exact schema (no prose, no markdown fences):\n"
        f"{schema_example}\n\n"
        "Rules:\n"
        "1. Return exactly one classification per input message; do not invent or drop ids.\n"
        "2. The category value MUST be one of the allowed categories verbatim.\n"
        "3. The confidence MUST be a number between 0.0 and 1.0 inclusive.\n\n"
        "Messages to classify:\n"
        f"{json.dumps(messages, ensure_ascii=False)}\n"
    )


def _coerce(items: list[dict], expected_ids: set[int]) -> list[dict]:
    cleaned = []
    seen_ids = set()
    for raw in items:
        try:
            mid = int(raw["id"])
            category = str(raw["category"])
            confidence = float(raw["confidence"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed classification item: {raw!r}: {e}")
        if mid not in expected_ids:
            raise ValueError(f"unexpected id {mid} not in input set")
        if mid in seen_ids:
            raise ValueError(f"duplicate classification for id {mid}")
        if category not in CATEGORIES:
            raise ValueError(f"category '{category}' for id {mid} not in vocabulary")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence {confidence} for id {mid} outside [0.0, 1.0]")

        needs_review = bool(raw.get("needs_human_review", False))
        if confidence < CONFIDENCE_THRESHOLD:
            needs_review = True

        cleaned.append({
            "id": mid,
            "category": category,
            "confidence": confidence,
            "needs_human_review": needs_review,
            "reason": str(raw.get("reason", "")).strip(),
        })
        seen_ids.add(mid)

    missing = expected_ids - seen_ids
    if missing:
        raise ValueError(f"missing classifications for ids: {sorted(missing)}")
    cleaned.sort(key=lambda c: c["id"])
    return cleaned


def run_classification(
    messages: list[dict],
    *,
    few_shot_block: str | None = None,
    stage: str = "initial_classification",
    input_artifacts: list[str] | None = None,
    output_artifact: str = "initial_classifications.json",
    few_shot_examples_included: bool | None = None,
) -> list[dict]:
    prompt = build_prompt(messages, few_shot_block=few_shot_block)
    if few_shot_examples_included is None:
        few_shot_examples_included = bool(few_shot_block)
    raw = call_llm(
        stage=stage,
        prompt=prompt,
        input_artifacts=input_artifacts or ["support_messages.json"],
        output_artifact=output_artifact,
        few_shot_examples_included=few_shot_examples_included,
        response_format={"type": "json_object"},
    )
    payload = json.loads(raw)
    items = payload.get("classifications")
    if not isinstance(items, list):
        raise ValueError(f"expected 'classifications' list in response; got keys={list(payload)}")
    expected_ids = {int(m["id"]) for m in messages}
    return _coerce(items, expected_ids)
