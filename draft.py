import json

from llm import call_llm
from vocab import NO_COMMITMENTS_INSTRUCTION

BATCHED_TIERS = {"low", "medium"}
INDIVIDUAL_TIERS = {"high", "critical"}

_RESPONSE_RULES = (
    "Each response MUST be:\n"
    "- 2 to 4 sentences (no fewer, no more)\n"
    "- professional in tone\n"
    "- empathetic to the user's situation\n"
    "- specific to the issue raised in the message (do not produce generic boilerplate)\n"
    "- free of unsupported promises\n"
)


def _build_batched_prompt(items: list[dict]) -> str:
    schema_example = (
        '{\n'
        '  "drafts": [\n'
        '    {"id": 1, "draft_response": "two-to-four sentence reply"}\n'
        '  ]\n'
        '}'
    )
    return (
        "You are a customer support agent for Deriv, a regulated online-trading platform.\n"
        "Draft a reply for EACH message in the list below.\n\n"
        f"{_RESPONSE_RULES}\n"
        f"{NO_COMMITMENTS_INSTRUCTION}\n\n"
        "Output a single JSON object with this exact schema (no prose, no markdown fences):\n"
        f"{schema_example}\n\n"
        "Rules:\n"
        "1. Return exactly one draft per input message; do not invent or drop ids.\n"
        "2. Use the user's own wording where helpful so the reply feels specific.\n"
        "3. The field name MUST be exactly 'draft_response' (singular, with underscore). "
        "Do not use 'draft', 'drafts', 'response', or any other key.\n\n"
        "Messages:\n"
        f"{json.dumps(items, ensure_ascii=False)}\n"
    )


_DRAFT_FIELD_ALIASES = ("draft_response", "response", "reply", "draft", "drafts")


def _extract_draft_text(raw_item: dict) -> str | None:
    for key in _DRAFT_FIELD_ALIASES:
        value = raw_item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _build_individual_prompt(item: dict, risk_level: str) -> str:
    schema_example = (
        '{\n'
        '  "draft_response": "two-to-four sentence reply"\n'
        '}'
    )
    return (
        "You are a customer support agent for Deriv, a regulated online-trading platform.\n"
        f"This message has been classified as {risk_level.upper()} risk and requires extra care.\n\n"
        f"{_RESPONSE_RULES}\n"
        "STRICT COMPLIANCE REQUIREMENT:\n"
        f"{NO_COMMITMENTS_INSTRUCTION}\n\n"
        "Output a single JSON object with this exact schema (no prose, no markdown fences):\n"
        f"{schema_example}\n\n"
        "Message to respond to:\n"
        f"{json.dumps(item, ensure_ascii=False)}\n"
    )


def _validate_text(text: str, mid: int) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"empty draft_response for id {mid}")
    return cleaned


def run_drafting(
    messages_by_id: dict[int, str],
    risk_scores: list[dict],
) -> list[dict]:
    risk_by_id = {r["id"]: r["risk_level"] for r in risk_scores}

    batched_inputs: list[dict] = []
    individual_inputs: list[tuple[int, str]] = []
    for r in risk_scores:
        mid = r["id"]
        level = r["risk_level"]
        if level in BATCHED_TIERS:
            batched_inputs.append({"id": mid, "message": messages_by_id[mid]})
        elif level in INDIVIDUAL_TIERS:
            individual_inputs.append((mid, level))
        else:
            raise ValueError(f"unexpected risk_level '{level}' for id {mid}")

    drafts_by_id: dict[int, dict] = {}

    if batched_inputs:
        prompt = _build_batched_prompt(batched_inputs)
        raw = call_llm(
            stage="response_drafting_batched",
            prompt=prompt,
            risk_tier="low_medium",
            input_artifacts=["initial_classifications.json", "risk_scores.json"],
            output_artifact="draft_responses.json",
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
        items = payload.get("drafts")
        if not isinstance(items, list):
            raise ValueError(f"expected 'drafts' list in batched response; got keys={list(payload)}")
        expected = {b["id"] for b in batched_inputs}
        seen: set[int] = set()
        for raw_item in items:
            try:
                mid = int(raw_item["id"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"malformed batched draft item (no id): {raw_item!r}: {e}")
            extracted = _extract_draft_text(raw_item)
            if extracted is None:
                raise ValueError(
                    f"batched draft for id {mid} missing a recognisable text field; "
                    f"keys present: {list(raw_item)}"
                )
            text = _validate_text(extracted, mid)
            if mid not in expected:
                raise ValueError(f"unexpected id {mid} in batched drafts")
            if mid in seen:
                raise ValueError(f"duplicate batched draft for id {mid}")
            seen.add(mid)
            drafts_by_id[mid] = {
                "id": mid,
                "risk_level": risk_by_id[mid],
                "draft_response": text,
                "drafting_mode": "batched",
            }
        missing = expected - seen
        if missing:
            raise ValueError(f"missing batched drafts for ids: {sorted(missing)}")

    for mid, level in individual_inputs:
        item = {"id": mid, "message": messages_by_id[mid]}
        prompt = _build_individual_prompt(item, level)
        raw = call_llm(
            stage="response_drafting_individual",
            prompt=prompt,
            risk_tier=level,
            message_id=mid,
            input_artifacts=["initial_classifications.json", "risk_scores.json"],
            output_artifact="draft_responses.json",
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
        text = _extract_draft_text(payload)
        if text is None:
            raise ValueError(
                f"individual draft for id {mid} missing a recognisable text field; "
                f"keys present: {list(payload)}"
            )
        drafts_by_id[mid] = {
            "id": mid,
            "risk_level": level,
            "draft_response": _validate_text(text, mid),
            "drafting_mode": "individual",
        }

    expected_all = {r["id"] for r in risk_scores}
    missing_all = expected_all - set(drafts_by_id)
    if missing_all:
        raise ValueError(f"missing drafts for ids: {sorted(missing_all)}")

    return [drafts_by_id[mid] for mid in sorted(drafts_by_id)]
