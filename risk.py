import json

from llm import call_llm
from vocab import RISK_LEVELS

RISK_CRITERIA = [
    "chargeback intent",
    "legal threat",
    "regulator or complaint language",
    "accusation of scam or fraud",
    "loss of funds claim",
    "urgent withdrawal or access-to-funds issue",
    "account suspension or frozen funds",
    "potential compliance or jurisdictional sensitivity",
]

_RISK_ORDER = ["low", "medium", "high", "critical"]

CATEGORY_DEFAULT_RISK = {
    "payments": "medium",
    "technical": "low",
    "compliance": "medium",
    "account": "medium",
    "product_query": "low",
    "escalation": "high",
}

ELEVATION_KEYWORDS = (
    "scam",
    "refund",
    "chargeback",
    "suspended",
    "locked",
    "frozen",
)


def _has_elevation_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ELEVATION_KEYWORDS)


def select_for_review(
    classifications: list[dict],
    messages_by_id: dict[int, str] | None = None,
) -> list[int]:
    selected: set[int] = set()
    for c in classifications:
        if c["category"] == "escalation" or c.get("needs_human_review"):
            selected.add(c["id"])
            continue
        if messages_by_id is not None and _has_elevation_keyword(messages_by_id.get(c["id"], "")):
            selected.add(c["id"])
    return sorted(selected)


def default_risk(message_text: str, category: str) -> str:
    base = CATEGORY_DEFAULT_RISK.get(category, "low")
    text = message_text.lower()
    if any(kw in text for kw in ELEVATION_KEYWORDS):
        idx = _RISK_ORDER.index(base)
        return _RISK_ORDER[min(idx + 1, len(_RISK_ORDER) - 1)]
    return base


def build_prompt(reviewed: list[dict]) -> str:
    criteria_block = "\n".join(f"- {c}" for c in RISK_CRITERIA)
    levels_block = ", ".join(_RISK_ORDER)
    schema_example = (
        '{\n'
        '  "scores": [\n'
        '    {"id": 14, "risk_level": "critical", '
        '"triggering_criteria": ["chargeback intent", "urgent withdrawal or access-to-funds issue"], '
        '"rationale": "short explanation"}\n'
        '  ]\n'
        '}'
    )
    return (
        "You are a risk-assessment system for support messages on a regulated online-trading platform.\n"
        "For each message provided, decide a risk level and identify which of the listed criteria apply.\n\n"
        "Risk criteria (cite by exact phrase in triggering_criteria when applicable):\n"
        f"{criteria_block}\n\n"
        f"Allowed risk levels (choose exactly one): {levels_block}.\n\n"
        "Guidelines:\n"
        "- 'critical' for messages combining urgent loss-of-funds OR explicit chargeback / "
        "legal-action threats. Any message that names 'chargeback' alongside urgency language "
        "('now', 'immediately', 'days', 'weeks') is critical.\n"
        "- 'high' for explicit complaint, scam accusation, or account-suspension distress without a chargeback/legal threat.\n"
        "- 'medium' for unresolved payments or compliance ambiguity without escalation language.\n"
        "- 'low' for routine ambiguity flagged for review without distress signals.\n\n"
        "Anchor example (illustrative only — do not echo this id):\n"
        '  Input:  {"id": 99, "message": "My withdrawal has been pending 12 days. I need my money back today or I am filing a chargeback through my bank."}\n'
        '  Output: {"id": 99, "risk_level": "critical", '
        '"triggering_criteria": ["chargeback intent", "urgent withdrawal or access-to-funds issue", "loss of funds claim"], '
        '"rationale": "Explicit chargeback threat combined with urgent withdrawal language and direct loss-of-funds claim."}\n\n'
        "Output a single JSON object with this exact schema (no prose, no markdown fences):\n"
        f"{schema_example}\n\n"
        "Rules:\n"
        "1. Return exactly one score per input message; do not invent or drop ids.\n"
        "2. risk_level MUST be one of the allowed values verbatim.\n"
        "3. triggering_criteria MUST be drawn from the list above (use the exact phrases).\n\n"
        "Messages to score:\n"
        f"{json.dumps(reviewed, ensure_ascii=False)}\n"
    )


def _coerce(items: list[dict], expected_ids: set[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for raw in items:
        try:
            mid = int(raw["id"])
            level = str(raw["risk_level"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed risk item: {raw!r}: {e}")
        if mid not in expected_ids:
            raise ValueError(f"unexpected id {mid} from risk model")
        if level not in RISK_LEVELS:
            raise ValueError(f"risk_level '{level}' for id {mid} not in vocabulary")
        triggering = raw.get("triggering_criteria") or []
        if not isinstance(triggering, list):
            raise ValueError(f"triggering_criteria for id {mid} must be a list")
        out[mid] = {
            "id": mid,
            "risk_level": level,
            "triggering_criteria": [str(t) for t in triggering],
            "rationale": str(raw.get("rationale", "")).strip(),
        }
    missing = expected_ids - set(out)
    if missing:
        raise ValueError(f"missing risk scores for ids: {sorted(missing)}")
    return out


def run_risk_scoring(
    messages_by_id: dict[int, str],
    classifications: list[dict],
) -> list[dict]:
    review_ids = select_for_review(classifications, messages_by_id)
    reviewed_payload = [{"id": mid, "message": messages_by_id[mid]} for mid in review_ids]

    llm_scores: dict[int, dict] = {}
    if reviewed_payload:
        prompt = build_prompt(reviewed_payload)
        raw = call_llm(
            stage="risk_scoring",
            prompt=prompt,
            input_artifacts=["initial_classifications.json"],
            output_artifact="risk_scores.json",
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
        items = payload.get("scores")
        if not isinstance(items, list):
            raise ValueError(f"expected 'scores' list in response; got keys={list(payload)}")
        llm_scores = _coerce(items, set(review_ids))

    results: list[dict] = []
    for c in classifications:
        mid = c["id"]
        if mid in llm_scores:
            results.append(llm_scores[mid])
            continue
        results.append({
            "id": mid,
            "risk_level": default_risk(messages_by_id[mid], c["category"]),
            "triggering_criteria": [],
            "rationale": (
                f"Default mapping: category={c['category']}, "
                "no Stage 2 review (category != escalation and not flagged)."
            ),
        })
    results.sort(key=lambda r: r["id"])
    return results
