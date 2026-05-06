import json
import re

from llm import call_llm
from vocab import COMPLIANCE_VIOLATIONS, NO_COMMITMENTS_INSTRUCTION

KEYWORD_CHECKS: tuple[str, ...] = (
    "will",
    "guarantee",
    "promise",
    "definitely",
    "refund you",
    "within 24 hours",
    "today",
)

KEYWORD_VIOLATION_HINT: dict[str, str] = {
    "will": "promise",
    "guarantee": "promise",
    "promise": "promise",
    "definitely": "promise",
    "refund you": "promise",
    "within 24 hours": "specific_timeline",
    "today": "specific_timeline",
}


def deterministic_keyword_hits(text: str) -> list[tuple[str, str]]:
    text_lower = text.lower()
    hits: list[tuple[str, str]] = []
    for kw in KEYWORD_CHECKS:
        if " " in kw:
            if kw in text_lower:
                hits.append((kw, KEYWORD_VIOLATION_HINT[kw]))
        else:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                hits.append((kw, KEYWORD_VIOLATION_HINT[kw]))
    return hits


def build_prompt(drafts: list[dict]) -> str:
    items = [{"id": d["id"], "draft_response": d["draft_response"]} for d in drafts]
    violations_block = ", ".join(sorted(COMPLIANCE_VIOLATIONS))
    schema_example = (
        '{\n'
        '  "results": [\n'
        '    {"id": 14, "passed": false, '
        '"violations": ["specific_timeline"], '
        '"evidence": "exact response excerpt", '
        '"recommended_fix": "string"}\n'
        '  ]\n'
        '}'
    )
    return (
        "You are a compliance reviewer for support responses on a regulated online-trading platform.\n"
        f"Policy: {NO_COMMITMENTS_INSTRUCTION}\n\n"
        "Review each draft response below and detect violations of these types:\n"
        "- promise: any explicit commitment, guarantee, or promise of action or outcome.\n"
        "- specific_timeline: any commitment to a specific timeframe (e.g., 'within 24 hours', 'today', 'by Friday', '3 business days').\n"
        "- liability_admission: any admission of fault, error, or liability by the platform.\n"
        "- non_compliant_financial_claim: any unsupported claim about returns, profits, refunds, or financial outcomes.\n"
        "- none: use this when the response is fully compliant.\n\n"
        f"Allowed violations vocabulary (use these exact strings): {violations_block}\n\n"
        "For each draft return:\n"
        "- id: integer matching the input id\n"
        "- passed: boolean (true iff no violations are present)\n"
        "- violations: list of violation types from the vocabulary above; use [\"none\"] when passed is true\n"
        "- evidence: an exact excerpt from the response that demonstrates the violation (or empty string when passed)\n"
        "- recommended_fix: a one-sentence rewrite suggestion (or empty string when passed)\n\n"
        "Output a single JSON object with this exact schema (no prose, no markdown fences):\n"
        f"{schema_example}\n\n"
        "Rules:\n"
        "1. Return exactly one result per input draft; do not invent or drop ids.\n"
        "2. Each violation MUST be one of the allowed vocabulary values verbatim.\n"
        "3. The evidence MUST be a verbatim substring of the draft_response when passed=false.\n\n"
        "Drafts to review:\n"
        f"{json.dumps(items, ensure_ascii=False)}\n"
    )


def _coerce_llm(items: list[dict], expected_ids: set[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for raw in items:
        try:
            mid = int(raw["id"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed compliance item: {raw!r}: {e}")
        if mid not in expected_ids:
            raise ValueError(f"unexpected id {mid} in compliance result")
        violations_raw = raw.get("violations") or []
        if not isinstance(violations_raw, list):
            raise ValueError(f"violations for id {mid} must be a list")
        clean_violations: list[str] = []
        for v in violations_raw:
            sv = str(v)
            if sv not in COMPLIANCE_VIOLATIONS:
                raise ValueError(f"violation '{sv}' for id {mid} not in vocabulary")
            clean_violations.append(sv)
        out[mid] = {
            "passed": bool(raw.get("passed", False)),
            "violations": clean_violations,
            "evidence": str(raw.get("evidence", "")).strip(),
            "recommended_fix": str(raw.get("recommended_fix", "")).strip(),
        }
    missing = expected_ids - set(out)
    if missing:
        raise ValueError(f"missing compliance results for ids: {sorted(missing)}")
    return out


def _merge(llm_result: dict, keyword_hits: list[tuple[str, str]]) -> dict:
    violations = [v for v in llm_result["violations"] if v != "none"]
    keyword_terms: list[str] = []
    for kw, hint in keyword_hits:
        keyword_terms.append(kw)
        if hint not in violations:
            violations.append(hint)

    passed = not violations
    if passed:
        violations = ["none"]

    evidence = llm_result["evidence"]
    if keyword_terms:
        kw_note = "keyword hits: " + ", ".join(f"'{k}'" for k in keyword_terms)
        evidence = f"{evidence} | {kw_note}".strip(" |")

    return {
        "passed": passed,
        "violations": violations,
        "evidence": evidence,
        "recommended_fix": llm_result["recommended_fix"],
        "keyword_hits": keyword_terms,
        "needs_human_review": not passed,
    }


def run_compliance_check(drafts: list[dict]) -> list[dict]:
    if not drafts:
        return []

    prompt = build_prompt(drafts)
    raw = call_llm(
        stage="compliance_check",
        prompt=prompt,
        input_artifacts=["draft_responses.json"],
        output_artifact="response_compliance.json",
        response_format={"type": "json_object"},
    )
    payload = json.loads(raw)
    items = payload.get("results")
    if not isinstance(items, list):
        raise ValueError(f"expected 'results' list in compliance response; got keys={list(payload)}")
    expected = {int(d["id"]) for d in drafts}
    llm_results = _coerce_llm(items, expected)

    final: list[dict] = []
    for d in drafts:
        mid = int(d["id"])
        text = d["draft_response"]
        merged = _merge(llm_results[mid], deterministic_keyword_hits(text))
        final.append({"id": mid, **merged})

    final.sort(key=lambda r: r["id"])
    return final
