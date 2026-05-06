import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from routing import ROUTING_BY_CATEGORY
from vocab import CATEGORIES, COMPLIANCE_VIOLATIONS, RISK_LEVELS


_FAILURES = 0


def _report(label: str, ok: bool, reason: str | None = None) -> None:
    global _FAILURES
    if ok:
        print(f"PASS {label}")
    else:
        print(f"FAIL {label} - {reason}")
        _FAILURES += 1


def _check(label: str, fn) -> None:
    try:
        fn()
        _report(label, True)
    except AssertionError as e:
        _report(label, False, str(e) or "assertion failed")
    except Exception as e:
        _report(label, False, f"unexpected {type(e).__name__}: {e}")


def _ensure(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise AssertionError(f"line {i} not valid JSON: {e}")
    return records


_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def _count_sentences(text: str) -> int:
    return sum(1 for _ in _SENTENCE_END.finditer(text.strip()))


REQUIRED = [
    "support_messages.json",
    "initial_classifications.json",
    "risk_scores.json",
    "draft_responses.json",
    "corrections.jsonl",
    "reclassified_outputs.json",
    "triage_output.json",
    "llm_calls.jsonl",
]

OPTIONAL = [
    "response_compliance.json",
    "analytics_summary.json",
    "routing_decisions.json",
    "correction_store_status.json",
]


def main() -> None:
    print("=" * 64)
    print("PIPELINE VALIDATION")
    print("=" * 64)

    for p in REQUIRED:
        _check(
            f"required artifact present: {p}",
            lambda path=p: _ensure(Path(path).exists(), f"file does not exist"),
        )

    for p in OPTIONAL:
        if Path(p).exists():
            print(f"PASS optional artifact present: {p}")
        else:
            print(f"SKIP optional artifact missing: {p}")

    if not Path("support_messages.json").exists():
        print()
        print("aborting further checks: support_messages.json is missing")
        sys.exit(1)

    messages = _load_json("support_messages.json")
    expected_ids = sorted(int(m["id"]) for m in messages)
    expected_id_set = set(expected_ids)
    msg_text_by_id = {int(m["id"]): m["message"] for m in messages}

    def check_initial() -> None:
        data = _load_json("initial_classifications.json")
        _ensure(isinstance(data, list), "not a list")
        ids = sorted(int(c["id"]) for c in data)
        _ensure(ids == expected_ids, f"id coverage mismatch: got {ids}, expected {expected_ids}")
        for c in data:
            mid = c["id"]
            _ensure(c["category"] in CATEGORIES, f"id {mid}: invalid category '{c['category']}'")
            conf = c["confidence"]
            _ensure(isinstance(conf, (int, float)), f"id {mid}: confidence not numeric")
            _ensure(0.0 <= conf <= 1.0, f"id {mid}: confidence {conf} outside [0.0, 1.0]")
            _ensure(isinstance(c.get("needs_human_review"), bool), f"id {mid}: needs_human_review missing/non-bool")
            if conf < 0.7:
                _ensure(c["needs_human_review"], f"id {mid}: confidence {conf} < 0.7 but not flagged for review")
            _ensure(isinstance(c.get("reason"), str), f"id {mid}: reason missing or non-string")
    _check("initial_classifications schema + coverage", check_initial)

    risk_data: list[dict] = []

    def check_risk() -> None:
        nonlocal risk_data
        risk_data = _load_json("risk_scores.json")
        _ensure(isinstance(risk_data, list), "not a list")
        ids = sorted(int(r["id"]) for r in risk_data)
        _ensure(ids == expected_ids, f"id coverage mismatch: got {ids}")
        for r in risk_data:
            mid = r["id"]
            _ensure(r["risk_level"] in RISK_LEVELS, f"id {mid}: invalid risk_level '{r['risk_level']}'")
            _ensure(isinstance(r.get("triggering_criteria"), list), f"id {mid}: triggering_criteria not a list")
            _ensure(isinstance(r.get("rationale"), str), f"id {mid}: rationale missing")
    _check("risk_scores schema + coverage", check_risk)

    def check_drafts() -> None:
        drafts = _load_json("draft_responses.json")
        _ensure(isinstance(drafts, list), "not a list")
        ids = sorted(int(d["id"]) for d in drafts)
        _ensure(ids == expected_ids, f"id coverage mismatch: got {ids}")
        risk_by_id = {int(r["id"]): r["risk_level"] for r in risk_data}
        for d in drafts:
            mid = d["id"]
            _ensure(d["drafting_mode"] in ("batched", "individual"),
                    f"id {mid}: invalid drafting_mode '{d['drafting_mode']}'")
            text = d.get("draft_response", "")
            _ensure(isinstance(text, str) and text.strip(), f"id {mid}: empty draft_response")
            sc = _count_sentences(text)
            _ensure(2 <= sc <= 4, f"id {mid}: draft has {sc} sentences (expected 2-4)")
            expected_risk = risk_by_id.get(mid)
            _ensure(d["risk_level"] == expected_risk,
                    f"id {mid}: draft risk_level '{d['risk_level']}' != risk_scores '{expected_risk}'")
            if expected_risk in ("low", "medium"):
                _ensure(d["drafting_mode"] == "batched",
                        f"id {mid}: risk={expected_risk} should be batched, got '{d['drafting_mode']}'")
            elif expected_risk in ("high", "critical"):
                _ensure(d["drafting_mode"] == "individual",
                        f"id {mid}: risk={expected_risk} should be individual, got '{d['drafting_mode']}'")
    _check("draft_responses schema + risk-tier dispatch + 2-4 sentence rule", check_drafts)

    correction_records: list[dict] = []

    def check_corrections() -> None:
        nonlocal correction_records
        correction_records = _load_jsonl("corrections.jsonl")
        for i, r in enumerate(correction_records, 1):
            for key in ("id", "message", "original_category", "corrected_category", "action", "timestamp"):
                _ensure(key in r, f"line {i}: missing field '{key}'")
            _ensure(r["action"] in ("accepted", "corrected", "skipped"),
                    f"line {i}: invalid action '{r['action']}'")
            _ensure(r["original_category"] in CATEGORIES,
                    f"line {i}: invalid original_category '{r['original_category']}'")
            if r["action"] == "corrected":
                _ensure(r["corrected_category"] in CATEGORIES,
                        f"line {i}: corrected action requires valid corrected_category")
            elif r["action"] == "skipped":
                _ensure(r["corrected_category"] is None,
                        f"line {i}: skipped action must have corrected_category=null")
            try:
                datetime.fromisoformat(r["timestamp"])
            except ValueError:
                raise AssertionError(f"line {i}: timestamp '{r['timestamp']}' not ISO-8601")
    _check("corrections.jsonl schema + action vocabulary", check_corrections)

    def check_reclassified() -> None:
        data = _load_json("reclassified_outputs.json")
        _ensure(isinstance(data, list), "not a list")
        ids = sorted(int(c["id"]) for c in data)
        _ensure(ids == expected_ids, f"id coverage mismatch")
        for c in data:
            mid = c["id"]
            _ensure(c["category"] in CATEGORIES, f"id {mid}: invalid category '{c['category']}'")
            conf = c["confidence"]
            _ensure(0.0 <= conf <= 1.0, f"id {mid}: confidence {conf} out of range")
            if conf < 0.7:
                _ensure(c["needs_human_review"], f"id {mid}: low conf not flagged")
    _check("reclassified_outputs schema + coverage", check_reclassified)

    if Path("response_compliance.json").exists():
        def check_compliance() -> None:
            data = _load_json("response_compliance.json")
            _ensure(isinstance(data, list), "not a list")
            ids = sorted(int(r["id"]) for r in data)
            _ensure(ids == expected_ids, f"id coverage mismatch")
            for r in data:
                mid = r["id"]
                _ensure(isinstance(r.get("passed"), bool), f"id {mid}: passed not bool")
                vs = r.get("violations")
                _ensure(isinstance(vs, list) and len(vs) >= 1,
                        f"id {mid}: violations missing or empty")
                for v in vs:
                    _ensure(v in COMPLIANCE_VIOLATIONS,
                            f"id {mid}: violation '{v}' not in vocabulary")
                if r["passed"]:
                    _ensure(vs == ["none"], f"id {mid}: passed=true but violations={vs}")
                else:
                    _ensure("none" not in vs, f"id {mid}: passed=false but 'none' in violations")
                _ensure(isinstance(r.get("evidence"), str), f"id {mid}: evidence missing")
                _ensure(isinstance(r.get("recommended_fix"), str), f"id {mid}: recommended_fix missing")
        _check("response_compliance schema + violation vocabulary", check_compliance)

    if Path("routing_decisions.json").exists():
        def check_routing() -> None:
            data = _load_json("routing_decisions.json")
            _ensure(isinstance(data, list), "not a list")
            ids = sorted(int(d["id"]) for d in data)
            _ensure(ids == expected_ids, f"id coverage mismatch")
            risk_by_id = {int(r["id"]): r["risk_level"] for r in risk_data}
            for d in data:
                mid = d["id"]
                cat = d["category"]
                _ensure(cat in CATEGORIES, f"id {mid}: invalid category '{cat}'")
                _ensure(d["risk_level"] in RISK_LEVELS, f"id {mid}: invalid risk_level")
                _ensure(d["risk_level"] == risk_by_id[mid],
                        f"id {mid}: routing risk_level '{d['risk_level']}' != risk_scores '{risk_by_id[mid]}'")
                rule = ROUTING_BY_CATEGORY[cat]
                _ensure(d["team"] == rule["team"], f"id {mid}: team mismatch")
                _ensure(d["escalation_path"] == rule["escalation_path"],
                        f"id {mid}: escalation_path mismatch")
                _ensure(isinstance(d.get("sla"), str) and d["sla"], f"id {mid}: sla empty")
        _check("routing_decisions schema + matrix consistency", check_routing)

    if Path("analytics_summary.json").exists():
        def check_analytics() -> None:
            data = _load_json("analytics_summary.json")
            for key in (
                "total_messages",
                "category_distribution",
                "average_confidence",
                "risk_tier_breakdown",
                "flagged_for_human_review",
                "operator_corrections",
                "accuracy",
                "most_common_operator_correction",
            ):
                _ensure(key in data, f"missing key '{key}'")
            _ensure(data["total_messages"] == len(expected_ids), "total_messages mismatch")
            _ensure(0.0 <= data["average_confidence"] <= 1.0, "average_confidence out of range")
            for cat in data["category_distribution"]:
                _ensure(cat in CATEGORIES, f"unknown category '{cat}' in distribution")
            for tier in data["risk_tier_breakdown"]:
                _ensure(tier in RISK_LEVELS, f"unknown risk tier '{tier}' in breakdown")
        _check("analytics_summary schema", check_analytics)

    if Path("correction_store_status.json").exists():
        def check_store_status() -> None:
            data = _load_json("correction_store_status.json")
            for key in (
                "prior_corrections_loaded",
                "prior_corrections_count",
                "few_shot_eligible_count",
                "source",
                "used_for_initial_few_shot",
                "timestamp",
            ):
                _ensure(key in data, f"missing key '{key}'")
            _ensure(isinstance(data["prior_corrections_loaded"], bool), "prior_corrections_loaded not bool")
            _ensure(isinstance(data["prior_corrections_count"], int), "prior_corrections_count not int")
            datetime.fromisoformat(data["timestamp"])
        _check("correction_store_status schema", check_store_status)

    def check_triage() -> None:
        data = _load_json("triage_output.json")
        _ensure("messages" in data and isinstance(data["messages"], list), "messages array missing")
        _ensure("summary" in data and isinstance(data["summary"], dict), "summary missing")
        ids = sorted(int(m["id"]) for m in data["messages"])
        _ensure(ids == expected_ids, "triage messages id coverage mismatch")
        for m in data["messages"]:
            mid = m["id"]
            for key in ("message", "initial_classification", "risk", "draft", "reclassification", "comparison"):
                _ensure(key in m, f"id {mid}: triage missing '{key}'")
            _ensure(m["message"] == msg_text_by_id[mid], f"id {mid}: triage message text drifted")
    _check("triage_output schema + per-message coverage", check_triage)

    def check_llm_log() -> None:
        records = _load_jsonl("llm_calls.jsonl")
        _ensure(records, "llm_calls.jsonl is empty")
        required_fields = (
            "stage", "risk_tier", "message_id", "timestamp",
            "provider", "model", "prompt_hash",
            "input_artifacts", "output_artifact", "few_shot_examples_included",
        )
        for i, r in enumerate(records, 1):
            for field in required_fields:
                _ensure(field in r, f"line {i}: missing '{field}'")
            try:
                datetime.fromisoformat(r["timestamp"])
            except ValueError:
                raise AssertionError(f"line {i}: timestamp not ISO-8601")
            _ensure(isinstance(r["input_artifacts"], list), f"line {i}: input_artifacts not a list")
            _ensure(isinstance(r["output_artifact"], str), f"line {i}: output_artifact not a string")
            _ensure(isinstance(r["few_shot_examples_included"], bool),
                    f"line {i}: few_shot_examples_included not bool")
            _ensure(r["provider"] == "groq", f"line {i}: provider != 'groq'")

        stages = Counter(r["stage"] for r in records)
        _ensure(stages.get("initial_classification", 0) >= 1, "no initial_classification record")
        _ensure(stages.get("risk_scoring", 0) >= 1, "no risk_scoring record")
        _ensure(stages.get("response_drafting_batched", 0) >= 1, "no response_drafting_batched record")
        _ensure(stages.get("reclassification", 0) >= 1, "no reclassification record")

        drafts = _load_json("draft_responses.json")
        individual_count = sum(1 for d in drafts if d["drafting_mode"] == "individual")
        if individual_count:
            _ensure(stages.get("response_drafting_individual", 0) >= individual_count,
                    f"{individual_count} individual drafts but only "
                    f"{stages.get('response_drafting_individual', 0)} log records")

        if Path("response_compliance.json").exists():
            _ensure(stages.get("compliance_check", 0) >= 1, "compliance file exists but no compliance_check record")

        reclass_records = [r for r in records if r["stage"] == "reclassification"]
        if reclass_records:
            latest = reclass_records[-1]
            _ensure(latest["few_shot_examples_included"] is True,
                    "latest reclassification record must have few_shot_examples_included=true")
    _check("llm_calls.jsonl audit log: schema + required stage records", check_llm_log)

    print()
    if _FAILURES:
        print(f"FAILED: {_FAILURES} check(s) failed")
        sys.exit(1)
    print("OK: all checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
