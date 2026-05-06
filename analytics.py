from collections import Counter


def _operator_label(correction: dict | None) -> tuple[str | None, str | None]:
    if not correction:
        return None, None
    action = correction.get("action")
    if action == "accepted":
        return correction.get("original_category"), action
    if action == "corrected":
        return correction.get("corrected_category"), action
    return None, action


def build_comparison(
    initial: list[dict],
    corrections: list[dict],
    reclassified: list[dict],
) -> list[dict]:
    initial_by_id = {c["id"]: c for c in initial}
    reclass_by_id = {c["id"]: c for c in reclassified}
    corr_by_id = {c["id"]: c for c in corrections}

    items: list[dict] = []
    for mid in sorted(initial_by_id):
        before = initial_by_id[mid]
        after = reclass_by_id.get(mid)
        if after is None:
            raise ValueError(f"reclassification missing for id {mid}")
        correction = corr_by_id.get(mid)
        operator_label, operator_action = _operator_label(correction)

        if operator_action == "corrected":
            moved_toward = (after["category"] == operator_label) and (before["category"] != operator_label)
        elif operator_action == "accepted":
            moved_toward = after["category"] == operator_label
        else:
            moved_toward = None

        items.append({
            "id": mid,
            "original_category": before["category"],
            "operator_action": operator_action,
            "operator_label": operator_label,
            "reclassified_category": after["category"],
            "category_changed": before["category"] != after["category"],
            "moved_toward_correction": moved_toward,
            "confidence_before": before["confidence"],
            "confidence_after": after["confidence"],
            "confidence_delta": round(after["confidence"] - before["confidence"], 4),
        })
    return items


def compute_accuracy_delta(comparison: list[dict]) -> dict:
    labeled = [c for c in comparison if c["operator_action"] in ("accepted", "corrected")]
    if not labeled:
        return {
            "labeled_count": 0,
            "accuracy_before": None,
            "accuracy_after": None,
            "accuracy_delta": None,
            "note": "no operator-provided labels (all skipped)",
        }
    n = len(labeled)
    before_correct = sum(1 for c in labeled if c["original_category"] == c["operator_label"])
    after_correct = sum(1 for c in labeled if c["reclassified_category"] == c["operator_label"])
    before = before_correct / n
    after = after_correct / n
    return {
        "labeled_count": n,
        "before_correct": before_correct,
        "after_correct": after_correct,
        "accuracy_before": round(before, 4),
        "accuracy_after": round(after, 4),
        "accuracy_delta": round(after - before, 4),
    }


def assemble_triage_output(
    messages: list[dict],
    classifications: list[dict],
    risk_scores: list[dict],
    drafts: list[dict],
    corrections: list[dict],
    reclassified: list[dict],
    comparison: list[dict],
    accuracy: dict,
    few_shot_block: str,
    compliance_results: list[dict] | None = None,
    analytics_summary: dict | None = None,
) -> dict:
    msg_by_id = {int(m["id"]): m["message"] for m in messages}
    cls_by_id = {c["id"]: c for c in classifications}
    risk_by_id = {r["id"]: r for r in risk_scores}
    draft_by_id = {d["id"]: d for d in drafts}
    corr_by_id = {c["id"]: c for c in corrections}
    reclass_by_id = {r["id"]: r for r in reclassified}
    cmp_by_id = {c["id"]: c for c in comparison}
    comp_by_id = {c["id"]: c for c in (compliance_results or [])}

    items = []
    for mid in sorted(msg_by_id):
        items.append({
            "id": mid,
            "message": msg_by_id[mid],
            "initial_classification": cls_by_id[mid],
            "risk": risk_by_id[mid],
            "draft": draft_by_id[mid],
            "compliance": comp_by_id.get(mid),
            "operator_correction": corr_by_id.get(mid),
            "reclassification": reclass_by_id[mid],
            "comparison": cmp_by_id[mid],
        })

    summary = {"total_messages": len(items), "accuracy": accuracy}
    if analytics_summary is not None:
        summary["analytics"] = analytics_summary

    return {
        "messages": items,
        "few_shot_block": few_shot_block,
        "summary": summary,
    }


def build_analytics_summary(
    classifications: list[dict],
    risk_scores: list[dict],
    corrections: list[dict],
    accuracy: dict,
    compliance_results: list[dict] | None = None,
) -> dict:
    n = len(classifications)
    cat_dist = Counter(c["category"] for c in classifications)
    risk_dist = Counter(r["risk_level"] for r in risk_scores)
    avg_conf = sum(c["confidence"] for c in classifications) / n if n else 0.0
    flagged = sum(1 for c in classifications if c["needs_human_review"])

    action_counts = Counter(c["action"] for c in corrections)
    correction_pairs: Counter = Counter()
    for c in corrections:
        if c["action"] == "corrected":
            pair = f"{c['original_category']} -> {c['corrected_category']}"
            correction_pairs[pair] += 1
    most_common_correction = None
    if correction_pairs:
        pair, count = correction_pairs.most_common(1)[0]
        most_common_correction = {"transition": pair, "count": count}

    summary = {
        "total_messages": n,
        "category_distribution": dict(cat_dist),
        "average_confidence": round(avg_conf, 4),
        "risk_tier_breakdown": dict(risk_dist),
        "flagged_for_human_review": flagged,
        "operator_corrections": {
            "accepted": action_counts.get("accepted", 0),
            "corrected": action_counts.get("corrected", 0),
            "skipped": action_counts.get("skipped", 0),
        },
        "accuracy": accuracy,
        "most_common_operator_correction": most_common_correction,
    }

    if compliance_results is not None:
        passed = sum(1 for r in compliance_results if r["passed"])
        violation_dist: Counter = Counter()
        for r in compliance_results:
            for v in r["violations"]:
                if v != "none":
                    violation_dist[v] += 1
        summary["compliance"] = {
            "checked": len(compliance_results),
            "passed": passed,
            "failed": len(compliance_results) - passed,
            "violation_distribution": dict(violation_dist),
        }

    return summary


def format_analytics_summary(summary: dict) -> str:
    lines = ["", "=" * 64, "ANALYTICS SUMMARY", "=" * 64]
    lines.append(f"total messages:                  {summary['total_messages']}")
    lines.append(f"average classification confidence: {summary['average_confidence']:.3f}")
    lines.append(f"flagged for human review:        {summary['flagged_for_human_review']}")

    lines.append("")
    lines.append("category distribution:")
    for cat, n in sorted(summary["category_distribution"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {cat:<14} {n}")

    lines.append("")
    lines.append("risk tier breakdown:")
    for tier in ("low", "medium", "high", "critical"):
        n = summary["risk_tier_breakdown"].get(tier, 0)
        if n:
            lines.append(f"  {tier:<8} {n}")

    lines.append("")
    oc = summary["operator_corrections"]
    lines.append(
        f"operator: accepted={oc['accepted']}, corrected={oc['corrected']}, skipped={oc['skipped']}"
    )
    if summary.get("most_common_operator_correction"):
        mc = summary["most_common_operator_correction"]
        lines.append(f"most common correction: {mc['transition']} (x{mc['count']})")

    acc = summary["accuracy"]
    if acc.get("labeled_count"):
        lines.append(
            f"accuracy: before={acc['accuracy_before']:.3f} "
            f"after={acc['accuracy_after']:.3f} "
            f"delta={acc['accuracy_delta']:+.3f} "
            f"(n={acc['labeled_count']})"
        )
    else:
        lines.append("accuracy delta: not computable (no operator-provided labels)")

    if summary.get("compliance"):
        c = summary["compliance"]
        lines.append("")
        lines.append(f"compliance: passed={c['passed']}/{c['checked']}, failed={c['failed']}")
        if c["violation_distribution"]:
            lines.append("violation distribution:")
            for vtype, n in sorted(c["violation_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"  {vtype:<32} {n}")

    lines.append("=" * 64)
    return "\n".join(lines)
