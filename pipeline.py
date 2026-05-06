import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from analytics import (
    assemble_triage_output,
    build_analytics_summary,
    build_comparison,
    compute_accuracy_delta,
    format_analytics_summary,
)
from classify import run_classification
from compliance import run_compliance_check
from correction_store import load_prior_corrections, write_status
from draft import run_drafting
from fewshot import build_few_shot_block, run_reclassification
from operator_review import collect_corrections
from risk import run_risk_scoring
from routing import run_routing
from state import PipelineState, StateMachine


def main() -> None:
    load_dotenv()
    sm = StateMachine(PipelineState.INIT)

    messages = json.loads(Path("support_messages.json").read_text(encoding="utf-8"))
    sm.advance(PipelineState.INPUTS_LOADED)
    print(f"loaded {len(messages)} messages")

    prior_corrections = load_prior_corrections()
    prior_few_shot = build_few_shot_block(prior_corrections)
    store_status = write_status(prior_corrections, used_for_initial_few_shot=bool(prior_few_shot))
    if prior_corrections:
        sm.advance(PipelineState.PRIOR_CORRECTIONS_LOADED)
        eligible = store_status["few_shot_eligible_count"]
        print(
            f"longitudinal store: loaded {len(prior_corrections)} prior record(s); "
            f"{eligible} eligible for few-shot "
            f"({'used' if prior_few_shot else 'not used'} for initial classification)"
        )
    else:
        print("longitudinal store: no prior corrections file found; initial classification runs zero-shot")

    classifications = run_classification(messages, few_shot_block=prior_few_shot or None)
    Path("initial_classifications.json").write_text(
        json.dumps(classifications, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    sm.advance(PipelineState.INITIAL_CLASSIFICATION_COMPLETE)
    flagged = sum(1 for c in classifications if c["needs_human_review"])
    print(f"classified {len(classifications)} messages ({flagged} flagged for human review)")

    messages_by_id = {int(m["id"]): m["message"] for m in messages}
    risk_scores = run_risk_scoring(messages_by_id, classifications)
    Path("risk_scores.json").write_text(
        json.dumps(risk_scores, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    sm.advance(PipelineState.RISK_SCORING_COMPLETE)
    dist = Counter(r["risk_level"] for r in risk_scores)
    dist_str = ", ".join(f"{k}={dist[k]}" for k in ("low", "medium", "high", "critical") if dist[k])
    print(f"risk-scored {len(risk_scores)} messages: {dist_str}")

    drafts = run_drafting(messages_by_id, risk_scores)
    Path("draft_responses.json").write_text(
        json.dumps(drafts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    sm.advance(PipelineState.RESPONSES_DRAFTED)
    mode_counts = Counter(d["drafting_mode"] for d in drafts)
    print(
        f"drafted {len(drafts)} responses "
        f"(batched={mode_counts.get('batched', 0)}, individual={mode_counts.get('individual', 0)})"
    )

    corrections = collect_corrections(messages_by_id, classifications)
    sm.advance(PipelineState.OPERATOR_CORRECTIONS_COLLECTED)
    action_counts = Counter(r["action"] for r in corrections)
    print(
        f"collected {len(corrections)} operator records "
        f"(accepted={action_counts.get('accepted', 0)}, "
        f"corrected={action_counts.get('corrected', 0)}, "
        f"skipped={action_counts.get('skipped', 0)})"
    )

    combined_corrections = prior_corrections + corrections
    few_shot_block = build_few_shot_block(combined_corrections)
    example_count = sum(
        1 for r in combined_corrections if r.get("action") in ("accepted", "corrected")
    )
    sm.advance(PipelineState.FEW_SHOT_BLOCK_BUILT)
    print(
        f"built few-shot block ({example_count} operator-validated examples; "
        f"{len(prior_corrections)} from prior runs + {len(corrections)} from this run)"
    )

    reclassified = run_reclassification(messages, few_shot_block)
    Path("reclassified_outputs.json").write_text(
        json.dumps(reclassified, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    sm.advance(PipelineState.RECLASSIFICATION_COMPLETE)
    flagged_after = sum(1 for c in reclassified if c["needs_human_review"])
    print(f"reclassified {len(reclassified)} messages ({flagged_after} flagged after few-shot pass)")

    comparison = build_comparison(classifications, corrections, reclassified)
    accuracy = compute_accuracy_delta(comparison)
    sm.advance(PipelineState.BEFORE_AFTER_COMPARISON_COMPLETE)
    if accuracy["labeled_count"]:
        print(
            f"accuracy on {accuracy['labeled_count']} operator-labeled messages: "
            f"before={accuracy['accuracy_before']:.3f}, "
            f"after={accuracy['accuracy_after']:.3f}, "
            f"delta={accuracy['accuracy_delta']:+.3f}"
        )
    else:
        print("accuracy delta: not computable (no operator-provided labels)")

    compliance_results = run_compliance_check(drafts)
    Path("response_compliance.json").write_text(
        json.dumps(compliance_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failed_count = sum(1 for r in compliance_results if not r["passed"])
    print(
        f"compliance check: {len(compliance_results) - failed_count}/{len(compliance_results)} passed, "
        f"{failed_count} failed (flagged for operator review)"
    )

    analytics_summary = build_analytics_summary(
        classifications,
        risk_scores,
        corrections,
        accuracy,
        compliance_results=compliance_results,
    )
    Path("analytics_summary.json").write_text(
        json.dumps(analytics_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    triage = assemble_triage_output(
        messages,
        classifications,
        risk_scores,
        drafts,
        corrections,
        reclassified,
        comparison,
        accuracy,
        few_shot_block,
        compliance_results=compliance_results,
        analytics_summary=analytics_summary,
    )
    Path("triage_output.json").write_text(
        json.dumps(triage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sm.advance(PipelineState.ANALYTICS_GENERATED)
    print(format_analytics_summary(analytics_summary))

    routing_decisions, critical_alerts = run_routing(messages_by_id, reclassified, risk_scores)
    Path("routing_decisions.json").write_text(
        json.dumps(routing_decisions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"routing: {len(routing_decisions)} decisions written; "
        f"{len(critical_alerts)} critical alert(s) printed"
    )


if __name__ == "__main__":
    main()
