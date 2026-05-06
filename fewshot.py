import json

from classify import run_classification


def build_few_shot_block(corrections: list[dict]) -> str:
    examples: list[tuple[str, str, str]] = []
    for r in corrections:
        action = r.get("action")
        if action == "corrected":
            label = r.get("corrected_category")
            tag = "operator corrected"
        elif action == "accepted":
            label = r.get("original_category")
            tag = "operator accepted"
        else:
            continue
        if not label:
            continue
        examples.append((str(r["message"]), str(label), tag))

    if not examples:
        return ""

    blocks = ["---"]
    for msg, label, tag in examples:
        blocks.append(f"Message: {json.dumps(msg, ensure_ascii=False)}")
        blocks.append(f"Category: {label}")
        blocks.append(f"Source: {tag}")
        blocks.append("---")
    return "\n".join(blocks)


def run_reclassification(
    messages: list[dict],
    few_shot_block: str,
) -> list[dict]:
    inputs = ["support_messages.json", "corrections.jsonl"] if few_shot_block else ["support_messages.json"]
    return run_classification(
        messages,
        few_shot_block=few_shot_block or None,
        stage="reclassification",
        input_artifacts=inputs,
        output_artifact="reclassified_outputs.json",
        few_shot_examples_included=True,
    )
