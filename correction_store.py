import json
from datetime import datetime, timezone
from pathlib import Path

CORRECTIONS_PATH = Path("corrections.jsonl")
STATUS_PATH = Path("correction_store_status.json")


def load_prior_corrections() -> list[dict]:
    if not CORRECTIONS_PATH.exists():
        return []
    records: list[dict] = []
    with CORRECTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"corrupt line in {CORRECTIONS_PATH}: {e}")
    return records


def write_status(
    prior_records: list[dict],
    *,
    used_for_initial_few_shot: bool,
) -> dict:
    eligible = [r for r in prior_records if r.get("action") in ("accepted", "corrected")]
    payload = {
        "prior_corrections_loaded": bool(prior_records),
        "prior_corrections_count": len(prior_records),
        "few_shot_eligible_count": len(eligible),
        "source": str(CORRECTIONS_PATH),
        "used_for_initial_few_shot": used_for_initial_few_shot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    STATUS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
