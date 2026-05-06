import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from vocab import CATEGORIES

CORRECTIONS_PATH = Path("corrections.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_response(raw: str) -> tuple[str, str | None] | str:
    text = raw.strip()
    lower = text.lower()
    if lower in ("y", "yes"):
        return ("accepted", None)
    if lower in ("skip", "s"):
        return ("skipped", None)
    if lower.startswith("correct to:"):
        target = text.split(":", 1)[1].strip().lower()
        if target in CATEGORIES:
            return ("corrected", target)
        return f"category '{target}' is not in the allowed vocabulary"
    return "unrecognised input"


def _print_header(total: int) -> None:
    print()
    print("=" * 64)
    print("OPERATOR CORRECTION INTERFACE")
    print("=" * 64)
    print(f"Reviewing {total} classifications.")
    print(f"Allowed categories: {', '.join(sorted(CATEGORIES))}")
    print("Commands: 'y' | 'correct to: <category>' | 'skip'")
    print()


def _format_item(c: dict, message_text: str) -> str:
    flagged = " [FLAGGED]" if c.get("needs_human_review") else ""
    lines = [
        f"--- id {c['id']}{flagged} ---",
        f"  message:    {message_text}",
        f"  category:   {c['category']}  (confidence={c['confidence']:.2f})",
    ]
    if c.get("reason"):
        lines.append(f"  reason:     {c['reason']}")
    return "\n".join(lines)


def _prompt_one(c: dict, message_text: str, *, interactive: bool) -> tuple[str, str | None]:
    print(_format_item(c, message_text))
    if not interactive:
        print("  (non-interactive stdin; auto-accepting)")
        return ("accepted", None)
    while True:
        try:
            raw = input("Classification correct? (y / correct to: [category] / skip): ")
        except EOFError:
            print("  EOF on stdin; treating as 'skip'")
            return ("skipped", None)
        parsed = _parse_response(raw)
        if isinstance(parsed, tuple):
            return parsed
        print(f"  invalid input ({parsed}); try again.")


def collect_corrections(
    messages_by_id: dict[int, str],
    classifications: list[dict],
) -> list[dict]:
    interactive = sys.stdin.isatty()

    _print_header(len(classifications))
    if not interactive:
        print("(stdin is not a TTY — auto-accepting every classification)\n")

    records: list[dict] = []
    with CORRECTIONS_PATH.open("a", encoding="utf-8") as f:
        for c in classifications:
            mid = c["id"]
            message_text = messages_by_id[mid]
            action, corrected = _prompt_one(c, message_text, interactive=interactive)

            if action == "accepted":
                corrected_category: str | None = c["category"]
            elif action == "corrected":
                corrected_category = corrected
            else:
                corrected_category = None

            record = {
                "id": mid,
                "message": message_text,
                "original_category": c["category"],
                "corrected_category": corrected_category,
                "action": action,
                "timestamp": _now_iso(),
            }
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records
