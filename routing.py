ROUTING_MATRIX: list[dict] = [
    {
        "category": "payments",
        "team": "Payments Support",
        "sla": "4 hours",
        "escalation_path": "Payments Lead -> Finance Operations",
    },
    {
        "category": "technical",
        "team": "Technical Support",
        "sla": "8 hours",
        "escalation_path": "Tech Lead -> Engineering On-Call",
    },
    {
        "category": "compliance",
        "team": "Compliance & Regulatory",
        "sla": "24 hours",
        "escalation_path": "Compliance Officer -> Legal",
    },
    {
        "category": "account",
        "team": "Account Operations",
        "sla": "8 hours",
        "escalation_path": "Account Manager -> Customer Success Lead",
    },
    {
        "category": "product_query",
        "team": "Product Support",
        "sla": "12 hours",
        "escalation_path": "Product Specialist -> Product Manager",
    },
    {
        "category": "escalation",
        "team": "Senior Escalation Desk",
        "sla": "1 hour",
        "escalation_path": "Escalation Manager -> Head of Customer Operations",
    },
]

ROUTING_BY_CATEGORY: dict[str, dict] = {r["category"]: r for r in ROUTING_MATRIX}

_RISK_SLA_OVERRIDE = {
    "critical": "30 minutes",
    "high": "2 hours",
}


def route_one(mid: int, category: str, risk_level: str) -> dict:
    if category not in ROUTING_BY_CATEGORY:
        raise ValueError(f"no routing rule for category '{category}' (id {mid})")
    rule = ROUTING_BY_CATEGORY[category]
    sla = _RISK_SLA_OVERRIDE.get(risk_level, rule["sla"])
    return {
        "id": mid,
        "category": category,
        "risk_level": risk_level,
        "team": rule["team"],
        "sla": sla,
        "escalation_path": rule["escalation_path"],
    }


def format_critical_alert(decision: dict, message_text: str, triggers: list[str]) -> str:
    lines = [
        "",
        ":rotating_light: *CRITICAL ESCALATION* :rotating_light:",
        f"> *Message #{decision['id']}*  |  category: `{decision['category']}`  |  risk: `{decision['risk_level']}`",
        f"> *Route to:* {decision['team']}   *SLA:* {decision['sla']}",
        f"> *Escalation path:* {decision['escalation_path']}",
    ]
    if triggers:
        lines.append("> *Triggering criteria:* " + ", ".join(triggers))
    lines.append(f"> *Message:* {message_text}")
    lines.append("")
    return "\n".join(lines)


def run_routing(
    messages_by_id: dict[int, str],
    final_classifications: list[dict],
    risk_scores: list[dict],
) -> tuple[list[dict], list[str]]:
    risk_by_id = {r["id"]: r for r in risk_scores}
    decisions: list[dict] = []
    alerts: list[str] = []

    for c in final_classifications:
        mid = c["id"]
        risk_record = risk_by_id.get(mid)
        if risk_record is None:
            raise ValueError(f"no risk score for id {mid}")
        decision = route_one(mid, c["category"], risk_record["risk_level"])
        decisions.append(decision)

        if risk_record["risk_level"] == "critical":
            alert = format_critical_alert(
                decision,
                messages_by_id[mid],
                risk_record.get("triggering_criteria", []),
            )
            alerts.append(alert)
            print(alert)

    decisions.sort(key=lambda d: d["id"])
    return decisions, alerts
