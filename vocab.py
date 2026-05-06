CATEGORIES = frozenset({
    "payments",
    "technical",
    "compliance",
    "account",
    "product_query",
    "escalation",
})

RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})

COMPLIANCE_VIOLATIONS = frozenset({
    "promise",
    "specific_timeline",
    "liability_admission",
    "non_compliant_financial_claim",
    "none",
})

CORRECTION_ACTIONS = frozenset({
    "accepted",
    "corrected",
    "skipped",
})

NO_COMMITMENTS_INSTRUCTION = "Do not include commitments, guarantees, specific timelines, admissions of liability, or promises of refund or reversal."
