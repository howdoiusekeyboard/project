from enum import Enum


class PipelineState(Enum):
    INIT = "init"
    INPUTS_LOADED = "inputs_loaded"
    PRIOR_CORRECTIONS_LOADED = "prior_corrections_loaded"
    INITIAL_CLASSIFICATION_COMPLETE = "initial_classification_complete"
    RISK_SCORING_COMPLETE = "risk_scoring_complete"
    RESPONSES_DRAFTED = "responses_drafted"
    COMPLIANCE_CHECK_COMPLETE = "compliance_check_complete"
    OPERATOR_CORRECTIONS_COLLECTED = "operator_corrections_collected"
    FEW_SHOT_BLOCK_BUILT = "few_shot_block_built"
    RECLASSIFICATION_COMPLETE = "reclassification_complete"
    BEFORE_AFTER_COMPARISON_COMPLETE = "before_after_comparison_complete"
    ANALYTICS_GENERATED = "analytics_generated"
    VALIDATION_COMPLETE = "validation_complete"
    RESULTS_FINALISED = "results_finalised"


_SKIPPABLE = {
    PipelineState.PRIOR_CORRECTIONS_LOADED,
    PipelineState.COMPLIANCE_CHECK_COMPLETE,
}

_ORDER = [
    PipelineState.INIT,
    PipelineState.INPUTS_LOADED,
    PipelineState.PRIOR_CORRECTIONS_LOADED,
    PipelineState.INITIAL_CLASSIFICATION_COMPLETE,
    PipelineState.RISK_SCORING_COMPLETE,
    PipelineState.RESPONSES_DRAFTED,
    PipelineState.COMPLIANCE_CHECK_COMPLETE,
    PipelineState.OPERATOR_CORRECTIONS_COLLECTED,
    PipelineState.FEW_SHOT_BLOCK_BUILT,
    PipelineState.RECLASSIFICATION_COMPLETE,
    PipelineState.BEFORE_AFTER_COMPARISON_COMPLETE,
    PipelineState.ANALYTICS_GENERATED,
    PipelineState.VALIDATION_COMPLETE,
    PipelineState.RESULTS_FINALISED,
]


def _build_allowed() -> dict:
    allowed: dict = {}
    n = len(_ORDER)
    for i, s in enumerate(_ORDER):
        targets = set()
        j = i + 1
        if j < n:
            targets.add(_ORDER[j])
            if _ORDER[j] in _SKIPPABLE and j + 1 < n:
                targets.add(_ORDER[j + 1])
                if _ORDER[j + 1] in _SKIPPABLE and j + 2 < n:
                    targets.add(_ORDER[j + 2])
        allowed[s] = frozenset(targets)
    return allowed


ALLOWED_TRANSITIONS = _build_allowed()


class StateMachine:
    def __init__(self, initial: PipelineState = PipelineState.INIT):
        self._current = initial

    @property
    def current(self) -> PipelineState:
        return self._current

    def advance(self, next_state: PipelineState) -> None:
        allowed = ALLOWED_TRANSITIONS.get(self._current, frozenset())
        assert next_state in allowed, (
            f"Illegal transition {self._current.name} -> {next_state.name}; "
            f"allowed: {[s.name for s in allowed]}"
        )
        self._current = next_state
