"""Tests for the FeedbackEngine module.

Tests every code path in feedback.py: enums, dataclasses, FeedbackEngine
methods, categorization, escalation logic, formatting, and edge cases.
"""

from __future__ import annotations

import pytest

from codebase_intelligence.constraints import (
    Constraint,
    ConstraintScope,
    ConstraintSet,
    ConstraintSeverity,
    ConstraintViolation,
    ErrorFormatConstraint,
    MustNotCrossConstraint,
    MustUseConstraint,
    NamingConstraint,
)
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import SemanticNode
from codebase_intelligence.feedback import (
    EscalationLevel,
    FeedbackEngine,
    RefinementContext,
    ViolationCategory,
    ViolationDiagnosis,
)
from codebase_intelligence.nodes import NodeType


# ── Helpers ───────────────────────────────────────────────────────────────


def _violation(
    name: str = "test_constraint",
    message: str = "Something wrong",
    severity: ConstraintSeverity = ConstraintSeverity.ERROR,
    file_path: str | None = "app.py",
    line_number: int | None = 10,
    node_id: str | None = "node_1",
    suggestion: str | None = "Fix it",
) -> ConstraintViolation:
    return ConstraintViolation(
        constraint_name=name,
        message=message,
        severity=severity,
        file_path=file_path,
        line_number=line_number,
        node_id=node_id,
        suggestion=suggestion,
    )


def _naming_constraint(name: str = "snake_case") -> NamingConstraint:
    return NamingConstraint(
        name=name,
        description="Snake case",
        pattern=r"^[a-z_][a-z0-9_]*$",
        node_types=[NodeType.FUNCTION],
    )


def _must_use_constraint(name: str = "require_docs") -> MustUseConstraint:
    return MustUseConstraint(
        name=name,
        description="Require docstrings",
        requirement="docstring",
        node_types=[NodeType.FUNCTION],
    )


def _must_not_cross_constraint(name: str = "boundary") -> MustNotCrossConstraint:
    return MustNotCrossConstraint(
        name=name,
        description="No crossing",
        source_pattern=r".*services.*",
        forbidden_targets=[r".*controllers.*"],
    )


def _error_format_constraint(name: str = "exc_naming") -> ErrorFormatConstraint:
    return ErrorFormatConstraint(
        name=name,
        description="Exception naming",
        exception_pattern=r"^[A-Z].*Error$",
    )


def _constraint_set(*constraints) -> ConstraintSet:
    return ConstraintSet(
        name="test_set",
        description="Test",
        constraints=list(constraints),
    )


# ── ViolationCategory Enum ───────────────────────────────────────────────


class TestViolationCategory:
    def test_all_values(self) -> None:
        values = {v.value for v in ViolationCategory}
        assert values == {"naming", "structural", "boundary", "error_format"}

    def test_string_enum(self) -> None:
        assert ViolationCategory.NAMING == "naming"
        assert ViolationCategory.STRUCTURAL == "structural"
        assert ViolationCategory.BOUNDARY == "boundary"
        assert ViolationCategory.ERROR_FORMAT == "error_format"

    def test_from_value(self) -> None:
        assert ViolationCategory("naming") == ViolationCategory.NAMING
        assert ViolationCategory("boundary") == ViolationCategory.BOUNDARY

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            ViolationCategory("invalid")


# ── EscalationLevel Enum ─────────────────────────────────────────────────


class TestEscalationLevel:
    def test_all_values(self) -> None:
        values = {v.value for v in EscalationLevel}
        assert values == {"hint", "explicit", "rewrite"}

    def test_string_enum(self) -> None:
        assert EscalationLevel.HINT == "hint"
        assert EscalationLevel.EXPLICIT == "explicit"
        assert EscalationLevel.REWRITE == "rewrite"

    def test_from_value(self) -> None:
        assert EscalationLevel("hint") == EscalationLevel.HINT
        assert EscalationLevel("rewrite") == EscalationLevel.REWRITE

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            EscalationLevel("panic")


# ── ViolationDiagnosis ───────────────────────────────────────────────────


class TestViolationDiagnosis:
    def test_basic_creation(self) -> None:
        v = _violation()
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="Bad name",
            suggestion="Fix name",
            escalation_level=EscalationLevel.HINT,
        )
        assert d.violation is v
        assert d.category == ViolationCategory.NAMING
        assert d.root_cause == "Bad name"
        assert d.suggestion == "Fix name"
        assert d.escalation_level == EscalationLevel.HINT
        assert d.confidence == 1.0

    def test_custom_confidence(self) -> None:
        v = _violation()
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.STRUCTURAL,
            root_cause="Missing docstring",
            suggestion="Add docstring",
            escalation_level=EscalationLevel.EXPLICIT,
            confidence=0.75,
        )
        assert d.confidence == 0.75

    def test_frozen(self) -> None:
        v = _violation()
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="r",
            suggestion="s",
            escalation_level=EscalationLevel.HINT,
        )
        with pytest.raises(AttributeError):
            d.confidence = 0.5  # type: ignore[misc]

    def test_all_categories(self) -> None:
        v = _violation()
        for cat in ViolationCategory:
            d = ViolationDiagnosis(
                violation=v,
                category=cat,
                root_cause="cause",
                suggestion="fix",
                escalation_level=EscalationLevel.HINT,
            )
            assert d.category == cat


# ── RefinementContext ────────────────────────────────────────────────────


class TestRefinementContext:
    def test_basic_creation(self) -> None:
        ctx = RefinementContext(original_request="Add endpoint")
        assert ctx.original_request == "Add endpoint"
        assert ctx.violations == []
        assert ctx.diagnoses == []
        assert ctx.attempt == 1
        assert ctx.max_attempts == 3
        assert ctx.history == []

    def test_full_creation(self) -> None:
        v = _violation()
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="r",
            suggestion="s",
            escalation_level=EscalationLevel.HINT,
        )
        ctx = RefinementContext(
            original_request="task",
            violations=[v],
            diagnoses=[d],
            attempt=2,
            max_attempts=5,
            history=[("code", [v])],
        )
        assert len(ctx.violations) == 1
        assert len(ctx.diagnoses) == 1
        assert ctx.attempt == 2
        assert ctx.max_attempts == 5
        assert len(ctx.history) == 1

    def test_mutable(self) -> None:
        ctx = RefinementContext(original_request="task")
        ctx.attempt = 3
        assert ctx.attempt == 3


# ── FeedbackEngine.diagnose ─────────────────────────────────────────────


class TestFeedbackEngineDiagnose:
    def test_empty_violations(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        diagnoses = engine.diagnose([], cs)
        assert diagnoses == []

    def test_naming_violation(self) -> None:
        engine = FeedbackEngine()
        nc = _naming_constraint(name="snake_case")
        cs = _constraint_set(nc)
        v = _violation(name="snake_case", message="Bad name")
        diagnoses = engine.diagnose([v], cs)
        assert len(diagnoses) == 1
        assert diagnoses[0].category == ViolationCategory.NAMING

    def test_must_use_violation(self) -> None:
        engine = FeedbackEngine()
        mu = _must_use_constraint(name="require_docs")
        cs = _constraint_set(mu)
        v = _violation(name="require_docs", message="Missing docstring")
        diagnoses = engine.diagnose([v], cs)
        assert len(diagnoses) == 1
        assert diagnoses[0].category == ViolationCategory.STRUCTURAL

    def test_boundary_violation(self) -> None:
        engine = FeedbackEngine()
        mnc = _must_not_cross_constraint(name="boundary")
        cs = _constraint_set(mnc)
        v = _violation(name="boundary", message="Cannot import")
        diagnoses = engine.diagnose([v], cs)
        assert len(diagnoses) == 1
        assert diagnoses[0].category == ViolationCategory.BOUNDARY

    def test_error_format_violation(self) -> None:
        engine = FeedbackEngine()
        ef = _error_format_constraint(name="exc_naming")
        cs = _constraint_set(ef)
        v = _violation(name="exc_naming", message="Bad exception")
        diagnoses = engine.diagnose([v], cs)
        assert len(diagnoses) == 1
        assert diagnoses[0].category == ViolationCategory.ERROR_FORMAT

    def test_unknown_constraint_infers_category(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="nonexistent", message="does not match pattern")
        diagnoses = engine.diagnose([v], cs)
        assert len(diagnoses) == 1
        assert diagnoses[0].category == ViolationCategory.NAMING

    def test_multiple_violations(self) -> None:
        engine = FeedbackEngine()
        nc = _naming_constraint(name="snake_case")
        mu = _must_use_constraint(name="require_docs")
        cs = _constraint_set(nc, mu)
        v1 = _violation(name="snake_case", message="Bad name")
        v2 = _violation(name="require_docs", message="Missing")
        diagnoses = engine.diagnose([v1, v2], cs)
        assert len(diagnoses) == 2
        assert diagnoses[0].category == ViolationCategory.NAMING
        assert diagnoses[1].category == ViolationCategory.STRUCTURAL

    def test_confidence_from_constraint(self) -> None:
        engine = FeedbackEngine()
        nc = NamingConstraint(
            name="custom",
            description="Custom",
            pattern=r".*",
            node_types=[NodeType.FUNCTION],
            confidence=0.8,
        )
        cs = _constraint_set(nc)
        v = _violation(name="custom")
        diagnoses = engine.diagnose([v], cs)
        assert diagnoses[0].confidence == 0.8

    def test_confidence_default_when_constraint_not_found(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="missing")
        diagnoses = engine.diagnose([v], cs)
        assert diagnoses[0].confidence == 1.0

    def test_suggestion_from_violation(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(suggestion="Rename to snake_case")
        diagnoses = engine.diagnose([v], cs)
        assert diagnoses[0].suggestion == "Rename to snake_case"

    def test_suggestion_fallback_naming(self) -> None:
        engine = FeedbackEngine()
        nc = _naming_constraint(name="nc")
        cs = _constraint_set(nc)
        v = _violation(name="nc", suggestion=None, message="bad")
        diagnoses = engine.diagnose([v], cs)
        assert "naming pattern" in diagnoses[0].suggestion.lower()

    def test_suggestion_fallback_structural(self) -> None:
        engine = FeedbackEngine()
        mu = _must_use_constraint(name="mu")
        cs = _constraint_set(mu)
        v = _violation(name="mu", suggestion=None, message="missing")
        diagnoses = engine.diagnose([v], cs)
        assert "required construct" in diagnoses[0].suggestion.lower()

    def test_suggestion_fallback_boundary(self) -> None:
        engine = FeedbackEngine()
        mnc = _must_not_cross_constraint(name="b")
        cs = _constraint_set(mnc)
        v = _violation(name="b", suggestion=None)
        diagnoses = engine.diagnose([v], cs)
        assert "import" in diagnoses[0].suggestion.lower()

    def test_suggestion_fallback_error_format(self) -> None:
        engine = FeedbackEngine()
        ef = _error_format_constraint(name="ef")
        cs = _constraint_set(ef)
        v = _violation(name="ef", suggestion=None)
        diagnoses = engine.diagnose([v], cs)
        assert "exception" in diagnoses[0].suggestion.lower()


# ── FeedbackEngine._categorize ──────────────────────────────────────────


class TestFeedbackEngineCategorize:
    def test_infer_naming_from_message(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="unknown", message="does not match pattern X")
        diag = engine.diagnose([v], cs)
        assert diag[0].category == ViolationCategory.NAMING

    def test_infer_boundary_from_message_import(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="unknown", message="cannot import from module")
        diag = engine.diagnose([v], cs)
        assert diag[0].category == ViolationCategory.BOUNDARY

    def test_infer_boundary_from_message_boundary(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="unknown", message="crossed boundary")
        diag = engine.diagnose([v], cs)
        assert diag[0].category == ViolationCategory.BOUNDARY

    def test_infer_error_format_from_message_exception(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="unknown", message="exception class bad")
        diag = engine.diagnose([v], cs)
        assert diag[0].category == ViolationCategory.ERROR_FORMAT

    def test_infer_error_format_from_message_inherit(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="unknown", message="must inherit from base")
        diag = engine.diagnose([v], cs)
        assert diag[0].category == ViolationCategory.ERROR_FORMAT

    def test_infer_error_format_from_message_error(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="unknown", message="error class issue")
        diag = engine.diagnose([v], cs)
        assert diag[0].category == ViolationCategory.ERROR_FORMAT

    def test_infer_structural_fallback(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="unknown", message="something is wrong here")
        diag = engine.diagnose([v], cs)
        assert diag[0].category == ViolationCategory.STRUCTURAL

    def test_infer_naming_from_naming_keyword(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(name="unknown", message="naming convention violated")
        diag = engine.diagnose([v], cs)
        assert diag[0].category == ViolationCategory.NAMING


# ── FeedbackEngine._escalation_for_attempt ───────────────────────────────


class TestEscalationForAttempt:
    def test_first_attempt_hint(self) -> None:
        engine = FeedbackEngine()
        v = _violation()
        cs = _constraint_set()
        ctx = RefinementContext(
            original_request="task",
            violations=[v],
            diagnoses=engine.diagnose([v], cs),
            attempt=1,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "review and fix" in prompt.lower()

    def test_middle_attempt_explicit(self) -> None:
        engine = FeedbackEngine()
        v = _violation()
        cs = _constraint_set()
        ctx = RefinementContext(
            original_request="task",
            violations=[v],
            diagnoses=engine.diagnose([v], cs),
            attempt=2,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "fix all" in prompt.lower()

    def test_final_attempt_rewrite(self) -> None:
        engine = FeedbackEngine()
        v = _violation()
        cs = _constraint_set()
        ctx = RefinementContext(
            original_request="task",
            violations=[v],
            diagnoses=engine.diagnose([v], cs),
            attempt=3,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "final attempt" in prompt.lower()

    def test_single_attempt_rewrite(self) -> None:
        engine = FeedbackEngine()
        v = _violation()
        cs = _constraint_set()
        ctx = RefinementContext(
            original_request="task",
            violations=[v],
            diagnoses=engine.diagnose([v], cs),
            attempt=1,
            max_attempts=1,
        )
        prompt = engine.build_refinement(ctx)
        assert "final attempt" in prompt.lower()

    def test_attempt_zero_hint(self) -> None:
        engine = FeedbackEngine()
        v = _violation()
        cs = _constraint_set()
        ctx = RefinementContext(
            original_request="task",
            violations=[v],
            diagnoses=engine.diagnose([v], cs),
            attempt=0,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "review and fix" in prompt.lower()


# ── FeedbackEngine.build_refinement ──────────────────────────────────────


class TestBuildRefinement:
    def test_hint_contains_violation_messages(self) -> None:
        engine = FeedbackEngine()
        v = _violation(message="BadFunc violates naming")
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="r",
            suggestion="s",
            escalation_level=EscalationLevel.HINT,
        )
        ctx = RefinementContext(
            original_request="task",
            diagnoses=[d],
            attempt=1,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "BadFunc violates naming" in prompt

    def test_explicit_contains_formatted_diagnoses(self) -> None:
        engine = FeedbackEngine()
        v = _violation(message="Missing docstring")
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.STRUCTURAL,
            root_cause="No docstring",
            suggestion="Add docstring",
            escalation_level=EscalationLevel.EXPLICIT,
        )
        ctx = RefinementContext(
            original_request="task",
            diagnoses=[d],
            attempt=2,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "STRUCTURAL" in prompt
        assert "Add docstring" in prompt

    def test_rewrite_contains_task(self) -> None:
        engine = FeedbackEngine()
        v = _violation()
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="r",
            suggestion="s",
            escalation_level=EscalationLevel.REWRITE,
        )
        ctx = RefinementContext(
            original_request="Add user endpoint",
            diagnoses=[d],
            attempt=3,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "Add user endpoint" in prompt

    def test_rewrite_with_history(self) -> None:
        engine = FeedbackEngine()
        v = _violation()
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="r",
            suggestion="s",
            escalation_level=EscalationLevel.REWRITE,
        )
        ctx = RefinementContext(
            original_request="task",
            diagnoses=[d],
            attempt=3,
            max_attempts=3,
            history=[("code1", [v]), ("code2", [v])],
        )
        prompt = engine.build_refinement(ctx)
        assert "2 previous attempt(s)" in prompt

    def test_rewrite_without_history(self) -> None:
        engine = FeedbackEngine()
        v = _violation()
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="r",
            suggestion="s",
            escalation_level=EscalationLevel.REWRITE,
        )
        ctx = RefinementContext(
            original_request="task",
            diagnoses=[d],
            attempt=3,
            max_attempts=3,
            history=[],
        )
        prompt = engine.build_refinement(ctx)
        assert "previous attempt" not in prompt

    def test_empty_diagnoses_hint(self) -> None:
        engine = FeedbackEngine()
        ctx = RefinementContext(
            original_request="task",
            diagnoses=[],
            attempt=1,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "Regenerate" in prompt

    def test_empty_diagnoses_explicit(self) -> None:
        engine = FeedbackEngine()
        ctx = RefinementContext(
            original_request="task",
            diagnoses=[],
            attempt=2,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "Apply each fix" in prompt

    def test_empty_diagnoses_rewrite(self) -> None:
        engine = FeedbackEngine()
        ctx = RefinementContext(
            original_request="task",
            diagnoses=[],
            attempt=3,
            max_attempts=3,
        )
        prompt = engine.build_refinement(ctx)
        assert "Rewrite" in prompt


# ── FeedbackEngine._format_diagnosis ─────────────────────────────────────


class TestFormatDiagnosis:
    def test_format_includes_category(self) -> None:
        engine = FeedbackEngine()
        v = _violation(message="Bad name")
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="Name wrong",
            suggestion="Fix name",
            escalation_level=EscalationLevel.EXPLICIT,
        )
        formatted = engine._format_diagnosis(d)
        assert "[NAMING]" in formatted
        assert "Bad name" in formatted
        assert "Name wrong" in formatted
        assert "Fix name" in formatted

    def test_format_all_categories(self) -> None:
        engine = FeedbackEngine()
        for cat in ViolationCategory:
            v = _violation(message="msg")
            d = ViolationDiagnosis(
                violation=v,
                category=cat,
                root_cause="cause",
                suggestion="fix",
                escalation_level=EscalationLevel.HINT,
            )
            formatted = engine._format_diagnosis(d)
            assert f"[{cat.value.upper()}]" in formatted

    def test_format_empty_root_cause(self) -> None:
        engine = FeedbackEngine()
        v = _violation(message="msg")
        d = ViolationDiagnosis(
            violation=v,
            category=ViolationCategory.NAMING,
            root_cause="",
            suggestion="fix",
            escalation_level=EscalationLevel.HINT,
        )
        formatted = engine._format_diagnosis(d)
        assert "[NAMING]" in formatted
        assert "fix" in formatted


# ── FeedbackEngine._determine_root_cause ─────────────────────────────────


class TestDetermineRootCause:
    def test_naming_root_cause(self) -> None:
        engine = FeedbackEngine()
        nc = _naming_constraint(name="nc")
        cs = _constraint_set(nc)
        v = _violation(name="nc", node_id="my_func")
        diag = engine.diagnose([v], cs)
        assert "naming convention" in diag[0].root_cause.lower()
        assert "my_func" in diag[0].root_cause

    def test_structural_root_cause(self) -> None:
        engine = FeedbackEngine()
        mu = _must_use_constraint(name="mu")
        cs = _constraint_set(mu)
        v = _violation(name="mu", node_id="some_func")
        diag = engine.diagnose([v], cs)
        assert "missing" in diag[0].root_cause.lower()

    def test_boundary_root_cause(self) -> None:
        engine = FeedbackEngine()
        mnc = _must_not_cross_constraint(name="b")
        cs = _constraint_set(mnc)
        v = _violation(name="b")
        diag = engine.diagnose([v], cs)
        assert "boundary" in diag[0].root_cause.lower()

    def test_error_format_root_cause(self) -> None:
        engine = FeedbackEngine()
        ef = _error_format_constraint(name="ef")
        cs = _constraint_set(ef)
        v = _violation(name="ef")
        diag = engine.diagnose([v], cs)
        assert "error format" in diag[0].root_cause.lower()

    def test_unknown_node_id(self) -> None:
        engine = FeedbackEngine()
        nc = _naming_constraint(name="nc")
        cs = _constraint_set(nc)
        v = _violation(name="nc", node_id=None)
        diag = engine.diagnose([v], cs)
        assert "unknown" in diag[0].root_cause.lower()


# ── FeedbackEngine._suggest_fix ──────────────────────────────────────────


class TestSuggestFix:
    def test_uses_violation_suggestion_when_present(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        v = _violation(suggestion="Do X")
        diag = engine.diagnose([v], cs)
        assert diag[0].suggestion == "Do X"

    def test_fallback_for_unknown_category(self) -> None:
        engine = FeedbackEngine()
        cs = _constraint_set()
        # Message that doesn't match any keyword -> STRUCTURAL
        v = _violation(
            name="unknown",
            message="something odd happened",
            suggestion=None,
        )
        diag = engine.diagnose([v], cs)
        assert "required construct" in diag[0].suggestion.lower()

    def test_categorize_unknown_constraint_subclass(self) -> None:
        """Constraint exists but is an unknown subclass -> falls through to _infer_category."""

        class CustomConstraint(Constraint):
            def __init__(self) -> None:
                super().__init__(name="custom", description="Custom")

            def validate(self, graph: SemanticGraph) -> list[ConstraintViolation]:
                return []

            def validate_node(self, node: SemanticNode) -> ConstraintViolation | None:
                return None

        engine = FeedbackEngine()
        custom = CustomConstraint()
        cs = _constraint_set(custom)
        v = _violation(name="custom", message="something odd")
        diag = engine.diagnose([v], cs)
        # Falls through isinstance checks -> _infer_category -> STRUCTURAL
        assert diag[0].category == ViolationCategory.STRUCTURAL
