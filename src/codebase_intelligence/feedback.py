"""Feedback engine for intelligent self-correction.

Diagnoses constraint violations, categorizes them, and builds
escalating refinement prompts for the agent retry loop.

The FeedbackEngine converts raw ConstraintViolations into actionable
ViolationDiagnoses and produces structured refinement prompts that
escalate in specificity across retry attempts.

Escalation Levels:
    - HINT: Gentle guidance toward the fix
    - EXPLICIT: Direct instruction with the exact fix
    - REWRITE: Full rewrite instruction with constraint rules

Usage:
    >>> engine = FeedbackEngine()
    >>> diagnoses = engine.diagnose(violations, constraints)
    >>> prompt = engine.build_refinement(context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from codebase_intelligence.constraints import (
    Constraint,
    ConstraintSet,
    ConstraintViolation,
    ErrorFormatConstraint,
    MustNotCrossConstraint,
    MustUseConstraint,
    NamingConstraint,
)


class ViolationCategory(str, Enum):
    """Category of a constraint violation.

    NAMING: Violations of naming conventions.
    STRUCTURAL: Missing required constructs (docstrings, type hints).
    BOUNDARY: Forbidden cross-module dependencies.
    ERROR_FORMAT: Exception class naming or inheritance issues.
    """

    NAMING = "naming"
    STRUCTURAL = "structural"
    BOUNDARY = "boundary"
    ERROR_FORMAT = "error_format"


class EscalationLevel(str, Enum):
    """How aggressively the refinement prompt guides the LLM.

    HINT: Gentle suggestion.
    EXPLICIT: Direct, specific instructions.
    REWRITE: Full rewrite instruction with rules inlined.
    """

    HINT = "hint"
    EXPLICIT = "explicit"
    REWRITE = "rewrite"


@dataclass(frozen=True)
class ViolationDiagnosis:
    """A diagnosed violation with root cause and suggested fix.

    Attributes:
        violation: The original constraint violation.
        category: What kind of violation this is.
        root_cause: Why the violation occurred.
        suggestion: How to fix it.
        escalation_level: Current escalation level.
        confidence: How confident the diagnosis is (0.0-1.0).
    """

    violation: ConstraintViolation
    category: ViolationCategory
    root_cause: str
    suggestion: str
    escalation_level: EscalationLevel
    confidence: float = 1.0


@dataclass
class RefinementContext:
    """Context for building a refinement prompt.

    Attributes:
        original_request: The original code generation task.
        violations: Violations from the last attempt.
        diagnoses: Diagnosed violations.
        attempt: Current attempt number (1-based).
        max_attempts: Maximum allowed attempts.
        history: Previous (source, violations) pairs.
    """

    original_request: str
    violations: list[ConstraintViolation] = field(default_factory=list)
    diagnoses: list[ViolationDiagnosis] = field(default_factory=list)
    attempt: int = 1
    max_attempts: int = 3
    history: list[tuple[str, list[ConstraintViolation]]] = field(default_factory=list)


class FeedbackEngine:
    """Diagnoses violations and builds escalating refinement prompts.

    The engine categorizes violations by type, determines root causes,
    and produces refinement prompts that escalate in specificity
    as retry attempts increase.

    Examples:
        >>> engine = FeedbackEngine()
        >>> diagnoses = engine.diagnose(violations, constraints)
        >>> context = RefinementContext(
        ...     original_request="Add user endpoint",
        ...     violations=violations,
        ...     diagnoses=diagnoses,
        ...     attempt=2,
        ...     max_attempts=3,
        ... )
        >>> prompt = engine.build_refinement(context)
    """

    def diagnose(
        self,
        violations: Sequence[ConstraintViolation],
        constraints: ConstraintSet,
    ) -> list[ViolationDiagnosis]:
        """Diagnose violations and determine root causes.

        Args:
            violations: Violations to diagnose.
            constraints: The constraint set for context.

        Returns:
            List of diagnosed violations with suggestions.
        """
        diagnoses: list[ViolationDiagnosis] = []
        for violation in violations:
            category = self._categorize(violation, constraints)
            root_cause = self._determine_root_cause(violation, category)
            suggestion = self._suggest_fix(violation, category)
            constraint = constraints.get(violation.constraint_name)
            confidence = constraint.confidence if constraint else 1.0
            diagnoses.append(
                ViolationDiagnosis(
                    violation=violation,
                    category=category,
                    root_cause=root_cause,
                    suggestion=suggestion,
                    escalation_level=EscalationLevel.HINT,
                    confidence=confidence,
                )
            )
        return diagnoses

    def build_refinement(self, context: RefinementContext) -> str:
        """Build a refinement prompt from diagnosed violations.

        Escalates the prompt style based on the attempt number.

        Args:
            context: Refinement context with diagnoses and attempt info.

        Returns:
            A refinement prompt string for the LLM.
        """
        level = self._escalation_for_attempt(context.attempt, context.max_attempts)

        if level == EscalationLevel.HINT:
            return self._build_hint(context)
        elif level == EscalationLevel.EXPLICIT:
            return self._build_explicit(context)
        return self._build_rewrite(context)

    def _categorize(
        self,
        violation: ConstraintViolation,
        constraints: ConstraintSet,
    ) -> ViolationCategory:
        """Map a violation to its category based on the constraint type.

        Args:
            violation: The violation to categorize.
            constraints: The constraint set for lookup.

        Returns:
            The violation category.
        """
        constraint = constraints.get(violation.constraint_name)
        if constraint is None:
            # Infer from violation message
            return self._infer_category(violation)

        if isinstance(constraint, NamingConstraint):
            return ViolationCategory.NAMING
        elif isinstance(constraint, MustUseConstraint):
            return ViolationCategory.STRUCTURAL
        elif isinstance(constraint, MustNotCrossConstraint):
            return ViolationCategory.BOUNDARY
        elif isinstance(constraint, ErrorFormatConstraint):
            return ViolationCategory.ERROR_FORMAT

        return self._infer_category(violation)

    def _infer_category(self, violation: ConstraintViolation) -> ViolationCategory:
        """Infer category from violation message when constraint is unknown."""
        msg = violation.message.lower()
        if "naming" in msg or "pattern" in msg or "match" in msg:
            return ViolationCategory.NAMING
        elif "import" in msg or "boundary" in msg or "cannot" in msg:
            return ViolationCategory.BOUNDARY
        elif "exception" in msg or "error" in msg or "inherit" in msg:
            return ViolationCategory.ERROR_FORMAT
        return ViolationCategory.STRUCTURAL

    def _escalation_for_attempt(
        self,
        attempt: int,
        max_attempts: int,
    ) -> EscalationLevel:
        """Determine escalation level based on attempt number.

        Args:
            attempt: Current attempt (1-based).
            max_attempts: Maximum attempts allowed.

        Returns:
            The appropriate escalation level.
        """
        if max_attempts <= 1:
            return EscalationLevel.REWRITE

        if attempt <= 1:
            return EscalationLevel.HINT
        elif attempt < max_attempts:
            return EscalationLevel.EXPLICIT
        return EscalationLevel.REWRITE

    def _determine_root_cause(
        self,
        violation: ConstraintViolation,
        category: ViolationCategory,
    ) -> str:
        """Determine the root cause of a violation."""
        if category == ViolationCategory.NAMING:
            return f"Name '{violation.node_id or 'unknown'}' does not follow naming convention"
        elif category == ViolationCategory.STRUCTURAL:
            return f"Required construct missing in '{violation.node_id or 'unknown'}'"
        elif category == ViolationCategory.BOUNDARY:
            return "Forbidden cross-boundary dependency detected"
        else:
            return "Exception class does not follow error format rules"

    def _suggest_fix(
        self,
        violation: ConstraintViolation,
        category: ViolationCategory,
    ) -> str:
        """Generate a fix suggestion for a violation."""
        if violation.suggestion:
            return violation.suggestion
        if category == ViolationCategory.NAMING:
            return "Rename to match the required naming pattern"
        elif category == ViolationCategory.STRUCTURAL:
            return "Add the required construct"
        elif category == ViolationCategory.BOUNDARY:
            return "Remove the forbidden import or restructure dependencies"
        else:
            return "Fix exception class naming or inheritance"

    def _format_diagnosis(self, diagnosis: ViolationDiagnosis) -> str:
        """Format a single diagnosis as a human-readable string.

        Args:
            diagnosis: The diagnosis to format.

        Returns:
            Formatted diagnosis string.
        """
        parts = [
            f"[{diagnosis.category.value.upper()}]",
            diagnosis.violation.message,
        ]
        if diagnosis.root_cause:
            parts.append(f"Root cause: {diagnosis.root_cause}")
        parts.append(f"Fix: {diagnosis.suggestion}")
        return " | ".join(parts)

    def _build_hint(self, context: RefinementContext) -> str:
        """Build a hint-level refinement prompt."""
        lines = [
            "Your previous code had some constraint violations.",
            "Please review and fix the following issues:",
            "",
        ]
        for diagnosis in context.diagnoses:
            lines.append(f"- {diagnosis.violation.message}")
        lines.append("")
        lines.append("Regenerate the code with these issues fixed.")
        return "\n".join(lines)

    def _build_explicit(self, context: RefinementContext) -> str:
        """Build an explicit-level refinement prompt."""
        lines = [
            "Your code violated project constraints. Fix ALL of the following:",
            "",
        ]
        for diagnosis in context.diagnoses:
            lines.append(f"- {self._format_diagnosis(diagnosis)}")
        lines.append("")
        lines.append("Apply each fix exactly as described and regenerate.")
        return "\n".join(lines)

    def _build_rewrite(self, context: RefinementContext) -> str:
        """Build a rewrite-level refinement prompt."""
        lines = [
            "FINAL ATTEMPT. Your code MUST satisfy ALL constraints.",
            f"Original task: {context.original_request}",
            "",
            "Violations that MUST be fixed:",
            "",
        ]
        for diagnosis in context.diagnoses:
            lines.append(f"- {self._format_diagnosis(diagnosis)}")

        if context.history:
            lines.append("")
            lines.append(f"You have failed {len(context.history)} previous attempt(s).")

        lines.append("")
        lines.append("Rewrite the code from scratch following ALL rules exactly.")
        return "\n".join(lines)
