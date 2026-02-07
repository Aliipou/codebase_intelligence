"""Constraint definition and compilation.

Constraints are hard rules derived from patterns that must be enforced
during code generation. They provide the guardrails that prevent LLM
output from violating codebase conventions.

Constraint Types:
    - MustUseConstraint: Requires certain constructs/patterns
    - MustNotCrossConstraint: Forbids crossing architectural boundaries
    - NamingConstraint: Enforces naming conventions
    - TypeConstraint: Enforces type usage patterns
    - DependencyConstraint: Controls allowed dependencies

The ConstraintCompiler converts patterns into testable constraints
and validates code against them.

Usage:
    >>> compiler = ConstraintCompiler()
    >>> constraints = compiler.compile_patterns(patterns)
    >>> violations = compiler.validate(generated_code, constraints)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    FunctionNode,
    NodeType,
    SemanticNode,
)
from codebase_intelligence.patterns import Pattern, PatternConfidence, PatternType


class ConstraintSeverity(str, Enum):
    """Severity level of a constraint violation.

    ERROR: Violation must be fixed, blocks merge
    WARNING: Should be fixed, but may be acceptable
    INFO: Informational, style preference
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ConstraintScope(str, Enum):
    """Scope at which a constraint applies.

    GLOBAL: Applies to entire codebase
    MODULE: Applies within a module
    CLASS: Applies within a class
    FUNCTION: Applies within a function
    """

    GLOBAL = "global"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"


@dataclass(frozen=True)
class ConstraintViolation:
    """Represents a constraint violation found during validation.

    Attributes:
        constraint_name: Name of the violated constraint.
        message: Human-readable description of the violation.
        severity: How serious the violation is.
        file_path: Path to the file containing the violation.
        line_number: Line number of the violation (if known).
        node_id: ID of the node that violates the constraint (if known).
        suggestion: Suggested fix for the violation.

    Examples:
        >>> violation = ConstraintViolation(
        ...     constraint_name="class_suffix_service",
        ...     message="Class 'UserHandler' should end with 'Service'",
        ...     severity=ConstraintSeverity.WARNING,
        ...     file_path="app/handlers.py",
        ...     line_number=42,
        ...     suggestion="Rename to 'UserService'"
        ... )
    """

    constraint_name: str
    message: str
    severity: ConstraintSeverity
    file_path: str | None = None
    line_number: int | None = None
    node_id: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert violation to dictionary for serialization."""
        return {
            "constraint_name": self.constraint_name,
            "message": self.message,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "node_id": self.node_id,
            "suggestion": self.suggestion,
        }

    def format_message(self) -> str:
        """Format a human-readable message with location info."""
        parts = [f"[{self.severity.value.upper()}]", self.message]
        if self.file_path:
            location = self.file_path
            if self.line_number:
                location += f":{self.line_number}"
            parts.append(f"({location})")
        return " ".join(parts)


class Constraint(ABC):
    """Abstract base class for all constraints.

    Constraints are immutable rules that can validate code entities
    and report violations.

    Attributes:
        name: Unique identifier for the constraint.
        description: Human-readable description.
        severity: How serious violations should be treated.
        scope: At what level this constraint applies.
        enabled: Whether the constraint is active.
    """

    def __init__(
        self,
        name: str,
        description: str,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        scope: ConstraintScope = ConstraintScope.GLOBAL,
        enabled: bool = True,
        confidence: float = 1.0,
        break_cost: float = 1.0,
    ) -> None:
        """Initialize constraint.

        Args:
            name: Unique identifier for the constraint.
            description: Human-readable description.
            severity: How serious violations should be treated.
            scope: At what level this constraint applies.
            enabled: Whether the constraint is active.
            confidence: How confident the system is in this constraint (0.0-1.0).
            break_cost: Penalty weight when this constraint is violated (0.0+).
        """
        self._name = name
        self._description = description
        self._severity = severity
        self._scope = scope
        self._enabled = enabled
        self._confidence = confidence
        self._break_cost = break_cost

    @property
    def name(self) -> str:
        """Return the constraint name."""
        return self._name

    @property
    def description(self) -> str:
        """Return the constraint description."""
        return self._description

    @property
    def severity(self) -> ConstraintSeverity:
        """Return the constraint severity."""
        return self._severity

    @property
    def scope(self) -> ConstraintScope:
        """Return the constraint scope."""
        return self._scope

    @property
    def enabled(self) -> bool:
        """Return whether constraint is enabled."""
        return self._enabled

    @property
    def confidence(self) -> float:
        """Return the confidence level (0.0-1.0)."""
        return self._confidence

    @property
    def break_cost(self) -> float:
        """Return the break cost penalty weight."""
        return self._break_cost

    @abstractmethod
    def validate(self, graph: SemanticGraph) -> list[ConstraintViolation]:
        """Validate a semantic graph against this constraint.

        Args:
            graph: The semantic graph to validate.

        Returns:
            List of violations found.
        """
        ...

    @abstractmethod
    def validate_node(self, node: SemanticNode) -> ConstraintViolation | None:
        """Validate a single node against this constraint.

        Args:
            node: The node to validate.

        Returns:
            A violation if the node violates the constraint, None otherwise.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert constraint to dictionary for serialization."""
        return {
            "name": self._name,
            "description": self._description,
            "severity": self._severity.value,
            "scope": self._scope.value,
            "enabled": self._enabled,
            "confidence": self._confidence,
            "break_cost": self._break_cost,
            "type": self.__class__.__name__,
        }


class NamingConstraint(Constraint):
    """Enforces naming conventions using regex patterns.

    Can be configured for different entity types (classes, functions,
    variables) with specific patterns they must match.

    Attributes:
        pattern: Regex pattern that names must match.
        node_types: Types of nodes this constraint applies to.
        case_sensitive: Whether pattern matching is case-sensitive.

    Examples:
        >>> # Classes must use PascalCase
        >>> constraint = NamingConstraint(
        ...     name="pascal_case_classes",
        ...     description="Classes must use PascalCase",
        ...     pattern=r"^[A-Z][a-zA-Z0-9]*$",
        ...     node_types=[NodeType.CLASS],
        ... )

        >>> # Functions must use snake_case
        >>> constraint = NamingConstraint(
        ...     name="snake_case_functions",
        ...     description="Functions must use snake_case",
        ...     pattern=r"^[a-z][a-z0-9_]*$",
        ...     node_types=[NodeType.FUNCTION],
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str,
        pattern: str,
        node_types: Sequence[NodeType],
        severity: ConstraintSeverity = ConstraintSeverity.WARNING,
        scope: ConstraintScope = ConstraintScope.GLOBAL,
        case_sensitive: bool = True,
        enabled: bool = True,
        exclude_patterns: Sequence[str] | None = None,
        confidence: float = 1.0,
        break_cost: float = 1.0,
    ) -> None:
        """Initialize naming constraint.

        Args:
            name: Unique identifier for the constraint.
            description: Human-readable description.
            pattern: Regex pattern that names must match.
            node_types: Types of nodes this constraint applies to.
            severity: How serious violations should be treated.
            scope: At what level this constraint applies.
            case_sensitive: Whether pattern matching is case-sensitive.
            enabled: Whether the constraint is active.
            exclude_patterns: Patterns to exclude from validation.
            confidence: How confident the system is in this constraint (0.0-1.0).
            break_cost: Penalty weight when this constraint is violated (0.0+).

        Raises:
            ValueError: If pattern is invalid regex.
        """
        super().__init__(name, description, severity, scope, enabled, confidence, break_cost)

        # Validate regex pattern
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            self._compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        self._pattern = pattern
        self._node_types = set(node_types)
        self._case_sensitive = case_sensitive
        self._exclude_patterns = [re.compile(p) for p in (exclude_patterns or [])]

    @property
    def pattern(self) -> str:
        """Return the regex pattern."""
        return self._pattern

    @property
    def node_types(self) -> set[NodeType]:
        """Return the applicable node types."""
        return self._node_types

    def validate(self, graph: SemanticGraph) -> list[ConstraintViolation]:
        """Validate all applicable nodes in the graph."""
        if not self._enabled:
            return []

        violations: list[ConstraintViolation] = []

        for node_type in self._node_types:
            for node in graph.get_nodes(node_type):
                violation = self.validate_node(node)
                if violation:
                    violations.append(violation)

        return violations

    def validate_node(self, node: SemanticNode) -> ConstraintViolation | None:
        """Validate a single node's name."""
        if not self._enabled:
            return None

        if node.node_type not in self._node_types:
            return None

        # Check exclusions
        for exclude in self._exclude_patterns:
            if exclude.match(node.name):
                return None

        if not self._compiled_pattern.match(node.name):
            return ConstraintViolation(
                constraint_name=self._name,
                message=f"'{node.name}' does not match pattern '{self._pattern}'",
                severity=self._severity,
                file_path=node.file_path,
                line_number=node.line_start,
                node_id=node.id,
                suggestion=f"Rename to match pattern: {self._pattern}",
            )

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert constraint to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "pattern": self._pattern,
                "node_types": [nt.value for nt in self._node_types],
                "case_sensitive": self._case_sensitive,
            }
        )
        return result


class MustUseConstraint(Constraint):
    """Enforces that certain constructs must be used.

    Examples include requiring type hints, docstrings, or specific
    decorators.

    Attributes:
        requirement: What must be present.
        node_types: Types of nodes this constraint applies to.
        check_function: Custom function to check the requirement.

    Examples:
        >>> # All public functions must have docstrings
        >>> constraint = MustUseConstraint(
        ...     name="require_docstrings",
        ...     description="Public functions must have docstrings",
        ...     requirement="docstring",
        ...     node_types=[NodeType.FUNCTION],
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str,
        requirement: str,
        node_types: Sequence[NodeType],
        severity: ConstraintSeverity = ConstraintSeverity.WARNING,
        scope: ConstraintScope = ConstraintScope.GLOBAL,
        enabled: bool = True,
        exclude_private: bool = True,
        exclude_dunder: bool = True,
        confidence: float = 1.0,
        break_cost: float = 1.0,
    ) -> None:
        """Initialize must-use constraint.

        Args:
            name: Unique identifier for the constraint.
            description: Human-readable description.
            requirement: What must be present (docstring, type_hints, etc).
            node_types: Types of nodes this constraint applies to.
            severity: How serious violations should be treated.
            scope: At what level this constraint applies.
            enabled: Whether the constraint is active.
            exclude_private: Whether to exclude private members.
            exclude_dunder: Whether to exclude dunder methods.
            confidence: How confident the system is in this constraint (0.0-1.0).
            break_cost: Penalty weight when this constraint is violated (0.0+).
        """
        super().__init__(name, description, severity, scope, enabled, confidence, break_cost)
        self._requirement = requirement
        self._node_types = set(node_types)
        self._exclude_private = exclude_private
        self._exclude_dunder = exclude_dunder

    @property
    def requirement(self) -> str:
        """Return the requirement type."""
        return self._requirement

    @property
    def node_types(self) -> set[NodeType]:
        """Return applicable node types."""
        return self._node_types

    def validate(self, graph: SemanticGraph) -> list[ConstraintViolation]:
        """Validate all applicable nodes in the graph."""
        if not self._enabled:
            return []

        violations: list[ConstraintViolation] = []

        for node_type in self._node_types:
            for node in graph.get_nodes(node_type):
                violation = self.validate_node(node)
                if violation:
                    violations.append(violation)

        return violations

    def validate_node(self, node: SemanticNode) -> ConstraintViolation | None:
        """Validate a single node for the required construct."""
        if not self._enabled:
            return None

        if node.node_type not in self._node_types:
            return None

        # Exclusions for functions
        if isinstance(node, FunctionNode):
            if self._exclude_private and node.is_private():
                return None
            if self._exclude_dunder and node.is_dunder():
                return None

        # Check the requirement
        if not self._check_requirement(node):
            return ConstraintViolation(
                constraint_name=self._name,
                message=f"'{node.name}' is missing required {self._requirement}",
                severity=self._severity,
                file_path=node.file_path,
                line_number=node.line_start,
                node_id=node.id,
                suggestion=f"Add {self._requirement} to '{node.name}'",
            )

        return None

    def _check_requirement(self, node: SemanticNode) -> bool:
        """Check if the node satisfies the requirement."""
        if self._requirement == "docstring":
            if isinstance(node, (FunctionNode, ClassNode)):
                return node.docstring is not None
        elif self._requirement == "type_hints":
            if isinstance(node, FunctionNode):
                return node.return_type is not None
        elif self._requirement == "decorators":
            if isinstance(node, FunctionNode):
                return len(node.decorators) > 0

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert constraint to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "requirement": self._requirement,
                "node_types": [nt.value for nt in self._node_types],
                "exclude_private": self._exclude_private,
                "exclude_dunder": self._exclude_dunder,
            }
        )
        return result


class MustNotCrossConstraint(Constraint):
    """Enforces architectural boundaries.

    Prevents code in one layer/module from depending on code in
    forbidden layers/modules. Used to enforce clean architecture.

    Attributes:
        source_pattern: Pattern matching source modules.
        forbidden_targets: Patterns for forbidden dependencies.
        allowed_targets: Patterns for allowed dependencies (optional).

    Examples:
        >>> # Services cannot import from controllers
        >>> constraint = MustNotCrossConstraint(
        ...     name="service_boundary",
        ...     description="Services cannot depend on controllers",
        ...     source_pattern=r".*/services/.*",
        ...     forbidden_targets=[r".*/controllers/.*", r".*/api/.*"],
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str,
        source_pattern: str,
        forbidden_targets: Sequence[str],
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        scope: ConstraintScope = ConstraintScope.GLOBAL,
        enabled: bool = True,
        allowed_targets: Sequence[str] | None = None,
        confidence: float = 1.0,
        break_cost: float = 1.0,
    ) -> None:
        """Initialize boundary constraint.

        Args:
            name: Unique identifier for the constraint.
            description: Human-readable description.
            source_pattern: Pattern matching source modules.
            forbidden_targets: Patterns for forbidden dependencies.
            severity: How serious violations should be treated.
            scope: At what level this constraint applies.
            enabled: Whether the constraint is active.
            allowed_targets: If set, only these patterns are allowed.
            confidence: How confident the system is in this constraint (0.0-1.0).
            break_cost: Penalty weight when this constraint is violated (0.0+).

        Raises:
            ValueError: If any pattern is invalid regex.
        """
        super().__init__(name, description, severity, scope, enabled, confidence, break_cost)

        try:
            self._source_pattern = re.compile(source_pattern)
            self._forbidden_targets = [re.compile(p) for p in forbidden_targets]
            self._allowed_targets = (
                [re.compile(p) for p in allowed_targets] if allowed_targets else None
            )
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        self._source_pattern_str = source_pattern
        self._forbidden_targets_str = list(forbidden_targets)

    @property
    def source_pattern(self) -> str:
        """Return the source pattern string."""
        return self._source_pattern_str

    @property
    def forbidden_targets(self) -> list[str]:
        """Return the forbidden target patterns."""
        return self._forbidden_targets_str

    def validate(self, graph: SemanticGraph) -> list[ConstraintViolation]:
        """Validate all edges for boundary violations."""
        if not self._enabled:
            return []

        violations: list[ConstraintViolation] = []

        # Check all import edges
        from codebase_intelligence.edges import EdgeType

        for edge in graph.get_edges(EdgeType.IMPORTS):
            source_node = graph.get_node(edge.source_id)
            target_node = graph.get_node(edge.target_id)

            if source_node is None or target_node is None:
                continue

            # Check if source matches
            if not self._source_pattern.search(source_node.file_path):
                continue

            # Check if target is forbidden
            target_path = target_node.file_path
            for forbidden in self._forbidden_targets:
                if forbidden.search(target_path):
                    violations.append(
                        ConstraintViolation(
                            constraint_name=self._name,
                            message=(
                                f"'{source_node.file_path}' cannot import "
                                f"from '{target_node.file_path}'"
                            ),
                            severity=self._severity,
                            file_path=source_node.file_path,
                            line_number=edge.line_number,
                            node_id=source_node.id,
                            suggestion="Remove forbidden import or restructure code",
                        )
                    )
                    break

        return violations

    def validate_node(self, node: SemanticNode) -> ConstraintViolation | None:
        """Boundary constraints work on edges, not individual nodes."""
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert constraint to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "source_pattern": self._source_pattern_str,
                "forbidden_targets": self._forbidden_targets_str,
            }
        )
        return result


@dataclass
class ErrorFormatConstraint(Constraint):
    """Enforces error handling format patterns.

    Validates that exception classes follow naming conventions and
    inherit from required base classes.

    Attributes:
        exception_pattern: Regex that exception class names must match.
        required_bases: Base classes that exceptions must inherit from.
        node_types: Always applies to CLASS nodes.

    Examples:
        >>> constraint = ErrorFormatConstraint(
        ...     name="exception_naming",
        ...     description="Exceptions must end with Error",
        ...     exception_pattern=r"^[A-Z][a-zA-Z]*Error$",
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str,
        exception_pattern: str,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        scope: ConstraintScope = ConstraintScope.GLOBAL,
        enabled: bool = True,
        required_bases: Sequence[str] | None = None,
        confidence: float = 1.0,
        break_cost: float = 1.0,
    ) -> None:
        """Initialize error format constraint.

        Args:
            name: Unique identifier.
            description: Human-readable description.
            exception_pattern: Regex for exception class names.
            severity: Violation severity.
            scope: Constraint scope.
            enabled: Whether active.
            required_bases: Base classes exceptions must extend.
            confidence: How confident the system is in this constraint (0.0-1.0).
            break_cost: Penalty weight when this constraint is violated (0.0+).

        Raises:
            ValueError: If exception_pattern is invalid regex.
        """
        super().__init__(name, description, severity, scope, enabled, confidence, break_cost)

        try:
            self._compiled_pattern = re.compile(exception_pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        self._exception_pattern = exception_pattern
        self._required_bases = list(required_bases) if required_bases else []

    @property
    def exception_pattern(self) -> str:
        """Return the exception naming pattern."""
        return self._exception_pattern

    @property
    def required_bases(self) -> list[str]:
        """Return the required base classes."""
        return self._required_bases

    def validate(self, graph: SemanticGraph) -> list[ConstraintViolation]:
        """Validate all exception classes in the graph."""
        if not self._enabled:
            return []

        violations: list[ConstraintViolation] = []
        for node in graph.get_nodes(NodeType.CLASS):
            # get_nodes(CLASS) always returns ClassNode instances
            class_node: ClassNode = node  # type: ignore[assignment]
            # Detect exception classes by name or base classes
            is_exception = self._is_exception_class(class_node)
            if not is_exception:
                continue

            violation = self.validate_node(node)
            if violation:
                violations.append(violation)

        return violations

    def validate_node(self, node: SemanticNode) -> ConstraintViolation | None:
        """Validate a single exception class node."""
        if not self._enabled:
            return None

        if not isinstance(node, ClassNode):
            return None

        if not self._is_exception_class(node):
            return None

        # Check naming pattern
        if not self._compiled_pattern.match(node.name):
            return ConstraintViolation(
                constraint_name=self._name,
                message=f"Exception '{node.name}' does not match pattern '{self._exception_pattern}'",
                severity=self._severity,
                file_path=node.file_path,
                line_number=node.line_start,
                node_id=node.id,
                suggestion=f"Rename to match pattern: {self._exception_pattern}",
            )

        # Check required bases
        if self._required_bases:
            has_required = any(base in node.bases for base in self._required_bases)
            if not has_required:
                return ConstraintViolation(
                    constraint_name=self._name,
                    message=(
                        f"Exception '{node.name}' must inherit from one of: "
                        f"{', '.join(self._required_bases)}"
                    ),
                    severity=self._severity,
                    file_path=node.file_path,
                    line_number=node.line_start,
                    node_id=node.id,
                    suggestion=f"Add base class: {self._required_bases[0]}",
                )

        return None

    def _is_exception_class(self, node: ClassNode) -> bool:
        """Determine if a class is an exception class."""
        exception_indicators = {"Exception", "Error", "BaseException"}
        # Check bases
        for base in node.bases:
            if any(ind in base for ind in exception_indicators):
                return True
        # Check name
        if node.name.endswith("Error") or node.name.endswith("Exception"):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert constraint to dictionary."""
        result = super().to_dict()
        result.update({
            "exception_pattern": self._exception_pattern,
            "required_bases": self._required_bases,
        })
        return result


@dataclass
class ConstraintSet:
    """A collection of constraints with metadata.

    Provides a way to group related constraints and apply them together.

    Attributes:
        name: Name of the constraint set.
        description: Description of the constraint set.
        constraints: List of constraints in the set.
        version: Version of the constraint set.

    Examples:
        >>> constraint_set = ConstraintSet(
        ...     name="fastapi_standards",
        ...     description="Standard constraints for FastAPI projects",
        ...     constraints=[naming_constraint, docstring_constraint],
        ... )
        >>> violations = constraint_set.validate(graph)
    """

    name: str
    description: str
    constraints: list[Constraint] = field(default_factory=list)
    version: str = "1.0.0"

    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the set.

        Args:
            constraint: The constraint to add.
        """
        self.constraints.append(constraint)

    def remove(self, constraint_name: str) -> bool:
        """Remove a constraint by name.

        Args:
            constraint_name: Name of the constraint to remove.

        Returns:
            True if constraint was found and removed.
        """
        for i, c in enumerate(self.constraints):
            if c.name == constraint_name:
                del self.constraints[i]
                return True
        return False

    def get(self, constraint_name: str) -> Constraint | None:
        """Get a constraint by name.

        Args:
            constraint_name: Name of the constraint to get.

        Returns:
            The constraint if found, None otherwise.
        """
        for c in self.constraints:
            if c.name == constraint_name:
                return c
        return None

    def validate(self, graph: SemanticGraph) -> list[ConstraintViolation]:
        """Validate a graph against all constraints in the set.

        Args:
            graph: The semantic graph to validate.

        Returns:
            List of all violations found.
        """
        violations: list[ConstraintViolation] = []
        for constraint in self.constraints:
            if constraint.enabled:
                violations.extend(constraint.validate(graph))
        return violations

    def get_errors(self, graph: SemanticGraph) -> list[ConstraintViolation]:
        """Get only ERROR severity violations.

        Args:
            graph: The semantic graph to validate.

        Returns:
            List of ERROR severity violations.
        """
        violations = self.validate(graph)
        return [v for v in violations if v.severity == ConstraintSeverity.ERROR]

    def enabled_count(self) -> int:
        """Count enabled constraints."""
        return sum(1 for c in self.constraints if c.enabled)


class ConstraintCompiler:
    """Compiles patterns into constraints.

    The compiler analyzes extracted patterns and generates appropriate
    constraints that can enforce those patterns.

    Attributes:
        _min_confidence: Minimum pattern confidence for compilation.
        _default_severity: Default severity for compiled constraints.

    Examples:
        >>> compiler = ConstraintCompiler(min_confidence=PatternConfidence.MEDIUM)
        >>> constraints = compiler.compile_patterns(patterns)
        >>> constraint_set = compiler.compile_to_set(patterns, "my_rules")
    """

    def __init__(
        self,
        min_confidence: PatternConfidence = PatternConfidence.MEDIUM,
        default_severity: ConstraintSeverity = ConstraintSeverity.WARNING,
    ) -> None:
        """Initialize the constraint compiler.

        Args:
            min_confidence: Minimum confidence level for pattern compilation.
            default_severity: Default severity for generated constraints.
        """
        self._min_confidence = min_confidence
        self._default_severity = default_severity

    def compile_patterns(self, patterns: Sequence[Pattern]) -> list[Constraint]:
        """Compile patterns into constraints.

        Args:
            patterns: List of patterns to compile.

        Returns:
            List of generated constraints.
        """
        constraints: list[Constraint] = []

        for pattern in patterns:
            if not self._meets_confidence(pattern):
                continue

            constraint = self._compile_pattern(pattern)
            if constraint:
                constraints.append(constraint)

        return constraints

    def compile_to_set(
        self,
        patterns: Sequence[Pattern],
        name: str,
        description: str = "",
    ) -> ConstraintSet:
        """Compile patterns into a constraint set.

        Args:
            patterns: List of patterns to compile.
            name: Name for the constraint set.
            description: Description for the constraint set.

        Returns:
            A ConstraintSet containing compiled constraints.
        """
        constraints = self.compile_patterns(patterns)
        return ConstraintSet(
            name=name,
            description=description or f"Constraints compiled from {len(patterns)} patterns",
            constraints=constraints,
        )

    def _meets_confidence(self, pattern: Pattern) -> bool:
        """Check if pattern meets minimum confidence threshold."""
        confidence_order = {
            PatternConfidence.LOW: 0,
            PatternConfidence.MEDIUM: 1,
            PatternConfidence.HIGH: 2,
        }
        return confidence_order[pattern.confidence] >= confidence_order[self._min_confidence]

    def _compile_pattern(self, pattern: Pattern) -> Constraint | None:
        """Compile a single pattern into a constraint."""
        if pattern.pattern_type == PatternType.NAMING:
            return self._compile_naming_pattern(pattern)
        elif pattern.pattern_type == PatternType.STRUCTURAL:
            return self._compile_structural_pattern(pattern)
        elif pattern.pattern_type == PatternType.FRAMEWORK:
            return self._compile_framework_pattern(pattern)

        return None

    def _compile_naming_pattern(self, pattern: Pattern) -> Constraint | None:
        """Compile a naming pattern into a NamingConstraint."""
        if not pattern.regex:
            return None

        # Determine node types from pattern name
        node_types: list[NodeType] = []
        if "class" in pattern.name:
            node_types.append(NodeType.CLASS)
        elif "function" in pattern.name:
            node_types.append(NodeType.FUNCTION)

        if not node_types:
            return None

        severity = (
            ConstraintSeverity.ERROR
            if pattern.confidence == PatternConfidence.HIGH
            else self._default_severity
        )

        return NamingConstraint(
            name=f"enforce_{pattern.name}",
            description=pattern.description,
            pattern=pattern.regex,
            node_types=node_types,
            severity=severity,
        )

    def _compile_structural_pattern(self, pattern: Pattern) -> Constraint | None:
        """Compile a structural pattern into a constraint."""
        # Structural patterns might generate MustNotCross constraints
        # for module organization
        return None

    def _compile_framework_pattern(self, pattern: Pattern) -> Constraint | None:
        """Compile a framework pattern into a constraint."""
        # Framework patterns might generate MustUse constraints
        # for required patterns like response_model
        if "response_model" in pattern.name:
            return MustUseConstraint(
                name="require_response_models",
                description="FastAPI endpoints should use response_model",
                requirement="decorators",  # Simplified check
                node_types=[NodeType.FUNCTION],
                severity=ConstraintSeverity.WARNING,
            )

        return None
