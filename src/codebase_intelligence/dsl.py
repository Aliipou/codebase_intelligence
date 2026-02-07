"""Constraint DSL for serialization and deserialization.

Provides JSON-based persistence for constraint definitions, enabling
constraints to be saved to files, loaded back, and shared across teams.

The DSL operates on plain dictionaries as the intermediate format,
with JSON as the file serialization layer.

Usage:
    >>> # Save constraints to file
    >>> ConstraintDSL.save(constraint_set, "constraints.json")
    >>>
    >>> # Load constraints from file
    >>> constraint_set = ConstraintDSL.load("constraints.json")
    >>>
    >>> # Convert to/from dictionaries
    >>> data = ConstraintDSL.to_dict(constraint_set)
    >>> constraint_set = ConstraintDSL.from_dict(data)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from codebase_intelligence.constraints import (
    Constraint,
    ConstraintScope,
    ConstraintSet,
    ConstraintSeverity,
    ErrorFormatConstraint,
    MustNotCrossConstraint,
    MustUseConstraint,
    NamingConstraint,
)
from codebase_intelligence.nodes import NodeType


class DSLError(Exception):
    """Raised when constraint DSL parsing or validation fails."""

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        self.message = message
        if path:
            super().__init__(f"{message} (in {path})")
        else:
            super().__init__(message)


class ConstraintDSL:
    """Serializes and deserializes constraint sets.

    Supports JSON file format for constraint persistence.
    All constraints are round-trippable: save → load produces
    equivalent constraint sets.

    Examples:
        >>> dsl = ConstraintDSL()
        >>> dsl.save(constraint_set, Path("rules.json"))
        >>> loaded = dsl.load(Path("rules.json"))
    """

    _NODE_TYPE_MAP: dict[str, NodeType] = {nt.value: nt for nt in NodeType}
    _SEVERITY_MAP: dict[str, ConstraintSeverity] = {s.value: s for s in ConstraintSeverity}
    _SCOPE_MAP: dict[str, ConstraintScope] = {s.value: s for s in ConstraintScope}

    def save(self, constraint_set: ConstraintSet, path: Path | str) -> None:
        """Save a constraint set to a JSON file.

        Args:
            constraint_set: The constraint set to save.
            path: File path to write to.
        """
        data = self.to_dict(constraint_set)
        path = Path(path)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path | str) -> ConstraintSet:
        """Load a constraint set from a JSON file.

        Args:
            path: File path to read from.

        Returns:
            The deserialized constraint set.

        Raises:
            DSLError: If the file is invalid or contains errors.
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Constraint file not found: {path}")

        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise DSLError(f"Invalid JSON: {e}", str(path)) from e

        return self.from_dict(data, str(path))

    def to_dict(self, constraint_set: ConstraintSet) -> dict[str, Any]:
        """Convert a constraint set to a dictionary.

        Args:
            constraint_set: The constraint set to convert.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "name": constraint_set.name,
            "description": constraint_set.description,
            "version": constraint_set.version,
            "constraints": [c.to_dict() for c in constraint_set.constraints],
        }

    def from_dict(
        self,
        data: dict[str, Any],
        source: str | None = None,
    ) -> ConstraintSet:
        """Create a constraint set from a dictionary.

        Args:
            data: Dictionary containing constraint set data.
            source: Optional source identifier for error messages.

        Returns:
            The deserialized constraint set.

        Raises:
            DSLError: If the data is invalid.
        """
        if not isinstance(data, dict):
            raise DSLError("Constraint data must be a dictionary", source)

        name = data.get("name", "")
        if not name:
            raise DSLError("Constraint set must have a 'name' field", source)

        description = data.get("description", "")
        version = data.get("version", "1.0.0")
        raw_constraints = data.get("constraints", [])

        if not isinstance(raw_constraints, list):
            raise DSLError("'constraints' must be a list", source)

        constraints: list[Constraint] = []
        for i, raw in enumerate(raw_constraints):
            constraint = self._parse_constraint(raw, i, source)
            constraints.append(constraint)

        return ConstraintSet(
            name=name,
            description=description,
            constraints=constraints,
            version=version,
        )

    def merge(self, *sets: ConstraintSet) -> ConstraintSet:
        """Merge multiple constraint sets into one.

        Later sets override earlier ones when constraint names collide.

        Args:
            *sets: Constraint sets to merge.

        Returns:
            A new merged constraint set.
        """
        seen: dict[str, Constraint] = {}
        for cs in sets:
            for constraint in cs.constraints:
                seen[constraint.name] = constraint

        return ConstraintSet(
            name=" + ".join(cs.name for cs in sets),
            description="Merged constraint set",
            constraints=list(seen.values()),
        )

    def _parse_constraint(
        self,
        raw: dict[str, Any],
        index: int,
        source: str | None,
    ) -> Constraint:
        """Parse a single constraint from a dictionary."""
        if not isinstance(raw, dict):
            raise DSLError(f"Constraint at index {index} must be a dictionary", source)

        constraint_type = raw.get("type", "")
        if not constraint_type:
            raise DSLError(f"Constraint at index {index} missing 'type' field", source)

        severity = self._parse_severity(raw.get("severity", "warning"), index, source)
        scope = self._parse_scope(raw.get("scope", "global"), index, source)
        enabled = raw.get("enabled", True)
        confidence = float(raw.get("confidence", 1.0))
        break_cost = float(raw.get("break_cost", 1.0))

        if constraint_type == "NamingConstraint":
            return self._parse_naming(raw, severity, scope, enabled, confidence, break_cost, index, source)
        elif constraint_type == "MustUseConstraint":
            return self._parse_must_use(raw, severity, scope, enabled, confidence, break_cost, index, source)
        elif constraint_type == "MustNotCrossConstraint":
            return self._parse_must_not_cross(raw, severity, scope, enabled, confidence, break_cost, index, source)
        elif constraint_type == "ErrorFormatConstraint":
            return self._parse_error_format(raw, severity, scope, enabled, confidence, break_cost, index, source)
        else:
            raise DSLError(
                f"Unknown constraint type '{constraint_type}' at index {index}",
                source,
            )

    def _parse_severity(
        self,
        value: str,
        index: int,
        source: str | None,
    ) -> ConstraintSeverity:
        """Parse severity from string."""
        if value not in self._SEVERITY_MAP:
            raise DSLError(
                f"Invalid severity '{value}' at index {index}. "
                f"Must be one of: {', '.join(self._SEVERITY_MAP)}",
                source,
            )
        return self._SEVERITY_MAP[value]

    def _parse_scope(
        self,
        value: str,
        index: int,
        source: str | None,
    ) -> ConstraintScope:
        """Parse scope from string."""
        if value not in self._SCOPE_MAP:
            raise DSLError(
                f"Invalid scope '{value}' at index {index}. "
                f"Must be one of: {', '.join(self._SCOPE_MAP)}",
                source,
            )
        return self._SCOPE_MAP[value]

    def _parse_node_types(
        self,
        values: list[str],
        index: int,
        source: str | None,
    ) -> list[NodeType]:
        """Parse node types from string list."""
        node_types: list[NodeType] = []
        for v in values:
            if v not in self._NODE_TYPE_MAP:
                raise DSLError(
                    f"Invalid node type '{v}' at index {index}. "
                    f"Must be one of: {', '.join(self._NODE_TYPE_MAP)}",
                    source,
                )
            node_types.append(self._NODE_TYPE_MAP[v])
        return node_types

    def _parse_naming(
        self,
        raw: dict[str, Any],
        severity: ConstraintSeverity,
        scope: ConstraintScope,
        enabled: bool,
        confidence: float,
        break_cost: float,
        index: int,
        source: str | None,
    ) -> NamingConstraint:
        """Parse a NamingConstraint from dictionary."""
        name = raw.get("name", "")
        if not name:
            raise DSLError(f"NamingConstraint at index {index} missing 'name'", source)

        pattern = raw.get("pattern", "")
        if not pattern:
            raise DSLError(f"NamingConstraint at index {index} missing 'pattern'", source)

        raw_types = raw.get("node_types", [])
        if not raw_types:
            raise DSLError(f"NamingConstraint at index {index} missing 'node_types'", source)

        node_types = self._parse_node_types(raw_types, index, source)
        case_sensitive = raw.get("case_sensitive", True)

        return NamingConstraint(
            name=name,
            description=raw.get("description", ""),
            pattern=pattern,
            node_types=node_types,
            severity=severity,
            scope=scope,
            enabled=enabled,
            case_sensitive=case_sensitive,
            confidence=confidence,
            break_cost=break_cost,
        )

    def _parse_must_use(
        self,
        raw: dict[str, Any],
        severity: ConstraintSeverity,
        scope: ConstraintScope,
        enabled: bool,
        confidence: float,
        break_cost: float,
        index: int,
        source: str | None,
    ) -> MustUseConstraint:
        """Parse a MustUseConstraint from dictionary."""
        name = raw.get("name", "")
        if not name:
            raise DSLError(f"MustUseConstraint at index {index} missing 'name'", source)

        requirement = raw.get("requirement", "")
        if not requirement:
            raise DSLError(
                f"MustUseConstraint at index {index} missing 'requirement'", source
            )

        raw_types = raw.get("node_types", [])
        if not raw_types:
            raise DSLError(f"MustUseConstraint at index {index} missing 'node_types'", source)

        node_types = self._parse_node_types(raw_types, index, source)

        return MustUseConstraint(
            name=name,
            description=raw.get("description", ""),
            requirement=requirement,
            node_types=node_types,
            severity=severity,
            scope=scope,
            enabled=enabled,
            exclude_private=raw.get("exclude_private", True),
            exclude_dunder=raw.get("exclude_dunder", True),
            confidence=confidence,
            break_cost=break_cost,
        )

    def _parse_must_not_cross(
        self,
        raw: dict[str, Any],
        severity: ConstraintSeverity,
        scope: ConstraintScope,
        enabled: bool,
        confidence: float,
        break_cost: float,
        index: int,
        source: str | None,
    ) -> MustNotCrossConstraint:
        """Parse a MustNotCrossConstraint from dictionary."""
        name = raw.get("name", "")
        if not name:
            raise DSLError(
                f"MustNotCrossConstraint at index {index} missing 'name'", source
            )

        source_pattern = raw.get("source_pattern", "")
        if not source_pattern:
            raise DSLError(
                f"MustNotCrossConstraint at index {index} missing 'source_pattern'",
                source,
            )

        forbidden = raw.get("forbidden_targets", [])
        if not forbidden:
            raise DSLError(
                f"MustNotCrossConstraint at index {index} missing 'forbidden_targets'",
                source,
            )

        return MustNotCrossConstraint(
            name=name,
            description=raw.get("description", ""),
            source_pattern=source_pattern,
            forbidden_targets=forbidden,
            severity=severity,
            scope=scope,
            enabled=enabled,
            confidence=confidence,
            break_cost=break_cost,
        )

    def _parse_error_format(
        self,
        raw: dict[str, Any],
        severity: ConstraintSeverity,
        scope: ConstraintScope,
        enabled: bool,
        confidence: float,
        break_cost: float,
        index: int,
        source: str | None,
    ) -> ErrorFormatConstraint:
        """Parse an ErrorFormatConstraint from dictionary."""
        name = raw.get("name", "")
        if not name:
            raise DSLError(
                f"ErrorFormatConstraint at index {index} missing 'name'", source
            )

        exception_pattern = raw.get("exception_pattern", "")
        if not exception_pattern:
            raise DSLError(
                f"ErrorFormatConstraint at index {index} missing 'exception_pattern'",
                source,
            )

        required_bases = raw.get("required_bases", [])

        return ErrorFormatConstraint(
            name=name,
            description=raw.get("description", ""),
            exception_pattern=exception_pattern,
            severity=severity,
            scope=scope,
            enabled=enabled,
            required_bases=required_bases or None,
            confidence=confidence,
            break_cost=break_cost,
        )
