"""Brutal unit tests for constraints.

Tests every code path, edge case, and validation rule for 100% coverage.
"""

from __future__ import annotations

import pytest

from codebase_intelligence.constraints import (
    Constraint,
    ConstraintCompiler,
    ConstraintScope,
    ConstraintSet,
    ConstraintSeverity,
    ConstraintViolation,
    ErrorFormatConstraint,
    MustNotCrossConstraint,
    MustUseConstraint,
    NamingConstraint,
)
from codebase_intelligence.edges import EdgeType, SemanticEdge
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    FunctionNode,
    ImportNode,
    ModuleNode,
    NodeType,
    SemanticNode,
)
from codebase_intelligence.patterns import Pattern, PatternConfidence, PatternType


# ── Helpers ───────────────────────────────────────────────────────────────


def _fn(name: str, fp: str = "app.py", ls: int = 10, le: int = 20,
        docstring: str | None = None, return_type: str | None = None,
        decorators: list[str] | None = None, params: list[str] | None = None) -> FunctionNode:
    return FunctionNode(
        name=name, file_path=fp, line_start=ls, line_end=le,
        docstring=docstring, return_type=return_type,
        decorators=decorators or [], parameters=params or [],
    )


def _cls(name: str, fp: str = "app.py", docstring: str | None = None) -> ClassNode:
    return ClassNode(name=name, file_path=fp, line_start=10, line_end=50, docstring=docstring)


def _mod(name: str, fp: str = "", imports: list[str] | None = None) -> ModuleNode:
    fp = fp or f"{name}.py"
    return ModuleNode(name=name, file_path=fp, line_start=1, line_end=100, imports=imports or [])


# ── Enums ─────────────────────────────────────────────────────────────────


class TestConstraintSeverity:
    def test_all_values(self) -> None:
        assert {s.value for s in ConstraintSeverity} == {"error", "warning", "info"}


class TestConstraintScope:
    def test_all_values(self) -> None:
        assert {s.value for s in ConstraintScope} == {"global", "module", "class", "function"}


# ── ConstraintViolation ──────────────────────────────────────────────────


class TestConstraintViolation:
    def test_basic_violation(self) -> None:
        v = ConstraintViolation(
            constraint_name="test",
            message="Something wrong",
            severity=ConstraintSeverity.ERROR,
        )
        assert v.constraint_name == "test"
        assert v.message == "Something wrong"
        assert v.severity == ConstraintSeverity.ERROR
        assert v.file_path is None
        assert v.line_number is None
        assert v.node_id is None
        assert v.suggestion is None

    def test_full_violation(self) -> None:
        v = ConstraintViolation(
            constraint_name="naming",
            message="Bad name",
            severity=ConstraintSeverity.WARNING,
            file_path="app.py",
            line_number=42,
            node_id="abc",
            suggestion="Rename it",
        )
        assert v.file_path == "app.py"
        assert v.line_number == 42
        assert v.node_id == "abc"
        assert v.suggestion == "Rename it"

    def test_to_dict(self) -> None:
        v = ConstraintViolation(
            constraint_name="test",
            message="msg",
            severity=ConstraintSeverity.ERROR,
            file_path="a.py",
            line_number=10,
            node_id="n1",
            suggestion="fix it",
        )
        d = v.to_dict()
        assert d["constraint_name"] == "test"
        assert d["severity"] == "error"
        assert d["file_path"] == "a.py"
        assert d["line_number"] == 10
        assert d["suggestion"] == "fix it"

    def test_to_dict_nulls(self) -> None:
        v = ConstraintViolation(
            constraint_name="test",
            message="msg",
            severity=ConstraintSeverity.INFO,
        )
        d = v.to_dict()
        assert d["file_path"] is None
        assert d["line_number"] is None

    def test_format_message_no_location(self) -> None:
        v = ConstraintViolation(
            constraint_name="test",
            message="Bad stuff",
            severity=ConstraintSeverity.ERROR,
        )
        assert v.format_message() == "[ERROR] Bad stuff"

    def test_format_message_with_file(self) -> None:
        v = ConstraintViolation(
            constraint_name="test",
            message="Bad stuff",
            severity=ConstraintSeverity.WARNING,
            file_path="app.py",
        )
        assert "(app.py)" in v.format_message()

    def test_format_message_with_file_and_line(self) -> None:
        v = ConstraintViolation(
            constraint_name="test",
            message="Bad stuff",
            severity=ConstraintSeverity.INFO,
            file_path="app.py",
            line_number=42,
        )
        msg = v.format_message()
        assert "[INFO]" in msg
        assert "(app.py:42)" in msg

    def test_frozen(self) -> None:
        v = ConstraintViolation(
            constraint_name="test", message="msg", severity=ConstraintSeverity.ERROR,
        )
        with pytest.raises(AttributeError):
            v.message = "changed"  # type: ignore[misc]


# ── NamingConstraint ─────────────────────────────────────────────────────


class TestNamingConstraint:
    def test_basic_creation(self) -> None:
        c = NamingConstraint(
            name="snake_case",
            description="Functions must use snake_case",
            pattern=r"^[a-z][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
        )
        assert c.name == "snake_case"
        assert c.pattern == r"^[a-z][a-z0-9_]*$"
        assert NodeType.FUNCTION in c.node_types

    def test_properties(self) -> None:
        c = NamingConstraint(
            name="t", description="d", pattern=r".", node_types=[NodeType.CLASS],
            severity=ConstraintSeverity.ERROR, scope=ConstraintScope.MODULE,
        )
        assert c.description == "d"
        assert c.severity == ConstraintSeverity.ERROR
        assert c.scope == ConstraintScope.MODULE
        assert c.enabled is True

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid regex"):
            NamingConstraint(
                name="bad", description="", pattern="[invalid",
                node_types=[NodeType.FUNCTION],
            )

    def test_validate_graph_passing(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("snake_case"))
        c = NamingConstraint(
            name="t", description="", pattern=r"^[a-z][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
        )
        assert c.validate(g) == []

    def test_validate_graph_violation(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("BadName"))
        c = NamingConstraint(
            name="t", description="", pattern=r"^[a-z][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
        )
        violations = c.validate(g)
        assert len(violations) == 1
        assert "BadName" in violations[0].message
        assert violations[0].suggestion is not None

    def test_validate_graph_disabled(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("BadName"))
        c = NamingConstraint(
            name="t", description="", pattern=r"^[a-z]", node_types=[NodeType.FUNCTION],
            enabled=False,
        )
        assert c.validate(g) == []

    def test_validate_node_wrong_type(self) -> None:
        c = NamingConstraint(
            name="t", description="", pattern=r"^[A-Z]",
            node_types=[NodeType.CLASS],
        )
        node = _fn("bad")
        assert c.validate_node(node) is None

    def test_validate_node_disabled(self) -> None:
        c = NamingConstraint(
            name="t", description="", pattern=r"^[a-z]",
            node_types=[NodeType.FUNCTION], enabled=False,
        )
        assert c.validate_node(_fn("BadName")) is None

    def test_exclude_patterns(self) -> None:
        c = NamingConstraint(
            name="t", description="", pattern=r"^[a-z][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            exclude_patterns=[r"^__.*__$"],
        )
        assert c.validate_node(_fn("__init__")) is None

    def test_exclude_patterns_no_match(self) -> None:
        """Exclude patterns exist but don't match — node still validated."""
        c = NamingConstraint(
            name="t", description="", pattern=r"^[a-z][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            exclude_patterns=[r"^__.*__$"],
        )
        # "BadName" doesn't match the exclude pattern, so it goes through validation
        v = c.validate_node(_fn("BadName"))
        assert v is not None

    def test_case_insensitive(self) -> None:
        c = NamingConstraint(
            name="t", description="", pattern=r"^[a-z][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION], case_sensitive=False,
        )
        assert c.validate_node(_fn("Snake_Case")) is None

    def test_to_dict(self) -> None:
        c = NamingConstraint(
            name="t", description="d", pattern=r"^x$",
            node_types=[NodeType.FUNCTION, NodeType.CLASS],
            case_sensitive=False,
        )
        d = c.to_dict()
        assert d["name"] == "t"
        assert d["type"] == "NamingConstraint"
        assert d["pattern"] == "^x$"
        assert d["case_sensitive"] is False
        assert "function" in d["node_types"]

    def test_multiple_node_types(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("BadFunc"))
        g.add_node(_cls("badclass"))
        c = NamingConstraint(
            name="t", description="", pattern=r"^[A-Z]",
            node_types=[NodeType.FUNCTION, NodeType.CLASS],
        )
        violations = c.validate(g)
        # badclass violates (starts with lowercase)
        assert any("badclass" in v.message for v in violations)


# ── MustUseConstraint ────────────────────────────────────────────────────


class TestMustUseConstraint:
    def test_basic_creation(self) -> None:
        c = MustUseConstraint(
            name="docstrings",
            description="Must have docstrings",
            requirement="docstring",
            node_types=[NodeType.FUNCTION],
        )
        assert c.name == "docstrings"
        assert c.requirement == "docstring"
        assert NodeType.FUNCTION in c.node_types

    def test_docstring_present(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("f", docstring="Doc here"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.FUNCTION],
        )
        assert c.validate(g) == []

    def test_docstring_missing(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("f"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.FUNCTION],
        )
        violations = c.validate(g)
        assert len(violations) == 1
        assert "docstring" in violations[0].message

    def test_type_hints_present(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("f", return_type="str"))
        c = MustUseConstraint(
            name="t", description="", requirement="type_hints",
            node_types=[NodeType.FUNCTION],
        )
        assert c.validate(g) == []

    def test_type_hints_missing(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("f"))
        c = MustUseConstraint(
            name="t", description="", requirement="type_hints",
            node_types=[NodeType.FUNCTION],
        )
        assert len(c.validate(g)) == 1

    def test_decorators_present(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("f", decorators=["staticmethod"]))
        c = MustUseConstraint(
            name="t", description="", requirement="decorators",
            node_types=[NodeType.FUNCTION],
        )
        assert c.validate(g) == []

    def test_decorators_missing(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("f"))
        c = MustUseConstraint(
            name="t", description="", requirement="decorators",
            node_types=[NodeType.FUNCTION],
        )
        assert len(c.validate(g)) == 1

    def test_exclude_private(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("_private"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.FUNCTION], exclude_private=True,
        )
        assert c.validate(g) == []

    def test_exclude_dunder(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("__init__"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.FUNCTION], exclude_dunder=True,
        )
        assert c.validate(g) == []

    def test_no_exclude_private(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("_private"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.FUNCTION], exclude_private=False,
        )
        assert len(c.validate(g)) == 1

    def test_disabled(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("f"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.FUNCTION], enabled=False,
        )
        assert c.validate(g) == []

    def test_validate_node_disabled(self) -> None:
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.FUNCTION], enabled=False,
        )
        assert c.validate_node(_fn("f")) is None

    def test_validate_node_wrong_type(self) -> None:
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.CLASS],
        )
        assert c.validate_node(_fn("f")) is None

    def test_class_docstring_check(self) -> None:
        g = SemanticGraph()
        g.add_node(_cls("C"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.CLASS],
        )
        assert len(c.validate(g)) == 1

    def test_class_docstring_present(self) -> None:
        g = SemanticGraph()
        g.add_node(_cls("C", docstring="Doc"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.CLASS],
        )
        assert c.validate(g) == []

    def test_unknown_requirement_passes(self) -> None:
        """Unknown requirement returns True (no violation)."""
        g = SemanticGraph()
        g.add_node(_fn("f"))
        c = MustUseConstraint(
            name="t", description="", requirement="unknown_thing",
            node_types=[NodeType.FUNCTION],
        )
        assert c.validate(g) == []

    def test_to_dict(self) -> None:
        c = MustUseConstraint(
            name="t", description="d", requirement="docstring",
            node_types=[NodeType.FUNCTION], exclude_private=True, exclude_dunder=False,
        )
        d = c.to_dict()
        assert d["requirement"] == "docstring"
        assert d["exclude_private"] is True
        assert d["exclude_dunder"] is False
        assert d["type"] == "MustUseConstraint"

    def test_non_function_node_for_type_hints(self) -> None:
        """type_hints check on a non-FunctionNode class passes."""
        g = SemanticGraph()
        g.add_node(_cls("C"))
        c = MustUseConstraint(
            name="t", description="", requirement="type_hints",
            node_types=[NodeType.CLASS],
        )
        # ClassNode doesn't have return_type, so _check_requirement
        # falls through to return True
        assert c.validate(g) == []

    def test_docstring_on_module_node(self) -> None:
        """docstring requirement on ModuleNode (not Function/Class) passes."""
        g = SemanticGraph()
        g.add_node(_mod("m"))
        c = MustUseConstraint(
            name="t", description="", requirement="docstring",
            node_types=[NodeType.MODULE],
        )
        assert c.validate(g) == []

    def test_decorators_on_class_node(self) -> None:
        """decorators requirement on ClassNode (not FunctionNode) passes."""
        g = SemanticGraph()
        g.add_node(_cls("C"))
        c = MustUseConstraint(
            name="t", description="", requirement="decorators",
            node_types=[NodeType.CLASS],
        )
        assert c.validate(g) == []


# ── MustNotCrossConstraint ────────────────────────────────────────────────


class TestMustNotCrossConstraint:
    def _boundary_graph(self) -> SemanticGraph:
        """Create graph: services/user.py → controllers/api.py (via IMPORTS edge)."""
        g = SemanticGraph()
        svc = _mod("user_svc", "services/user.py")
        ctrl = _mod("api_ctrl", "controllers/api.py")
        g.add_node(svc)
        g.add_node(ctrl)
        # Create import edge with an ImportNode as proxy
        imp = ImportNode(
            name="api_ctrl",
            file_path="controllers/api.py",
            line_start=1,
            line_end=1,
            module="controllers.api",
        )
        g.add_node(imp)
        edge = SemanticEdge(
            source_id=svc.id,
            target_id=imp.id,
            edge_type=EdgeType.IMPORTS,
            line_number=5,
        )
        g.add_edge(edge)
        return g

    def test_basic_creation(self) -> None:
        c = MustNotCrossConstraint(
            name="boundary",
            description="Services cannot import controllers",
            source_pattern=r".*/services/.*",
            forbidden_targets=[r".*/controllers/.*"],
        )
        assert c.name == "boundary"
        assert c.source_pattern == r".*/services/.*"
        assert c.forbidden_targets == [r".*/controllers/.*"]

    def test_violation_detected(self) -> None:
        g = self._boundary_graph()
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*services.*",
            forbidden_targets=[r".*controllers.*"],
        )
        violations = c.validate(g)
        assert len(violations) >= 1
        assert "cannot import" in violations[0].message

    def test_no_violation(self) -> None:
        g = self._boundary_graph()
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*api.*",  # Source doesn't match services
            forbidden_targets=[r".*controllers.*"],
        )
        assert c.validate(g) == []

    def test_disabled(self) -> None:
        g = self._boundary_graph()
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*services.*",
            forbidden_targets=[r".*controllers.*"],
            enabled=False,
        )
        assert c.validate(g) == []

    def test_validate_node_returns_none(self) -> None:
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*", forbidden_targets=[r".*"],
        )
        assert c.validate_node(_fn("f")) is None

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid regex"):
            MustNotCrossConstraint(
                name="t", description="",
                source_pattern="[invalid",
                forbidden_targets=[r".*"],
            )

    def test_to_dict(self) -> None:
        c = MustNotCrossConstraint(
            name="t", description="d",
            source_pattern=r".*svc.*",
            forbidden_targets=[r".*ctrl.*"],
        )
        d = c.to_dict()
        assert d["source_pattern"] == r".*svc.*"
        assert d["forbidden_targets"] == [r".*ctrl.*"]
        assert d["type"] == "MustNotCrossConstraint"

    def test_allowed_targets(self) -> None:
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*",
            forbidden_targets=[r".*"],
            allowed_targets=[r".*ok.*"],
        )
        # Just verify it doesn't crash
        assert c is not None

    def test_dangling_edge_skipped(self) -> None:
        """IMPORTS edge with missing node is skipped (line 577)."""
        g = SemanticGraph()
        a = _mod("a", "services/a.py")
        b = _mod("b", "controllers/b.py")
        g.add_node(a)
        g.add_node(b)
        edge = SemanticEdge(
            source_id=a.id, target_id=b.id,
            edge_type=EdgeType.IMPORTS, line_number=1,
        )
        g.add_edge(edge)
        # Corrupt the graph: remove node from internal storage without cascade
        del g._nodes[b.id]
        if b.node_type in g._nodes_by_type:
            g._nodes_by_type[b.node_type].discard(b.id)
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*services.*",
            forbidden_targets=[r".*controllers.*"],
        )
        # Should not crash — just skip the dangling edge
        assert c.validate(g) == []

    def test_source_matches_but_target_not_forbidden(self) -> None:
        """Source matches pattern but target is not in forbidden list."""
        g = SemanticGraph()
        a = _mod("a", "services/a.py")
        b = _mod("b", "utils/b.py")
        g.add_node(a)
        g.add_node(b)
        edge = SemanticEdge(
            source_id=a.id, target_id=b.id,
            edge_type=EdgeType.IMPORTS, line_number=1,
        )
        g.add_edge(edge)
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*services.*",
            forbidden_targets=[r".*controllers.*"],
        )
        assert c.validate(g) == []

    def test_multiple_forbidden_targets_first_no_match(self) -> None:
        """Multiple forbidden patterns, first doesn't match but second does."""
        g = SemanticGraph()
        a = _mod("a", "services/a.py")
        b = _mod("b", "controllers/b.py")
        g.add_node(a)
        g.add_node(b)
        edge = SemanticEdge(
            source_id=a.id, target_id=b.id,
            edge_type=EdgeType.IMPORTS, line_number=1,
        )
        g.add_edge(edge)
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*services.*",
            forbidden_targets=[r".*repositories.*", r".*controllers.*"],
        )
        violations = c.validate(g)
        assert len(violations) == 1

    def test_source_not_matching_skips(self) -> None:
        """If source node doesn't match source_pattern, skip it."""
        g = SemanticGraph()
        a = _mod("a", "allowed/a.py")
        b = _mod("b", "controllers/b.py")
        g.add_node(a)
        g.add_node(b)
        imp = ImportNode(
            name="b", file_path="controllers/b.py",
            line_start=1, line_end=1, module="controllers.b",
        )
        g.add_node(imp)
        g.add_edge(SemanticEdge(
            source_id=a.id, target_id=imp.id,
            edge_type=EdgeType.IMPORTS, line_number=1,
        ))
        c = MustNotCrossConstraint(
            name="t", description="",
            source_pattern=r".*services.*",
            forbidden_targets=[r".*controllers.*"],
        )
        assert c.validate(g) == []


# ── ErrorFormatConstraint ─────────────────────────────────────────────────


def _exc_cls(
    name: str,
    bases: list[str] | None = None,
    fp: str = "errors.py",
) -> ClassNode:
    return ClassNode(
        name=name,
        file_path=fp,
        line_start=10,
        line_end=20,
        bases=bases or [],
    )


class TestErrorFormatConstraint:
    def test_basic_creation(self) -> None:
        c = ErrorFormatConstraint(
            name="exc_naming",
            description="Exception naming",
            exception_pattern=r"^[A-Z].*Error$",
        )
        assert c.name == "exc_naming"
        assert c.exception_pattern == r"^[A-Z].*Error$"
        assert c.required_bases == []

    def test_properties(self) -> None:
        c = ErrorFormatConstraint(
            name="t",
            description="d",
            exception_pattern=r".*",
            severity=ConstraintSeverity.ERROR,
            scope=ConstraintScope.MODULE,
            required_bases=["BaseError"],
        )
        assert c.description == "d"
        assert c.severity == ConstraintSeverity.ERROR
        assert c.scope == ConstraintScope.MODULE
        assert c.required_bases == ["BaseError"]
        assert c.enabled is True

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid regex"):
            ErrorFormatConstraint(
                name="bad",
                description="",
                exception_pattern="[invalid",
            )

    def test_validate_passes_matching_pattern(self) -> None:
        g = SemanticGraph()
        g.add_node(_exc_cls("NotFoundError", bases=["Exception"]))
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z].*Error$",
        )
        assert c.validate(g) == []

    def test_validate_violation_nonmatching_pattern(self) -> None:
        g = SemanticGraph()
        g.add_node(_exc_cls("bad_error", bases=["Exception"]))
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z].*Error$",
            severity=ConstraintSeverity.ERROR,
        )
        violations = c.validate(g)
        assert len(violations) == 1
        assert "bad_error" in violations[0].message
        assert violations[0].suggestion is not None

    def test_validate_disabled(self) -> None:
        g = SemanticGraph()
        g.add_node(_exc_cls("bad_error", bases=["Exception"]))
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z]",
            enabled=False,
        )
        assert c.validate(g) == []

    def test_validate_skips_non_exception_class(self) -> None:
        g = SemanticGraph()
        g.add_node(_cls("RegularClass"))
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z].*Error$",
        )
        assert c.validate(g) == []

    def test_validate_required_bases_pass(self) -> None:
        g = SemanticGraph()
        g.add_node(_exc_cls("AppError", bases=["BaseError"]))
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z].*Error$",
            required_bases=["BaseError"],
        )
        assert c.validate(g) == []

    def test_validate_required_bases_violation(self) -> None:
        g = SemanticGraph()
        g.add_node(_exc_cls("AppError", bases=["Exception"]))
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z].*Error$",
            required_bases=["BaseError"],
            severity=ConstraintSeverity.ERROR,
        )
        violations = c.validate(g)
        assert len(violations) == 1
        assert "must inherit" in violations[0].message
        assert violations[0].suggestion is not None

    def test_validate_node_disabled(self) -> None:
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r".*",
            enabled=False,
        )
        assert c.validate_node(_exc_cls("FooError", bases=["Exception"])) is None

    def test_validate_node_not_class(self) -> None:
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r".*",
        )
        assert c.validate_node(_fn("func")) is None

    def test_validate_node_not_exception(self) -> None:
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r".*",
        )
        assert c.validate_node(_cls("PlainClass")) is None

    def test_validate_node_pass(self) -> None:
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z].*Error$",
        )
        assert c.validate_node(_exc_cls("AppError", bases=["Exception"])) is None

    def test_validate_node_violation(self) -> None:
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z].*Error$",
        )
        v = c.validate_node(_exc_cls("bad_exc", bases=["Exception"]))
        assert v is not None
        assert "bad_exc" in v.message

    def test_is_exception_by_base_error(self) -> None:
        c = ErrorFormatConstraint(name="t", description="", exception_pattern=r".*")
        assert c._is_exception_class(_exc_cls("Foo", bases=["SomeError"])) is True

    def test_is_exception_by_base_exception(self) -> None:
        c = ErrorFormatConstraint(name="t", description="", exception_pattern=r".*")
        assert c._is_exception_class(_exc_cls("Foo", bases=["Exception"])) is True

    def test_is_exception_by_base_base_exception(self) -> None:
        c = ErrorFormatConstraint(name="t", description="", exception_pattern=r".*")
        assert c._is_exception_class(_exc_cls("Foo", bases=["BaseException"])) is True

    def test_is_exception_by_name_ending_error(self) -> None:
        c = ErrorFormatConstraint(name="t", description="", exception_pattern=r".*")
        assert c._is_exception_class(_exc_cls("CustomError")) is True

    def test_is_exception_by_name_ending_exception(self) -> None:
        c = ErrorFormatConstraint(name="t", description="", exception_pattern=r".*")
        assert c._is_exception_class(_exc_cls("CustomException")) is True

    def test_is_exception_by_name_with_non_indicator_bases(self) -> None:
        """Bases exist but don't contain indicators — falls through to name check."""
        c = ErrorFormatConstraint(name="t", description="", exception_pattern=r".*")
        assert c._is_exception_class(_exc_cls("CustomError", bases=["SomeBase", "Mixin"])) is True

    def test_is_not_exception(self) -> None:
        c = ErrorFormatConstraint(name="t", description="", exception_pattern=r".*")
        assert c._is_exception_class(_exc_cls("RegularClass")) is False

    def test_to_dict(self) -> None:
        c = ErrorFormatConstraint(
            name="exc_fmt",
            description="d",
            exception_pattern=r"^[A-Z].*Error$",
            required_bases=["BaseError", "AppException"],
        )
        d = c.to_dict()
        assert d["type"] == "ErrorFormatConstraint"
        assert d["name"] == "exc_fmt"
        assert d["exception_pattern"] == r"^[A-Z].*Error$"
        assert d["required_bases"] == ["BaseError", "AppException"]

    def test_to_dict_no_required_bases(self) -> None:
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r".*",
        )
        d = c.to_dict()
        assert d["required_bases"] == []

    def test_required_bases_none_becomes_empty_list(self) -> None:
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r".*",
            required_bases=None,
        )
        assert c.required_bases == []

    def test_multiple_exception_classes(self) -> None:
        g = SemanticGraph()
        g.add_node(_exc_cls("GoodError", bases=["Exception"]))
        g.add_node(_exc_cls("bad_error", bases=["Exception"]))
        c = ErrorFormatConstraint(
            name="t",
            description="",
            exception_pattern=r"^[A-Z].*Error$",
        )
        violations = c.validate(g)
        assert len(violations) == 1
        assert "bad_error" in violations[0].message


# ── ConstraintSet ─────────────────────────────────────────────────────────


class TestConstraintSet:
    def test_basic_creation(self) -> None:
        cs = ConstraintSet(name="test", description="Test set")
        assert cs.name == "test"
        assert cs.constraints == []
        assert cs.version == "1.0.0"

    def test_add_constraint(self) -> None:
        cs = ConstraintSet(name="t", description="")
        c = NamingConstraint(
            name="nc", description="", pattern=r".", node_types=[NodeType.FUNCTION],
        )
        cs.add(c)
        assert len(cs.constraints) == 1

    def test_remove_constraint(self) -> None:
        cs = ConstraintSet(name="t", description="")
        c = NamingConstraint(
            name="nc", description="", pattern=r".", node_types=[NodeType.FUNCTION],
        )
        cs.add(c)
        assert cs.remove("nc") is True
        assert len(cs.constraints) == 0

    def test_remove_nonexistent(self) -> None:
        cs = ConstraintSet(name="t", description="")
        assert cs.remove("nonexistent") is False

    def test_get_constraint(self) -> None:
        cs = ConstraintSet(name="t", description="")
        c = NamingConstraint(
            name="nc", description="", pattern=r".", node_types=[NodeType.FUNCTION],
        )
        cs.add(c)
        assert cs.get("nc") is c

    def test_get_nonexistent(self) -> None:
        cs = ConstraintSet(name="t", description="")
        assert cs.get("nonexistent") is None

    def test_validate(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("BadName"))
        cs = ConstraintSet(name="t", description="")
        cs.add(NamingConstraint(
            name="nc", description="", pattern=r"^[a-z]",
            node_types=[NodeType.FUNCTION],
        ))
        violations = cs.validate(g)
        assert len(violations) == 1

    def test_validate_skips_disabled(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("BadName"))
        cs = ConstraintSet(name="t", description="")
        cs.add(NamingConstraint(
            name="nc", description="", pattern=r"^[a-z]",
            node_types=[NodeType.FUNCTION], enabled=False,
        ))
        assert cs.validate(g) == []

    def test_get_errors(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("BadName"))
        cs = ConstraintSet(name="t", description="")
        cs.add(NamingConstraint(
            name="err", description="", pattern=r"^[a-z]",
            node_types=[NodeType.FUNCTION], severity=ConstraintSeverity.ERROR,
        ))
        cs.add(NamingConstraint(
            name="warn", description="", pattern=r"^[a-z]",
            node_types=[NodeType.FUNCTION], severity=ConstraintSeverity.WARNING,
        ))
        errors = cs.get_errors(g)
        assert all(v.severity == ConstraintSeverity.ERROR for v in errors)

    def test_remove_second_constraint(self) -> None:
        """Remove a constraint that's not the first — exercises loop iteration."""
        cs = ConstraintSet(name="t", description="")
        cs.add(NamingConstraint(
            name="first", description="", pattern=r".", node_types=[NodeType.FUNCTION],
        ))
        cs.add(NamingConstraint(
            name="second", description="", pattern=r".", node_types=[NodeType.FUNCTION],
        ))
        assert cs.remove("second") is True
        assert len(cs.constraints) == 1
        assert cs.constraints[0].name == "first"

    def test_get_second_constraint(self) -> None:
        """Get a constraint that's not the first — exercises loop iteration."""
        cs = ConstraintSet(name="t", description="")
        cs.add(NamingConstraint(
            name="first", description="", pattern=r".", node_types=[NodeType.FUNCTION],
        ))
        cs.add(NamingConstraint(
            name="second", description="", pattern=r".", node_types=[NodeType.CLASS],
        ))
        c = cs.get("second")
        assert c is not None
        assert c.name == "second"

    def test_enabled_count(self) -> None:
        cs = ConstraintSet(name="t", description="")
        cs.add(NamingConstraint(
            name="a", description="", pattern=r".", node_types=[NodeType.FUNCTION],
        ))
        cs.add(NamingConstraint(
            name="b", description="", pattern=r".", node_types=[NodeType.FUNCTION],
            enabled=False,
        ))
        assert cs.enabled_count() == 1


# ── Constraint.to_dict ───────────────────────────────────────────────────


class TestConstraintConfidenceBreakCost:
    def test_default_confidence(self) -> None:
        c = NamingConstraint(
            name="t", description="d", pattern=r".", node_types=[NodeType.FUNCTION],
        )
        assert c.confidence == 1.0

    def test_default_break_cost(self) -> None:
        c = NamingConstraint(
            name="t", description="d", pattern=r".", node_types=[NodeType.FUNCTION],
        )
        assert c.break_cost == 1.0

    def test_custom_confidence(self) -> None:
        c = NamingConstraint(
            name="t", description="d", pattern=r".", node_types=[NodeType.FUNCTION],
            confidence=0.75,
        )
        assert c.confidence == 0.75

    def test_custom_break_cost(self) -> None:
        c = NamingConstraint(
            name="t", description="d", pattern=r".", node_types=[NodeType.FUNCTION],
            break_cost=2.5,
        )
        assert c.break_cost == 2.5

    def test_confidence_in_to_dict(self) -> None:
        c = NamingConstraint(
            name="t", description="d", pattern=r".", node_types=[NodeType.FUNCTION],
            confidence=0.9, break_cost=1.5,
        )
        d = c.to_dict()
        assert d["confidence"] == 0.9
        assert d["break_cost"] == 1.5

    def test_confidence_must_use(self) -> None:
        c = MustUseConstraint(
            name="t", description="d", requirement="docstring",
            node_types=[NodeType.FUNCTION], confidence=0.6, break_cost=3.0,
        )
        assert c.confidence == 0.6
        assert c.break_cost == 3.0

    def test_confidence_must_not_cross(self) -> None:
        c = MustNotCrossConstraint(
            name="t", description="d", source_pattern=r".*",
            forbidden_targets=[r".*"], confidence=0.5, break_cost=4.0,
        )
        assert c.confidence == 0.5
        assert c.break_cost == 4.0

    def test_confidence_error_format(self) -> None:
        c = ErrorFormatConstraint(
            name="t", description="d", exception_pattern=r".*",
            confidence=0.3, break_cost=5.0,
        )
        assert c.confidence == 0.3
        assert c.break_cost == 5.0


class TestConstraintBaseToDict:
    def test_base_to_dict(self) -> None:
        c = NamingConstraint(
            name="t", description="d", pattern=r".",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.WARNING,
            scope=ConstraintScope.MODULE,
        )
        d = c.to_dict()
        assert d["name"] == "t"
        assert d["description"] == "d"
        assert d["severity"] == "warning"
        assert d["scope"] == "module"
        assert d["enabled"] is True


# ── ConstraintCompiler ────────────────────────────────────────────────────


class TestConstraintCompiler:
    def test_basic_compile(self) -> None:
        patterns = [
            Pattern(
                name="class_suffix_service",
                pattern_type=PatternType.NAMING,
                description="Service classes",
                regex=r"^[A-Z].*Service$",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        constraints = compiler.compile_patterns(patterns)
        assert len(constraints) == 1

    def test_filters_by_confidence(self) -> None:
        patterns = [
            Pattern(
                name="class_suffix_service",
                pattern_type=PatternType.NAMING,
                description="",
                regex=r"^.*$",
                confidence=PatternConfidence.LOW,
            ),
        ]
        compiler = ConstraintCompiler(min_confidence=PatternConfidence.HIGH)
        assert compiler.compile_patterns(patterns) == []

    def test_medium_confidence_threshold(self) -> None:
        patterns = [
            Pattern(
                name="class_suffix_service",
                pattern_type=PatternType.NAMING,
                description="",
                regex=r"^.*$",
                confidence=PatternConfidence.MEDIUM,
            ),
        ]
        compiler = ConstraintCompiler(min_confidence=PatternConfidence.MEDIUM)
        assert len(compiler.compile_patterns(patterns)) == 1

    def test_low_confidence_passes_low_threshold(self) -> None:
        patterns = [
            Pattern(
                name="class_suffix_service",
                pattern_type=PatternType.NAMING,
                description="",
                regex=r"^.*$",
                confidence=PatternConfidence.LOW,
            ),
        ]
        compiler = ConstraintCompiler(min_confidence=PatternConfidence.LOW)
        assert len(compiler.compile_patterns(patterns)) == 1

    def test_compile_to_set(self) -> None:
        patterns = [
            Pattern(
                name="class_suffix_service",
                pattern_type=PatternType.NAMING,
                description="Service",
                regex=r"^[A-Z].*Service$",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        cs = compiler.compile_to_set(patterns, "my_set", "My description")
        assert cs.name == "my_set"
        assert cs.description == "My description"
        assert len(cs.constraints) == 1

    def test_compile_to_set_default_description(self) -> None:
        compiler = ConstraintCompiler()
        cs = compiler.compile_to_set([], "my_set")
        assert "0 patterns" in cs.description

    def test_naming_pattern_class(self) -> None:
        patterns = [
            Pattern(
                name="class_suffix_service",
                pattern_type=PatternType.NAMING,
                description="",
                regex=r"^.*Service$",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        constraints = compiler.compile_patterns(patterns)
        assert len(constraints) == 1
        assert isinstance(constraints[0], NamingConstraint)

    def test_naming_pattern_function(self) -> None:
        patterns = [
            Pattern(
                name="function_prefix_get",
                pattern_type=PatternType.NAMING,
                description="",
                regex=r"^get_.*$",
                confidence=PatternConfidence.MEDIUM,
            ),
        ]
        compiler = ConstraintCompiler()
        constraints = compiler.compile_patterns(patterns)
        assert len(constraints) == 1

    def test_naming_pattern_no_regex_skipped(self) -> None:
        patterns = [
            Pattern(
                name="class_naming_something",
                pattern_type=PatternType.NAMING,
                description="",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        assert compiler.compile_patterns(patterns) == []

    def test_naming_pattern_unknown_type_skipped(self) -> None:
        """Naming pattern with neither 'class' nor 'function' in name."""
        patterns = [
            Pattern(
                name="variable_naming",
                pattern_type=PatternType.NAMING,
                description="",
                regex=r"^.*$",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        assert compiler.compile_patterns(patterns) == []

    def test_structural_pattern_returns_none(self) -> None:
        patterns = [
            Pattern(
                name="module_dir_services",
                pattern_type=PatternType.STRUCTURAL,
                description="",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        assert compiler.compile_patterns(patterns) == []

    def test_framework_response_model_pattern(self) -> None:
        patterns = [
            Pattern(
                name="fastapi_response_models",
                pattern_type=PatternType.FRAMEWORK,
                description="",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        constraints = compiler.compile_patterns(patterns)
        assert len(constraints) == 1
        assert isinstance(constraints[0], MustUseConstraint)

    def test_framework_non_response_model_skipped(self) -> None:
        patterns = [
            Pattern(
                name="fastapi_get_endpoints",
                pattern_type=PatternType.FRAMEWORK,
                description="",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        assert compiler.compile_patterns(patterns) == []

    def test_dependency_pattern_skipped(self) -> None:
        patterns = [
            Pattern(
                name="dependency_fastapi",
                pattern_type=PatternType.DEPENDENCY,
                description="",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        assert compiler.compile_patterns(patterns) == []

    def test_behavioral_pattern_skipped(self) -> None:
        patterns = [
            Pattern(
                name="async_codebase",
                pattern_type=PatternType.BEHAVIORAL,
                description="",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        assert compiler.compile_patterns(patterns) == []

    def test_high_confidence_naming_gets_error_severity(self) -> None:
        patterns = [
            Pattern(
                name="class_suffix_service",
                pattern_type=PatternType.NAMING,
                description="",
                regex=r"^.*$",
                confidence=PatternConfidence.HIGH,
            ),
        ]
        compiler = ConstraintCompiler()
        constraints = compiler.compile_patterns(patterns)
        assert constraints[0].severity == ConstraintSeverity.ERROR

    def test_medium_confidence_naming_gets_default_severity(self) -> None:
        patterns = [
            Pattern(
                name="class_suffix_service",
                pattern_type=PatternType.NAMING,
                description="",
                regex=r"^.*$",
                confidence=PatternConfidence.MEDIUM,
            ),
        ]
        compiler = ConstraintCompiler(default_severity=ConstraintSeverity.INFO)
        constraints = compiler.compile_patterns(patterns)
        assert constraints[0].severity == ConstraintSeverity.INFO
