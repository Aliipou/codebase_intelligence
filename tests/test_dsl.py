"""Tests for constraint DSL serialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from codebase_intelligence.constraints import (
    ConstraintScope,
    ConstraintSet,
    ConstraintSeverity,
    ErrorFormatConstraint,
    MustNotCrossConstraint,
    MustUseConstraint,
    NamingConstraint,
)
from codebase_intelligence.dsl import ConstraintDSL, DSLError
from codebase_intelligence.nodes import NodeType


# ── Helpers ───────────────────────────────────────────────────────────────


def _naming_constraint(
    name: str = "pascal_classes",
    pattern: str = r"^[A-Z][a-zA-Z0-9]*$",
    node_types: list[NodeType] | None = None,
    severity: ConstraintSeverity = ConstraintSeverity.WARNING,
    scope: ConstraintScope = ConstraintScope.GLOBAL,
    enabled: bool = True,
    case_sensitive: bool = True,
) -> NamingConstraint:
    return NamingConstraint(
        name=name,
        description="PascalCase classes",
        pattern=pattern,
        node_types=node_types or [NodeType.CLASS],
        severity=severity,
        scope=scope,
        enabled=enabled,
        case_sensitive=case_sensitive,
    )


def _must_use_constraint(
    name: str = "require_docstrings",
    requirement: str = "docstring",
    node_types: list[NodeType] | None = None,
    severity: ConstraintSeverity = ConstraintSeverity.WARNING,
    scope: ConstraintScope = ConstraintScope.GLOBAL,
    enabled: bool = True,
    exclude_private: bool = True,
    exclude_dunder: bool = True,
) -> MustUseConstraint:
    return MustUseConstraint(
        name=name,
        description="Require docstrings",
        requirement=requirement,
        node_types=node_types or [NodeType.FUNCTION],
        severity=severity,
        scope=scope,
        enabled=enabled,
        exclude_private=exclude_private,
        exclude_dunder=exclude_dunder,
    )


def _must_not_cross_constraint(
    name: str = "service_boundary",
    source_pattern: str = r".*/services/.*",
    forbidden_targets: list[str] | None = None,
    severity: ConstraintSeverity = ConstraintSeverity.ERROR,
    scope: ConstraintScope = ConstraintScope.GLOBAL,
    enabled: bool = True,
) -> MustNotCrossConstraint:
    return MustNotCrossConstraint(
        name=name,
        description="Services cannot depend on controllers",
        source_pattern=source_pattern,
        forbidden_targets=forbidden_targets or [r".*/controllers/.*"],
        severity=severity,
        scope=scope,
        enabled=enabled,
    )


def _constraint_set(
    name: str = "test_set",
    description: str = "A test constraint set",
    version: str = "1.0.0",
    constraints: list | None = None,
) -> ConstraintSet:
    return ConstraintSet(
        name=name,
        description=description,
        version=version,
        constraints=constraints if constraints is not None else [],
    )


# ── DSLError ──────────────────────────────────────────────────────────────


class TestDSLError:
    """Tests for the DSLError exception class."""

    def test_message_only(self) -> None:
        err = DSLError("something broke")
        assert err.message == "something broke"
        assert err.path is None
        assert str(err) == "something broke"

    def test_message_with_path(self) -> None:
        err = DSLError("bad value", path="rules.json")
        assert err.message == "bad value"
        assert err.path == "rules.json"
        assert str(err) == "bad value (in rules.json)"

    def test_path_none_explicitly(self) -> None:
        err = DSLError("oops", path=None)
        assert err.path is None
        assert str(err) == "oops"

    def test_empty_string_path_is_falsy(self) -> None:
        err = DSLError("oops", path="")
        assert err.path == ""
        # empty string is falsy, so no "(in ...)" suffix
        assert str(err) == "oops"

    def test_is_exception(self) -> None:
        with pytest.raises(DSLError):
            raise DSLError("fail")


# ── ConstraintDSL save / load ────────────────────────────────────────────


class TestConstraintDSLSaveLoad:
    """Tests for save() and load() file operations."""

    def test_save_creates_file(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(constraints=[_naming_constraint()])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.json"
            dsl.save(cs, p)
            assert p.exists()
            data = json.loads(p.read_text(encoding="utf-8"))
            assert data["name"] == "test_set"
            assert len(data["constraints"]) == 1

    def test_save_accepts_str_path(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set()
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "out.json")
            dsl.save(cs, p)
            assert Path(p).exists()

    def test_load_returns_constraint_set(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(constraints=[_naming_constraint()])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.json"
            dsl.save(cs, p)
            loaded = dsl.load(p)
            assert loaded.name == "test_set"
            assert len(loaded.constraints) == 1

    def test_load_accepts_str_path(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(constraints=[_naming_constraint()])
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "out.json")
            dsl.save(cs, p)
            loaded = dsl.load(p)
            assert loaded.name == "test_set"

    def test_load_file_not_found(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(FileNotFoundError, match="Constraint file not found"):
            dsl.load(Path("nonexistent_file_abc123.json"))

    def test_load_invalid_json(self) -> None:
        dsl = ConstraintDSL()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{not valid json!!!", encoding="utf-8")
            with pytest.raises(DSLError, match="Invalid JSON"):
                dsl.load(p)

    def test_load_invalid_json_includes_path(self) -> None:
        dsl = ConstraintDSL()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("<<< not json >>>", encoding="utf-8")
            with pytest.raises(DSLError) as exc_info:
                dsl.load(p)
            assert exc_info.value.path == str(p)

    def test_round_trip_naming_constraint(self) -> None:
        dsl = ConstraintDSL()
        original = _naming_constraint(
            name="snake_funcs",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
            scope=ConstraintScope.MODULE,
            enabled=False,
            case_sensitive=False,
        )
        cs = _constraint_set(constraints=[original])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.json"
            dsl.save(cs, p)
            loaded = dsl.load(p)

        c = loaded.constraints[0]
        assert isinstance(c, NamingConstraint)
        assert c.name == "snake_funcs"
        assert c.pattern == r"^[a-z_][a-z0-9_]*$"
        assert c.node_types == {NodeType.FUNCTION}
        assert c.severity == ConstraintSeverity.ERROR
        assert c.scope == ConstraintScope.MODULE
        assert c.enabled is False

    def test_round_trip_must_use_constraint(self) -> None:
        dsl = ConstraintDSL()
        original = _must_use_constraint(
            name="type_hints_required",
            requirement="type_hints",
            node_types=[NodeType.FUNCTION, NodeType.METHOD],
            severity=ConstraintSeverity.INFO,
            scope=ConstraintScope.CLASS,
            enabled=True,
            exclude_private=False,
            exclude_dunder=False,
        )
        cs = _constraint_set(constraints=[original])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.json"
            dsl.save(cs, p)
            loaded = dsl.load(p)

        c = loaded.constraints[0]
        assert isinstance(c, MustUseConstraint)
        assert c.name == "type_hints_required"
        assert c.requirement == "type_hints"
        assert c.severity == ConstraintSeverity.INFO
        assert c.scope == ConstraintScope.CLASS

    def test_round_trip_must_not_cross_constraint(self) -> None:
        dsl = ConstraintDSL()
        original = _must_not_cross_constraint(
            name="no_cross",
            source_pattern=r".*core.*",
            forbidden_targets=[r".*ui.*", r".*api.*"],
            severity=ConstraintSeverity.WARNING,
            scope=ConstraintScope.FUNCTION,
        )
        cs = _constraint_set(constraints=[original])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.json"
            dsl.save(cs, p)
            loaded = dsl.load(p)

        c = loaded.constraints[0]
        assert isinstance(c, MustNotCrossConstraint)
        assert c.name == "no_cross"
        assert c.source_pattern == r".*core.*"
        assert c.forbidden_targets == [r".*ui.*", r".*api.*"]
        assert c.severity == ConstraintSeverity.WARNING
        assert c.scope == ConstraintScope.FUNCTION

    def test_round_trip_multiple_constraints(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(
            constraints=[
                _naming_constraint(),
                _must_use_constraint(),
                _must_not_cross_constraint(),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.json"
            dsl.save(cs, p)
            loaded = dsl.load(p)

        assert len(loaded.constraints) == 3
        assert isinstance(loaded.constraints[0], NamingConstraint)
        assert isinstance(loaded.constraints[1], MustUseConstraint)
        assert isinstance(loaded.constraints[2], MustNotCrossConstraint)

    def test_round_trip_preserves_version_and_description(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(
            name="versioned",
            description="My custom rules",
            version="2.5.1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.json"
            dsl.save(cs, p)
            loaded = dsl.load(p)

        assert loaded.name == "versioned"
        assert loaded.description == "My custom rules"
        assert loaded.version == "2.5.1"


# ── ConstraintDSL to_dict / from_dict ────────────────────────────────────


class TestConstraintDSLToFromDict:
    """Tests for to_dict() and from_dict() dictionary operations."""

    def test_to_dict_basic(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(name="basic", description="desc", version="1.2.3")
        d = dsl.to_dict(cs)
        assert d["name"] == "basic"
        assert d["description"] == "desc"
        assert d["version"] == "1.2.3"
        assert d["constraints"] == []

    def test_to_dict_with_constraints(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(constraints=[_naming_constraint()])
        d = dsl.to_dict(cs)
        assert len(d["constraints"]) == 1
        assert d["constraints"][0]["type"] == "NamingConstraint"

    def test_from_dict_data_not_dict(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="must be a dictionary"):
            dsl.from_dict("not a dict")  # type: ignore[arg-type]

    def test_from_dict_data_not_dict_list(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="must be a dictionary"):
            dsl.from_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_from_dict_missing_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="must have a 'name' field"):
            dsl.from_dict({})

    def test_from_dict_empty_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="must have a 'name' field"):
            dsl.from_dict({"name": ""})

    def test_from_dict_constraints_not_list(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="'constraints' must be a list"):
            dsl.from_dict({"name": "test", "constraints": "not a list"})

    def test_from_dict_defaults(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({"name": "minimal"})
        assert cs.name == "minimal"
        assert cs.description == ""
        assert cs.version == "1.0.0"
        assert cs.constraints == []

    def test_from_dict_with_source_none(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({"name": "test"}, source=None)
        assert cs.name == "test"

    def test_from_dict_with_source_string(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError) as exc_info:
            dsl.from_dict({}, source="my_file.json")
        assert exc_info.value.path == "my_file.json"

    def test_from_dict_full(self) -> None:
        dsl = ConstraintDSL()
        data = {
            "name": "full_test",
            "description": "Complete test",
            "version": "3.0.0",
            "constraints": [
                {
                    "type": "NamingConstraint",
                    "name": "pascal",
                    "pattern": r"^[A-Z]",
                    "node_types": ["class"],
                    "severity": "error",
                    "scope": "module",
                    "enabled": False,
                    "case_sensitive": False,
                },
            ],
        }
        cs = dsl.from_dict(data)
        assert cs.name == "full_test"
        assert cs.description == "Complete test"
        assert cs.version == "3.0.0"
        assert len(cs.constraints) == 1
        c = cs.constraints[0]
        assert isinstance(c, NamingConstraint)
        assert c.enabled is False

    def test_from_dict_round_trip(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(
            constraints=[
                _naming_constraint(),
                _must_use_constraint(),
                _must_not_cross_constraint(),
            ],
        )
        d = dsl.to_dict(cs)
        loaded = dsl.from_dict(d)
        assert loaded.name == cs.name
        assert len(loaded.constraints) == len(cs.constraints)


# ── ConstraintDSL merge ──────────────────────────────────────────────────


class TestConstraintDSLMerge:
    """Tests for merge() combining multiple constraint sets."""

    def test_merge_empty(self) -> None:
        dsl = ConstraintDSL()
        result = dsl.merge()
        assert result.constraints == []
        assert result.name == ""
        assert result.description == "Merged constraint set"

    def test_merge_single(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(name="only", constraints=[_naming_constraint()])
        result = dsl.merge(cs)
        assert result.name == "only"
        assert len(result.constraints) == 1

    def test_merge_two_disjoint(self) -> None:
        dsl = ConstraintDSL()
        cs1 = _constraint_set(name="A", constraints=[_naming_constraint(name="a")])
        cs2 = _constraint_set(name="B", constraints=[_must_use_constraint(name="b")])
        result = dsl.merge(cs1, cs2)
        assert result.name == "A + B"
        assert len(result.constraints) == 2

    def test_merge_overlapping_names_last_wins(self) -> None:
        dsl = ConstraintDSL()
        c1 = _naming_constraint(name="shared_name")
        c2 = _must_use_constraint(name="shared_name")
        cs1 = _constraint_set(name="A", constraints=[c1])
        cs2 = _constraint_set(name="B", constraints=[c2])
        result = dsl.merge(cs1, cs2)
        assert len(result.constraints) == 1
        assert isinstance(result.constraints[0], MustUseConstraint)

    def test_merge_three_sets(self) -> None:
        dsl = ConstraintDSL()
        cs1 = _constraint_set(name="A", constraints=[_naming_constraint(name="a1")])
        cs2 = _constraint_set(name="B", constraints=[_must_use_constraint(name="b1")])
        cs3 = _constraint_set(
            name="C",
            constraints=[_must_not_cross_constraint(name="c1")],
        )
        result = dsl.merge(cs1, cs2, cs3)
        assert result.name == "A + B + C"
        assert len(result.constraints) == 3

    def test_merge_preserves_order_of_last_seen(self) -> None:
        dsl = ConstraintDSL()
        c1 = _naming_constraint(name="x")
        c2 = _must_use_constraint(name="y")
        c3 = _must_not_cross_constraint(name="x")  # overrides c1
        cs1 = _constraint_set(name="A", constraints=[c1, c2])
        cs2 = _constraint_set(name="B", constraints=[c3])
        result = dsl.merge(cs1, cs2)
        assert len(result.constraints) == 2
        names = [c.name for c in result.constraints]
        assert "x" in names
        assert "y" in names
        # "x" should be the MustNotCross variant (overridden)
        x_constraint = [c for c in result.constraints if c.name == "x"][0]
        assert isinstance(x_constraint, MustNotCrossConstraint)


# ── ConstraintDSL parsing (individual constraint types) ──────────────────


class TestConstraintDSLParsing:
    """Tests for _parse_constraint and related parsing internals."""

    # -- _parse_constraint top-level errors --

    def test_parse_constraint_not_dict(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="Constraint at index 0 must be a dictionary"):
            dsl.from_dict({"name": "test", "constraints": ["not a dict"]})

    def test_parse_constraint_not_dict_number(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="Constraint at index 0 must be a dictionary"):
            dsl.from_dict({"name": "test", "constraints": [42]})

    def test_parse_constraint_missing_type(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'type' field"):
            dsl.from_dict({"name": "test", "constraints": [{"name": "foo"}]})

    def test_parse_constraint_empty_type(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'type' field"):
            dsl.from_dict({"name": "test", "constraints": [{"type": ""}]})

    def test_parse_constraint_unknown_type(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="Unknown constraint type 'FooConstraint'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{"type": "FooConstraint", "name": "x"}],
            })

    def test_parse_constraint_unknown_type_with_source(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError) as exc_info:
            dsl.from_dict(
                {
                    "name": "test",
                    "constraints": [{"type": "BadType"}],
                },
                source="somefile.json",
            )
        assert exc_info.value.path == "somefile.json"

    # -- severity parsing --

    def test_parse_severity_valid_values(self) -> None:
        dsl = ConstraintDSL()
        for sev in ("error", "warning", "info"):
            cs = dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "pattern": "^x",
                    "node_types": ["class"],
                    "severity": sev,
                }],
            })
            assert cs.constraints[0].severity.value == sev

    def test_parse_severity_invalid(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="Invalid severity 'critical'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "pattern": "^x",
                    "node_types": ["class"],
                    "severity": "critical",
                }],
            })

    def test_parse_severity_default_is_warning(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": ["class"],
            }],
        })
        assert cs.constraints[0].severity == ConstraintSeverity.WARNING

    # -- scope parsing --

    def test_parse_scope_valid_values(self) -> None:
        dsl = ConstraintDSL()
        for sc in ("global", "module", "class", "function"):
            cs = dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "pattern": "^x",
                    "node_types": ["class"],
                    "scope": sc,
                }],
            })
            assert cs.constraints[0].scope.value == sc

    def test_parse_scope_invalid(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="Invalid scope 'project'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "pattern": "^x",
                    "node_types": ["class"],
                    "scope": "project",
                }],
            })

    def test_parse_scope_default_is_global(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": ["class"],
            }],
        })
        assert cs.constraints[0].scope == ConstraintScope.GLOBAL

    # -- node_types parsing --

    def test_parse_node_types_all_valid(self) -> None:
        dsl = ConstraintDSL()
        all_types = [nt.value for nt in NodeType]
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": all_types,
            }],
        })
        c = cs.constraints[0]
        assert isinstance(c, NamingConstraint)
        assert c.node_types == set(NodeType)

    def test_parse_node_types_invalid(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="Invalid node type 'alien'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "pattern": "^x",
                    "node_types": ["class", "alien"],
                }],
            })

    # -- NamingConstraint parsing --

    def test_parse_naming_missing_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="NamingConstraint at index 0 missing 'name'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "pattern": "^x",
                    "node_types": ["class"],
                }],
            })

    def test_parse_naming_empty_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="NamingConstraint at index 0 missing 'name'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "",
                    "pattern": "^x",
                    "node_types": ["class"],
                }],
            })

    def test_parse_naming_missing_pattern(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'pattern'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "node_types": ["class"],
                }],
            })

    def test_parse_naming_empty_pattern(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'pattern'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "pattern": "",
                    "node_types": ["class"],
                }],
            })

    def test_parse_naming_missing_node_types(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'node_types'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "pattern": "^x",
                }],
            })

    def test_parse_naming_empty_node_types(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'node_types'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "NamingConstraint",
                    "name": "nc",
                    "pattern": "^x",
                    "node_types": [],
                }],
            })

    def test_parse_naming_case_sensitive_true(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": ["class"],
                "case_sensitive": True,
            }],
        })
        c = cs.constraints[0]
        assert isinstance(c, NamingConstraint)

    def test_parse_naming_case_sensitive_false(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": ["class"],
                "case_sensitive": False,
            }],
        })
        c = cs.constraints[0]
        assert isinstance(c, NamingConstraint)

    def test_parse_naming_case_sensitive_default(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": ["class"],
            }],
        })
        # default case_sensitive is True (checked by round-trip to_dict)
        d = dsl.to_dict(cs)
        assert d["constraints"][0]["case_sensitive"] is True

    def test_parse_naming_with_description(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "description": "A naming rule",
                "pattern": "^x",
                "node_types": ["class"],
            }],
        })
        assert cs.constraints[0].description == "A naming rule"

    def test_parse_naming_description_default_empty(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": ["class"],
            }],
        })
        assert cs.constraints[0].description == ""

    def test_parse_naming_enabled_default_true(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": ["class"],
            }],
        })
        assert cs.constraints[0].enabled is True

    def test_parse_naming_enabled_false(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^x",
                "node_types": ["class"],
                "enabled": False,
            }],
        })
        assert cs.constraints[0].enabled is False

    # -- MustUseConstraint parsing --

    def test_parse_must_use_missing_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="MustUseConstraint at index 0 missing 'name'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustUseConstraint",
                    "requirement": "docstring",
                    "node_types": ["function"],
                }],
            })

    def test_parse_must_use_empty_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="MustUseConstraint at index 0 missing 'name'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustUseConstraint",
                    "name": "",
                    "requirement": "docstring",
                    "node_types": ["function"],
                }],
            })

    def test_parse_must_use_missing_requirement(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'requirement'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustUseConstraint",
                    "name": "mu",
                    "node_types": ["function"],
                }],
            })

    def test_parse_must_use_empty_requirement(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'requirement'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustUseConstraint",
                    "name": "mu",
                    "requirement": "",
                    "node_types": ["function"],
                }],
            })

    def test_parse_must_use_missing_node_types(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'node_types'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustUseConstraint",
                    "name": "mu",
                    "requirement": "docstring",
                }],
            })

    def test_parse_must_use_empty_node_types(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'node_types'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustUseConstraint",
                    "name": "mu",
                    "requirement": "docstring",
                    "node_types": [],
                }],
            })

    def test_parse_must_use_exclude_private_default(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "mu",
                "requirement": "docstring",
                "node_types": ["function"],
            }],
        })
        d = dsl.to_dict(cs)
        assert d["constraints"][0]["exclude_private"] is True

    def test_parse_must_use_exclude_private_false(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "mu",
                "requirement": "docstring",
                "node_types": ["function"],
                "exclude_private": False,
            }],
        })
        d = dsl.to_dict(cs)
        assert d["constraints"][0]["exclude_private"] is False

    def test_parse_must_use_exclude_dunder_default(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "mu",
                "requirement": "docstring",
                "node_types": ["function"],
            }],
        })
        d = dsl.to_dict(cs)
        assert d["constraints"][0]["exclude_dunder"] is True

    def test_parse_must_use_exclude_dunder_false(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "mu",
                "requirement": "docstring",
                "node_types": ["function"],
                "exclude_dunder": False,
            }],
        })
        d = dsl.to_dict(cs)
        assert d["constraints"][0]["exclude_dunder"] is False

    def test_parse_must_use_with_description(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "mu",
                "description": "Must have docs",
                "requirement": "docstring",
                "node_types": ["function"],
            }],
        })
        assert cs.constraints[0].description == "Must have docs"

    def test_parse_must_use_invalid_node_type(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="Invalid node type 'bogus'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustUseConstraint",
                    "name": "mu",
                    "requirement": "docstring",
                    "node_types": ["bogus"],
                }],
            })

    # -- MustNotCrossConstraint parsing --

    def test_parse_must_not_cross_missing_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(
            DSLError,
            match="MustNotCrossConstraint at index 0 missing 'name'",
        ):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustNotCrossConstraint",
                    "source_pattern": ".*",
                    "forbidden_targets": [".*"],
                }],
            })

    def test_parse_must_not_cross_empty_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(
            DSLError,
            match="MustNotCrossConstraint at index 0 missing 'name'",
        ):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustNotCrossConstraint",
                    "name": "",
                    "source_pattern": ".*",
                    "forbidden_targets": [".*"],
                }],
            })

    def test_parse_must_not_cross_missing_source_pattern(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'source_pattern'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustNotCrossConstraint",
                    "name": "mnc",
                    "forbidden_targets": [".*"],
                }],
            })

    def test_parse_must_not_cross_empty_source_pattern(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'source_pattern'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustNotCrossConstraint",
                    "name": "mnc",
                    "source_pattern": "",
                    "forbidden_targets": [".*"],
                }],
            })

    def test_parse_must_not_cross_missing_forbidden_targets(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'forbidden_targets'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustNotCrossConstraint",
                    "name": "mnc",
                    "source_pattern": ".*",
                }],
            })

    def test_parse_must_not_cross_empty_forbidden_targets(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'forbidden_targets'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "MustNotCrossConstraint",
                    "name": "mnc",
                    "source_pattern": ".*",
                    "forbidden_targets": [],
                }],
            })

    def test_parse_must_not_cross_valid(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustNotCrossConstraint",
                "name": "mnc",
                "source_pattern": r".*/a/.*",
                "forbidden_targets": [r".*/b/.*", r".*/c/.*"],
                "severity": "error",
                "scope": "global",
                "enabled": True,
                "description": "No crossing",
            }],
        })
        c = cs.constraints[0]
        assert isinstance(c, MustNotCrossConstraint)
        assert c.name == "mnc"
        assert c.source_pattern == r".*/a/.*"
        assert c.forbidden_targets == [r".*/b/.*", r".*/c/.*"]
        assert c.severity == ConstraintSeverity.ERROR
        assert c.scope == ConstraintScope.GLOBAL
        assert c.enabled is True
        assert c.description == "No crossing"

    def test_parse_must_not_cross_default_description(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustNotCrossConstraint",
                "name": "mnc",
                "source_pattern": ".*",
                "forbidden_targets": [".*"],
            }],
        })
        assert cs.constraints[0].description == ""

    # -- Index in error messages --

    def test_error_at_nonzero_index(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="at index 1"):
            dsl.from_dict({
                "name": "test",
                "constraints": [
                    {
                        "type": "NamingConstraint",
                        "name": "ok",
                        "pattern": "^x",
                        "node_types": ["class"],
                    },
                    {
                        "type": "NamingConstraint",
                        # missing name
                        "pattern": "^x",
                        "node_types": ["class"],
                    },
                ],
            })

    def test_error_index_for_not_dict(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="at index 2"):
            dsl.from_dict({
                "name": "test",
                "constraints": [
                    {
                        "type": "NamingConstraint",
                        "name": "a",
                        "pattern": "^x",
                        "node_types": ["class"],
                    },
                    {
                        "type": "NamingConstraint",
                        "name": "b",
                        "pattern": "^y",
                        "node_types": ["function"],
                    },
                    None,  # not a dict
                ],
            })


# ── Branch partials (defaults, edge-case combinations) ───────────────────


class TestBranchPartials:
    """Tests exercising remaining branch partial combinations."""

    def test_enabled_default_must_use(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "mu",
                "requirement": "docstring",
                "node_types": ["function"],
            }],
        })
        assert cs.constraints[0].enabled is True

    def test_enabled_false_must_use(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "mu",
                "requirement": "docstring",
                "node_types": ["function"],
                "enabled": False,
            }],
        })
        assert cs.constraints[0].enabled is False

    def test_enabled_default_must_not_cross(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustNotCrossConstraint",
                "name": "mnc",
                "source_pattern": ".*",
                "forbidden_targets": [".*"],
            }],
        })
        assert cs.constraints[0].enabled is True

    def test_enabled_false_must_not_cross(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustNotCrossConstraint",
                "name": "mnc",
                "source_pattern": ".*",
                "forbidden_targets": [".*"],
                "enabled": False,
            }],
        })
        assert cs.constraints[0].enabled is False

    def test_severity_info_scope_function(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustNotCrossConstraint",
                "name": "mnc",
                "source_pattern": ".*",
                "forbidden_targets": [".*"],
                "severity": "info",
                "scope": "function",
            }],
        })
        assert cs.constraints[0].severity == ConstraintSeverity.INFO
        assert cs.constraints[0].scope == ConstraintScope.FUNCTION

    def test_multiple_node_types_naming(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "nc",
                "pattern": "^[a-z]",
                "node_types": ["function", "method", "variable"],
            }],
        })
        c = cs.constraints[0]
        assert isinstance(c, NamingConstraint)
        assert c.node_types == {NodeType.FUNCTION, NodeType.METHOD, NodeType.VARIABLE}

    def test_multiple_node_types_must_use(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "mu",
                "requirement": "type_hints",
                "node_types": ["function", "class"],
            }],
        })
        c = cs.constraints[0]
        assert isinstance(c, MustUseConstraint)
        assert c.node_types == {NodeType.FUNCTION, NodeType.CLASS}

    def test_dsl_error_raised_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise DSLError("generic error")

    def test_load_chained_exception_from_json_decode(self) -> None:
        dsl = ConstraintDSL()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{bad}", encoding="utf-8")
            with pytest.raises(DSLError) as exc_info:
                dsl.load(p)
            # verify the chain: __cause__ should be json.JSONDecodeError
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)

    def test_save_json_is_well_formatted(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(constraints=[_naming_constraint()])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fmt.json"
            dsl.save(cs, p)
            text = p.read_text(encoding="utf-8")
            # should be indented with 2 spaces
            assert "\n  " in text

    def test_from_dict_constraints_key_absent_defaults_to_empty(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({"name": "noconstraints"})
        assert cs.constraints == []

    def test_node_type_map_coverage(self) -> None:
        """Ensure the NODE_TYPE_MAP is populated for all NodeType values."""
        dsl = ConstraintDSL()
        for nt in NodeType:
            assert nt.value in dsl._NODE_TYPE_MAP
            assert dsl._NODE_TYPE_MAP[nt.value] is nt

    def test_severity_map_coverage(self) -> None:
        """Ensure the SEVERITY_MAP is populated for all ConstraintSeverity values."""
        dsl = ConstraintDSL()
        for s in ConstraintSeverity:
            assert s.value in dsl._SEVERITY_MAP
            assert dsl._SEVERITY_MAP[s.value] is s

    def test_scope_map_coverage(self) -> None:
        """Ensure the SCOPE_MAP is populated for all ConstraintScope values."""
        dsl = ConstraintDSL()
        for s in ConstraintScope:
            assert s.value in dsl._SCOPE_MAP
            assert dsl._SCOPE_MAP[s.value] is s

    def test_to_dict_constraint_calls_constraint_to_dict(self) -> None:
        """to_dict delegates to each constraint's to_dict method."""
        dsl = ConstraintDSL()
        nc = _naming_constraint()
        mu = _must_use_constraint()
        mnc = _must_not_cross_constraint()
        cs = _constraint_set(constraints=[nc, mu, mnc])
        d = dsl.to_dict(cs)
        assert d["constraints"][0]["type"] == "NamingConstraint"
        assert d["constraints"][1]["type"] == "MustUseConstraint"
        assert d["constraints"][2]["type"] == "MustNotCrossConstraint"

    def test_merge_with_empty_constraint_sets(self) -> None:
        dsl = ConstraintDSL()
        cs1 = _constraint_set(name="A", constraints=[])
        cs2 = _constraint_set(name="B", constraints=[])
        result = dsl.merge(cs1, cs2)
        assert result.name == "A + B"
        assert result.constraints == []

    def test_from_dict_not_dict_with_source(self) -> None:
        """Ensure source path propagates in non-dict error."""
        dsl = ConstraintDSL()
        with pytest.raises(DSLError) as exc_info:
            dsl.from_dict(42, source="input.json")  # type: ignore[arg-type]
        assert exc_info.value.path == "input.json"
        assert "must be a dictionary" in exc_info.value.message

    def test_from_dict_empty_name_with_source(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError) as exc_info:
            dsl.from_dict({"name": ""}, source="file.json")
        assert exc_info.value.path == "file.json"

    def test_from_dict_constraints_not_list_with_source(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError) as exc_info:
            dsl.from_dict(
                {"name": "test", "constraints": 123},
                source="origin.json",
            )
        assert exc_info.value.path == "origin.json"

    def test_full_round_trip_all_fields_naming(self) -> None:
        """Full round-trip for NamingConstraint preserving all fields."""
        dsl = ConstraintDSL()
        raw = {
            "name": "my_set",
            "description": "full set",
            "version": "9.9.9",
            "constraints": [{
                "type": "NamingConstraint",
                "name": "strict",
                "description": "Strict naming",
                "pattern": r"^[A-Z][a-z]+$",
                "node_types": ["class", "function"],
                "severity": "error",
                "scope": "class",
                "enabled": False,
                "case_sensitive": False,
            }],
        }
        cs = dsl.from_dict(raw)
        d = dsl.to_dict(cs)
        assert d["name"] == "my_set"
        assert d["description"] == "full set"
        assert d["version"] == "9.9.9"
        c = d["constraints"][0]
        assert c["type"] == "NamingConstraint"
        assert c["name"] == "strict"
        assert c["description"] == "Strict naming"
        assert c["pattern"] == r"^[A-Z][a-z]+$"
        assert set(c["node_types"]) == {"class", "function"}
        assert c["severity"] == "error"
        assert c["scope"] == "class"
        assert c["enabled"] is False
        assert c["case_sensitive"] is False

    def test_full_round_trip_all_fields_must_use(self) -> None:
        """Full round-trip for MustUseConstraint preserving all fields."""
        dsl = ConstraintDSL()
        raw = {
            "name": "my_set",
            "description": "",
            "version": "1.0.0",
            "constraints": [{
                "type": "MustUseConstraint",
                "name": "docs",
                "description": "Need docs",
                "requirement": "type_hints",
                "node_types": ["method"],
                "severity": "info",
                "scope": "module",
                "enabled": True,
                "exclude_private": False,
                "exclude_dunder": False,
            }],
        }
        cs = dsl.from_dict(raw)
        d = dsl.to_dict(cs)
        c = d["constraints"][0]
        assert c["type"] == "MustUseConstraint"
        assert c["requirement"] == "type_hints"
        assert c["node_types"] == ["method"]
        assert c["exclude_private"] is False
        assert c["exclude_dunder"] is False

    def test_full_round_trip_all_fields_must_not_cross(self) -> None:
        """Full round-trip for MustNotCrossConstraint preserving all fields."""
        dsl = ConstraintDSL()
        raw = {
            "name": "my_set",
            "description": "",
            "version": "1.0.0",
            "constraints": [{
                "type": "MustNotCrossConstraint",
                "name": "boundary",
                "description": "No cross",
                "source_pattern": r"^core/.*",
                "forbidden_targets": [r"^ui/.*", r"^api/.*"],
                "severity": "warning",
                "scope": "module",
                "enabled": False,
            }],
        }
        cs = dsl.from_dict(raw)
        d = dsl.to_dict(cs)
        c = d["constraints"][0]
        assert c["type"] == "MustNotCrossConstraint"
        assert c["source_pattern"] == r"^core/.*"
        assert c["forbidden_targets"] == [r"^ui/.*", r"^api/.*"]
        assert c["severity"] == "warning"
        assert c["scope"] == "module"
        assert c["enabled"] is False


# ── ErrorFormatConstraint DSL ─────────────────────────────────────────────


def _error_format_constraint(
    name: str = "exc_naming",
    exception_pattern: str = r"^[A-Z].*Error$",
    required_bases: list[str] | None = None,
    severity: ConstraintSeverity = ConstraintSeverity.WARNING,
    scope: ConstraintScope = ConstraintScope.GLOBAL,
    enabled: bool = True,
) -> ErrorFormatConstraint:
    return ErrorFormatConstraint(
        name=name,
        description="Exception naming rules",
        exception_pattern=exception_pattern,
        severity=severity,
        scope=scope,
        enabled=enabled,
        required_bases=required_bases,
    )


class TestErrorFormatConstraintDSL:
    """Tests for ErrorFormatConstraint serialization and deserialization."""

    def test_round_trip(self) -> None:
        dsl = ConstraintDSL()
        original = _error_format_constraint(
            name="exc_fmt",
            exception_pattern=r"^[A-Z].*Error$",
            required_bases=["BaseError"],
            severity=ConstraintSeverity.ERROR,
            scope=ConstraintScope.MODULE,
            enabled=False,
        )
        cs = _constraint_set(constraints=[original])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.json"
            dsl.save(cs, p)
            loaded = dsl.load(p)

        c = loaded.constraints[0]
        assert isinstance(c, ErrorFormatConstraint)
        assert c.name == "exc_fmt"
        assert c.exception_pattern == r"^[A-Z].*Error$"
        assert c.required_bases == ["BaseError"]
        assert c.severity == ConstraintSeverity.ERROR
        assert c.scope == ConstraintScope.MODULE
        assert c.enabled is False

    def test_round_trip_no_required_bases(self) -> None:
        dsl = ConstraintDSL()
        original = _error_format_constraint(required_bases=None)
        cs = _constraint_set(constraints=[original])
        d = dsl.to_dict(cs)
        loaded = dsl.from_dict(d)
        c = loaded.constraints[0]
        assert isinstance(c, ErrorFormatConstraint)
        assert c.required_bases == []

    def test_parse_missing_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="ErrorFormatConstraint at index 0 missing 'name'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "ErrorFormatConstraint",
                    "exception_pattern": ".*",
                }],
            })

    def test_parse_empty_name(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="ErrorFormatConstraint at index 0 missing 'name'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "ErrorFormatConstraint",
                    "name": "",
                    "exception_pattern": ".*",
                }],
            })

    def test_parse_missing_exception_pattern(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'exception_pattern'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "ErrorFormatConstraint",
                    "name": "ef",
                }],
            })

    def test_parse_empty_exception_pattern(self) -> None:
        dsl = ConstraintDSL()
        with pytest.raises(DSLError, match="missing 'exception_pattern'"):
            dsl.from_dict({
                "name": "test",
                "constraints": [{
                    "type": "ErrorFormatConstraint",
                    "name": "ef",
                    "exception_pattern": "",
                }],
            })

    def test_parse_valid_with_all_fields(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "ErrorFormatConstraint",
                "name": "ef",
                "description": "Custom errors",
                "exception_pattern": r"^App.*Error$",
                "required_bases": ["BaseError"],
                "severity": "error",
                "scope": "module",
                "enabled": False,
            }],
        })
        c = cs.constraints[0]
        assert isinstance(c, ErrorFormatConstraint)
        assert c.name == "ef"
        assert c.description == "Custom errors"
        assert c.exception_pattern == r"^App.*Error$"
        assert c.required_bases == ["BaseError"]
        assert c.severity == ConstraintSeverity.ERROR
        assert c.scope == ConstraintScope.MODULE
        assert c.enabled is False

    def test_parse_defaults(self) -> None:
        dsl = ConstraintDSL()
        cs = dsl.from_dict({
            "name": "test",
            "constraints": [{
                "type": "ErrorFormatConstraint",
                "name": "ef",
                "exception_pattern": ".*",
            }],
        })
        c = cs.constraints[0]
        assert isinstance(c, ErrorFormatConstraint)
        assert c.description == ""
        assert c.required_bases == []
        assert c.severity == ConstraintSeverity.WARNING
        assert c.scope == ConstraintScope.GLOBAL
        assert c.enabled is True

    def test_full_round_trip_to_dict_from_dict(self) -> None:
        dsl = ConstraintDSL()
        raw = {
            "name": "my_set",
            "description": "rules",
            "version": "2.0.0",
            "constraints": [{
                "type": "ErrorFormatConstraint",
                "name": "exc_fmt",
                "description": "Exception rules",
                "exception_pattern": r"^[A-Z].*Error$",
                "required_bases": ["AppError", "BaseException"],
                "severity": "info",
                "scope": "class",
                "enabled": True,
            }],
        }
        cs = dsl.from_dict(raw)
        d = dsl.to_dict(cs)
        c = d["constraints"][0]
        assert c["type"] == "ErrorFormatConstraint"
        assert c["name"] == "exc_fmt"
        assert c["exception_pattern"] == r"^[A-Z].*Error$"
        assert c["required_bases"] == ["AppError", "BaseException"]
        assert c["severity"] == "info"
        assert c["scope"] == "class"
        assert c["enabled"] is True

    def test_mixed_constraints_with_error_format(self) -> None:
        dsl = ConstraintDSL()
        cs = _constraint_set(
            constraints=[
                _naming_constraint(),
                _error_format_constraint(),
                _must_not_cross_constraint(),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rt.json"
            dsl.save(cs, p)
            loaded = dsl.load(p)

        assert len(loaded.constraints) == 3
        assert isinstance(loaded.constraints[0], NamingConstraint)
        assert isinstance(loaded.constraints[1], ErrorFormatConstraint)
        assert isinstance(loaded.constraints[2], MustNotCrossConstraint)
