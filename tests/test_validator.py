"""Tests for code validator."""

from __future__ import annotations

import pytest

from codebase_intelligence.constraints import (
    ConstraintSet,
    ConstraintSeverity,
    ConstraintViolation,
    MustUseConstraint,
    NamingConstraint,
)
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    FunctionNode,
    ModuleNode,
    NodeType,
)
from codebase_intelligence.parser import ASTParser
from codebase_intelligence.validator import (
    CodeValidator,
    LintResult,
    TestResult,
    ValidationMetrics,
    ValidationResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _fn(
    name: str,
    fp: str = "app.py",
    ls: int = 10,
    le: int = 20,
    docstring: str | None = None,
    return_type: str | None = None,
    decorators: list[str] | None = None,
    params: list[str] | None = None,
) -> FunctionNode:
    return FunctionNode(
        name=name,
        file_path=fp,
        line_start=ls,
        line_end=le,
        docstring=docstring,
        return_type=return_type,
        decorators=decorators or [],
        parameters=params or [],
    )


def _cls(
    name: str,
    fp: str = "app.py",
    ls: int = 10,
    le: int = 50,
    docstring: str | None = None,
) -> ClassNode:
    return ClassNode(
        name=name, file_path=fp, line_start=ls, line_end=le, docstring=docstring
    )


def _mod(name: str, fp: str = "", imports: list[str] | None = None) -> ModuleNode:
    fp = fp or f"{name}.py"
    return ModuleNode(
        name=name, file_path=fp, line_start=1, line_end=100, imports=imports or []
    )


def _empty_constraint_set(name: str = "empty") -> ConstraintSet:
    return ConstraintSet(name=name, description="Empty set for testing")


# ── ValidationMetrics ─────────────────────────────────────────────────────


class TestValidationMetrics:
    def test_default_values(self) -> None:
        m = ValidationMetrics()
        assert m.constraints_checked == 0
        assert m.constraints_passed == 0
        assert m.constraints_failed == 0
        assert m.error_count == 0
        assert m.warning_count == 0
        assert m.info_count == 0
        assert m.nodes_in_generated == 0

    def test_pass_rate_zero_checked(self) -> None:
        m = ValidationMetrics(constraints_checked=0)
        assert m.pass_rate == 1.0

    def test_pass_rate_with_checked_and_passed(self) -> None:
        m = ValidationMetrics(constraints_checked=10, constraints_passed=7)
        assert m.pass_rate == pytest.approx(0.7)

    def test_pass_rate_all_passed(self) -> None:
        m = ValidationMetrics(constraints_checked=5, constraints_passed=5)
        assert m.pass_rate == pytest.approx(1.0)

    def test_pass_rate_none_passed(self) -> None:
        m = ValidationMetrics(constraints_checked=3, constraints_passed=0)
        assert m.pass_rate == pytest.approx(0.0)

    def test_to_dict(self) -> None:
        m = ValidationMetrics(
            constraints_checked=4,
            constraints_passed=3,
            constraints_failed=1,
            error_count=1,
            warning_count=2,
            info_count=0,
            nodes_in_generated=5,
        )
        d = m.to_dict()
        assert d["constraints_checked"] == 4
        assert d["constraints_passed"] == 3
        assert d["constraints_failed"] == 1
        assert d["error_count"] == 1
        assert d["warning_count"] == 2
        assert d["info_count"] == 0
        assert d["nodes_in_generated"] == 5
        assert d["pass_rate"] == pytest.approx(0.75)

    def test_to_dict_includes_pass_rate(self) -> None:
        m = ValidationMetrics(constraints_checked=0)
        d = m.to_dict()
        assert "pass_rate" in d
        assert d["pass_rate"] == 1.0

    def test_frozen(self) -> None:
        m = ValidationMetrics()
        with pytest.raises(AttributeError):
            m.constraints_checked = 5  # type: ignore[misc]


# ── ValidationResult ─────────────────────────────────────────────────────


class TestValidationResult:
    def test_default_values(self) -> None:
        r = ValidationResult()
        assert r.is_valid is True
        assert r.violations == []
        assert isinstance(r.metrics, ValidationMetrics)
        assert r.parse_error is None
        assert r.generated_graph is None

    def test_errors_property_filters_error_severity(self) -> None:
        error_v = ConstraintViolation(
            constraint_name="e1",
            message="error one",
            severity=ConstraintSeverity.ERROR,
        )
        warning_v = ConstraintViolation(
            constraint_name="w1",
            message="warning one",
            severity=ConstraintSeverity.WARNING,
        )
        info_v = ConstraintViolation(
            constraint_name="i1",
            message="info one",
            severity=ConstraintSeverity.INFO,
        )
        r = ValidationResult(violations=[error_v, warning_v, info_v])
        errors = r.errors
        assert len(errors) == 1
        assert errors[0].severity == ConstraintSeverity.ERROR
        assert errors[0].constraint_name == "e1"

    def test_warnings_property_filters_warning_severity(self) -> None:
        error_v = ConstraintViolation(
            constraint_name="e1",
            message="error one",
            severity=ConstraintSeverity.ERROR,
        )
        warning_v = ConstraintViolation(
            constraint_name="w1",
            message="warning one",
            severity=ConstraintSeverity.WARNING,
        )
        info_v = ConstraintViolation(
            constraint_name="i1",
            message="info one",
            severity=ConstraintSeverity.INFO,
        )
        r = ValidationResult(violations=[error_v, warning_v, info_v])
        warnings = r.warnings
        assert len(warnings) == 1
        assert warnings[0].severity == ConstraintSeverity.WARNING
        assert warnings[0].constraint_name == "w1"

    def test_errors_empty_when_no_violations(self) -> None:
        r = ValidationResult()
        assert r.errors == []

    def test_warnings_empty_when_no_violations(self) -> None:
        r = ValidationResult()
        assert r.warnings == []

    def test_to_dict_without_parse_error(self) -> None:
        r = ValidationResult()
        d = r.to_dict()
        assert d["is_valid"] is True
        assert d["violations"] == []
        assert "metrics" in d
        assert d["parse_error"] is None

    def test_to_dict_with_parse_error(self) -> None:
        r = ValidationResult(is_valid=False, parse_error="Syntax error at line 5")
        d = r.to_dict()
        assert d["is_valid"] is False
        assert d["parse_error"] == "Syntax error at line 5"

    def test_to_dict_with_violations(self) -> None:
        v = ConstraintViolation(
            constraint_name="test_c",
            message="bad name",
            severity=ConstraintSeverity.WARNING,
            file_path="test.py",
            line_number=10,
        )
        r = ValidationResult(violations=[v])
        d = r.to_dict()
        assert len(d["violations"]) == 1
        assert d["violations"][0]["constraint_name"] == "test_c"
        assert d["violations"][0]["severity"] == "warning"

    def test_to_dict_metrics_included(self) -> None:
        m = ValidationMetrics(constraints_checked=2, constraints_passed=1)
        r = ValidationResult(metrics=m)
        d = r.to_dict()
        assert d["metrics"]["constraints_checked"] == 2
        assert d["metrics"]["constraints_passed"] == 1


# ── CodeValidator.validate ───────────────────────────────────────────────


class TestCodeValidatorValidate:
    def test_valid_python_no_constraints(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        result = validator.validate(
            source="def hello():\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.is_valid is True
        assert result.violations == []
        assert result.parse_error is None
        assert result.generated_graph is not None

    def test_invalid_python_syntax_error(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        result = validator.validate(
            source="def hello( :\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.is_valid is False
        assert result.parse_error is not None
        assert "Syntax error" in result.parse_error
        assert result.generated_graph is None

    def test_syntax_error_returns_early(self) -> None:
        """When parse fails, no constraints or consistency checks run."""
        validator = CodeValidator()
        naming = NamingConstraint(
            name="snake_case",
            description="Must use snake_case",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        cs = ConstraintSet(
            name="strict", description="Strict rules", constraints=[naming]
        )
        result = validator.validate(
            source="def ???(): pass",
            file_path="test.py",
            constraints=cs,
        )
        assert result.is_valid is False
        assert result.parse_error is not None
        assert result.violations == []

    def test_naming_constraint_violation(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="snake_case_funcs",
            description="Functions must be snake_case",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        cs = ConstraintSet(
            name="style", description="Style rules", constraints=[naming]
        )
        result = validator.validate(
            source="def BadName():\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any(v.constraint_name == "snake_case_funcs" for v in result.violations)
        assert result.metrics.error_count > 0

    def test_naming_constraint_no_violation(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="snake_case_funcs",
            description="Functions must be snake_case",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        cs = ConstraintSet(
            name="style", description="Style rules", constraints=[naming]
        )
        result = validator.validate(
            source="def good_name():\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.is_valid is True
        assert result.metrics.error_count == 0

    def test_warning_violations_do_not_invalidate(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="pascal_classes",
            description="Classes should be PascalCase",
            pattern=r"^[A-Z][a-zA-Z0-9]*$",
            node_types=[NodeType.CLASS],
            severity=ConstraintSeverity.WARNING,
        )
        cs = ConstraintSet(
            name="style", description="Style rules", constraints=[naming]
        )
        result = validator.validate(
            source="class bad_class:\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        # Warnings do not make is_valid False
        assert result.is_valid is True
        assert result.metrics.warning_count > 0
        assert result.metrics.error_count == 0

    def test_consistency_class_name_shadowing(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()

        # Build an original graph with a class named "User" in another file
        original = SemanticGraph()
        original.add_node(
            _cls("User", fp="models.py", ls=1, le=10)
        )

        result = validator.validate(
            source="class User:\n    pass\n",
            file_path="test.py",
            constraints=cs,
            original_graph=original,
        )
        # Shadowing produces a WARNING, so is_valid stays True
        assert result.is_valid is True
        shadow_violations = [
            v for v in result.violations if v.constraint_name == "consistency_no_shadow"
        ]
        assert len(shadow_violations) >= 1
        assert "Class 'User'" in shadow_violations[0].message
        assert shadow_violations[0].severity == ConstraintSeverity.WARNING

    def test_consistency_function_name_shadowing(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()

        original = SemanticGraph()
        original.add_node(
            _fn("process_data", fp="utils.py", ls=1, le=5)
        )

        result = validator.validate(
            source="def process_data():\n    pass\n",
            file_path="test.py",
            constraints=cs,
            original_graph=original,
        )
        shadow_violations = [
            v for v in result.violations if v.constraint_name == "consistency_no_shadow"
        ]
        assert len(shadow_violations) >= 1
        assert "Function 'process_data'" in shadow_violations[0].message
        assert shadow_violations[0].severity == ConstraintSeverity.WARNING

    def test_consistency_no_shadow_when_names_differ(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()

        original = SemanticGraph()
        original.add_node(_cls("ExistingClass", fp="other.py", ls=1, le=10))
        original.add_node(_fn("existing_func", fp="other.py", ls=1, le=5))

        result = validator.validate(
            source="class NewClass:\n    pass\n\ndef new_func():\n    pass\n",
            file_path="test.py",
            constraints=cs,
            original_graph=original,
        )
        shadow_violations = [
            v for v in result.violations if v.constraint_name == "consistency_no_shadow"
        ]
        assert len(shadow_violations) == 0

    def test_consistency_no_shadow_same_file(self) -> None:
        """Names in the same file_path as generated code are excluded from shadowing."""
        validator = CodeValidator()
        cs = _empty_constraint_set()

        original = SemanticGraph()
        # This class is in test.py, same as the generated code file_path
        original.add_node(_cls("User", fp="test.py", ls=1, le=10))

        result = validator.validate(
            source="class User:\n    pass\n",
            file_path="test.py",
            constraints=cs,
            original_graph=original,
        )
        shadow_violations = [
            v for v in result.violations if v.constraint_name == "consistency_no_shadow"
        ]
        assert len(shadow_violations) == 0

    def test_no_original_graph_skips_consistency(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        result = validator.validate(
            source="class User:\n    pass\n",
            file_path="test.py",
            constraints=cs,
            original_graph=None,
        )
        shadow_violations = [
            v for v in result.violations if v.constraint_name == "consistency_no_shadow"
        ]
        assert len(shadow_violations) == 0

    def test_mixed_errors_and_warnings(self) -> None:
        validator = CodeValidator()
        naming_error = NamingConstraint(
            name="snake_case_funcs",
            description="Functions must be snake_case",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        naming_warning = NamingConstraint(
            name="pascal_classes",
            description="Classes should be PascalCase",
            pattern=r"^[A-Z][a-zA-Z0-9]*$",
            node_types=[NodeType.CLASS],
            severity=ConstraintSeverity.WARNING,
        )
        cs = ConstraintSet(
            name="mixed",
            description="Mixed rules",
            constraints=[naming_error, naming_warning],
        )
        source = "class bad_class:\n    pass\n\ndef BadFunc():\n    pass\n"
        result = validator.validate(
            source=source, file_path="test.py", constraints=cs
        )
        assert result.is_valid is False
        assert result.metrics.error_count >= 1
        assert result.metrics.warning_count >= 1

    def test_generated_graph_is_set_on_success(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        result = validator.validate(
            source="x = 1\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.generated_graph is not None

    def test_multiple_constraint_violations(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="snake_case_funcs",
            description="Functions must be snake_case",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        cs = ConstraintSet(
            name="style", description="Style rules", constraints=[naming]
        )
        source = "def BadOne():\n    pass\n\ndef BadTwo():\n    pass\n"
        result = validator.validate(
            source=source, file_path="test.py", constraints=cs
        )
        assert result.is_valid is False
        error_violations = [
            v
            for v in result.violations
            if v.severity == ConstraintSeverity.ERROR
        ]
        assert len(error_violations) >= 2

    def test_consistency_with_both_class_and_function_shadow(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()

        original = SemanticGraph()
        original.add_node(_cls("Foo", fp="other.py", ls=1, le=10))
        original.add_node(_fn("bar", fp="other.py", ls=20, le=30))

        source = "class Foo:\n    pass\n\ndef bar():\n    pass\n"
        result = validator.validate(
            source=source,
            file_path="test.py",
            constraints=cs,
            original_graph=original,
        )
        shadow_violations = [
            v for v in result.violations if v.constraint_name == "consistency_no_shadow"
        ]
        assert len(shadow_violations) == 2
        messages = [v.message for v in shadow_violations]
        assert any("Class 'Foo'" in m for m in messages)
        assert any("Function 'bar'" in m for m in messages)


# ── CodeValidator.validate_source ────────────────────────────────────────


class TestCodeValidatorValidateSource:
    def test_valid_python(self) -> None:
        validator = CodeValidator()
        result = validator.validate_source(
            source="def hello():\n    pass\n",
            file_path="test.py",
        )
        assert result.is_valid is True
        assert result.parse_error is None
        assert result.generated_graph is not None
        assert result.metrics.nodes_in_generated > 0

    def test_invalid_python(self) -> None:
        validator = CodeValidator()
        result = validator.validate_source(
            source="def (\n",
            file_path="test.py",
        )
        assert result.is_valid is False
        assert result.parse_error is not None
        assert result.generated_graph is None

    def test_empty_source(self) -> None:
        validator = CodeValidator()
        result = validator.validate_source(
            source="",
            file_path="test.py",
        )
        # Empty source is valid Python
        assert result.is_valid is True
        assert result.parse_error is None
        assert result.metrics.nodes_in_generated > 0  # At least module node

    def test_complex_valid_source(self) -> None:
        validator = CodeValidator()
        source = (
            "class MyClass:\n"
            "    def method(self):\n"
            "        pass\n"
            "\n"
            "def standalone():\n"
            "    return 42\n"
        )
        result = validator.validate_source(source=source, file_path="test.py")
        assert result.is_valid is True
        # Module + class + function inside class + standalone function = at least 4
        assert result.metrics.nodes_in_generated >= 4

    def test_violations_list_is_empty(self) -> None:
        """validate_source does not run constraint checks."""
        validator = CodeValidator()
        result = validator.validate_source(
            source="def BadName():\n    pass\n",
            file_path="test.py",
        )
        assert result.violations == []


# ── _compute_metrics ─────────────────────────────────────────────────────


class TestComputeMetrics:
    """Tests for _compute_metrics through the validate() public API."""

    def test_no_violations_all_pass(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="snake_case",
            description="snake_case funcs",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        cs = ConstraintSet(
            name="test", description="test", constraints=[naming]
        )
        result = validator.validate(
            source="def good_name():\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.metrics.constraints_checked == 1
        assert result.metrics.constraints_passed == 1
        assert result.metrics.constraints_failed == 0
        assert result.metrics.error_count == 0
        assert result.metrics.warning_count == 0
        assert result.metrics.info_count == 0

    def test_one_error_violation(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="snake_case",
            description="snake_case funcs",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        cs = ConstraintSet(
            name="test", description="test", constraints=[naming]
        )
        result = validator.validate(
            source="def BadName():\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.metrics.constraints_checked == 1
        assert result.metrics.constraints_failed >= 1
        assert result.metrics.error_count >= 1

    def test_warning_counted(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="pascal_classes",
            description="PascalCase classes",
            pattern=r"^[A-Z][a-zA-Z0-9]*$",
            node_types=[NodeType.CLASS],
            severity=ConstraintSeverity.WARNING,
        )
        cs = ConstraintSet(
            name="test", description="test", constraints=[naming]
        )
        result = validator.validate(
            source="class bad_name:\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.metrics.warning_count >= 1
        assert result.metrics.error_count == 0

    def test_info_counted(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="info_naming",
            description="Info-level naming",
            pattern=r"^[A-Z][a-zA-Z0-9]*$",
            node_types=[NodeType.CLASS],
            severity=ConstraintSeverity.INFO,
        )
        cs = ConstraintSet(
            name="test", description="test", constraints=[naming]
        )
        result = validator.validate(
            source="class bad_name:\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.metrics.info_count >= 1
        assert result.metrics.error_count == 0
        assert result.metrics.warning_count == 0

    def test_nodes_in_generated_counted(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        result = validator.validate(
            source="def a():\n    pass\n\ndef b():\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        # At least module node + 2 function nodes
        assert result.metrics.nodes_in_generated >= 3

    def test_distinct_failed_constraints_counted(self) -> None:
        """Multiple violations from same constraint count as one failed constraint."""
        validator = CodeValidator()
        naming = NamingConstraint(
            name="snake_case",
            description="snake_case funcs",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        cs = ConstraintSet(
            name="test", description="test", constraints=[naming]
        )
        source = "def BadOne():\n    pass\n\ndef BadTwo():\n    pass\n"
        result = validator.validate(
            source=source, file_path="test.py", constraints=cs
        )
        # Both violations come from the same constraint "snake_case"
        assert result.metrics.constraints_failed == 1
        assert result.metrics.constraints_passed == 0

    def test_multiple_constraints_mixed_failures(self) -> None:
        validator = CodeValidator()
        func_naming = NamingConstraint(
            name="snake_funcs",
            description="snake funcs",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        class_naming = NamingConstraint(
            name="pascal_classes",
            description="pascal classes",
            pattern=r"^[A-Z][a-zA-Z0-9]*$",
            node_types=[NodeType.CLASS],
            severity=ConstraintSeverity.WARNING,
        )
        cs = ConstraintSet(
            name="test",
            description="test",
            constraints=[func_naming, class_naming],
        )
        # bad function name (violates snake_funcs), good class name (passes pascal_classes)
        source = "class GoodClass:\n    pass\n\ndef BadFunc():\n    pass\n"
        result = validator.validate(
            source=source, file_path="test.py", constraints=cs
        )
        assert result.metrics.constraints_checked == 2
        assert result.metrics.constraints_failed == 1
        assert result.metrics.constraints_passed == 1

    def test_disabled_constraints_not_counted(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="disabled_rule",
            description="Disabled",
            pattern=r"^[a-z_]+$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
            enabled=False,
        )
        cs = ConstraintSet(
            name="test", description="test", constraints=[naming]
        )
        result = validator.validate(
            source="def BadName():\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        # Disabled constraint is not counted
        assert result.metrics.constraints_checked == 0
        assert result.is_valid is True

    def test_consistency_violations_counted_in_metrics(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        original = SemanticGraph()
        original.add_node(_cls("Foo", fp="other.py", ls=1, le=10))

        result = validator.validate(
            source="class Foo:\n    pass\n",
            file_path="test.py",
            constraints=cs,
            original_graph=original,
        )
        # Consistency violations contribute to warning count
        assert result.metrics.warning_count >= 1

    def test_pass_rate_in_result_metrics(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        result = validator.validate(
            source="x = 1\n",
            file_path="test.py",
            constraints=cs,
        )
        # No constraints checked means pass_rate = 1.0
        assert result.metrics.pass_rate == 1.0


# ── CodeValidator with MustUseConstraint ─────────────────────────────────


class TestCodeValidatorMustUse:
    def test_must_use_docstring_violation(self) -> None:
        validator = CodeValidator()
        must_doc = MustUseConstraint(
            name="require_docstrings",
            description="Public functions need docstrings",
            requirement="docstring",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
            exclude_private=True,
            exclude_dunder=True,
        )
        cs = ConstraintSet(
            name="docs", description="Doc rules", constraints=[must_doc]
        )
        result = validator.validate(
            source="def public_func():\n    pass\n",
            file_path="test.py",
            constraints=cs,
        )
        assert result.is_valid is False
        assert result.metrics.error_count >= 1

    def test_must_use_docstring_passes_with_docstring(self) -> None:
        validator = CodeValidator()
        must_doc = MustUseConstraint(
            name="require_docstrings",
            description="Public functions need docstrings",
            requirement="docstring",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
            exclude_private=True,
            exclude_dunder=True,
        )
        cs = ConstraintSet(
            name="docs", description="Doc rules", constraints=[must_doc]
        )
        result = validator.validate(
            source='def public_func():\n    """A docstring."""\n    pass\n',
            file_path="test.py",
            constraints=cs,
        )
        assert result.is_valid is True
        assert result.metrics.error_count == 0


# ── Edge case: constraint + consistency together ─────────────────────────


class TestValidatorIntegration:
    def test_constraints_plus_consistency_both_contribute(self) -> None:
        validator = CodeValidator()
        naming = NamingConstraint(
            name="snake_funcs",
            description="snake_case funcs",
            pattern=r"^[a-z_][a-z0-9_]*$",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        cs = ConstraintSet(
            name="style", description="Style rules", constraints=[naming]
        )
        original = SemanticGraph()
        original.add_node(_fn("BadFunc", fp="other.py", ls=1, le=5))

        # Generated code has a function 'BadFunc' which:
        # 1) violates snake_case naming (ERROR)
        # 2) shadows existing function name (WARNING)
        result = validator.validate(
            source="def BadFunc():\n    pass\n",
            file_path="test.py",
            constraints=cs,
            original_graph=original,
        )
        assert result.is_valid is False
        assert result.metrics.error_count >= 1
        assert result.metrics.warning_count >= 1
        assert len(result.violations) >= 2

    def test_validator_reuse(self) -> None:
        """Validator can be reused for multiple validations."""
        validator = CodeValidator()
        cs = _empty_constraint_set()

        r1 = validator.validate(
            source="def a():\n    pass\n",
            file_path="test1.py",
            constraints=cs,
        )
        r2 = validator.validate(
            source="def b():\n    pass\n",
            file_path="test2.py",
            constraints=cs,
        )
        assert r1.is_valid is True
        assert r2.is_valid is True
        # Each result has its own graph
        assert r1.generated_graph is not r2.generated_graph

    def test_shadow_violation_has_correct_fields(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        original = SemanticGraph()
        original.add_node(_cls("Widget", fp="models.py", ls=1, le=10))

        result = validator.validate(
            source="class Widget:\n    pass\n",
            file_path="test.py",
            constraints=cs,
            original_graph=original,
        )
        shadow = [
            v for v in result.violations if v.constraint_name == "consistency_no_shadow"
        ]
        assert len(shadow) == 1
        v = shadow[0]
        assert v.file_path == "test.py"
        assert v.line_number is not None
        assert v.node_id is not None
        assert v.suggestion is not None
        assert "Widget" in v.suggestion


# ── LintResult ───────────────────────────────────────────────────────────


class TestLintResult:
    def test_defaults(self) -> None:
        r = LintResult()
        assert r.issues == ()
        assert r.tool == "ruff"
        assert r.returncode == 0

    def test_issue_count(self) -> None:
        r = LintResult(issues=("line1", "line2", "line3"))
        assert r.issue_count == 3

    def test_issue_count_zero(self) -> None:
        r = LintResult()
        assert r.issue_count == 0

    def test_passed_true(self) -> None:
        r = LintResult()
        assert r.passed is True

    def test_passed_false(self) -> None:
        r = LintResult(issues=("something",))
        assert r.passed is False

    def test_frozen(self) -> None:
        r = LintResult()
        with pytest.raises(AttributeError):
            r.tool = "other"  # type: ignore[misc]


# ── TestResult ───────────────────────────────────────────────────────────


class TestTestResult:
    def test_defaults(self) -> None:
        r = TestResult()
        assert r.passed == 0
        assert r.failed == 0
        assert r.errors == 0
        assert r.output == ""
        assert r.returncode == 0

    def test_total(self) -> None:
        r = TestResult(passed=5, failed=2, errors=1)
        assert r.total == 8

    def test_all_passed_true(self) -> None:
        r = TestResult(passed=3, failed=0, errors=0)
        assert r.all_passed is True

    def test_all_passed_false_with_failures(self) -> None:
        r = TestResult(passed=3, failed=1, errors=0)
        assert r.all_passed is False

    def test_all_passed_false_with_errors(self) -> None:
        r = TestResult(passed=3, failed=0, errors=1)
        assert r.all_passed is False

    def test_all_passed_false_when_no_tests(self) -> None:
        r = TestResult(passed=0, failed=0, errors=0)
        assert r.all_passed is False

    def test_frozen(self) -> None:
        r = TestResult()
        with pytest.raises(AttributeError):
            r.passed = 10  # type: ignore[misc]


# ── ValidationMetrics extended fields ────────────────────────────────────


class TestValidationMetricsExtended:
    def test_new_default_values(self) -> None:
        m = ValidationMetrics()
        assert m.lint_issue_count == 0
        assert m.test_pass_count == 0
        assert m.test_fail_count == 0
        assert m.lines_added == 0

    def test_to_dict_includes_new_fields(self) -> None:
        m = ValidationMetrics(
            lint_issue_count=3,
            test_pass_count=10,
            test_fail_count=2,
            lines_added=50,
        )
        d = m.to_dict()
        assert d["lint_issue_count"] == 3
        assert d["test_pass_count"] == 10
        assert d["test_fail_count"] == 2
        assert d["lines_added"] == 50


# ── CodeValidator.lint ───────────────────────────────────────────────────


class TestCodeValidatorLint:
    def test_lint_clean_code(self) -> None:
        from unittest.mock import patch, MagicMock

        validator = CodeValidator()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            result = validator.lint("x = 1\n")
        assert result.passed is True
        assert result.issue_count == 0
        assert result.tool == "ruff"
        assert result.returncode == 0

    def test_lint_with_issues(self) -> None:
        from unittest.mock import patch, MagicMock

        validator = CodeValidator()
        mock_result = MagicMock()
        mock_result.stdout = "file.py:1:1: E501 line too long\nfile.py:2:1: F401 unused import\nFound 2 errors.\n"
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = validator.lint("x = 1\n")
        assert result.passed is False
        assert result.issue_count == 2
        assert result.returncode == 1

    def test_lint_tool_not_found(self) -> None:
        from unittest.mock import patch

        validator = CodeValidator()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = validator.lint("x = 1\n")
        assert result.returncode == -1
        assert result.tool == "ruff"

    def test_lint_timeout(self) -> None:
        import subprocess
        from unittest.mock import patch

        validator = CodeValidator()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ruff", timeout=30)):
            result = validator.lint("x = 1\n")
        assert result.returncode == -2

    def test_lint_custom_tool(self) -> None:
        from unittest.mock import patch, MagicMock

        validator = CodeValidator()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            result = validator.lint("x = 1\n", tool="flake8")
        assert result.tool == "flake8"

    def test_lint_filters_found_line(self) -> None:
        from unittest.mock import patch, MagicMock

        validator = CodeValidator()
        mock_result = MagicMock()
        mock_result.stdout = "file.py:1:1: E501 too long\nFound 1 error.\n"
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = validator.lint("x = 1\n")
        assert result.issue_count == 1
        # "Found 1 error." line is filtered out
        assert all("Found" not in issue for issue in result.issues)


# ── CodeValidator.run_tests ──────────────────────────────────────────────


class TestCodeValidatorRunTests:
    def test_run_tests_all_pass(self) -> None:
        from unittest.mock import patch, MagicMock

        validator = CodeValidator()
        mock_result = MagicMock()
        mock_result.stdout = "===== 5 passed in 0.5s =====\n"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            result = validator.run_tests(["python", "-m", "pytest"])
        assert result.passed == 5
        assert result.failed == 0
        assert result.errors == 0
        assert result.all_passed is True
        assert result.returncode == 0

    def test_run_tests_with_failures(self) -> None:
        from unittest.mock import patch, MagicMock

        validator = CodeValidator()
        mock_result = MagicMock()
        mock_result.stdout = "===== 3 passed, 2 failed in 1.0s =====\n"
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = validator.run_tests(["pytest"])
        assert result.passed == 3
        assert result.failed == 2
        assert result.all_passed is False

    def test_run_tests_with_errors(self) -> None:
        from unittest.mock import patch, MagicMock

        validator = CodeValidator()
        mock_result = MagicMock()
        mock_result.stdout = "===== 1 passed, 1 error in 0.5s =====\n"
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = validator.run_tests(["pytest"])
        assert result.passed == 1
        assert result.errors == 1

    def test_run_tests_command_not_found(self) -> None:
        from unittest.mock import patch

        validator = CodeValidator()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = validator.run_tests(["nonexistent_runner"])
        assert result.returncode == -1
        assert "not found" in result.output

    def test_run_tests_timeout(self) -> None:
        import subprocess
        from unittest.mock import patch

        validator = CodeValidator()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=120)):
            result = validator.run_tests(["pytest"], timeout=120)
        assert result.returncode == -2
        assert "timed out" in result.output

    def test_run_tests_with_working_dir(self) -> None:
        from unittest.mock import patch, MagicMock

        validator = CodeValidator()
        mock_result = MagicMock()
        mock_result.stdout = "===== 1 passed in 0.1s =====\n"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            validator.run_tests(["pytest"], working_dir="/tmp/proj")
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["cwd"] == "/tmp/proj"


# ── _parse_pytest_output ─────────────────────────────────────────────────


class TestParsePytestOutput:
    def test_all_passed(self) -> None:
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output("===== 10 passed in 0.5s =====\n")
        assert p == 10
        assert f == 0
        assert e == 0

    def test_mixed_results(self) -> None:
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output(
            "===== 5 passed, 3 failed, 1 error in 2.0s =====\n"
        )
        assert p == 5
        assert f == 3
        assert e == 1

    def test_no_summary_line(self) -> None:
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output("some random output\n")
        assert p == 0
        assert f == 0
        assert e == 0

    def test_empty_output(self) -> None:
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output("")
        assert p == 0
        assert f == 0
        assert e == 0

    def test_only_failed(self) -> None:
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output("===== 2 failed in 0.5s =====\n")
        assert p == 0
        assert f == 2
        assert e == 0

    def test_keyword_at_start_no_number(self) -> None:
        """'passed' at i=0 has no preceding digit — should not crash."""
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output("passed something\n")
        assert p == 0

    def test_non_digit_before_keyword(self) -> None:
        """Word before 'failed' is not a digit — should be ignored."""
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output("tests failed in 0.5s\n")
        assert f == 0

    def test_passed_comma_variant(self) -> None:
        """'passed,' (with comma suffix) matched by startswith."""
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output("===== 3 passed, 1 failed, in 0.5s =====\n")
        assert p == 3
        assert f == 1

    def test_error_keyword_without_preceding_digit(self) -> None:
        """'error' appears but preceding word is not a digit."""
        validator = CodeValidator()
        p, f, e = validator._parse_pytest_output("some error occurred\n")
        assert e == 0


# ── Validate lines_added metric ──────────────────────────────────────────


class TestValidateLinesAdded:
    def test_lines_added_counted(self) -> None:
        validator = CodeValidator()
        cs = _empty_constraint_set()
        source = "def a():\n    pass\n\ndef b():\n    pass\n"
        result = validator.validate(
            source=source,
            file_path="test.py",
            constraints=cs,
        )
        assert result.metrics.lines_added == 5
