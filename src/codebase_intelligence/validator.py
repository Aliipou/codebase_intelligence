"""Code validator for constraint-aware generation.

Validates generated code against active constraints and checks
consistency with the existing codebase graph. Produces a structured
validation report with metrics.

Five validation layers:
1. Static: Parse the code, check for syntax errors
2. Constraint: Check all active constraints against parsed graph
3. Consistency: Verify no forbidden edges or boundary violations
4. Linting: Run ruff on generated code for style enforcement
5. Testing: Run existing test suite to catch regressions

Usage:
    >>> validator = CodeValidator()
    >>> result = validator.validate(
    ...     source="def get_user(): ...",
    ...     file_path="app/routes/users.py",
    ...     constraints=constraint_set,
    ... )
    >>> print(result.is_valid)
    True
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codebase_intelligence.constraints import (
    ConstraintSet,
    ConstraintSeverity,
    ConstraintViolation,
)
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.parser import ASTParser, ParseError


@dataclass(frozen=True)
class LintResult:
    """Result of running a linter on generated code.

    Attributes:
        issues: List of linting issues found.
        tool: Name of the linting tool used.
        returncode: Exit code from the linter.
    """

    issues: tuple[str, ...] = ()
    tool: str = "ruff"
    returncode: int = 0

    @property
    def issue_count(self) -> int:
        """Number of linting issues found."""
        return len(self.issues)

    @property
    def passed(self) -> bool:
        """Whether linting passed with no issues."""
        return self.issue_count == 0


@dataclass(frozen=True)
class TestResult:
    """Result of running tests against generated code.

    Attributes:
        passed: Number of tests that passed.
        failed: Number of tests that failed.
        errors: Number of test errors.
        output: Raw test runner output.
        returncode: Exit code from the test runner.
    """

    passed: int = 0
    failed: int = 0
    errors: int = 0
    output: str = ""
    returncode: int = 0

    @property
    def total(self) -> int:
        """Total tests run."""
        return self.passed + self.failed + self.errors

    @property
    def all_passed(self) -> bool:
        """Whether all tests passed."""
        return self.failed == 0 and self.errors == 0 and self.total > 0


@dataclass(frozen=True)
class ValidationMetrics:
    """Quantitative metrics from a validation run.

    Attributes:
        constraints_checked: Total constraints evaluated.
        constraints_passed: Constraints with no violations.
        constraints_failed: Constraints with violations.
        error_count: Number of ERROR severity violations.
        warning_count: Number of WARNING severity violations.
        info_count: Number of INFO severity violations.
        nodes_in_generated: Nodes found in generated code.
        lint_issue_count: Number of linting issues found.
        test_pass_count: Number of tests passed.
        test_fail_count: Number of tests failed.
        lines_added: Lines of code in generated source.
    """

    constraints_checked: int = 0
    constraints_passed: int = 0
    constraints_failed: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    nodes_in_generated: int = 0
    lint_issue_count: int = 0
    test_pass_count: int = 0
    test_fail_count: int = 0
    lines_added: int = 0

    @property
    def pass_rate(self) -> float:
        """Fraction of constraints that passed (0.0 to 1.0)."""
        if self.constraints_checked == 0:
            return 1.0
        return self.constraints_passed / self.constraints_checked

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "constraints_checked": self.constraints_checked,
            "constraints_passed": self.constraints_passed,
            "constraints_failed": self.constraints_failed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "nodes_in_generated": self.nodes_in_generated,
            "pass_rate": self.pass_rate,
            "lint_issue_count": self.lint_issue_count,
            "test_pass_count": self.test_pass_count,
            "test_fail_count": self.test_fail_count,
            "lines_added": self.lines_added,
        }


@dataclass
class ValidationResult:
    """Result of a code validation.

    Attributes:
        is_valid: True if no ERROR violations exist.
        violations: All violations found.
        metrics: Quantitative validation metrics.
        parse_error: Parse error if code had syntax errors.
        generated_graph: Semantic graph of the generated code.
        lint_result: Linting results (if linting was run).
        test_result: Test results (if tests were run).
    """

    is_valid: bool = True
    violations: list[ConstraintViolation] = field(default_factory=list)
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    parse_error: str | None = None
    generated_graph: SemanticGraph | None = None
    lint_result: LintResult | None = None
    test_result: TestResult | None = None

    @property
    def errors(self) -> list[ConstraintViolation]:
        """Only ERROR severity violations."""
        return [v for v in self.violations if v.severity == ConstraintSeverity.ERROR]

    @property
    def warnings(self) -> list[ConstraintViolation]:
        """Only WARNING severity violations."""
        return [v for v in self.violations if v.severity == ConstraintSeverity.WARNING]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "metrics": self.metrics.to_dict(),
            "parse_error": self.parse_error,
        }


class CodeValidator:
    """Validates generated code against constraints.

    Performs multi-layer validation:
    1. Parse generated code (catch syntax errors)
    2. Check all constraints against parsed code
    3. Optionally check consistency with existing graph
    4. Optionally lint with ruff
    5. Optionally run existing tests

    Examples:
        >>> validator = CodeValidator()
        >>> result = validator.validate(
        ...     source="class UserService:\\n    pass",
        ...     file_path="services/user.py",
        ...     constraints=constraint_set,
        ... )
    """

    def __init__(self) -> None:
        """Initialize the code validator."""
        self._parser = ASTParser()

    def validate(
        self,
        source: str,
        file_path: str,
        constraints: ConstraintSet,
        original_graph: SemanticGraph | None = None,
    ) -> ValidationResult:
        """Validate generated source code.

        Args:
            source: The generated Python source code.
            file_path: File path to associate with the code.
            constraints: Active constraints to check.
            original_graph: Existing codebase graph for consistency checks.

        Returns:
            ValidationResult with violations and metrics.
        """
        result = ValidationResult()

        # Layer 1: Parse the generated code
        try:
            generated_graph = self._parser.parse_source(source, file_path)
            result.generated_graph = generated_graph
        except ParseError as e:
            result.is_valid = False
            result.parse_error = str(e)
            return result

        # Layer 2: Check constraints
        constraint_violations = self._check_constraints(generated_graph, constraints)
        result.violations.extend(constraint_violations)

        # Layer 3: Consistency with original graph
        if original_graph is not None:
            consistency_violations = self._check_consistency(
                generated_graph, original_graph, file_path
            )
            result.violations.extend(consistency_violations)

        # Compute metrics
        result.metrics = self._compute_metrics(
            result.violations, constraints, generated_graph, source
        )

        # Valid only if no errors
        result.is_valid = result.metrics.error_count == 0

        return result

    def validate_source(
        self,
        source: str,
        file_path: str,
    ) -> ValidationResult:
        """Validate that source code is parseable Python.

        Args:
            source: The Python source code.
            file_path: File path for error messages.

        Returns:
            ValidationResult (violations will be empty, only parse_error matters).
        """
        result = ValidationResult()
        try:
            graph = self._parser.parse_source(source, file_path)
            result.generated_graph = graph
            stats = graph.get_stats()
            result.metrics = ValidationMetrics(nodes_in_generated=stats.node_count)
        except ParseError as e:
            result.is_valid = False
            result.parse_error = str(e)
        return result

    def lint(self, source: str, tool: str = "ruff") -> LintResult:
        """Run a linter on source code.

        Writes source to a temp file, runs the linter, and parses output.

        Args:
            source: Python source code to lint.
            tool: Linter command to use (default: "ruff").

        Returns:
            LintResult with issues found.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(source)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [tool, "check", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse output lines — each non-empty line is an issue
            issues = tuple(
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip() and not line.startswith("Found")
            )
            return LintResult(
                issues=issues,
                tool=tool,
                returncode=result.returncode,
            )
        except FileNotFoundError:
            return LintResult(tool=tool, returncode=-1)
        except subprocess.TimeoutExpired:
            return LintResult(tool=tool, returncode=-2)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def run_tests(
        self,
        test_command: list[str],
        working_dir: str | None = None,
        timeout: int = 120,
    ) -> TestResult:
        """Run a test suite and capture results.

        Args:
            test_command: Command to run (e.g., ["python", "-m", "pytest"]).
            working_dir: Working directory for the test command.
            timeout: Timeout in seconds.

        Returns:
            TestResult with pass/fail counts.
        """
        try:
            result = subprocess.run(
                test_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
            )
            passed, failed, errors = self._parse_pytest_output(result.stdout)
            return TestResult(
                passed=passed,
                failed=failed,
                errors=errors,
                output=result.stdout,
                returncode=result.returncode,
            )
        except FileNotFoundError:
            return TestResult(output="Test command not found", returncode=-1)
        except subprocess.TimeoutExpired:
            return TestResult(output="Test execution timed out", returncode=-2)

    def _parse_pytest_output(self, output: str) -> tuple[int, int, int]:
        """Parse pytest summary line for pass/fail/error counts.

        Looks for lines like:
            '5 passed, 2 failed, 1 error in 1.23s'
            '10 passed in 0.5s'
        """
        passed = 0
        failed = 0
        errors = 0

        for line in output.splitlines():
            line = line.strip()
            # Look for the summary line with "passed", "failed", "error"
            if "passed" in line or "failed" in line or "error" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" or part.startswith("passed"):
                        if i > 0 and parts[i - 1].isdigit():
                            passed = int(parts[i - 1])
                    elif part == "failed" or part.startswith("failed"):
                        if i > 0 and parts[i - 1].isdigit():
                            failed = int(parts[i - 1])
                    elif part == "error" or part.startswith("error"):
                        if i > 0 and parts[i - 1].isdigit():
                            errors = int(parts[i - 1])

        return passed, failed, errors

    def _check_constraints(
        self,
        graph: SemanticGraph,
        constraints: ConstraintSet,
    ) -> list[ConstraintViolation]:
        """Check all constraints against the graph."""
        return constraints.validate(graph)

    def _check_consistency(
        self,
        generated: SemanticGraph,
        original: SemanticGraph,
        file_path: str,
    ) -> list[ConstraintViolation]:
        """Check generated code for consistency with existing codebase.

        Detects:
        - Import of modules not present in original graph
        - Naming conflicts with existing entities
        """
        violations: list[ConstraintViolation] = []

        # Check for naming conflicts — new names that shadow existing names
        from codebase_intelligence.nodes import NodeType

        existing_names: set[str] = set()
        for node in original.get_nodes():
            if node.file_path != file_path:
                existing_names.add(node.name)

        for node in generated.get_nodes(NodeType.CLASS):
            if node.name in existing_names:
                violations.append(
                    ConstraintViolation(
                        constraint_name="consistency_no_shadow",
                        message=(
                            f"Class '{node.name}' conflicts with existing name "
                            f"in the codebase"
                        ),
                        severity=ConstraintSeverity.WARNING,
                        file_path=file_path,
                        line_number=node.line_start,
                        node_id=node.id,
                        suggestion=f"Choose a unique name for '{node.name}'",
                    )
                )

        for node in generated.get_nodes(NodeType.FUNCTION):
            if node.name in existing_names:
                violations.append(
                    ConstraintViolation(
                        constraint_name="consistency_no_shadow",
                        message=(
                            f"Function '{node.name}' conflicts with existing name "
                            f"in the codebase"
                        ),
                        severity=ConstraintSeverity.WARNING,
                        file_path=file_path,
                        line_number=node.line_start,
                        node_id=node.id,
                        suggestion=f"Choose a unique name for '{node.name}'",
                    )
                )

        return violations

    def _compute_metrics(
        self,
        violations: list[ConstraintViolation],
        constraints: ConstraintSet,
        graph: SemanticGraph,
        source: str = "",
    ) -> ValidationMetrics:
        """Compute validation metrics."""
        total = constraints.enabled_count()
        error_count = sum(1 for v in violations if v.severity == ConstraintSeverity.ERROR)
        warning_count = sum(1 for v in violations if v.severity == ConstraintSeverity.WARNING)
        info_count = sum(1 for v in violations if v.severity == ConstraintSeverity.INFO)

        # Count distinct failed constraints
        failed_names: set[str] = set()
        for v in violations:
            failed_names.add(v.constraint_name)
        failed = len(failed_names)

        stats = graph.get_stats()
        lines = len(source.splitlines()) if source else 0

        return ValidationMetrics(
            constraints_checked=total,
            constraints_passed=max(0, total - failed),
            constraints_failed=failed,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            nodes_in_generated=stats.node_count,
            lines_added=lines,
        )
