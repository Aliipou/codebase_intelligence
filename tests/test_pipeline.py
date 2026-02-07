"""Tests for end-to-end pipeline."""

from __future__ import annotations

import os
import tempfile

import pytest

from codebase_intelligence.compiler import CompiledPrompt
from codebase_intelligence.constraints import (
    ConstraintSet,
    ConstraintSeverity,
    NamingConstraint,
)
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.llm import LLMError, LLMResponse, MessageRole, StubLLMProvider
from codebase_intelligence.nodes import NodeType
from codebase_intelligence.pipeline import (
    GenerationResult,
    Pipeline,
    PipelineConfig,
    PipelineError,
)
from codebase_intelligence.validator import ValidationResult


# ── PipelineConfig ───────────────────────────────────────────────────────


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.max_retries == 3
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 4096
        assert cfg.validate_output is True
        assert cfg.relevant_files is None

    def test_custom_values(self) -> None:
        cfg = PipelineConfig(max_retries=1, temperature=0.8, validate_output=False)
        assert cfg.max_retries == 1
        assert cfg.temperature == 0.8
        assert cfg.validate_output is False

    def test_frozen(self) -> None:
        cfg = PipelineConfig()
        with pytest.raises(AttributeError):
            cfg.max_retries = 10  # type: ignore[misc]

    def test_relevant_files(self) -> None:
        cfg = PipelineConfig(relevant_files=["a.py", "b.py"])
        assert cfg.relevant_files == ["a.py", "b.py"]


# ── GenerationResult ────────────────────────────────────────────────────


class TestGenerationResult:
    def test_defaults(self) -> None:
        r = GenerationResult()
        assert r.source == ""
        assert r.validation is None
        assert r.attempts == 0
        assert r.prompt is None
        assert r.llm_response is None
        assert r.is_valid is False

    def test_to_dict_no_validation(self) -> None:
        r = GenerationResult(source="code", is_valid=True, attempts=1)
        d = r.to_dict()
        assert d["source"] == "code"
        assert d["is_valid"] is True
        assert d["attempts"] == 1
        assert d["validation"] is None

    def test_to_dict_with_validation(self) -> None:
        v = ValidationResult(is_valid=True)
        r = GenerationResult(source="x", validation=v, attempts=2)
        d = r.to_dict()
        assert d["validation"] is not None
        assert d["validation"]["is_valid"] is True


# ── PipelineError ────────────────────────────────────────────────────────


class TestPipelineError:
    def test_creation(self) -> None:
        e = PipelineError("something broke")
        assert str(e) == "something broke"


# ── Pipeline init ────────────────────────────────────────────────────────


class TestPipelineInit:
    def test_default_config(self) -> None:
        provider = StubLLMProvider()
        p = Pipeline(llm=provider)
        assert p.config.max_retries == 3

    def test_custom_config(self) -> None:
        cfg = PipelineConfig(max_retries=5)
        p = Pipeline(llm=StubLLMProvider(), config=cfg)
        assert p.config.max_retries == 5


# ── analyze_repo ─────────────────────────────────────────────────────────


class TestAnalyzeRepo:
    def test_valid_directory(self) -> None:
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "app.py"), "w") as f:
            f.write("def hello():\n    pass\n")
        p = Pipeline(llm=StubLLMProvider())
        graph, constraints = p.analyze_repo(tmpdir)
        assert graph.get_stats().node_count > 0
        assert isinstance(constraints, ConstraintSet)

    def test_not_a_directory(self) -> None:
        p = Pipeline(llm=StubLLMProvider())
        with pytest.raises(PipelineError, match="Not a directory"):
            p.analyze_repo("/nonexistent/path/xyz")

    def test_recursive_false(self) -> None:
        tmpdir = tempfile.mkdtemp()
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "mod.py"), "w") as f:
            f.write("x = 1\n")
        p = Pipeline(llm=StubLLMProvider())
        graph, _ = p.analyze_repo(tmpdir, recursive=False)
        # non-recursive should not find sub/mod.py
        assert graph.get_stats().node_count == 0

    def test_empty_directory(self) -> None:
        tmpdir = tempfile.mkdtemp()
        p = Pipeline(llm=StubLLMProvider())
        graph, constraints = p.analyze_repo(tmpdir)
        assert graph.get_stats().node_count == 0

    def test_parse_failure_raises_pipeline_error(self) -> None:
        from unittest.mock import patch

        tmpdir = tempfile.mkdtemp()
        p = Pipeline(llm=StubLLMProvider())
        with patch.object(p._parser, "parse_directory", side_effect=RuntimeError("boom")):
            with pytest.raises(PipelineError, match="Failed to parse repository"):
                p.analyze_repo(tmpdir)


# ── generate ─────────────────────────────────────────────────────────────


class TestGenerate:
    def _empty_graph_and_constraints(self) -> tuple[SemanticGraph, ConstraintSet]:
        return SemanticGraph(), ConstraintSet(name="t", description="t")

    def test_simple_valid_generation(self) -> None:
        provider = StubLLMProvider(responses=["def hello():\n    pass\n"])
        p = Pipeline(llm=provider)
        graph, cs = self._empty_graph_and_constraints()
        result = p.generate("make hello", graph, cs)
        assert result.is_valid is True
        assert result.attempts == 1
        assert result.source == "def hello():\n    pass\n"
        assert result.prompt is not None
        assert result.llm_response is not None

    def test_validation_disabled(self) -> None:
        cfg = PipelineConfig(validate_output=False)
        provider = StubLLMProvider(responses=["not even python!!!"])
        p = Pipeline(llm=provider, config=cfg)
        graph, cs = self._empty_graph_and_constraints()
        result = p.generate("do thing", graph, cs)
        assert result.is_valid is True
        assert result.validation is None
        assert result.attempts == 1

    def test_syntax_error_retries_then_fails(self) -> None:
        cfg = PipelineConfig(max_retries=2)
        # Both attempts return invalid syntax
        provider = StubLLMProvider(responses=["def !!!"])
        p = Pipeline(llm=provider, config=cfg)
        graph, cs = self._empty_graph_and_constraints()
        result = p.generate("broken", graph, cs)
        assert result.is_valid is False
        assert result.attempts == 2

    def test_retry_succeeds_on_second_attempt(self) -> None:
        cfg = PipelineConfig(max_retries=3)
        # First attempt: syntax error; second: valid code
        provider = StubLLMProvider(responses=["def !!!", "def ok():\n    pass\n"])
        p = Pipeline(llm=provider, config=cfg)
        graph, cs = self._empty_graph_and_constraints()
        result = p.generate("thing", graph, cs)
        assert result.is_valid is True
        assert result.attempts == 2

    def test_llm_error_non_retryable(self) -> None:
        provider = StubLLMProvider(error=LLMError("fail", retryable=False))
        p = Pipeline(llm=provider)
        graph, cs = self._empty_graph_and_constraints()
        result = p.generate("task", graph, cs)
        assert result.is_valid is False
        assert result.source == ""

    def test_llm_error_retryable_exhausts_retries(self) -> None:
        cfg = PipelineConfig(max_retries=2)
        provider = StubLLMProvider(error=LLMError("timeout", retryable=True))
        p = Pipeline(llm=provider, config=cfg)
        graph, cs = self._empty_graph_and_constraints()
        result = p.generate("task", graph, cs)
        assert result.is_valid is False
        assert result.attempts == 2

    def test_relevant_files_passed_to_infer(self) -> None:
        provider = StubLLMProvider(responses=["x = 1\n"])
        p = Pipeline(llm=provider)
        graph, cs = self._empty_graph_and_constraints()
        result = p.generate("task", graph, cs, relevant_files=["app/routes.py"])
        assert result.is_valid is True

    def test_relevant_files_from_config(self) -> None:
        cfg = PipelineConfig(relevant_files=["config.py"])
        provider = StubLLMProvider(responses=["x = 1\n"])
        p = Pipeline(llm=provider, config=cfg)
        graph, cs = self._empty_graph_and_constraints()
        result = p.generate("task", graph, cs)
        assert result.is_valid is True

    def test_naming_constraint_violation_fails(self) -> None:
        provider = StubLLMProvider(responses=["class bad_name:\n    pass\n"])
        cfg = PipelineConfig(max_retries=1)
        p = Pipeline(llm=provider, config=cfg)
        graph = SemanticGraph()
        cs = ConstraintSet(name="t", description="t")
        cs.add(NamingConstraint(
            name="pascal_classes",
            description="PascalCase classes",
            pattern=r"^[A-Z][a-zA-Z0-9]+$",
            node_types=[NodeType.CLASS],
            severity=ConstraintSeverity.ERROR,
        ))
        result = p.generate("make class", graph, cs)
        assert result.is_valid is False
        assert result.validation is not None
        assert len(result.validation.violations) > 0


# ── _infer_file_path ─────────────────────────────────────────────────────


class TestInferFilePath:
    def test_one_relevant_file(self) -> None:
        p = Pipeline(llm=StubLLMProvider())
        assert p._infer_file_path("task", ["single.py"]) == "single.py"

    def test_no_relevant_files(self) -> None:
        p = Pipeline(llm=StubLLMProvider())
        assert p._infer_file_path("task", None) == "generated.py"

    def test_empty_relevant_files(self) -> None:
        p = Pipeline(llm=StubLLMProvider())
        assert p._infer_file_path("task", []) == "generated.py"

    def test_multiple_relevant_files(self) -> None:
        p = Pipeline(llm=StubLLMProvider())
        assert p._infer_file_path("task", ["a.py", "b.py"]) == "generated.py"


# ── _build_request / _build_retry_request ────────────────────────────────


class TestBuildRequest:
    def test_build_request_structure(self) -> None:
        provider = StubLLMProvider()
        p = Pipeline(llm=provider)
        graph, cs = SemanticGraph(), ConstraintSet(name="t", description="t")
        prompt = p._compiler.compile("task", graph, cs)
        req = p._build_request(prompt)
        assert len(req.messages) == 2
        assert req.messages[0].role == MessageRole.SYSTEM
        assert req.messages[1].role == MessageRole.USER

    def test_build_retry_request_structure(self) -> None:
        p = Pipeline(llm=StubLLMProvider())
        graph, cs = SemanticGraph(), ConstraintSet(name="t", description="t")
        prompt = p._compiler.compile("task", graph, cs)
        validation = ValidationResult(is_valid=False)
        req = p._build_retry_request(prompt, "bad code", validation)
        assert len(req.messages) == 4
        assert req.messages[2].role == MessageRole.ASSISTANT
        assert req.messages[2].content == "bad code"
        assert req.messages[3].role == MessageRole.USER
        assert "violations" in req.messages[3].content
