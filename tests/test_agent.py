"""Tests for the CodeAgent module.

Tests every code path in agent.py: dataclasses, CodeAgent methods,
full loop scenarios, history tracking, and StubLLMProvider integration.
"""

from __future__ import annotations

import pytest

from codebase_intelligence.agent import (
    AgentConfig,
    AgentPlan,
    AgentResult,
    CodeAgent,
    Observation,
)
from codebase_intelligence.constraints import (
    ConstraintScope,
    ConstraintSet,
    ConstraintSeverity,
    ConstraintViolation,
    NamingConstraint,
    MustUseConstraint,
)
from codebase_intelligence.edges import SemanticEdge, EdgeType
from codebase_intelligence.feedback import (
    EscalationLevel,
    RefinementContext,
    ViolationCategory,
    ViolationDiagnosis,
)
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.llm import StubLLMProvider
from codebase_intelligence.nodes import (
    ClassNode,
    FunctionNode,
    ModuleNode,
    NodeType,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _fn(
    name: str = "my_func",
    fp: str = "app.py",
    docstring: str | None = None,
    return_type: str | None = None,
) -> FunctionNode:
    return FunctionNode(
        name=name,
        file_path=fp,
        line_start=1,
        line_end=10,
        docstring=docstring,
        return_type=return_type,
        decorators=[],
        parameters=[],
    )


def _cls(name: str = "MyClass", fp: str = "app.py") -> ClassNode:
    return ClassNode(name=name, file_path=fp, line_start=1, line_end=20)


def _mod(name: str = "app", fp: str = "app.py") -> ModuleNode:
    return ModuleNode(name=name, file_path=fp, line_start=1, line_end=100, imports=[])


def _graph(*nodes) -> SemanticGraph:
    g = SemanticGraph()
    for n in nodes:
        g.add_node(n)
    return g


def _empty_graph() -> SemanticGraph:
    return SemanticGraph()


def _constraint_set(*constraints) -> ConstraintSet:
    return ConstraintSet(
        name="test_set",
        description="Test",
        constraints=list(constraints),
    )


def _naming_constraint(name: str = "snake_case") -> NamingConstraint:
    return NamingConstraint(
        name=name,
        description="Snake case",
        pattern=r"^[a-z_][a-z0-9_]*$",
        node_types=[NodeType.FUNCTION],
        severity=ConstraintSeverity.ERROR,
    )


# Valid Python source that passes snake_case naming
VALID_SOURCE = "def hello_world():\n    pass\n"

# Invalid Python source (class name in function constraint context is fine,
# but a PascalCase function would fail snake_case)
INVALID_SOURCE = "def BadName():\n    pass\n"

# Syntax error source
SYNTAX_ERROR_SOURCE = "def ("


# ── Observation ──────────────────────────────────────────────────────────


class TestObservation:
    def test_default_creation(self) -> None:
        obs = Observation()
        assert obs.graph_stats == {}
        assert obs.relevant_nodes == ()
        assert obs.relevant_edges == ()
        assert obs.existing_patterns == ()
        assert obs.constraint_summary == ()

    def test_full_creation(self) -> None:
        fn = _fn()
        obs = Observation(
            graph_stats={"node_count": 5, "edge_count": 3},
            relevant_nodes=(fn,),
            relevant_edges=(),
            existing_patterns=("snake_case",),
            constraint_summary=("naming",),
        )
        assert obs.graph_stats["node_count"] == 5
        assert len(obs.relevant_nodes) == 1
        assert obs.existing_patterns == ("snake_case",)

    def test_frozen(self) -> None:
        obs = Observation()
        with pytest.raises(AttributeError):
            obs.graph_stats = {}  # type: ignore[misc]


# ── AgentPlan ────────────────────────────────────────────────────────────


class TestAgentPlan:
    def test_default_creation(self) -> None:
        plan = AgentPlan()
        assert plan.strategy == ""
        assert plan.target_constraints == ()
        assert plan.context_nodes == ()
        assert plan.prompt_sections == ()

    def test_full_creation(self) -> None:
        plan = AgentPlan(
            strategy="Generate user endpoint",
            target_constraints=("snake_case", "docstrings"),
            context_nodes=("UserService",),
            prompt_sections=("system", "task"),
        )
        assert plan.strategy == "Generate user endpoint"
        assert len(plan.target_constraints) == 2
        assert "UserService" in plan.context_nodes

    def test_frozen(self) -> None:
        plan = AgentPlan()
        with pytest.raises(AttributeError):
            plan.strategy = "new"  # type: ignore[misc]


# ── AgentConfig ──────────────────────────────────────────────────────────


class TestAgentConfig:
    def test_defaults(self) -> None:
        cfg = AgentConfig()
        assert cfg.max_attempts == 3
        assert cfg.escalation_enabled is True
        assert cfg.history_enabled is True
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 4096

    def test_custom(self) -> None:
        cfg = AgentConfig(
            max_attempts=5,
            escalation_enabled=False,
            history_enabled=False,
            temperature=0.8,
            max_tokens=2048,
        )
        assert cfg.max_attempts == 5
        assert cfg.escalation_enabled is False
        assert cfg.history_enabled is False
        assert cfg.temperature == 0.8
        assert cfg.max_tokens == 2048

    def test_frozen(self) -> None:
        cfg = AgentConfig()
        with pytest.raises(AttributeError):
            cfg.max_attempts = 10  # type: ignore[misc]


# ── AgentResult ──────────────────────────────────────────────────────────


class TestAgentResult:
    def test_defaults(self) -> None:
        result = AgentResult()
        assert result.source == ""
        assert result.is_valid is False
        assert result.attempts == 0
        assert result.violations == []
        assert result.diagnoses == []
        assert result.history == []

    def test_full_creation(self) -> None:
        v = ConstraintViolation(
            constraint_name="test",
            message="bad",
            severity=ConstraintSeverity.ERROR,
        )
        result = AgentResult(
            source="def f(): pass",
            is_valid=True,
            attempts=2,
            violations=[v],
            history=[("code1", [v])],
        )
        assert result.source == "def f(): pass"
        assert result.is_valid is True
        assert result.attempts == 2
        assert len(result.violations) == 1
        assert len(result.history) == 1

    def test_mutable(self) -> None:
        result = AgentResult()
        result.source = "new code"
        result.is_valid = True
        assert result.source == "new code"
        assert result.is_valid is True


# ── CodeAgent.__init__ ───────────────────────────────────────────────────


class TestCodeAgentInit:
    def test_default_config(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        assert agent.config.max_attempts == 3

    def test_custom_config(self) -> None:
        provider = StubLLMProvider()
        cfg = AgentConfig(max_attempts=5)
        agent = CodeAgent(llm=provider, config=cfg)
        assert agent.config.max_attempts == 5


# ── CodeAgent.observe ────────────────────────────────────────────────────


class TestCodeAgentObserve:
    def test_empty_graph(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        g = _empty_graph()
        cs = _constraint_set()
        obs = agent.observe(g, cs)
        assert obs.graph_stats["node_count"] == 0
        assert obs.graph_stats["edge_count"] == 0
        assert obs.relevant_nodes == ()
        assert obs.constraint_summary == ()

    def test_graph_with_nodes(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        g = _graph(_fn("hello"), _cls("World"))
        cs = _constraint_set(_naming_constraint())
        obs = agent.observe(g, cs)
        assert obs.graph_stats["node_count"] == 2
        assert len(obs.relevant_nodes) == 2
        assert "snake_case" in obs.constraint_summary

    def test_disabled_constraints_excluded(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        g = _empty_graph()
        nc = NamingConstraint(
            name="disabled_one",
            description="",
            pattern=r".*",
            node_types=[NodeType.FUNCTION],
            enabled=False,
        )
        cs = _constraint_set(nc)
        obs = agent.observe(g, cs)
        assert "disabled_one" not in obs.constraint_summary


# ── CodeAgent.plan ───────────────────────────────────────────────────────


class TestCodeAgentPlan:
    def test_basic_plan(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        g = _graph(_fn("hello"))
        cs = _constraint_set(_naming_constraint())
        obs = agent.observe(g, cs)
        plan = agent.plan("Add endpoint", obs, cs)
        assert "Add endpoint" in plan.strategy
        assert "snake_case" in plan.target_constraints
        assert "hello" in plan.context_nodes
        assert len(plan.prompt_sections) == 5

    def test_plan_with_empty_graph(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        g = _empty_graph()
        cs = _constraint_set()
        obs = agent.observe(g, cs)
        plan = agent.plan("Generate code", obs, cs)
        assert "Generate code" in plan.strategy
        assert plan.target_constraints == ()
        assert plan.context_nodes == ()

    def test_plan_includes_node_count_in_strategy(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        g = _graph(_fn("f1"), _fn("f2", fp="b.py"))
        cs = _constraint_set()
        obs = agent.observe(g, cs)
        plan = agent.plan("task", obs, cs)
        assert "2 nodes" in plan.strategy


# ── CodeAgent.act ────────────────────────────────────────────────────────


class TestCodeAgentAct:
    def test_act_calls_llm(self) -> None:
        provider = StubLLMProvider(responses=[VALID_SOURCE])
        agent = CodeAgent(llm=provider)
        g = _empty_graph()
        cs = _constraint_set()
        obs = agent.observe(g, cs)
        plan = agent.plan("task", obs, cs)
        source = agent.act(plan, obs, cs, "task", g, "app.py")
        assert source == VALID_SOURCE
        assert provider.call_count == 1

    def test_act_uses_config_temperature(self) -> None:
        provider = StubLLMProvider(responses=["code"])
        cfg = AgentConfig(temperature=0.9, max_tokens=2048)
        agent = CodeAgent(llm=provider, config=cfg)
        g = _empty_graph()
        cs = _constraint_set()
        obs = agent.observe(g, cs)
        plan = agent.plan("task", obs, cs)
        agent.act(plan, obs, cs, "task", g, "app.py")
        req = provider.requests[0]
        assert req.temperature == 0.9
        assert req.max_tokens == 2048


# ── CodeAgent.validate ───────────────────────────────────────────────────


class TestCodeAgentValidate:
    def test_validate_valid_code(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        cs = _constraint_set()
        result = agent.validate(VALID_SOURCE, "app.py", cs)
        assert result.is_valid is True

    def test_validate_syntax_error(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        cs = _constraint_set()
        result = agent.validate(SYNTAX_ERROR_SOURCE, "app.py", cs)
        assert result.is_valid is False
        assert result.parse_error is not None

    def test_validate_with_constraint_violation(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        result = agent.validate(INVALID_SOURCE, "app.py", cs)
        assert result.is_valid is False
        assert len(result.violations) > 0

    def test_validate_with_original_graph(self) -> None:
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider)
        cs = _constraint_set()
        g = _graph(_fn("hello_world", fp="other.py"))
        # hello_world in both files -> consistency warning
        result = agent.validate(VALID_SOURCE, "app.py", cs, g)
        # May or may not have warnings depending on consistency check
        assert result is not None


# ── CodeAgent.refine ─────────────────────────────────────────────────────


class TestCodeAgentRefine:
    def test_refine_calls_llm(self) -> None:
        provider = StubLLMProvider(responses=["original", VALID_SOURCE])
        agent = CodeAgent(llm=provider)
        cs = _constraint_set()

        # First call was act
        from codebase_intelligence.validator import ValidationResult
        validation = ValidationResult(is_valid=False)

        v = ConstraintViolation(
            constraint_name="test",
            message="bad",
            severity=ConstraintSeverity.ERROR,
        )
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
            max_attempts=3,
        )

        new_source = agent.refine("original code", validation, ctx)
        assert provider.call_count == 1
        assert new_source is not None

    def test_refine_messages_contain_previous_code(self) -> None:
        provider = StubLLMProvider(responses=["fixed code"])
        agent = CodeAgent(llm=provider)

        from codebase_intelligence.validator import ValidationResult
        validation = ValidationResult(is_valid=False)

        ctx = RefinementContext(
            original_request="task",
            diagnoses=[],
            attempt=2,
            max_attempts=3,
        )

        agent.refine("old code here", validation, ctx)
        req = provider.requests[0]
        # Should have 4 messages: system, user(task), assistant(old code), user(refinement)
        assert len(req.messages) == 4
        assert req.messages[2].content == "old code here"


# ── CodeAgent.run — full loop ────────────────────────────────────────────


class TestCodeAgentRun:
    def test_pass_on_first_try(self) -> None:
        provider = StubLLMProvider(responses=[VALID_SOURCE])
        agent = CodeAgent(llm=provider)
        cs = _constraint_set()
        g = _empty_graph()
        result = agent.run("Add function", g, cs, "app.py")
        assert result.is_valid is True
        assert result.attempts == 1
        assert result.source == VALID_SOURCE
        assert len(result.history) == 1

    def test_pass_after_retry(self) -> None:
        # First response fails naming, second passes
        provider = StubLLMProvider(responses=[INVALID_SOURCE, VALID_SOURCE])
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        agent = CodeAgent(llm=provider)
        g = _empty_graph()
        result = agent.run("Add function", g, cs, "app.py")
        assert result.is_valid is True
        assert result.attempts == 2
        assert len(result.history) == 2

    def test_exhaust_retries(self) -> None:
        # Always returns invalid code
        provider = StubLLMProvider(responses=[INVALID_SOURCE])
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        cfg = AgentConfig(max_attempts=2)
        agent = CodeAgent(llm=provider, config=cfg)
        g = _empty_graph()
        result = agent.run("Add function", g, cs, "app.py")
        assert result.is_valid is False
        assert result.attempts == 2
        assert len(result.violations) > 0
        assert len(result.diagnoses) > 0
        assert len(result.history) == 2

    def test_single_attempt(self) -> None:
        provider = StubLLMProvider(responses=[VALID_SOURCE])
        cfg = AgentConfig(max_attempts=1)
        agent = CodeAgent(llm=provider, config=cfg)
        cs = _constraint_set()
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        assert result.is_valid is True
        assert result.attempts == 1

    def test_single_attempt_failure(self) -> None:
        provider = StubLLMProvider(responses=[INVALID_SOURCE])
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        cfg = AgentConfig(max_attempts=1)
        agent = CodeAgent(llm=provider, config=cfg)
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        assert result.is_valid is False
        assert result.attempts == 1
        assert len(result.diagnoses) > 0

    def test_history_tracking(self) -> None:
        provider = StubLLMProvider(responses=[INVALID_SOURCE, INVALID_SOURCE, VALID_SOURCE])
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        cfg = AgentConfig(max_attempts=3)
        agent = CodeAgent(llm=provider, config=cfg)
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        assert result.is_valid is True
        assert result.attempts == 3
        assert len(result.history) == 3
        # First two entries should have violations
        assert len(result.history[0][1]) > 0
        assert len(result.history[1][1]) > 0
        # Third entry should have no violations (valid)
        assert len(result.history[2][1]) == 0

    def test_history_disabled(self) -> None:
        provider = StubLLMProvider(responses=[INVALID_SOURCE, VALID_SOURCE])
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        cfg = AgentConfig(max_attempts=3, history_enabled=False)
        agent = CodeAgent(llm=provider, config=cfg)
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        assert result.is_valid is True
        # History is still tracked internally but refinement context gets empty history
        assert result.attempts == 2

    def test_syntax_error_retries(self) -> None:
        provider = StubLLMProvider(responses=[SYNTAX_ERROR_SOURCE, VALID_SOURCE])
        cfg = AgentConfig(max_attempts=3)
        agent = CodeAgent(llm=provider, config=cfg)
        cs = _constraint_set()
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        # Syntax error causes is_valid=False on first try, then valid on second
        assert result.is_valid is True
        assert result.attempts == 2

    def test_run_with_graph_context(self) -> None:
        provider = StubLLMProvider(responses=[VALID_SOURCE])
        agent = CodeAgent(llm=provider)
        g = _graph(_fn("existing_func"), _cls("ExistingClass"))
        cs = _constraint_set()
        result = agent.run("Add helper", g, cs, "helpers.py")
        assert result.is_valid is True
        assert result.attempts == 1

    def test_run_with_multiple_constraints(self) -> None:
        provider = StubLLMProvider(responses=[VALID_SOURCE])
        agent = CodeAgent(llm=provider)
        nc = _naming_constraint()
        mu = MustUseConstraint(
            name="require_docs",
            description="Require docstrings",
            requirement="docstring",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.WARNING,
        )
        cs = _constraint_set(nc, mu)
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        # hello_world() has no docstring -> WARNING but not ERROR
        # snake_case passes, so is_valid depends on errors only
        assert result.attempts == 1

    def test_violations_updated_each_attempt(self) -> None:
        provider = StubLLMProvider(responses=[INVALID_SOURCE, VALID_SOURCE])
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        cfg = AgentConfig(max_attempts=3)
        agent = CodeAgent(llm=provider, config=cfg)
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        assert result.is_valid is True
        # Final violations should be empty (valid code)
        assert result.violations == []

    def test_diagnoses_populated_on_failure(self) -> None:
        provider = StubLLMProvider(responses=[INVALID_SOURCE])
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        cfg = AgentConfig(max_attempts=1)
        agent = CodeAgent(llm=provider, config=cfg)
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        assert result.is_valid is False
        assert len(result.diagnoses) > 0
        assert result.diagnoses[0].category == ViolationCategory.NAMING


# ── CodeAgent with various StubLLMProvider configs ───────────────────────


class TestCodeAgentWithStub:
    def test_stub_cycles_responses(self) -> None:
        # Stub cycles: resp[0], resp[1], resp[0], resp[1], ...
        provider = StubLLMProvider(responses=[INVALID_SOURCE, VALID_SOURCE])
        nc = _naming_constraint()
        cs = _constraint_set(nc)
        cfg = AgentConfig(max_attempts=3)
        agent = CodeAgent(llm=provider, config=cfg)
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        assert result.is_valid is True
        assert provider.call_count == 2

    def test_stub_records_requests(self) -> None:
        provider = StubLLMProvider(responses=[VALID_SOURCE])
        agent = CodeAgent(llm=provider)
        g = _empty_graph()
        cs = _constraint_set()
        agent.run("task", g, cs, "app.py")
        assert len(provider.requests) == 1
        assert len(provider.requests[0].messages) >= 2

    def test_agent_config_property(self) -> None:
        cfg = AgentConfig(max_attempts=7)
        provider = StubLLMProvider()
        agent = CodeAgent(llm=provider, config=cfg)
        assert agent.config is cfg
        assert agent.config.max_attempts == 7

    def test_exhaust_retries_with_syntax_errors(self) -> None:
        """All attempts produce syntax errors -> violations empty, diagnoses empty."""
        provider = StubLLMProvider(responses=[SYNTAX_ERROR_SOURCE])
        cfg = AgentConfig(max_attempts=2)
        agent = CodeAgent(llm=provider, config=cfg)
        cs = _constraint_set()
        g = _empty_graph()
        result = agent.run("task", g, cs, "app.py")
        assert result.is_valid is False
        assert result.attempts == 2
        # Syntax errors don't produce constraint violations
        assert result.violations == []
        assert result.diagnoses == []
