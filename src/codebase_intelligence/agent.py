"""Code agent with observe/plan/act/validate/refine loop.

Implements an autonomous agent that generates code through iterative
refinement. The agent observes the codebase graph, plans a generation
strategy, acts by calling the LLM, validates output against constraints,
and refines via the FeedbackEngine on failure.

Usage:
    >>> from codebase_intelligence.agent import CodeAgent, AgentConfig
    >>> from codebase_intelligence.llm import StubLLMProvider
    >>>
    >>> provider = StubLLMProvider(responses=["def hello(): pass"])
    >>> agent = CodeAgent(llm=provider)
    >>> result = agent.run(
    ...     request="Add a hello function",
    ...     graph=graph,
    ...     constraints=constraint_set,
    ...     file_path="app.py",
    ... )
    >>> print(result.is_valid)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codebase_intelligence.compiler import PromptCompiler
from codebase_intelligence.constraints import (
    Constraint,
    ConstraintSet,
    ConstraintViolation,
)
from codebase_intelligence.feedback import (
    EscalationLevel,
    FeedbackEngine,
    RefinementContext,
    ViolationDiagnosis,
)
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.llm import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    MessageRole,
)
from codebase_intelligence.nodes import SemanticNode
from codebase_intelligence.edges import SemanticEdge
from codebase_intelligence.validator import CodeValidator, ValidationResult


@dataclass(frozen=True)
class Observation:
    """Result of observing the codebase state.

    Attributes:
        graph_stats: Summary statistics of the graph.
        relevant_nodes: Nodes relevant to the task.
        relevant_edges: Edges relevant to the task.
        existing_patterns: Detected patterns summary.
        constraint_summary: Summary of active constraints.
    """

    graph_stats: dict[str, int] = field(default_factory=dict)
    relevant_nodes: tuple[SemanticNode, ...] = ()
    relevant_edges: tuple[SemanticEdge, ...] = ()
    existing_patterns: tuple[str, ...] = ()
    constraint_summary: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentPlan:
    """The agent's plan for code generation.

    Attributes:
        strategy: Description of the generation strategy.
        target_constraints: Constraints that apply to this task.
        context_nodes: Nodes providing context for generation.
        prompt_sections: Prompt section descriptions.
    """

    strategy: str = ""
    target_constraints: tuple[str, ...] = ()
    context_nodes: tuple[str, ...] = ()
    prompt_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the CodeAgent.

    Attributes:
        max_attempts: Maximum generation attempts.
        escalation_enabled: Whether to escalate refinement prompts.
        history_enabled: Whether to track attempt history.
        temperature: LLM sampling temperature.
        max_tokens: Max tokens for LLM response.
    """

    max_attempts: int = 3
    escalation_enabled: bool = True
    history_enabled: bool = True
    temperature: float = 0.2
    max_tokens: int = 4096


@dataclass
class AgentResult:
    """Result of a full agent run.

    Attributes:
        source: The generated source code.
        is_valid: Whether the code passed validation.
        attempts: Number of generation attempts made.
        violations: Violations from the final attempt.
        diagnoses: Diagnoses from the final attempt.
        history: All (source, violations) pairs from each attempt.
    """

    source: str = ""
    is_valid: bool = False
    attempts: int = 0
    violations: list[ConstraintViolation] = field(default_factory=list)
    diagnoses: list[ViolationDiagnosis] = field(default_factory=list)
    history: list[tuple[str, list[ConstraintViolation]]] = field(default_factory=list)


class CodeAgent:
    """Autonomous code generation agent with feedback loop.

    The agent follows an observe -> plan -> act -> validate -> refine
    cycle, using the FeedbackEngine for intelligent self-correction.

    Examples:
        >>> agent = CodeAgent(llm=provider, config=AgentConfig(max_attempts=5))
        >>> result = agent.run("Add user endpoint", graph, constraints, "routes.py")
        >>> if result.is_valid:
        ...     print(result.source)
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the code agent.

        Args:
            llm: LLM provider for code generation.
            config: Agent configuration.
        """
        self._llm = llm
        self._config = config or AgentConfig()
        self._compiler = PromptCompiler()
        self._validator = CodeValidator()
        self._feedback = FeedbackEngine()

    @property
    def config(self) -> AgentConfig:
        """Return the agent configuration."""
        return self._config

    def observe(
        self,
        graph: SemanticGraph,
        constraints: ConstraintSet,
    ) -> Observation:
        """Analyze the codebase graph and constraints.

        Args:
            graph: The semantic graph to observe.
            constraints: Active constraint set.

        Returns:
            An Observation summarizing the current state.
        """
        stats = graph.get_stats()
        graph_stats = {
            "node_count": stats.node_count,
            "edge_count": stats.edge_count,
        }

        nodes = tuple(graph.get_nodes())
        edges = tuple(graph.get_edges())

        constraint_names = tuple(
            c.name for c in constraints.constraints if c.enabled
        )

        return Observation(
            graph_stats=graph_stats,
            relevant_nodes=nodes,
            relevant_edges=edges,
            constraint_summary=constraint_names,
        )

    def plan(
        self,
        request: str,
        observation: Observation,
        constraints: ConstraintSet,
    ) -> AgentPlan:
        """Determine a generation strategy based on the observation.

        Args:
            request: The code generation task.
            observation: Current codebase observation.
            constraints: Active constraint set.

        Returns:
            An AgentPlan describing the strategy.
        """
        target_constraints = tuple(
            c.name for c in constraints.constraints if c.enabled
        )

        context_nodes = tuple(
            n.name for n in observation.relevant_nodes
        )

        strategy = f"Generate code for: {request}"
        if observation.graph_stats.get("node_count", 0) > 0:
            strategy += f" (context: {observation.graph_stats['node_count']} nodes)"

        return AgentPlan(
            strategy=strategy,
            target_constraints=target_constraints,
            context_nodes=context_nodes,
            prompt_sections=("system", "task", "context", "constraints", "output"),
        )

    def act(
        self,
        plan: AgentPlan,
        observation: Observation,
        constraints: ConstraintSet,
        request: str,
        graph: SemanticGraph,
        file_path: str,
    ) -> str:
        """Compile a prompt and call the LLM to generate code.

        Args:
            plan: The generation plan.
            observation: Current observation.
            constraints: Active constraint set.
            request: The task description.
            graph: The semantic graph.
            file_path: Target file path.

        Returns:
            Generated source code.
        """
        prompt = self._compiler.compile(
            task=request,
            graph=graph,
            constraints=constraints,
            relevant_files=[file_path],
        )

        messages = (
            LLMMessage(role=MessageRole.SYSTEM, content=prompt.system_message()),
            LLMMessage(role=MessageRole.USER, content=prompt.user_message()),
        )

        llm_request = LLMRequest(
            messages=messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        response = self._llm.complete(llm_request)
        return response.content

    def validate(
        self,
        source: str,
        file_path: str,
        constraints: ConstraintSet,
        graph: SemanticGraph | None = None,
    ) -> ValidationResult:
        """Validate generated code against constraints.

        Args:
            source: Generated source code.
            file_path: File path for the code.
            constraints: Active constraint set.
            graph: Optional original graph for consistency checks.

        Returns:
            Validation result.
        """
        return self._validator.validate(
            source=source,
            file_path=file_path,
            constraints=constraints,
            original_graph=graph,
        )

    def refine(
        self,
        source: str,
        validation_result: ValidationResult,
        context: RefinementContext,
    ) -> str:
        """Use the FeedbackEngine to refine and regenerate code.

        Args:
            source: Previous generated code.
            validation_result: Validation result with violations.
            context: Refinement context.

        Returns:
            New generated source code.
        """
        refinement_prompt = self._feedback.build_refinement(context)

        messages = (
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="You are a code generator. Fix ALL constraint violations.",
            ),
            LLMMessage(role=MessageRole.USER, content=context.original_request),
            LLMMessage(role=MessageRole.ASSISTANT, content=source),
            LLMMessage(role=MessageRole.USER, content=refinement_prompt),
        )

        llm_request = LLMRequest(
            messages=messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        response = self._llm.complete(llm_request)
        return response.content

    def run(
        self,
        request: str,
        graph: SemanticGraph,
        constraints: ConstraintSet,
        file_path: str,
    ) -> AgentResult:
        """Execute the full observe -> plan -> act -> validate -> refine loop.

        Args:
            request: The code generation task.
            graph: Semantic graph of the codebase.
            constraints: Active constraint set.
            file_path: Target file path.

        Returns:
            AgentResult with generated code and metadata.
        """
        result = AgentResult()
        history: list[tuple[str, list[ConstraintViolation]]] = []

        # Observe
        observation = self.observe(graph, constraints)

        # Plan
        plan = self.plan(request, observation, constraints)

        for attempt in range(1, self._config.max_attempts + 1):
            result.attempts = attempt

            # Act (first attempt) or Refine (subsequent attempts)
            if attempt == 1:
                source = self.act(plan, observation, constraints, request, graph, file_path)
            else:
                diagnoses = self._feedback.diagnose(
                    result.violations, constraints
                )
                result.diagnoses = diagnoses

                refinement_context = RefinementContext(
                    original_request=request,
                    violations=result.violations,
                    diagnoses=diagnoses,
                    attempt=attempt,
                    max_attempts=self._config.max_attempts,
                    history=history if self._config.history_enabled else [],
                )

                source = self.refine(result.source, validation, refinement_context)

            result.source = source

            # Validate
            validation = self.validate(source, file_path, constraints, graph)
            result.violations = list(validation.violations)

            # Track history
            if self._config.history_enabled:
                history.append((source, list(validation.violations)))

            if validation.is_valid:
                result.is_valid = True
                result.history = history
                return result

        # Exhausted all attempts
        result.is_valid = False
        result.history = history

        # Final diagnoses
        if result.violations:
            result.diagnoses = self._feedback.diagnose(result.violations, constraints)

        return result
