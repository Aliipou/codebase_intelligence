"""Prompt compiler for constraint-aware code generation.

Converts a semantic graph, active constraints, and a task description
into a structured prompt suitable for any chat-completion LLM.

The compiler performs graph slicing to extract only relevant context,
formats constraints as enforceable rules, and assembles everything
into a token-budgeted prompt.

Usage:
    >>> compiler = PromptCompiler()
    >>> prompt = compiler.compile(
    ...     task="Add a delete endpoint for users",
    ...     graph=graph,
    ...     constraints=constraint_set,
    ...     relevant_files=["app/routes/users.py"],
    ... )
    >>> print(prompt.render())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from codebase_intelligence.constraints import (
    Constraint,
    ConstraintSet,
    ErrorFormatConstraint,
    MustNotCrossConstraint,
    MustUseConstraint,
    NamingConstraint,
)
from codebase_intelligence.edges import EdgeType
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    EndpointNode,
    FunctionNode,
    ImportNode,
    ModuleNode,
    NodeType,
    SemanticNode,
    VariableNode,
)


class SectionKind(str, Enum):
    """Kind of prompt section for ordering and filtering."""

    SYSTEM = "system"
    TASK = "task"
    CONTEXT = "context"
    CONSTRAINTS = "constraints"
    EXAMPLES = "examples"
    OUTPUT_FORMAT = "output_format"


@dataclass(frozen=True)
class PromptSection:
    """A single section of a compiled prompt.

    Attributes:
        kind: Category of this section.
        heading: Display heading for the section.
        content: The section body text.
        priority: Higher priority sections are kept when trimming.
        token_estimate: Estimated token count for this section.
    """

    kind: SectionKind
    heading: str
    content: str
    priority: int = 0
    token_estimate: int = 0


@dataclass
class CompiledPrompt:
    """A fully compiled prompt ready for LLM consumption.

    Attributes:
        task: The original task description.
        sections: Ordered prompt sections.
        max_tokens: Target max tokens for LLM response.
    """

    task: str
    sections: list[PromptSection] = field(default_factory=list)
    max_tokens: int = 4096

    def render(self) -> str:
        """Render the full prompt as a single string."""
        parts: list[str] = []
        for section in sorted(self.sections, key=lambda s: -s.priority):
            parts.append(f"## {section.heading}\n\n{section.content}")
        return "\n\n".join(parts)

    def system_message(self) -> str:
        """Extract system-level instructions."""
        system_parts = [
            s.content
            for s in self.sections
            if s.kind == SectionKind.SYSTEM
        ]
        return "\n\n".join(system_parts)

    def user_message(self) -> str:
        """Extract user-level message (task + context)."""
        non_system = [
            s for s in sorted(self.sections, key=lambda s: -s.priority)
            if s.kind != SectionKind.SYSTEM
        ]
        parts: list[str] = []
        for section in non_system:
            parts.append(f"## {section.heading}\n\n{section.content}")
        return "\n\n".join(parts)

    @property
    def total_token_estimate(self) -> int:
        """Total estimated tokens across all sections."""
        return sum(s.token_estimate for s in self.sections)


@dataclass(frozen=True)
class GraphSlice:
    """A relevant subset of the semantic graph.

    Attributes:
        nodes: Nodes in this slice.
        edges: Edges in this slice.
        file_paths: Source files represented.
    """

    nodes: tuple[SemanticNode, ...]
    edges: tuple[tuple[str, str, str], ...] = ()
    file_paths: tuple[str, ...] = ()


class PromptCompiler:
    """Compiles task + graph + constraints into LLM prompts.

    The compiler:
    1. Slices the graph to relevant context
    2. Formats code structure information
    3. Translates constraints into natural language rules
    4. Assembles a token-budgeted prompt

    Attributes:
        _token_budget: Maximum tokens for the context window.
        _chars_per_token: Rough character-to-token ratio.

    Examples:
        >>> compiler = PromptCompiler(token_budget=8000)
        >>> prompt = compiler.compile(
        ...     task="Add logging to all service methods",
        ...     graph=graph,
        ...     constraints=constraints,
        ... )
    """

    def __init__(
        self,
        token_budget: int = 8000,
        chars_per_token: int = 4,
        max_response_tokens: int = 4096,
    ) -> None:
        """Initialize the prompt compiler.

        Args:
            token_budget: Maximum tokens for the full prompt.
            chars_per_token: Rough conversion factor for token estimation.
            max_response_tokens: Max tokens for LLM response.
        """
        self._token_budget = token_budget
        self._chars_per_token = chars_per_token
        self._max_response_tokens = max_response_tokens

    def compile(
        self,
        task: str,
        graph: SemanticGraph,
        constraints: ConstraintSet,
        relevant_files: list[str] | None = None,
    ) -> CompiledPrompt:
        """Compile a full prompt from task, graph, and constraints.

        Args:
            task: Natural language task description.
            graph: The semantic graph of the codebase.
            constraints: Active constraint set.
            relevant_files: File paths to focus on (optional).

        Returns:
            A compiled prompt ready for LLM consumption.
        """
        prompt = CompiledPrompt(task=task, max_tokens=self._max_response_tokens)

        # System instructions
        prompt.sections.append(self._build_system_section())

        # Task description
        prompt.sections.append(self._build_task_section(task))

        # Graph context
        graph_slice = self.slice_graph(graph, relevant_files)
        context_section = self._build_context_section(graph_slice)
        prompt.sections.append(context_section)

        # Constraints
        constraint_section = self._build_constraint_section(constraints)
        prompt.sections.append(constraint_section)

        # Output format
        prompt.sections.append(self._build_output_section())

        return prompt

    def slice_graph(
        self,
        graph: SemanticGraph,
        relevant_files: list[str] | None = None,
    ) -> GraphSlice:
        """Extract a relevant subgraph for the task.

        If relevant_files is provided, includes only nodes from those files
        plus their immediate dependencies. Otherwise includes the full graph.

        Args:
            graph: The full semantic graph.
            relevant_files: Files to focus on.

        Returns:
            A GraphSlice containing relevant nodes and edges.
        """
        if relevant_files is None:
            nodes = tuple(graph.get_nodes())
            edges: list[tuple[str, str, str]] = []
            for edge in graph.get_edges():
                edges.append((edge.source_id, edge.target_id, edge.edge_type.value))
            file_paths = tuple(sorted({n.file_path for n in nodes}))
            return GraphSlice(nodes=nodes, edges=tuple(edges), file_paths=file_paths)

        # Collect nodes from relevant files
        node_ids: set[str] = set()
        nodes_list: list[SemanticNode] = []
        for node in graph.get_nodes():
            if node.file_path in relevant_files:
                node_ids.add(node.id)
                nodes_list.append(node)

        # Expand to include direct dependencies (1-hop neighbors)
        expanded_ids: set[str] = set(node_ids)
        for nid in node_ids:
            for successor in graph.get_successors(nid):
                expanded_ids.add(successor.id)

        # Add neighbor nodes not yet included
        for node in graph.get_nodes():
            if node.id in expanded_ids and node.id not in node_ids:
                nodes_list.append(node)

        # Collect edges between included nodes
        edges_list: list[tuple[str, str, str]] = []
        for edge in graph.get_edges():
            if edge.source_id in expanded_ids and edge.target_id in expanded_ids:
                edges_list.append(
                    (edge.source_id, edge.target_id, edge.edge_type.value)
                )

        file_paths = tuple(sorted({n.file_path for n in nodes_list}))
        return GraphSlice(
            nodes=tuple(nodes_list),
            edges=tuple(edges_list),
            file_paths=file_paths,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // self._chars_per_token)

    def format_constraint(self, constraint: Constraint) -> str:
        """Format a single constraint as a natural language rule.

        Args:
            constraint: The constraint to format.

        Returns:
            Human-readable rule text.
        """
        if isinstance(constraint, NamingConstraint):
            types_str = ", ".join(nt.value for nt in constraint.node_types)
            return (
                f"NAMING RULE [{constraint.severity.value.upper()}]: "
                f"{constraint.description} "
                f"(applies to: {types_str}, pattern: {constraint.pattern})"
            )
        elif isinstance(constraint, MustUseConstraint):
            types_str = ", ".join(nt.value for nt in constraint.node_types)
            return (
                f"REQUIRED [{constraint.severity.value.upper()}]: "
                f"{constraint.description} "
                f"(requirement: {constraint.requirement}, applies to: {types_str})"
            )
        elif isinstance(constraint, MustNotCrossConstraint):
            return (
                f"BOUNDARY [{constraint.severity.value.upper()}]: "
                f"{constraint.description} "
                f"(forbidden: {', '.join(constraint.forbidden_targets)})"
            )
        elif isinstance(constraint, ErrorFormatConstraint):
            bases_str = (
                f", required bases: {', '.join(constraint.required_bases)}"
                if constraint.required_bases
                else ""
            )
            return (
                f"ERROR FORMAT [{constraint.severity.value.upper()}]: "
                f"{constraint.description} "
                f"(pattern: {constraint.exception_pattern}{bases_str})"
            )
        return f"RULE [{constraint.severity.value.upper()}]: {constraint.description}"

    def format_node(self, node: SemanticNode) -> str:
        """Format a node as a concise description.

        Args:
            node: The node to format.

        Returns:
            One-line description of the node.
        """
        if isinstance(node, ModuleNode):
            pkg = " [package]" if node.is_package else ""
            return f"module {node.name}{pkg} ({node.file_path})"
        elif isinstance(node, ClassNode):
            bases = f"({', '.join(node.bases)})" if node.bases else ""
            tags: list[str] = []
            if node.is_pydantic:
                tags.append("pydantic")
            if node.is_dataclass:
                tags.append("dataclass")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            return f"class {node.name}{bases}{tag_str}"
        elif isinstance(node, FunctionNode):
            params = ", ".join(node.parameters)
            ret = f" -> {node.return_type}" if node.return_type else ""
            prefix = "async " if node.is_async else ""
            return f"{prefix}def {node.name}({params}){ret}"
        elif isinstance(node, EndpointNode):
            return f"{node.http_method} {node.path} -> {node.name}()"
        elif isinstance(node, ImportNode):
            if node.is_from_import:
                names = ", ".join(node.imported_names) if node.imported_names else "*"
                return f"from {node.module} import {names}"
            return f"import {node.module}"
        elif isinstance(node, VariableNode):
            ann = f": {node.type_annotation}" if node.type_annotation else ""
            return f"{node.name}{ann} [{node.scope}]"
        return f"{node.node_type.value} {node.name}"

    def _build_system_section(self) -> PromptSection:
        """Build the system instruction section."""
        content = (
            "You are a code generator that MUST follow the project's conventions.\n"
            "Generate ONLY the requested code. Do not explain or add commentary.\n"
            "Follow ALL constraints listed below — violations will be rejected.\n"
            "Match the existing code style, patterns, and architecture exactly."
        )
        return PromptSection(
            kind=SectionKind.SYSTEM,
            heading="System Instructions",
            content=content,
            priority=100,
            token_estimate=self.estimate_tokens(content),
        )

    def _build_task_section(self, task: str) -> PromptSection:
        """Build the task description section."""
        content = f"Generate code for the following task:\n\n{task}"
        return PromptSection(
            kind=SectionKind.TASK,
            heading="Task",
            content=content,
            priority=90,
            token_estimate=self.estimate_tokens(content),
        )

    def _build_context_section(self, graph_slice: GraphSlice) -> PromptSection:
        """Build the codebase context section from a graph slice."""
        parts: list[str] = []

        if graph_slice.file_paths:
            parts.append(f"Files: {', '.join(graph_slice.file_paths)}")

        # Group nodes by type
        by_type: dict[str, list[str]] = {}
        for node in graph_slice.nodes:
            key = node.node_type.value
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(self.format_node(node))

        for type_name, descriptions in sorted(by_type.items()):
            parts.append(f"\n### {type_name.title()}s")
            for desc in descriptions:
                parts.append(f"- {desc}")

        content = "\n".join(parts) if parts else "No codebase context available."
        return PromptSection(
            kind=SectionKind.CONTEXT,
            heading="Codebase Context",
            content=content,
            priority=70,
            token_estimate=self.estimate_tokens(content),
        )

    def _build_constraint_section(self, constraints: ConstraintSet) -> PromptSection:
        """Build the constraints section."""
        rules: list[str] = []
        for constraint in constraints.constraints:
            if constraint.enabled:
                rules.append(f"- {self.format_constraint(constraint)}")

        content = "\n".join(rules) if rules else "No active constraints."
        return PromptSection(
            kind=SectionKind.CONSTRAINTS,
            heading="Constraints (MUST follow)",
            content=content,
            priority=80,
            token_estimate=self.estimate_tokens(content),
        )

    def _build_output_section(self) -> PromptSection:
        """Build the output format section."""
        content = (
            "Respond with ONLY valid Python code.\n"
            "Include necessary imports at the top.\n"
            "Follow the constraints above exactly.\n"
            "Do not include markdown code fences or explanations."
        )
        return PromptSection(
            kind=SectionKind.OUTPUT_FORMAT,
            heading="Output Format",
            content=content,
            priority=60,
            token_estimate=self.estimate_tokens(content),
        )
