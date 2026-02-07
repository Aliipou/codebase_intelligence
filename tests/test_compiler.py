"""Tests for prompt compiler."""

from __future__ import annotations

import pytest

from codebase_intelligence.compiler import (
    CompiledPrompt,
    GraphSlice,
    PromptCompiler,
    PromptSection,
    SectionKind,
)
from codebase_intelligence.constraints import (
    Constraint,
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
    DecoratorNode,
    EndpointNode,
    FunctionNode,
    ImportNode,
    ModuleNode,
    NodeType,
    SemanticNode,
    VariableNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module(
    name: str = "app",
    file_path: str = "app.py",
    line_start: int = 1,
    line_end: int = 100,
    is_package: bool = False,
    **kwargs,
) -> ModuleNode:
    return ModuleNode(
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        is_package=is_package,
        **kwargs,
    )


def _make_class(
    name: str = "MyClass",
    file_path: str = "app.py",
    line_start: int = 10,
    line_end: int = 50,
    bases: list[str] | None = None,
    is_pydantic: bool = False,
    is_dataclass: bool = False,
) -> ClassNode:
    return ClassNode(
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        bases=bases or [],
        is_pydantic=is_pydantic,
        is_dataclass=is_dataclass,
    )


def _make_function(
    name: str = "my_func",
    file_path: str = "app.py",
    line_start: int = 10,
    line_end: int = 20,
    parameters: list[str] | None = None,
    return_type: str | None = None,
    is_async: bool = False,
) -> FunctionNode:
    return FunctionNode(
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        parameters=parameters or [],
        return_type=return_type,
        is_async=is_async,
    )


def _make_endpoint(
    name: str = "get_users",
    file_path: str = "app.py",
    line_start: int = 10,
    line_end: int = 20,
    http_method: str = "GET",
    path: str = "/users",
) -> EndpointNode:
    return EndpointNode(
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        http_method=http_method,
        path=path,
    )


def _make_import(
    name: str = "os_import",
    file_path: str = "app.py",
    line_start: int = 1,
    line_end: int = 1,
    module: str = "os",
    is_from_import: bool = False,
    imported_names: list[str] | None = None,
) -> ImportNode:
    return ImportNode(
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        module=module,
        is_from_import=is_from_import,
        imported_names=imported_names or [],
    )


def _make_variable(
    name: str = "count",
    file_path: str = "app.py",
    line_start: int = 5,
    line_end: int = 5,
    type_annotation: str | None = None,
    scope: str = "local",
) -> VariableNode:
    return VariableNode(
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        type_annotation=type_annotation,
        scope=scope,
    )


def _make_graph_with_nodes(*nodes: SemanticNode) -> SemanticGraph:
    """Build a SemanticGraph containing the given nodes."""
    g = SemanticGraph()
    for n in nodes:
        g.add_node(n)
    return g


def _empty_constraint_set() -> ConstraintSet:
    return ConstraintSet(name="empty", description="empty set", constraints=[])


# ---------------------------------------------------------------------------
# TestSectionKind
# ---------------------------------------------------------------------------


class TestSectionKind:
    """Tests for SectionKind enum."""

    def test_all_section_kinds_exist(self) -> None:
        expected = {"SYSTEM", "TASK", "CONTEXT", "CONSTRAINTS", "EXAMPLES", "OUTPUT_FORMAT"}
        actual = {sk.name for sk in SectionKind}
        assert actual == expected

    def test_section_kind_values(self) -> None:
        assert SectionKind.SYSTEM.value == "system"
        assert SectionKind.TASK.value == "task"
        assert SectionKind.CONTEXT.value == "context"
        assert SectionKind.CONSTRAINTS.value == "constraints"
        assert SectionKind.EXAMPLES.value == "examples"
        assert SectionKind.OUTPUT_FORMAT.value == "output_format"

    def test_section_kind_is_string_enum(self) -> None:
        assert SectionKind.SYSTEM == "system"
        assert SectionKind.TASK == "task"


# ---------------------------------------------------------------------------
# TestPromptSection
# ---------------------------------------------------------------------------


class TestPromptSection:
    """Tests for PromptSection dataclass."""

    def test_create_with_all_fields(self) -> None:
        section = PromptSection(
            kind=SectionKind.SYSTEM,
            heading="System Instructions",
            content="Follow the rules.",
            priority=100,
            token_estimate=25,
        )
        assert section.kind == SectionKind.SYSTEM
        assert section.heading == "System Instructions"
        assert section.content == "Follow the rules."
        assert section.priority == 100
        assert section.token_estimate == 25

    def test_default_priority_and_tokens(self) -> None:
        section = PromptSection(
            kind=SectionKind.TASK,
            heading="Task",
            content="Do something.",
        )
        assert section.priority == 0
        assert section.token_estimate == 0

    def test_frozen(self) -> None:
        section = PromptSection(
            kind=SectionKind.TASK,
            heading="Task",
            content="stuff",
        )
        with pytest.raises(AttributeError):
            section.priority = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestCompiledPrompt
# ---------------------------------------------------------------------------


class TestCompiledPrompt:
    """Tests for CompiledPrompt."""

    def test_defaults(self) -> None:
        prompt = CompiledPrompt(task="do stuff")
        assert prompt.task == "do stuff"
        assert prompt.sections == []
        assert prompt.max_tokens == 4096

    def test_render_empty(self) -> None:
        prompt = CompiledPrompt(task="t")
        assert prompt.render() == ""

    def test_render_single_section(self) -> None:
        prompt = CompiledPrompt(task="t", sections=[
            PromptSection(kind=SectionKind.TASK, heading="Task", content="Do it.", priority=10),
        ])
        assert prompt.render() == "## Task\n\nDo it."

    def test_render_sorts_by_priority_desc(self) -> None:
        s1 = PromptSection(kind=SectionKind.TASK, heading="Low", content="low", priority=1)
        s2 = PromptSection(kind=SectionKind.CONTEXT, heading="High", content="high", priority=10)
        prompt = CompiledPrompt(task="t", sections=[s1, s2])
        rendered = prompt.render()
        assert rendered.index("High") < rendered.index("Low")

    def test_system_message_extracts_system_sections(self) -> None:
        sys_section = PromptSection(
            kind=SectionKind.SYSTEM, heading="Sys", content="system content", priority=100,
        )
        task_section = PromptSection(
            kind=SectionKind.TASK, heading="Task", content="task content", priority=90,
        )
        prompt = CompiledPrompt(task="t", sections=[sys_section, task_section])
        assert prompt.system_message() == "system content"

    def test_system_message_multiple_system_sections(self) -> None:
        s1 = PromptSection(kind=SectionKind.SYSTEM, heading="A", content="alpha", priority=100)
        s2 = PromptSection(kind=SectionKind.SYSTEM, heading="B", content="beta", priority=99)
        prompt = CompiledPrompt(task="t", sections=[s1, s2])
        msg = prompt.system_message()
        assert "alpha" in msg
        assert "beta" in msg

    def test_system_message_empty_when_no_system(self) -> None:
        prompt = CompiledPrompt(task="t", sections=[
            PromptSection(kind=SectionKind.TASK, heading="Task", content="x", priority=1),
        ])
        assert prompt.system_message() == ""

    def test_user_message_excludes_system(self) -> None:
        sys_section = PromptSection(
            kind=SectionKind.SYSTEM, heading="Sys", content="system", priority=100,
        )
        task_section = PromptSection(
            kind=SectionKind.TASK, heading="Task", content="task", priority=90,
        )
        ctx_section = PromptSection(
            kind=SectionKind.CONTEXT, heading="Ctx", content="context", priority=70,
        )
        prompt = CompiledPrompt(task="t", sections=[sys_section, task_section, ctx_section])
        user_msg = prompt.user_message()
        assert "system" not in user_msg
        assert "task" in user_msg
        assert "context" in user_msg

    def test_user_message_sorted_by_priority_desc(self) -> None:
        low = PromptSection(kind=SectionKind.CONTEXT, heading="Low", content="low", priority=1)
        high = PromptSection(kind=SectionKind.TASK, heading="High", content="high", priority=90)
        prompt = CompiledPrompt(task="t", sections=[low, high])
        user_msg = prompt.user_message()
        assert user_msg.index("High") < user_msg.index("Low")

    def test_user_message_empty_when_only_system(self) -> None:
        prompt = CompiledPrompt(task="t", sections=[
            PromptSection(kind=SectionKind.SYSTEM, heading="Sys", content="sys", priority=100),
        ])
        assert prompt.user_message() == ""

    def test_total_token_estimate(self) -> None:
        s1 = PromptSection(kind=SectionKind.TASK, heading="A", content="a", token_estimate=10)
        s2 = PromptSection(kind=SectionKind.CONTEXT, heading="B", content="b", token_estimate=20)
        prompt = CompiledPrompt(task="t", sections=[s1, s2])
        assert prompt.total_token_estimate == 30

    def test_total_token_estimate_empty(self) -> None:
        prompt = CompiledPrompt(task="t")
        assert prompt.total_token_estimate == 0


# ---------------------------------------------------------------------------
# TestGraphSlice
# ---------------------------------------------------------------------------


class TestGraphSlice:
    """Tests for GraphSlice dataclass."""

    def test_create_with_all_fields(self) -> None:
        node = _make_module()
        gs = GraphSlice(
            nodes=(node,),
            edges=(("a", "b", "contains"),),
            file_paths=("app.py",),
        )
        assert len(gs.nodes) == 1
        assert gs.edges == (("a", "b", "contains"),)
        assert gs.file_paths == ("app.py",)

    def test_defaults(self) -> None:
        node = _make_module()
        gs = GraphSlice(nodes=(node,))
        assert gs.edges == ()
        assert gs.file_paths == ()

    def test_empty(self) -> None:
        gs = GraphSlice(nodes=())
        assert gs.nodes == ()
        assert gs.edges == ()
        assert gs.file_paths == ()

    def test_frozen(self) -> None:
        gs = GraphSlice(nodes=())
        with pytest.raises(AttributeError):
            gs.nodes = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestPromptCompiler
# ---------------------------------------------------------------------------


class TestPromptCompiler:
    """Tests for PromptCompiler."""

    def test_default_init(self) -> None:
        compiler = PromptCompiler()
        assert compiler._token_budget == 8000
        assert compiler._chars_per_token == 4
        assert compiler._max_response_tokens == 4096

    def test_custom_init(self) -> None:
        compiler = PromptCompiler(
            token_budget=16000,
            chars_per_token=3,
            max_response_tokens=2048,
        )
        assert compiler._token_budget == 16000
        assert compiler._chars_per_token == 3
        assert compiler._max_response_tokens == 2048

    # -- compile ---------------------------------------------------------------

    def test_compile_returns_compiled_prompt(self) -> None:
        compiler = PromptCompiler()
        graph = SemanticGraph()
        cs = _empty_constraint_set()
        prompt = compiler.compile(task="Add a feature", graph=graph, constraints=cs)
        assert isinstance(prompt, CompiledPrompt)
        assert prompt.task == "Add a feature"
        assert prompt.max_tokens == 4096

    def test_compile_has_all_five_sections(self) -> None:
        compiler = PromptCompiler()
        graph = SemanticGraph()
        cs = _empty_constraint_set()
        prompt = compiler.compile(task="task", graph=graph, constraints=cs)
        kinds = [s.kind for s in prompt.sections]
        assert SectionKind.SYSTEM in kinds
        assert SectionKind.TASK in kinds
        assert SectionKind.CONTEXT in kinds
        assert SectionKind.CONSTRAINTS in kinds
        assert SectionKind.OUTPUT_FORMAT in kinds
        assert len(prompt.sections) == 5

    def test_compile_system_section_content(self) -> None:
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t", graph=SemanticGraph(), constraints=_empty_constraint_set(),
        )
        sys_msg = prompt.system_message()
        assert "code generator" in sys_msg
        assert "MUST follow" in sys_msg

    def test_compile_task_section_content(self) -> None:
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="Build a REST endpoint",
            graph=SemanticGraph(),
            constraints=_empty_constraint_set(),
        )
        task_sections = [s for s in prompt.sections if s.kind == SectionKind.TASK]
        assert len(task_sections) == 1
        assert "Build a REST endpoint" in task_sections[0].content

    def test_compile_output_section_content(self) -> None:
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t", graph=SemanticGraph(), constraints=_empty_constraint_set(),
        )
        out = [s for s in prompt.sections if s.kind == SectionKind.OUTPUT_FORMAT]
        assert len(out) == 1
        assert "valid Python code" in out[0].content

    def test_compile_with_graph_context(self) -> None:
        mod = _make_module()
        func = _make_function()
        graph = _make_graph_with_nodes(mod, func)
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t", graph=graph, constraints=_empty_constraint_set(),
        )
        ctx = [s for s in prompt.sections if s.kind == SectionKind.CONTEXT]
        assert len(ctx) == 1
        assert "app.py" in ctx[0].content

    def test_compile_with_relevant_files(self) -> None:
        mod_a = _make_module(name="mod_a", file_path="a.py")
        mod_b = _make_module(name="mod_b", file_path="b.py", line_start=1, line_end=50)
        graph = _make_graph_with_nodes(mod_a, mod_b)
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t",
            graph=graph,
            constraints=_empty_constraint_set(),
            relevant_files=["a.py"],
        )
        ctx = [s for s in prompt.sections if s.kind == SectionKind.CONTEXT]
        assert "a.py" in ctx[0].content

    def test_compile_with_constraints(self) -> None:
        naming = NamingConstraint(
            name="snake_funcs",
            description="Functions must be snake_case",
            pattern=r"^[a-z_]+$",
            node_types=[NodeType.FUNCTION],
        )
        cs = ConstraintSet(name="rules", description="rules", constraints=[naming])
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t", graph=SemanticGraph(), constraints=cs,
        )
        constraint_section = [s for s in prompt.sections if s.kind == SectionKind.CONSTRAINTS]
        assert len(constraint_section) == 1
        assert "snake_funcs" not in constraint_section[0].content  # name not shown
        assert "NAMING RULE" in constraint_section[0].content

    def test_compile_max_response_tokens_propagated(self) -> None:
        compiler = PromptCompiler(max_response_tokens=2048)
        prompt = compiler.compile(
            task="t", graph=SemanticGraph(), constraints=_empty_constraint_set(),
        )
        assert prompt.max_tokens == 2048

    def test_compile_disabled_constraint_excluded(self) -> None:
        naming = NamingConstraint(
            name="disabled_rule",
            description="should not appear",
            pattern=r"^.*$",
            node_types=[NodeType.FUNCTION],
            enabled=False,
        )
        cs = ConstraintSet(name="rules", description="rules", constraints=[naming])
        compiler = PromptCompiler()
        prompt = compiler.compile(task="t", graph=SemanticGraph(), constraints=cs)
        constraint_section = [s for s in prompt.sections if s.kind == SectionKind.CONSTRAINTS]
        assert "No active constraints." in constraint_section[0].content

    # -- estimate_tokens -------------------------------------------------------

    def test_estimate_tokens_basic(self) -> None:
        compiler = PromptCompiler(chars_per_token=4)
        assert compiler.estimate_tokens("abcdefgh") == 2  # 8 / 4

    def test_estimate_tokens_empty_string_returns_one(self) -> None:
        compiler = PromptCompiler(chars_per_token=4)
        assert compiler.estimate_tokens("") == 1  # max(1, 0)

    def test_estimate_tokens_short_string_returns_one(self) -> None:
        compiler = PromptCompiler(chars_per_token=4)
        assert compiler.estimate_tokens("ab") == 1  # max(1, 0)

    def test_estimate_tokens_custom_ratio(self) -> None:
        compiler = PromptCompiler(chars_per_token=2)
        assert compiler.estimate_tokens("abcdef") == 3

    # -- slice_graph -----------------------------------------------------------

    def test_slice_graph_no_relevant_files_full_graph(self) -> None:
        mod = _make_module()
        func = _make_function()
        graph = _make_graph_with_nodes(mod, func)
        edge = SemanticEdge(
            source_id=mod.id, target_id=func.id, edge_type=EdgeType.CONTAINS,
        )
        graph.add_edge(edge)

        compiler = PromptCompiler()
        gs = compiler.slice_graph(graph)

        assert len(gs.nodes) == 2
        assert len(gs.edges) == 1
        assert gs.edges[0] == (mod.id, func.id, "contains")
        assert "app.py" in gs.file_paths

    def test_slice_graph_no_relevant_files_empty_graph(self) -> None:
        compiler = PromptCompiler()
        gs = compiler.slice_graph(SemanticGraph())
        assert gs.nodes == ()
        assert gs.edges == ()
        assert gs.file_paths == ()

    def test_slice_graph_with_relevant_files(self) -> None:
        mod_a = _make_module(name="mod_a", file_path="a.py")
        func_a = _make_function(name="func_a", file_path="a.py", line_start=10, line_end=20)
        mod_b = _make_module(name="mod_b", file_path="b.py")
        func_b = _make_function(name="func_b", file_path="b.py", line_start=10, line_end=20)
        graph = _make_graph_with_nodes(mod_a, func_a, mod_b, func_b)

        # Edge from a -> b (dependency)
        edge_ab = SemanticEdge(
            source_id=func_a.id, target_id=func_b.id, edge_type=EdgeType.CALLS,
        )
        graph.add_edge(edge_ab)

        compiler = PromptCompiler()
        gs = compiler.slice_graph(graph, relevant_files=["a.py"])

        # Should include a.py nodes and 1-hop dependency (func_b)
        node_ids = {n.id for n in gs.nodes}
        assert mod_a.id in node_ids
        assert func_a.id in node_ids
        assert func_b.id in node_ids
        # mod_b is not directly connected to a.py nodes via outgoing edges
        # so it should NOT be included
        assert mod_b.id not in node_ids

        # Edge between included nodes should be included
        edge_tuples = gs.edges
        assert (func_a.id, func_b.id, "calls") in edge_tuples

    def test_slice_graph_relevant_files_empty_list(self) -> None:
        mod = _make_module()
        graph = _make_graph_with_nodes(mod)
        compiler = PromptCompiler()
        gs = compiler.slice_graph(graph, relevant_files=[])
        # No nodes match the empty relevant_files list
        assert gs.nodes == ()

    def test_slice_graph_relevant_files_no_match(self) -> None:
        mod = _make_module(file_path="app.py")
        graph = _make_graph_with_nodes(mod)
        compiler = PromptCompiler()
        gs = compiler.slice_graph(graph, relevant_files=["nonexistent.py"])
        assert gs.nodes == ()

    def test_slice_graph_relevant_files_sorts_file_paths(self) -> None:
        mod_b = _make_module(name="mod_b", file_path="b.py")
        mod_a = _make_module(name="mod_a", file_path="a.py")
        graph = _make_graph_with_nodes(mod_b, mod_a)
        compiler = PromptCompiler()
        gs = compiler.slice_graph(graph, relevant_files=["b.py", "a.py"])
        assert gs.file_paths == ("a.py", "b.py")

    def test_slice_graph_no_relevant_files_sorts_file_paths(self) -> None:
        mod_b = _make_module(name="mod_b", file_path="b.py")
        mod_a = _make_module(name="mod_a", file_path="a.py")
        graph = _make_graph_with_nodes(mod_b, mod_a)
        compiler = PromptCompiler()
        gs = compiler.slice_graph(graph)
        assert gs.file_paths == ("a.py", "b.py")

    def test_slice_graph_edges_filtered_to_expanded_ids(self) -> None:
        """Edges where either endpoint is outside expanded_ids are excluded."""
        n1 = _make_function(name="f1", file_path="a.py", line_start=1, line_end=5)
        n2 = _make_function(name="f2", file_path="b.py", line_start=1, line_end=5)
        n3 = _make_function(name="f3", file_path="c.py", line_start=1, line_end=5)
        graph = _make_graph_with_nodes(n1, n2, n3)

        # n1 -> n2 (included: n1 in relevant, n2 is 1-hop)
        graph.add_edge(SemanticEdge(
            source_id=n1.id, target_id=n2.id, edge_type=EdgeType.CALLS,
        ))
        # n2 -> n3 (excluded: n3 not in expanded set when relevant=["a.py"])
        graph.add_edge(SemanticEdge(
            source_id=n2.id, target_id=n3.id, edge_type=EdgeType.CALLS,
        ))

        compiler = PromptCompiler()
        gs = compiler.slice_graph(graph, relevant_files=["a.py"])
        edge_tuples = gs.edges
        assert (n1.id, n2.id, "calls") in edge_tuples
        assert (n2.id, n3.id, "calls") not in edge_tuples


# ---------------------------------------------------------------------------
# TestFormatConstraint
# ---------------------------------------------------------------------------


class TestFormatConstraint:
    """Tests for PromptCompiler.format_constraint()."""

    def test_naming_constraint(self) -> None:
        c = NamingConstraint(
            name="pascal_classes",
            description="Classes must be PascalCase",
            pattern=r"^[A-Z][a-zA-Z0-9]*$",
            node_types=[NodeType.CLASS],
            severity=ConstraintSeverity.WARNING,
        )
        compiler = PromptCompiler()
        result = compiler.format_constraint(c)
        assert result.startswith("NAMING RULE [WARNING]:")
        assert "Classes must be PascalCase" in result
        assert "class" in result
        assert r"^[A-Z][a-zA-Z0-9]*$" in result

    def test_naming_constraint_multiple_types(self) -> None:
        c = NamingConstraint(
            name="multi",
            description="Multi",
            pattern=r".*",
            node_types=[NodeType.CLASS, NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        compiler = PromptCompiler()
        result = compiler.format_constraint(c)
        assert "NAMING RULE [ERROR]:" in result
        # Both types should appear (order may vary since node_types is a set)
        assert "class" in result
        assert "function" in result

    def test_must_use_constraint(self) -> None:
        c = MustUseConstraint(
            name="docstrings",
            description="Public functions need docstrings",
            requirement="docstring",
            node_types=[NodeType.FUNCTION],
            severity=ConstraintSeverity.ERROR,
        )
        compiler = PromptCompiler()
        result = compiler.format_constraint(c)
        assert result.startswith("REQUIRED [ERROR]:")
        assert "Public functions need docstrings" in result
        assert "docstring" in result
        assert "function" in result

    def test_must_not_cross_constraint(self) -> None:
        c = MustNotCrossConstraint(
            name="boundary",
            description="Services cannot import controllers",
            source_pattern=r".*/services/.*",
            forbidden_targets=[r".*/controllers/.*", r".*/api/.*"],
            severity=ConstraintSeverity.ERROR,
        )
        compiler = PromptCompiler()
        result = compiler.format_constraint(c)
        assert result.startswith("BOUNDARY [ERROR]:")
        assert "Services cannot import controllers" in result
        assert ".*/controllers/.*" in result
        assert ".*/api/.*" in result

    def test_error_format_constraint(self) -> None:
        c = ErrorFormatConstraint(
            name="exc_naming",
            description="Exception naming rules",
            exception_pattern=r"^[A-Z].*Error$",
            severity=ConstraintSeverity.ERROR,
        )
        compiler = PromptCompiler()
        result = compiler.format_constraint(c)
        assert result.startswith("ERROR FORMAT [ERROR]:")
        assert "Exception naming rules" in result
        assert r"^[A-Z].*Error$" in result

    def test_error_format_constraint_with_bases(self) -> None:
        c = ErrorFormatConstraint(
            name="exc_naming",
            description="Exception naming rules",
            exception_pattern=r"^[A-Z].*Error$",
            severity=ConstraintSeverity.WARNING,
            required_bases=["BaseError", "AppException"],
        )
        compiler = PromptCompiler()
        result = compiler.format_constraint(c)
        assert "ERROR FORMAT [WARNING]:" in result
        assert "required bases:" in result
        assert "BaseError" in result
        assert "AppException" in result

    def test_unknown_constraint_type_fallback(self) -> None:
        """A custom constraint subclass falls through to the default branch."""

        class CustomConstraint(Constraint):
            def validate(self, graph):
                return []

            def validate_node(self, node):
                return None

        c = CustomConstraint(
            name="custom",
            description="A custom rule",
            severity=ConstraintSeverity.INFO,
        )
        compiler = PromptCompiler()
        result = compiler.format_constraint(c)
        assert result == "RULE [INFO]: A custom rule"


# ---------------------------------------------------------------------------
# TestFormatNode
# ---------------------------------------------------------------------------


class TestFormatNode:
    """Tests for PromptCompiler.format_node()."""

    # -- ModuleNode --

    def test_module_node_non_package(self) -> None:
        node = _make_module(name="utils", file_path="utils.py", is_package=False)
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "module utils (utils.py)"

    def test_module_node_package(self) -> None:
        node = _make_module(name="pkg", file_path="pkg/__init__.py", is_package=True)
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "module pkg [package] (pkg/__init__.py)"

    # -- ClassNode --

    def test_class_node_plain(self) -> None:
        node = _make_class(name="Foo")
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "class Foo"

    def test_class_node_with_bases(self) -> None:
        node = _make_class(name="Child", bases=["Base", "Mixin"])
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "class Child(Base, Mixin)"

    def test_class_node_pydantic(self) -> None:
        node = _make_class(name="UserModel", is_pydantic=True)
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "class UserModel [pydantic]"

    def test_class_node_dataclass(self) -> None:
        node = _make_class(name="Config", is_dataclass=True)
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "class Config [dataclass]"

    def test_class_node_pydantic_and_dataclass(self) -> None:
        node = _make_class(name="Both", is_pydantic=True, is_dataclass=True)
        compiler = PromptCompiler()
        result = compiler.format_node(node)
        assert "pydantic" in result
        assert "dataclass" in result
        assert result == "class Both [pydantic, dataclass]"

    def test_class_node_with_bases_and_tags(self) -> None:
        node = _make_class(name="X", bases=["BaseModel"], is_pydantic=True)
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "class X(BaseModel) [pydantic]"

    # -- FunctionNode --

    def test_function_node_simple(self) -> None:
        node = _make_function(name="greet")
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "def greet()"

    def test_function_node_with_params(self) -> None:
        node = _make_function(name="add", parameters=["a", "b"])
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "def add(a, b)"

    def test_function_node_with_return_type(self) -> None:
        node = _make_function(name="get_name", return_type="str")
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "def get_name() -> str"

    def test_function_node_async(self) -> None:
        node = _make_function(name="fetch", is_async=True)
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "async def fetch()"

    def test_function_node_async_with_params_and_return(self) -> None:
        node = _make_function(
            name="process",
            parameters=["self", "data"],
            return_type="Result",
            is_async=True,
        )
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "async def process(self, data) -> Result"

    def test_function_node_no_return_type(self) -> None:
        node = _make_function(name="void_func", return_type=None)
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "def void_func()"

    # -- EndpointNode --

    def test_endpoint_node(self) -> None:
        node = _make_endpoint(name="list_users", http_method="GET", path="/api/users")
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "GET /api/users -> list_users()"

    def test_endpoint_node_post(self) -> None:
        node = _make_endpoint(name="create_user", http_method="POST", path="/users")
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "POST /users -> create_user()"

    # -- ImportNode --

    def test_import_node_regular(self) -> None:
        node = _make_import(module="os", is_from_import=False)
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "import os"

    def test_import_node_from_import_with_names(self) -> None:
        node = _make_import(
            module="os.path",
            is_from_import=True,
            imported_names=["join", "exists"],
        )
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "from os.path import join, exists"

    def test_import_node_from_import_no_names_star(self) -> None:
        node = _make_import(
            module="utils",
            is_from_import=True,
            imported_names=[],
        )
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "from utils import *"

    # -- VariableNode --

    def test_variable_node_no_annotation(self) -> None:
        node = _make_variable(name="count", scope="module")
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "count [module]"

    def test_variable_node_with_annotation(self) -> None:
        node = _make_variable(name="total", type_annotation="int", scope="local")
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "total: int [local]"

    def test_variable_node_class_scope(self) -> None:
        node = _make_variable(name="value", type_annotation="str", scope="class")
        compiler = PromptCompiler()
        assert compiler.format_node(node) == "value: str [class]"

    # -- Unknown node type (fallback) --

    def test_unknown_node_type_fallback(self) -> None:
        """DecoratorNode is not handled by any isinstance branch -- hits fallback."""
        node = DecoratorNode(
            name="my_decorator",
            file_path="app.py",
            line_start=1,
            line_end=1,
            decorator_name="my_decorator",
        )
        compiler = PromptCompiler()
        result = compiler.format_node(node)
        assert result == "decorator my_decorator"


# ---------------------------------------------------------------------------
# TestBranchPartials — internal _build_* methods via compile()
# ---------------------------------------------------------------------------


class TestBranchPartials:
    """Test internal builder branches via integration through compile()."""

    def test_context_section_no_nodes_shows_fallback(self) -> None:
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t",
            graph=SemanticGraph(),
            constraints=_empty_constraint_set(),
        )
        ctx = [s for s in prompt.sections if s.kind == SectionKind.CONTEXT]
        assert ctx[0].content == "No codebase context available."

    def test_context_section_groups_by_node_type(self) -> None:
        mod = _make_module(name="app", file_path="app.py")
        cls = _make_class(name="Svc", file_path="app.py")
        func = _make_function(name="run", file_path="app.py")
        graph = _make_graph_with_nodes(mod, cls, func)
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t", graph=graph, constraints=_empty_constraint_set(),
        )
        ctx = [s for s in prompt.sections if s.kind == SectionKind.CONTEXT][0].content
        assert "### Modules" in ctx or "### Module" in ctx
        assert "### Class" in ctx
        assert "### Function" in ctx

    def test_context_section_with_file_paths(self) -> None:
        mod = _make_module(name="app", file_path="app.py")
        graph = _make_graph_with_nodes(mod)
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t", graph=graph, constraints=_empty_constraint_set(),
        )
        ctx = [s for s in prompt.sections if s.kind == SectionKind.CONTEXT][0].content
        assert "Files:" in ctx
        assert "app.py" in ctx

    def test_constraint_section_with_enabled_and_disabled(self) -> None:
        enabled = NamingConstraint(
            name="enabled_rule",
            description="must follow",
            pattern=r".*",
            node_types=[NodeType.CLASS],
            enabled=True,
        )
        disabled = NamingConstraint(
            name="disabled_rule",
            description="should not show",
            pattern=r".*",
            node_types=[NodeType.CLASS],
            enabled=False,
        )
        cs = ConstraintSet(
            name="mixed", description="mixed", constraints=[enabled, disabled],
        )
        compiler = PromptCompiler()
        prompt = compiler.compile(task="t", graph=SemanticGraph(), constraints=cs)
        constraint_content = [
            s for s in prompt.sections if s.kind == SectionKind.CONSTRAINTS
        ][0].content
        assert "must follow" in constraint_content
        assert "should not show" not in constraint_content

    def test_constraint_section_empty_constraints(self) -> None:
        cs = ConstraintSet(name="empty", description="", constraints=[])
        compiler = PromptCompiler()
        prompt = compiler.compile(task="t", graph=SemanticGraph(), constraints=cs)
        constraint_content = [
            s for s in prompt.sections if s.kind == SectionKind.CONSTRAINTS
        ][0].content
        assert constraint_content == "No active constraints."

    def test_section_priorities_ordering(self) -> None:
        """Verify the priority hierarchy: system > task > constraints > context > output."""
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="t", graph=SemanticGraph(), constraints=_empty_constraint_set(),
        )
        priority_map = {s.kind: s.priority for s in prompt.sections}
        assert priority_map[SectionKind.SYSTEM] > priority_map[SectionKind.TASK]
        assert priority_map[SectionKind.TASK] > priority_map[SectionKind.CONSTRAINTS]
        assert priority_map[SectionKind.CONSTRAINTS] > priority_map[SectionKind.CONTEXT]
        assert priority_map[SectionKind.CONTEXT] > priority_map[SectionKind.OUTPUT_FORMAT]

    def test_token_estimates_are_positive(self) -> None:
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="Build something cool",
            graph=SemanticGraph(),
            constraints=_empty_constraint_set(),
        )
        for section in prompt.sections:
            assert section.token_estimate >= 1

    def test_render_full_prompt_is_nonempty(self) -> None:
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="Hello",
            graph=SemanticGraph(),
            constraints=_empty_constraint_set(),
        )
        rendered = prompt.render()
        assert len(rendered) > 0
        assert "## System Instructions" in rendered
        assert "## Task" in rendered

    def test_compile_with_all_node_types(self) -> None:
        """Compile with every node type to ensure context section formats all."""
        mod = _make_module(name="app", file_path="app.py", is_package=True)
        cls = _make_class(name="User", file_path="app.py", bases=["BaseModel"], is_pydantic=True)
        func = _make_function(
            name="get_user", file_path="app.py", parameters=["self", "uid"],
            return_type="User", is_async=True,
        )
        ep = _make_endpoint(name="list_ep", file_path="app.py", http_method="GET", path="/ep")
        imp = _make_import(name="os_imp", file_path="app.py", module="os")
        var = _make_variable(name="MAX", file_path="app.py", type_annotation="int", scope="module")

        graph = _make_graph_with_nodes(mod, cls, func, ep, imp, var)
        compiler = PromptCompiler()
        prompt = compiler.compile(
            task="enhance",
            graph=graph,
            constraints=_empty_constraint_set(),
        )
        ctx_content = [s for s in prompt.sections if s.kind == SectionKind.CONTEXT][0].content

        # All node descriptions should appear
        assert "[package]" in ctx_content
        assert "class User(BaseModel)" in ctx_content
        assert "async def get_user" in ctx_content
        assert "GET /ep" in ctx_content
        assert "import os" in ctx_content
        assert "MAX: int [module]" in ctx_content

    def test_compile_with_all_constraint_types(self) -> None:
        naming = NamingConstraint(
            name="naming", description="d1", pattern=r".*",
            node_types=[NodeType.CLASS], severity=ConstraintSeverity.WARNING,
        )
        must_use = MustUseConstraint(
            name="must_use", description="d2", requirement="docstring",
            node_types=[NodeType.FUNCTION], severity=ConstraintSeverity.ERROR,
        )
        boundary = MustNotCrossConstraint(
            name="boundary", description="d3",
            source_pattern=r".*", forbidden_targets=[r"bad"],
            severity=ConstraintSeverity.INFO,
        )
        cs = ConstraintSet(
            name="all",
            description="all types",
            constraints=[naming, must_use, boundary],
        )
        compiler = PromptCompiler()
        prompt = compiler.compile(task="t", graph=SemanticGraph(), constraints=cs)
        content = [s for s in prompt.sections if s.kind == SectionKind.CONSTRAINTS][0].content
        assert "NAMING RULE" in content
        assert "REQUIRED" in content
        assert "BOUNDARY" in content

    def test_slice_graph_neighbor_expansion_adds_nodes_from_other_files(self) -> None:
        """When slicing with relevant_files, nodes from other files may be
        pulled in via 1-hop expansion."""
        local_func = _make_function(name="caller", file_path="local.py", line_start=1, line_end=5)
        remote_cls = _make_class(name="RemoteSvc", file_path="remote.py", line_start=1, line_end=50)
        graph = _make_graph_with_nodes(local_func, remote_cls)
        graph.add_edge(SemanticEdge(
            source_id=local_func.id, target_id=remote_cls.id, edge_type=EdgeType.USES_TYPE,
        ))

        compiler = PromptCompiler()
        gs = compiler.slice_graph(graph, relevant_files=["local.py"])
        node_ids = {n.id for n in gs.nodes}
        assert local_func.id in node_ids
        assert remote_cls.id in node_ids
        # Both files should appear in file_paths
        assert "local.py" in gs.file_paths
        assert "remote.py" in gs.file_paths

    def test_context_section_multiple_nodes_same_type(self) -> None:
        """395->397: Multiple nodes of the same type are grouped."""
        from codebase_intelligence.nodes import FunctionNode
        from codebase_intelligence.compiler import GraphSlice

        fn1 = FunctionNode(name="foo", file_path="a.py", line_start=1, line_end=2)
        fn2 = FunctionNode(name="bar", file_path="a.py", line_start=3, line_end=4)
        gs = GraphSlice(nodes=(fn1, fn2), file_paths=("a.py",))
        compiler = PromptCompiler()
        section = compiler._build_context_section(gs)
        assert "foo" in section.content
        assert "bar" in section.content
