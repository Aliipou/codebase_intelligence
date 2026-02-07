"""Brutal unit tests for semantic graph.

Tests every code path, edge case, and validation rule for 100% coverage.
"""

from __future__ import annotations

import pytest

from codebase_intelligence.edges import EdgeType, SemanticEdge, create_contains_edge
from codebase_intelligence.graph import GraphStats, SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    FunctionNode,
    ModuleNode,
    NodeType,
    SemanticNode,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _mod(name: str = "app", line_start: int = 1, line_end: int = 100) -> ModuleNode:
    return ModuleNode(name=name, file_path=f"{name}.py", line_start=line_start, line_end=line_end)


def _cls(name: str = "MyClass", fp: str = "app.py", ls: int = 10, le: int = 50) -> ClassNode:
    return ClassNode(name=name, file_path=fp, line_start=ls, line_end=le)


def _fn(name: str = "my_func", fp: str = "app.py", ls: int = 20, le: int = 30) -> FunctionNode:
    return FunctionNode(name=name, file_path=fp, line_start=ls, line_end=le)


def _edge(src: str, tgt: str, etype: EdgeType = EdgeType.CALLS, **kw) -> SemanticEdge:  # type: ignore[no-untyped-def]
    return SemanticEdge(source_id=src, target_id=tgt, edge_type=etype, **kw)


def _populated_graph() -> tuple[SemanticGraph, ModuleNode, ClassNode, FunctionNode, FunctionNode]:
    """Create a populated graph: module → class → func_a, func_b; func_a calls func_b."""
    g = SemanticGraph()
    m = _mod("app")
    c = _cls("Service", "app.py", 10, 80)
    fa = _fn("get_data", "app.py", 20, 30)
    fb = _fn("process", "app.py", 40, 50)
    g.add_node(m)
    g.add_node(c)
    g.add_node(fa)
    g.add_node(fb)
    g.add_edge(_edge(m.id, c.id, EdgeType.CONTAINS))
    g.add_edge(_edge(c.id, fa.id, EdgeType.CONTAINS))
    g.add_edge(_edge(c.id, fb.id, EdgeType.CONTAINS))
    g.add_edge(_edge(fa.id, fb.id, EdgeType.CALLS, line_number=25))
    return g, m, c, fa, fb


# ── GraphStats ────────────────────────────────────────────────────────────


class TestGraphStats:
    """Tests for GraphStats model."""

    def test_default_values(self) -> None:
        stats = GraphStats()
        assert stats.node_count == 0
        assert stats.edge_count == 0
        assert stats.nodes_by_type == {}
        assert stats.edges_by_type == {}
        assert stats.connected_components == 0
        assert stats.max_depth == 0

    def test_custom_values(self) -> None:
        stats = GraphStats(
            node_count=10,
            edge_count=20,
            nodes_by_type={"module": 2, "function": 8},
            edges_by_type={"calls": 15, "contains": 5},
            connected_components=1,
            max_depth=3,
        )
        assert stats.node_count == 10
        assert stats.edge_count == 20
        assert stats.nodes_by_type == {"module": 2, "function": 8}


# ── SemanticGraph: Init & Basics ──────────────────────────────────────────


class TestSemanticGraphBasics:
    """Tests for basic graph operations."""

    def test_empty_graph(self) -> None:
        g = SemanticGraph()
        assert len(g) == 0
        assert repr(g) == "SemanticGraph(nodes=0, edges=0)"

    def test_add_node(self) -> None:
        g = SemanticGraph()
        m = _mod()
        g.add_node(m)
        assert len(g) == 1
        assert g.has_node(m.id)
        assert g.get_node(m.id) is m

    def test_add_none_node_raises(self) -> None:
        g = SemanticGraph()
        with pytest.raises(ValueError, match="Cannot add None"):
            g.add_node(None)  # type: ignore[arg-type]

    def test_add_duplicate_node_replaces(self) -> None:
        """Adding a node with same ID replaces the old one."""
        g = SemanticGraph()
        m1 = _mod("app")
        g.add_node(m1)
        # Create another node with different name but same ID
        m2 = ModuleNode(
            id=m1.id,
            name="app_v2",
            file_path="app_v2.py",
            line_start=1,
            line_end=200,
        )
        g.add_node(m2)
        assert len(g) == 1
        assert g.get_node(m1.id).name == "app_v2"

    def test_add_duplicate_node_updates_type_index(self) -> None:
        """Replacing a node updates the type index correctly."""
        g = SemanticGraph()
        # Add a function node
        f = _fn("foo")
        g.add_node(f)
        assert list(g.get_nodes(NodeType.FUNCTION))

        # Replace with a class node using same ID
        c = ClassNode(
            id=f.id,
            name="Foo",
            file_path="app.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(c)
        # Old type should be gone, new type should be present
        assert f.id not in {n.id for n in g.get_nodes(NodeType.FUNCTION)}
        assert f.id in {n.id for n in g.get_nodes(NodeType.CLASS)}

    def test_add_edge(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        e = _edge(a.id, b.id)
        g.add_edge(e)
        assert g.has_edge(e.id)
        assert g.get_edge(e.id) is e

    def test_add_none_edge_raises(self) -> None:
        g = SemanticGraph()
        with pytest.raises(ValueError, match="Cannot add None"):
            g.add_edge(None)  # type: ignore[arg-type]

    def test_add_edge_missing_source_raises(self) -> None:
        g = SemanticGraph()
        b = _fn("b")
        g.add_node(b)
        with pytest.raises(KeyError, match="Source node"):
            g.add_edge(_edge("nonexistent", b.id))

    def test_add_edge_missing_target_raises(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        g.add_node(a)
        with pytest.raises(KeyError, match="Target node"):
            g.add_edge(_edge(a.id, "nonexistent"))

    def test_add_duplicate_edge_replaces(self) -> None:
        """Adding an edge with same ID replaces the old one."""
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)

        e1 = SemanticEdge(
            id="edge_1",
            source_id=a.id,
            target_id=b.id,
            edge_type=EdgeType.CALLS,
            weight=0.5,
        )
        g.add_edge(e1)

        e2 = SemanticEdge(
            id="edge_1",
            source_id=a.id,
            target_id=b.id,
            edge_type=EdgeType.CALLS,
            weight=0.9,
        )
        g.add_edge(e2)

        fetched = g.get_edge("edge_1")
        assert fetched is not None
        assert fetched.weight == 0.9

    def test_get_node_not_found(self) -> None:
        g = SemanticGraph()
        assert g.get_node("nonexistent") is None

    def test_get_edge_not_found(self) -> None:
        g = SemanticGraph()
        assert g.get_edge("nonexistent") is None

    def test_has_node_false(self) -> None:
        g = SemanticGraph()
        assert g.has_node("x") is False

    def test_has_edge_false(self) -> None:
        g = SemanticGraph()
        assert g.has_edge("x") is False

    def test_contains_dunder(self) -> None:
        """Test __contains__ protocol."""
        g = SemanticGraph()
        m = _mod()
        g.add_node(m)
        assert m.id in g
        assert "nonexistent" not in g

    def test_repr(self) -> None:
        g, _, _, _, _ = _populated_graph()
        assert "nodes=4" in repr(g)
        assert "edges=4" in repr(g)


# ── Remove Operations ────────────────────────────────────────────────────


class TestSemanticGraphRemove:
    """Tests for remove_node and remove_edge."""

    def test_remove_node(self) -> None:
        g = SemanticGraph()
        m = _mod()
        g.add_node(m)
        removed = g.remove_node(m.id)
        assert removed is m
        assert not g.has_node(m.id)
        assert len(g) == 0

    def test_remove_node_not_found(self) -> None:
        g = SemanticGraph()
        assert g.remove_node("nonexistent") is None

    def test_remove_node_cascades_edges(self) -> None:
        """Removing a node removes all connected edges."""
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        e = _edge(a.id, b.id)
        g.add_edge(e)

        g.remove_node(a.id)
        assert not g.has_edge(e.id)
        assert g.has_node(b.id)

    def test_remove_node_cascades_incoming_edges(self) -> None:
        """Removing target node removes incoming edges."""
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        e = _edge(a.id, b.id)
        g.add_edge(e)

        g.remove_node(b.id)
        assert not g.has_edge(e.id)
        assert g.has_node(a.id)

    def test_remove_edge(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        e = _edge(a.id, b.id)
        g.add_edge(e)

        removed = g.remove_edge(e.id)
        assert removed is e
        assert not g.has_edge(e.id)
        assert g.has_node(a.id)
        assert g.has_node(b.id)

    def test_remove_edge_not_found(self) -> None:
        g = SemanticGraph()
        assert g.remove_edge("nonexistent") is None


# ── Iteration ─────────────────────────────────────────────────────────────


class TestSemanticGraphIteration:
    """Tests for get_nodes and get_edges iteration."""

    def test_get_nodes_all(self) -> None:
        g, _, _, _, _ = _populated_graph()
        all_nodes = list(g.get_nodes())
        assert len(all_nodes) == 4

    def test_get_nodes_by_type(self) -> None:
        g, _, _, _, _ = _populated_graph()
        modules = list(g.get_nodes(NodeType.MODULE))
        assert len(modules) == 1
        functions = list(g.get_nodes(NodeType.FUNCTION))
        assert len(functions) == 2

    def test_get_nodes_by_type_empty(self) -> None:
        g = SemanticGraph()
        assert list(g.get_nodes(NodeType.ENDPOINT)) == []

    def test_get_edges_all(self) -> None:
        g, _, _, _, _ = _populated_graph()
        all_edges = list(g.get_edges())
        assert len(all_edges) == 4

    def test_get_edges_by_type(self) -> None:
        g, _, _, _, _ = _populated_graph()
        contains = list(g.get_edges(EdgeType.CONTAINS))
        assert len(contains) == 3
        calls = list(g.get_edges(EdgeType.CALLS))
        assert len(calls) == 1

    def test_get_edges_by_type_empty(self) -> None:
        g = SemanticGraph()
        assert list(g.get_edges(EdgeType.IMPORTS)) == []


# ── Traversal ─────────────────────────────────────────────────────────────


class TestSemanticGraphTraversal:
    """Tests for successors, predecessors, outgoing, incoming edges."""

    def test_get_successors(self) -> None:
        g, m, c, fa, fb = _populated_graph()
        succ = g.get_successors(m.id)
        assert len(succ) == 1
        assert succ[0].id == c.id

    def test_get_successors_with_type_filter(self) -> None:
        g, _, c, fa, fb = _populated_graph()
        # c has CONTAINS edges to fa, fb
        succ = g.get_successors(c.id, EdgeType.CONTAINS)
        assert len(succ) == 2
        # c has no CALLS edges
        succ = g.get_successors(c.id, EdgeType.CALLS)
        assert len(succ) == 0

    def test_get_successors_nonexistent_node(self) -> None:
        g = SemanticGraph()
        assert g.get_successors("nonexistent") == []

    def test_get_predecessors(self) -> None:
        g, m, c, _, _ = _populated_graph()
        preds = g.get_predecessors(c.id)
        assert len(preds) == 1
        assert preds[0].id == m.id

    def test_get_predecessors_with_type_filter(self) -> None:
        g, _, c, fa, fb = _populated_graph()
        preds = g.get_predecessors(fb.id, EdgeType.CALLS)
        assert len(preds) == 1
        assert preds[0].id == fa.id

    def test_get_predecessors_nonexistent_node(self) -> None:
        g = SemanticGraph()
        assert g.get_predecessors("nonexistent") == []

    def test_get_outgoing_edges(self) -> None:
        g, _, c, _, _ = _populated_graph()
        out = g.get_outgoing_edges(c.id)
        assert len(out) == 2  # Two CONTAINS edges

    def test_get_outgoing_edges_with_type(self) -> None:
        g, _, c, _, _ = _populated_graph()
        out = g.get_outgoing_edges(c.id, EdgeType.CONTAINS)
        assert len(out) == 2
        out = g.get_outgoing_edges(c.id, EdgeType.CALLS)
        assert len(out) == 0

    def test_get_outgoing_edges_nonexistent(self) -> None:
        g = SemanticGraph()
        assert g.get_outgoing_edges("nonexistent") == []

    def test_get_incoming_edges(self) -> None:
        g, _, _, _, fb = _populated_graph()
        inc = g.get_incoming_edges(fb.id)
        assert len(inc) == 2  # CONTAINS from c + CALLS from fa

    def test_get_incoming_edges_with_type(self) -> None:
        g, _, _, _, fb = _populated_graph()
        inc = g.get_incoming_edges(fb.id, EdgeType.CALLS)
        assert len(inc) == 1
        inc = g.get_incoming_edges(fb.id, EdgeType.CONTAINS)
        assert len(inc) == 1

    def test_get_incoming_edges_nonexistent(self) -> None:
        g = SemanticGraph()
        assert g.get_incoming_edges("nonexistent") == []


# ── Path Finding ──────────────────────────────────────────────────────────


class TestSemanticGraphFindPath:
    """Tests for find_path BFS."""

    def test_find_direct_path(self) -> None:
        g, _, _, fa, fb = _populated_graph()
        path = g.find_path(fa.id, fb.id)
        assert path is not None
        assert len(path) == 2
        assert path[0].id == fa.id
        assert path[1].id == fb.id

    def test_find_multi_hop_path(self) -> None:
        g, m, c, fa, fb = _populated_graph()
        # m → c → fa → fb
        path = g.find_path(m.id, fb.id)
        assert path is not None
        assert len(path) >= 2
        assert path[0].id == m.id
        assert path[-1].id == fb.id

    def test_find_path_same_node(self) -> None:
        g, m, _, _, _ = _populated_graph()
        path = g.find_path(m.id, m.id)
        assert path is not None
        assert len(path) == 1
        assert path[0].id == m.id

    def test_find_path_no_path(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        # No edge between them
        assert g.find_path(a.id, b.id) is None

    def test_find_path_source_nonexistent(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        g.add_node(a)
        assert g.find_path("nonexistent", a.id) is None

    def test_find_path_target_nonexistent(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        g.add_node(a)
        assert g.find_path(a.id, "nonexistent") is None

    def test_find_path_with_edge_type_filter(self) -> None:
        g, m, c, fa, fb = _populated_graph()
        # Path via CONTAINS only: m → c → fa (or fb)
        path = g.find_path(m.id, fa.id, edge_types=[EdgeType.CONTAINS])
        assert path is not None
        assert path[0].id == m.id
        assert path[-1].id == fa.id

    def test_find_path_bfs_skips_already_visited(self) -> None:
        """BFS should skip already-visited nodes when popped from queue.

        Diamond: a→b→d→e→target and a→c→d.
        d gets enqueued twice (via b and c). First pop of d enqueues e.
        Second pop of d hits 'continue' because d is already visited.
        """
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        c = _fn("c", ls=60, le=70)
        d = _fn("d", ls=80, le=90)
        e = _fn("e", ls=100, le=110)
        target = _fn("target", ls=120, le=130)
        g.add_node(a)
        g.add_node(b)
        g.add_node(c)
        g.add_node(d)
        g.add_node(e)
        g.add_node(target)
        g.add_edge(_edge(a.id, b.id, EdgeType.CALLS, line_number=1))
        g.add_edge(_edge(a.id, c.id, EdgeType.CALLS, line_number=2))
        g.add_edge(_edge(b.id, d.id, EdgeType.CALLS, line_number=3))
        g.add_edge(_edge(c.id, d.id, EdgeType.CALLS, line_number=4))
        g.add_edge(_edge(d.id, e.id, EdgeType.CALLS, line_number=5))
        g.add_edge(_edge(e.id, target.id, EdgeType.CALLS, line_number=6))
        path = g.find_path(a.id, target.id)
        assert path is not None
        assert path[0].id == a.id
        assert path[-1].id == target.id

    def test_find_path_filtered_no_path(self) -> None:
        g, m, _, _, _ = _populated_graph()
        # No IMPORTS edges exist
        path = g.find_path(m.id, _populated_graph()[3].id, edge_types=[EdgeType.IMPORTS])
        # Rebuild for clean IDs
        g2 = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g2.add_node(a)
        g2.add_node(b)
        g2.add_edge(_edge(a.id, b.id, EdgeType.CALLS))
        assert g2.find_path(a.id, b.id, edge_types=[EdgeType.IMPORTS]) is None


# ── Cycle Detection ──────────────────────────────────────────────────────


class TestSemanticGraphFindCycles:
    """Tests for find_cycles DFS."""

    def test_no_cycles(self) -> None:
        g, _, _, _, _ = _populated_graph()
        cycles = g.find_cycles()
        assert cycles == []

    def test_simple_cycle(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        g.add_edge(_edge(a.id, b.id, EdgeType.CALLS, line_number=1))
        g.add_edge(_edge(b.id, a.id, EdgeType.CALLS, line_number=2))
        cycles = g.find_cycles()
        assert len(cycles) >= 1

    def test_cycle_with_edge_type_filter(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        g.add_edge(_edge(a.id, b.id, EdgeType.CALLS, line_number=1))
        g.add_edge(_edge(b.id, a.id, EdgeType.IMPORTS, line_number=2))
        # Cycle only exists if both types are followed
        cycles_calls = g.find_cycles(edge_types=[EdgeType.CALLS])
        assert len(cycles_calls) == 0  # No cycle with CALLS only
        # Unfiltered finds the cycle (a→b CALLS, b→a IMPORTS)
        cycles_all = g.find_cycles()
        assert len(cycles_all) >= 1

    def test_find_cycles_empty_graph(self) -> None:
        g = SemanticGraph()
        assert g.find_cycles() == []

    def test_three_node_cycle(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        c = _fn("c", ls=60, le=70)
        g.add_node(a)
        g.add_node(b)
        g.add_node(c)
        g.add_edge(_edge(a.id, b.id, EdgeType.CALLS, line_number=1))
        g.add_edge(_edge(b.id, c.id, EdgeType.CALLS, line_number=2))
        g.add_edge(_edge(c.id, a.id, EdgeType.CALLS, line_number=3))
        cycles = g.find_cycles()
        assert len(cycles) >= 1


# ── Subgraph ─────────────────────────────────────────────────────────────


class TestSemanticGraphSubgraph:
    """Tests for get_subgraph."""

    def test_subgraph_with_subset(self) -> None:
        g, m, c, fa, fb = _populated_graph()
        sub = g.get_subgraph({c.id, fa.id, fb.id})
        assert len(sub) == 3
        # Edges between these nodes should be preserved
        edges = list(sub.get_edges())
        assert len(edges) >= 2  # CONTAINS c→fa, c→fb, CALLS fa→fb

    def test_subgraph_excludes_edges_crossing_boundary(self) -> None:
        g, m, c, fa, _ = _populated_graph()
        # Subgraph with just m and fa — no direct edge between them
        sub = g.get_subgraph({m.id, fa.id})
        assert len(sub) == 2
        # No edge should be present since m→c→fa, and c is excluded
        edges = list(sub.get_edges())
        assert len(edges) == 0

    def test_subgraph_empty_set(self) -> None:
        g, _, _, _, _ = _populated_graph()
        sub = g.get_subgraph(set())
        assert len(sub) == 0

    def test_subgraph_nonexistent_ids_ignored(self) -> None:
        g, m, _, _, _ = _populated_graph()
        sub = g.get_subgraph({m.id, "nonexistent"})
        assert len(sub) == 1


# ── Neighborhood ─────────────────────────────────────────────────────────


class TestSemanticGraphNeighborhood:
    """Tests for get_neighborhood."""

    def test_neighborhood_depth_1(self) -> None:
        g, _, c, fa, fb = _populated_graph()
        # Neighborhood of c at depth 1: m (predecessor), fa, fb (successors)
        sub = g.get_neighborhood(c.id, depth=1)
        assert len(sub) >= 3  # c + at least m + fa + fb

    def test_neighborhood_depth_2(self) -> None:
        g, m, c, fa, fb = _populated_graph()
        sub = g.get_neighborhood(fa.id, depth=2)
        # depth 1: c (predecessor via CONTAINS), fb (successor via CALLS)
        # depth 2: m (c's predecessor), nothing new from fb
        assert sub.has_node(fa.id)
        assert sub.has_node(c.id)
        assert sub.has_node(fb.id)

    def test_neighborhood_nonexistent_raises(self) -> None:
        g = SemanticGraph()
        with pytest.raises(KeyError, match="not found"):
            g.get_neighborhood("nonexistent")

    def test_neighborhood_depth_0(self) -> None:
        """Depth 0 should only include the center node."""
        g, m, _, _, _ = _populated_graph()
        sub = g.get_neighborhood(m.id, depth=0)
        assert len(sub) == 1
        assert sub.has_node(m.id)

    def test_neighborhood_with_edge_type_filter(self) -> None:
        g, m, c, fa, fb = _populated_graph()
        # Only follow CALLS edges from fa
        sub = g.get_neighborhood(fa.id, depth=1, edge_types=[EdgeType.CALLS])
        assert sub.has_node(fa.id)
        assert sub.has_node(fb.id)
        # c should NOT be in neighborhood (CONTAINS edge filtered out)
        assert not sub.has_node(c.id)


# ── NetworkX Conversion ──────────────────────────────────────────────────


class TestSemanticGraphToNetworkX:
    """Tests for to_networkx."""

    def test_empty_graph_conversion(self) -> None:
        g = SemanticGraph()
        nx_g = g.to_networkx()
        assert len(nx_g.nodes) == 0
        assert len(nx_g.edges) == 0

    def test_populated_graph_conversion(self) -> None:
        g, m, c, fa, fb = _populated_graph()
        nx_g = g.to_networkx()
        assert len(nx_g.nodes) == 4
        assert len(nx_g.edges) == 4

    def test_node_attributes(self) -> None:
        g = SemanticGraph()
        m = _mod("myapp")
        g.add_node(m)
        nx_g = g.to_networkx()
        attrs = nx_g.nodes[m.id]
        assert attrs["name"] == "myapp"
        assert attrs["node_type"] == "module"
        assert attrs["file_path"] == "myapp.py"

    def test_edge_attributes(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        g.add_edge(_edge(a.id, b.id, EdgeType.CALLS))
        nx_g = g.to_networkx()
        edge_data = nx_g.edges[a.id, b.id]
        assert edge_data["edge_type"] == "calls"
        assert edge_data["weight"] == 1.0


# ── Stats ─────────────────────────────────────────────────────────────────


class TestSemanticGraphStats:
    """Tests for get_stats."""

    def test_empty_graph_stats(self) -> None:
        g = SemanticGraph()
        stats = g.get_stats()
        assert stats.node_count == 0
        assert stats.edge_count == 0
        assert stats.connected_components == 0
        assert stats.max_depth == 0

    def test_populated_graph_stats(self) -> None:
        g, _, _, _, _ = _populated_graph()
        stats = g.get_stats()
        assert stats.node_count == 4
        assert stats.edge_count == 4
        assert stats.connected_components == 1
        assert "module" in stats.nodes_by_type
        assert "contains" in stats.edges_by_type

    def test_max_depth_calculation(self) -> None:
        """module → class → function gives depth 3."""
        g, _, _, _, _ = _populated_graph()
        stats = g.get_stats()
        assert stats.max_depth >= 2  # At least module→class→function

    def test_disconnected_components(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        # No edges → 2 disconnected components
        stats = g.get_stats()
        assert stats.connected_components == 2

    def test_max_depth_no_contains_edges(self) -> None:
        """Graph with no CONTAINS edges has max_depth 0."""
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        g.add_edge(_edge(a.id, b.id, EdgeType.CALLS))
        stats = g.get_stats()
        # All nodes are roots, none have CONTAINS children
        # Each root contributes depth 1 via calc_depth
        assert stats.max_depth == 1


# ── _calculate_max_depth edge cases ─────────────────────────────────────


class TestMaxDepthEdgeCases:
    """Targeted tests for _calculate_max_depth internals."""

    def test_single_node_depth(self) -> None:
        g = SemanticGraph()
        m = _mod()
        g.add_node(m)
        # Single node is a root with depth 1
        assert g._calculate_max_depth() == 1

    def test_chain_depth(self) -> None:
        """m CONTAINS c CONTAINS f → depth 3."""
        g = SemanticGraph()
        m = _mod()
        c = _cls("C", "app.py", 5, 50)
        f = _fn("f", "app.py", 10, 20)
        g.add_node(m)
        g.add_node(c)
        g.add_node(f)
        g.add_edge(_edge(m.id, c.id, EdgeType.CONTAINS))
        g.add_edge(_edge(c.id, f.id, EdgeType.CONTAINS))
        assert g._calculate_max_depth() == 3

    def test_no_nodes_depth(self) -> None:
        g = SemanticGraph()
        assert g._calculate_max_depth() == 0


# ── _remove_edge_from_indices ────────────────────────────────────────────


class TestRemoveEdgeFromIndices:
    """Ensure _remove_edge_from_indices properly cleans up all indices."""

    def test_indices_cleaned_after_edge_removal(self) -> None:
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        e = _edge(a.id, b.id, EdgeType.CALLS)
        g.add_edge(e)

        g.remove_edge(e.id)
        assert g.get_outgoing_edges(a.id) == []
        assert g.get_incoming_edges(b.id) == []
        assert list(g.get_edges(EdgeType.CALLS)) == []


# ── Branch Partial Coverage ──────────────────────────────────────────────


class TestBranchPartials:
    """Target remaining branch partials for 100% branch coverage."""

    def test_get_successors_dangling_edge(self) -> None:
        """304->301: target_node is None in get_successors."""
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        e = _edge(a.id, b.id, EdgeType.CALLS)
        g.add_edge(e)
        # Corrupt: remove b from _nodes without cascade
        del g._nodes[b.id]
        if b.node_type in g._nodes_by_type:
            g._nodes_by_type[b.node_type].discard(b.id)
        # get_successors should skip the dangling edge
        assert g.get_successors(a.id) == []

    def test_get_predecessors_dangling_edge(self) -> None:
        """326->323: source_node is None in get_predecessors."""
        g = SemanticGraph()
        a = _fn("a")
        b = _fn("b", ls=40, le=50)
        g.add_node(a)
        g.add_node(b)
        e = _edge(a.id, b.id, EdgeType.CALLS)
        g.add_edge(e)
        # Corrupt: remove a from _nodes without cascade
        del g._nodes[a.id]
        if a.node_type in g._nodes_by_type:
            g._nodes_by_type[a.node_type].discard(a.id)
        # get_predecessors should skip the dangling edge
        assert g.get_predecessors(b.id) == []

    def test_find_path_inner_loop_visited_skip(self) -> None:
        """410->401: BFS inner loop encounters edge to already-visited node."""
        g = SemanticGraph()
        a = _fn("a", ls=1, le=5)
        b = _fn("b", ls=10, le=15)
        c = _fn("c", ls=20, le=25)
        d = _fn("d", ls=30, le=35)
        for n in [a, b, c, d]:
            g.add_node(n)
        # a -> b -> c -> a (cycle), c -> d (target)
        g.add_edge(_edge(a.id, b.id, EdgeType.CALLS))
        g.add_edge(_edge(b.id, c.id, EdgeType.CALLS))
        g.add_edge(_edge(c.id, a.id, EdgeType.CALLS))  # back edge
        g.add_edge(_edge(c.id, d.id, EdgeType.CALLS))  # to target
        # BFS: pop a (visited={a}), enqueue b.
        # pop b (visited={a,b}), enqueue c.
        # pop c (visited={a,b,c}), edges: c->a (a in visited! 410->401), c->d (target found).
        path = g.find_path(a.id, d.id)
        assert path is not None
        assert len(path) == 4  # a -> b -> c -> d

    def test_get_neighborhood_edge_type_filter_excludes(self) -> None:
        """514->513: edge_types filter excludes some edges in get_neighborhood."""
        g = SemanticGraph()
        a = _fn("a", ls=1, le=5)
        b = _fn("b", ls=10, le=15)
        c = _fn("c", ls=20, le=25)
        for n in [a, b, c]:
            g.add_node(n)
        g.add_edge(_edge(a.id, b.id, EdgeType.CALLS))
        g.add_edge(_edge(a.id, c.id, EdgeType.IMPORTS))
        # With CALLS filter, only b should be in neighborhood, not c
        neighborhood = g.get_neighborhood(a.id, depth=1, edge_types=[EdgeType.CALLS])
        node_ids = {n.id for n in neighborhood.get_nodes()}
        assert b.id in node_ids
        assert c.id not in node_ids
