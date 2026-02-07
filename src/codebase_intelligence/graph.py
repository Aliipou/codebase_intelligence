"""Semantic graph implementation.

The SemanticGraph is the core data structure that represents a codebase
as a directed graph of nodes (code entities) and edges (relationships).

Key Features:
    - Efficient node and edge lookup by ID
    - Graph traversal methods (successors, predecessors, paths)
    - Subgraph extraction for focused analysis
    - Cycle detection for dependency analysis
    - Integration with NetworkX for advanced algorithms

Usage:
    >>> graph = SemanticGraph()
    >>> graph.add_node(module_node)
    >>> graph.add_node(function_node)
    >>> graph.add_edge(contains_edge)
    >>> successors = graph.get_successors(module_node.id)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Sequence

import networkx as nx
from pydantic import BaseModel, Field

from codebase_intelligence.edges import EdgeType, SemanticEdge
from codebase_intelligence.nodes import NodeType, SemanticNode


class GraphStats(BaseModel):
    """Statistics about a semantic graph.

    Provides insight into the size and structure of the graph
    for analysis and debugging purposes.

    Attributes:
        node_count: Total number of nodes.
        edge_count: Total number of edges.
        nodes_by_type: Count of nodes grouped by type.
        edges_by_type: Count of edges grouped by type.
        connected_components: Number of weakly connected components.
        max_depth: Maximum depth of the containment hierarchy.
    """

    node_count: int = 0
    edge_count: int = 0
    nodes_by_type: dict[str, int] = Field(default_factory=dict)
    edges_by_type: dict[str, int] = Field(default_factory=dict)
    connected_components: int = 0
    max_depth: int = 0


class SemanticGraph:
    """A directed graph representing code structure and relationships.

    The SemanticGraph stores nodes (code entities like modules, classes,
    functions) and edges (relationships like contains, calls, imports).
    It provides efficient lookup, traversal, and analysis operations.

    Thread Safety:
        This class is NOT thread-safe. External synchronization is
        required for concurrent access.

    Attributes:
        _nodes: Internal storage mapping node IDs to nodes.
        _edges: Internal storage mapping edge IDs to edges.
        _outgoing: Index of outgoing edges by source node ID.
        _incoming: Index of incoming edges by target node ID.

    Examples:
        >>> graph = SemanticGraph()

        >>> # Add a module and a function
        >>> module = ModuleNode(name="app", file_path="app.py", line_start=1, line_end=100)
        >>> func = FunctionNode(name="main", file_path="app.py", line_start=10, line_end=20)
        >>> graph.add_node(module)
        >>> graph.add_node(func)

        >>> # Connect them with a containment edge
        >>> edge = SemanticEdge(
        ...     source_id=module.id,
        ...     target_id=func.id,
        ...     edge_type=EdgeType.CONTAINS
        ... )
        >>> graph.add_edge(edge)

        >>> # Query relationships
        >>> children = graph.get_successors(module.id, EdgeType.CONTAINS)
        >>> assert func.id in [n.id for n in children]
    """

    def __init__(self) -> None:
        """Initialize an empty semantic graph."""
        self._nodes: dict[str, SemanticNode] = {}
        self._edges: dict[str, SemanticEdge] = {}
        self._outgoing: dict[str, list[SemanticEdge]] = defaultdict(list)
        self._incoming: dict[str, list[SemanticEdge]] = defaultdict(list)
        self._nodes_by_type: dict[NodeType, set[str]] = defaultdict(set)
        self._edges_by_type: dict[EdgeType, set[str]] = defaultdict(set)

    def add_node(self, node: SemanticNode) -> None:
        """Add a node to the graph.

        If a node with the same ID already exists, it will be replaced.

        Args:
            node: The semantic node to add.

        Raises:
            ValueError: If node is None.
        """
        if node is None:
            raise ValueError("Cannot add None as a node")

        # Remove from type index if updating existing node
        if node.id in self._nodes:
            old_node = self._nodes[node.id]
            self._nodes_by_type[old_node.node_type].discard(node.id)

        self._nodes[node.id] = node
        self._nodes_by_type[node.node_type].add(node.id)

    def add_edge(self, edge: SemanticEdge) -> None:
        """Add an edge to the graph.

        Both source and target nodes must exist in the graph.
        If an edge with the same ID already exists, it will be replaced.

        Args:
            edge: The semantic edge to add.

        Raises:
            ValueError: If edge is None.
            KeyError: If source or target node doesn't exist.
        """
        if edge is None:
            raise ValueError("Cannot add None as an edge")

        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node '{edge.source_id}' not found in graph")

        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node '{edge.target_id}' not found in graph")

        # Remove old edge from indices if updating
        if edge.id in self._edges:
            self._remove_edge_from_indices(self._edges[edge.id])

        self._edges[edge.id] = edge
        self._outgoing[edge.source_id].append(edge)
        self._incoming[edge.target_id].append(edge)
        self._edges_by_type[edge.edge_type].add(edge.id)

    def _remove_edge_from_indices(self, edge: SemanticEdge) -> None:
        """Remove an edge from all internal indices."""
        self._outgoing[edge.source_id] = [
            e for e in self._outgoing[edge.source_id] if e.id != edge.id
        ]
        self._incoming[edge.target_id] = [
            e for e in self._incoming[edge.target_id] if e.id != edge.id
        ]
        self._edges_by_type[edge.edge_type].discard(edge.id)

    def remove_node(self, node_id: str) -> SemanticNode | None:
        """Remove a node and all its connected edges from the graph.

        Args:
            node_id: ID of the node to remove.

        Returns:
            The removed node, or None if not found.
        """
        if node_id not in self._nodes:
            return None

        node = self._nodes[node_id]

        # Remove all connected edges
        edges_to_remove = list(self._outgoing[node_id]) + list(self._incoming[node_id])
        for edge in edges_to_remove:
            self.remove_edge(edge.id)

        # Remove from indices
        self._nodes_by_type[node.node_type].discard(node_id)
        del self._nodes[node_id]
        del self._outgoing[node_id]
        del self._incoming[node_id]

        return node

    def remove_edge(self, edge_id: str) -> SemanticEdge | None:
        """Remove an edge from the graph.

        Args:
            edge_id: ID of the edge to remove.

        Returns:
            The removed edge, or None if not found.
        """
        if edge_id not in self._edges:
            return None

        edge = self._edges[edge_id]
        self._remove_edge_from_indices(edge)
        del self._edges[edge_id]

        return edge

    def get_node(self, node_id: str) -> SemanticNode | None:
        """Get a node by its ID.

        Args:
            node_id: The unique identifier of the node.

        Returns:
            The node if found, None otherwise.
        """
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> SemanticEdge | None:
        """Get an edge by its ID.

        Args:
            edge_id: The unique identifier of the edge.

        Returns:
            The edge if found, None otherwise.
        """
        return self._edges.get(edge_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The node ID to check.

        Returns:
            True if node exists, False otherwise.
        """
        return node_id in self._nodes

    def has_edge(self, edge_id: str) -> bool:
        """Check if an edge exists in the graph.

        Args:
            edge_id: The edge ID to check.

        Returns:
            True if edge exists, False otherwise.
        """
        return edge_id in self._edges

    def get_nodes(self, node_type: NodeType | None = None) -> Iterator[SemanticNode]:
        """Iterate over nodes in the graph.

        Args:
            node_type: If provided, only yield nodes of this type.

        Yields:
            SemanticNode instances matching the filter criteria.
        """
        if node_type is None:
            yield from self._nodes.values()
        else:
            for node_id in self._nodes_by_type.get(node_type, set()):
                yield self._nodes[node_id]

    def get_edges(self, edge_type: EdgeType | None = None) -> Iterator[SemanticEdge]:
        """Iterate over edges in the graph.

        Args:
            edge_type: If provided, only yield edges of this type.

        Yields:
            SemanticEdge instances matching the filter criteria.
        """
        if edge_type is None:
            yield from self._edges.values()
        else:
            for edge_id in self._edges_by_type.get(edge_type, set()):
                yield self._edges[edge_id]

    def get_successors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[SemanticNode]:
        """Get all nodes connected by outgoing edges from a node.

        Args:
            node_id: ID of the source node.
            edge_type: If provided, only follow edges of this type.

        Returns:
            List of nodes connected by outgoing edges.
        """
        result = []
        for edge in self._outgoing.get(node_id, []):
            if edge_type is None or edge.edge_type == edge_type:
                target_node = self._nodes.get(edge.target_id)
                if target_node is not None:
                    result.append(target_node)
        return result

    def get_predecessors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[SemanticNode]:
        """Get all nodes connected by incoming edges to a node.

        Args:
            node_id: ID of the target node.
            edge_type: If provided, only follow edges of this type.

        Returns:
            List of nodes connected by incoming edges.
        """
        result = []
        for edge in self._incoming.get(node_id, []):
            if edge_type is None or edge.edge_type == edge_type:
                source_node = self._nodes.get(edge.source_id)
                if source_node is not None:
                    result.append(source_node)
        return result

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[SemanticEdge]:
        """Get all outgoing edges from a node.

        Args:
            node_id: ID of the source node.
            edge_type: If provided, only return edges of this type.

        Returns:
            List of outgoing edges.
        """
        edges = self._outgoing.get(node_id, [])
        if edge_type is None:
            return list(edges)
        return [e for e in edges if e.edge_type == edge_type]

    def get_incoming_edges(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[SemanticEdge]:
        """Get all incoming edges to a node.

        Args:
            node_id: ID of the target node.
            edge_type: If provided, only return edges of this type.

        Returns:
            List of incoming edges.
        """
        edges = self._incoming.get(node_id, [])
        if edge_type is None:
            return list(edges)
        return [e for e in edges if e.edge_type == edge_type]

    def find_path(
        self,
        source_id: str,
        target_id: str,
        edge_types: Sequence[EdgeType] | None = None,
    ) -> list[SemanticNode] | None:
        """Find a path between two nodes using BFS.

        Args:
            source_id: ID of the starting node.
            target_id: ID of the destination node.
            edge_types: If provided, only traverse edges of these types.

        Returns:
            List of nodes forming the path (including source and target),
            or None if no path exists.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        if source_id == target_id:
            return [self._nodes[source_id]]

        visited: set[str] = set()
        queue: list[tuple[str, list[str]]] = [(source_id, [source_id])]

        while queue:
            current_id, path = queue.pop(0)

            if current_id in visited:
                continue
            visited.add(current_id)

            for edge in self._outgoing.get(current_id, []):
                if edge_types is not None and edge.edge_type not in edge_types:
                    continue

                next_id = edge.target_id
                if next_id == target_id:
                    path_nodes = path + [next_id]
                    return [self._nodes[nid] for nid in path_nodes]

                if next_id not in visited:
                    queue.append((next_id, path + [next_id]))

        return None

    def find_cycles(self, edge_types: Sequence[EdgeType] | None = None) -> list[list[str]]:
        """Find all cycles in the graph.

        Uses DFS to detect back edges that form cycles.

        Args:
            edge_types: If provided, only consider edges of these types.

        Returns:
            List of cycles, where each cycle is a list of node IDs.
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node_id: str) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for edge in self._outgoing.get(node_id, []):
                if edge_types is not None and edge.edge_type not in edge_types:
                    continue

                next_id = edge.target_id
                if next_id not in visited:
                    dfs(next_id)
                elif next_id in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(next_id)
                    cycles.append(path[cycle_start:] + [next_id])

            path.pop()
            rec_stack.remove(node_id)

        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)

        return cycles

    def get_subgraph(self, node_ids: set[str]) -> "SemanticGraph":
        """Extract a subgraph containing only the specified nodes.

        Edges are included only if both endpoints are in the subgraph.

        Args:
            node_ids: Set of node IDs to include.

        Returns:
            A new SemanticGraph containing the subgraph.
        """
        subgraph = SemanticGraph()

        for node_id in node_ids:
            if node_id in self._nodes:
                subgraph.add_node(self._nodes[node_id])

        for edge in self._edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                subgraph.add_edge(edge)

        return subgraph

    def get_neighborhood(
        self,
        node_id: str,
        depth: int = 1,
        edge_types: Sequence[EdgeType] | None = None,
    ) -> "SemanticGraph":
        """Extract the neighborhood subgraph around a node.

        Includes all nodes reachable within the specified depth,
        following edges in both directions.

        Args:
            node_id: ID of the center node.
            depth: Maximum distance from the center node.
            edge_types: If provided, only follow edges of these types.

        Returns:
            A new SemanticGraph containing the neighborhood.

        Raises:
            KeyError: If the center node doesn't exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in graph")

        node_ids: set[str] = {node_id}
        frontier: set[str] = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()

            for current_id in frontier:
                # Add successors
                for edge in self._outgoing.get(current_id, []):
                    if edge_types is None or edge.edge_type in edge_types:
                        next_frontier.add(edge.target_id)

                # Add predecessors
                for edge in self._incoming.get(current_id, []):
                    if edge_types is None or edge.edge_type in edge_types:
                        next_frontier.add(edge.source_id)

            next_frontier -= node_ids
            node_ids.update(next_frontier)
            frontier = next_frontier

        return self.get_subgraph(node_ids)

    def to_networkx(self) -> nx.DiGraph:
        """Convert to a NetworkX directed graph.

        Useful for advanced graph algorithms not implemented here.

        Returns:
            A NetworkX DiGraph with nodes and edges from this graph.
        """
        G = nx.DiGraph()

        for node in self._nodes.values():
            G.add_node(
                node.id,
                name=node.name,
                node_type=node.node_type.value,
                file_path=node.file_path,
            )

        for edge in self._edges.values():
            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
            )

        return G

    def get_stats(self) -> GraphStats:
        """Calculate statistics about the graph.

        Returns:
            GraphStats with counts and metrics about the graph.
        """
        nodes_by_type = {
            node_type.value: len(node_ids)
            for node_type, node_ids in self._nodes_by_type.items()
            if node_ids
        }

        edges_by_type = {
            edge_type.value: len(edge_ids)
            for edge_type, edge_ids in self._edges_by_type.items()
            if edge_ids
        }

        # Calculate connected components using NetworkX
        nx_graph = self.to_networkx()
        connected_components = nx.number_weakly_connected_components(nx_graph)

        # Calculate max depth of containment hierarchy
        max_depth = self._calculate_max_depth()

        return GraphStats(
            node_count=len(self._nodes),
            edge_count=len(self._edges),
            nodes_by_type=nodes_by_type,
            edges_by_type=edges_by_type,
            connected_components=connected_components,
            max_depth=max_depth,
        )

    def _calculate_max_depth(self) -> int:
        """Calculate the maximum depth of the containment hierarchy."""
        max_depth = 0

        # Find root nodes (no incoming CONTAINS edges)
        roots: set[str] = set()
        for node_id in self._nodes:
            has_parent = any(
                e.edge_type == EdgeType.CONTAINS for e in self._incoming.get(node_id, [])
            )
            if not has_parent:
                roots.add(node_id)

        def calc_depth(node_id: str, current_depth: int) -> int:
            max_child_depth = current_depth
            for edge in self._outgoing.get(node_id, []):
                if edge.edge_type == EdgeType.CONTAINS:
                    child_depth = calc_depth(edge.target_id, current_depth + 1)
                    max_child_depth = max(max_child_depth, child_depth)
            return max_child_depth

        for root_id in roots:
            depth = calc_depth(root_id, 1)
            max_depth = max(max_depth, depth)

        return max_depth

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if a node ID is in the graph."""
        return node_id in self._nodes

    def __repr__(self) -> str:
        """Return string representation of the graph."""
        return f"SemanticGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
