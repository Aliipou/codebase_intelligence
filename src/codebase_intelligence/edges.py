"""Semantic graph edge definitions.

Edges represent relationships between code entities:
- Import dependencies
- Function calls
- Class inheritance
- Type usage
- Containment (module contains class, class contains method)

Each edge is directed, typed, and may carry additional metadata
about the nature of the relationship.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class EdgeType(str, Enum):
    """Types of edges (relationships) in the semantic graph.

    Categories:
        Structural: CONTAINS, DEFINED_IN
        Dependency: IMPORTS, CALLS, USES_TYPE
        Inheritance: INHERITS, IMPLEMENTS
        Reference: REFERENCES, DECORATES
    """

    # Structural relationships
    CONTAINS = "contains"
    DEFINED_IN = "defined_in"

    # Dependency relationships
    IMPORTS = "imports"
    CALLS = "calls"
    USES_TYPE = "uses_type"

    # Inheritance relationships
    INHERITS = "inherits"
    IMPLEMENTS = "implements"

    # Reference relationships
    REFERENCES = "references"
    DECORATES = "decorates"
    INSTANTIATES = "instantiates"

    # FastAPI-specific
    DEPENDS_ON = "depends_on"
    ROUTES_TO = "routes_to"


class EdgeCategory(str, Enum):
    """High-level categorization of edge types."""

    STRUCTURAL = "structural"
    DEPENDENCY = "dependency"
    INHERITANCE = "inheritance"
    REFERENCE = "reference"
    FRAMEWORK = "framework"


# Mapping from edge types to categories
EDGE_TYPE_CATEGORIES: dict[EdgeType, EdgeCategory] = {
    EdgeType.CONTAINS: EdgeCategory.STRUCTURAL,
    EdgeType.DEFINED_IN: EdgeCategory.STRUCTURAL,
    EdgeType.IMPORTS: EdgeCategory.DEPENDENCY,
    EdgeType.CALLS: EdgeCategory.DEPENDENCY,
    EdgeType.USES_TYPE: EdgeCategory.DEPENDENCY,
    EdgeType.INHERITS: EdgeCategory.INHERITANCE,
    EdgeType.IMPLEMENTS: EdgeCategory.INHERITANCE,
    EdgeType.REFERENCES: EdgeCategory.REFERENCE,
    EdgeType.DECORATES: EdgeCategory.REFERENCE,
    EdgeType.INSTANTIATES: EdgeCategory.REFERENCE,
    EdgeType.DEPENDS_ON: EdgeCategory.FRAMEWORK,
    EdgeType.ROUTES_TO: EdgeCategory.FRAMEWORK,
}


class SemanticEdge(BaseModel):
    """Represents a directed relationship between two semantic nodes.

    An edge connects a source node to a target node with a specific
    relationship type. Edges are immutable once created.

    Attributes:
        id: Unique identifier for the edge (auto-generated).
        source_id: ID of the source node (where the edge originates).
        target_id: ID of the target node (where the edge points).
        edge_type: Type of relationship this edge represents.
        weight: Strength/importance of the relationship (0.0-1.0).
        metadata: Additional edge-specific information.
        line_number: Source line where this relationship is established.
        is_conditional: Whether this edge represents a conditional relationship.

    Examples:
        >>> # A function calls another function
        >>> edge = SemanticEdge(
        ...     source_id="func_a_id",
        ...     target_id="func_b_id",
        ...     edge_type=EdgeType.CALLS,
        ...     line_number=42
        ... )

        >>> # A class inherits from a base class
        >>> edge = SemanticEdge(
        ...     source_id="child_class_id",
        ...     target_id="parent_class_id",
        ...     edge_type=EdgeType.INHERITS
        ... )
    """

    id: str = Field(default="")
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    line_number: int | None = Field(default=None, ge=1)
    is_conditional: bool = False

    model_config = {"frozen": True}

    def model_post_init(self, __context: Any) -> None:
        """Generate ID after initialization if not provided."""
        if not self.id:
            object.__setattr__(self, "id", self._generate_id())

    def _generate_id(self) -> str:
        """Generate a unique ID based on edge properties.

        The ID is a hash of source, target, type, and line number
        to ensure uniqueness even for multiple edges between same nodes.
        """
        unique_str = f"{self.source_id}:{self.target_id}:{self.edge_type.value}:{self.line_number}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

    @model_validator(mode="after")
    def validate_no_self_loop(self) -> "SemanticEdge":
        """Ensure edge doesn't connect a node to itself."""
        if self.source_id == self.target_id:
            raise ValueError("Self-loops are not allowed: source_id cannot equal target_id")
        return self

    @property
    def category(self) -> EdgeCategory:
        """Get the high-level category of this edge type."""
        return EDGE_TYPE_CATEGORIES[self.edge_type]

    def is_structural(self) -> bool:
        """Check if this is a structural relationship (containment, definition)."""
        return self.category == EdgeCategory.STRUCTURAL

    def is_dependency(self) -> bool:
        """Check if this is a dependency relationship (imports, calls, uses)."""
        return self.category == EdgeCategory.DEPENDENCY

    def is_inheritance(self) -> bool:
        """Check if this is an inheritance relationship."""
        return self.category == EdgeCategory.INHERITANCE

    def reversed(self) -> "SemanticEdge":
        """Create a new edge with source and target swapped.

        Useful for traversing the graph in reverse direction.
        Note: The reversed edge gets a new ID.

        Returns:
            A new SemanticEdge with source and target swapped.
        """
        return SemanticEdge(
            source_id=self.target_id,
            target_id=self.source_id,
            edge_type=self.edge_type,
            weight=self.weight,
            metadata=self.metadata,
            line_number=self.line_number,
            is_conditional=self.is_conditional,
        )

    def with_weight(self, new_weight: float) -> "SemanticEdge":
        """Create a new edge with updated weight.

        Args:
            new_weight: The new weight value (0.0-1.0).

        Returns:
            A new SemanticEdge with the updated weight.
        """
        return SemanticEdge(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            edge_type=self.edge_type,
            weight=new_weight,
            metadata=self.metadata,
            line_number=self.line_number,
            is_conditional=self.is_conditional,
        )

    def describes_same_relationship(self, other: "SemanticEdge") -> bool:
        """Check if two edges describe the same logical relationship.

        Two edges describe the same relationship if they have the same
        source, target, and type (regardless of other attributes).

        Args:
            other: Another edge to compare with.

        Returns:
            True if edges represent the same relationship.
        """
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.edge_type == other.edge_type
        )


def create_contains_edge(container_id: str, contained_id: str) -> SemanticEdge:
    """Factory function to create a CONTAINS edge.

    Args:
        container_id: ID of the containing node (e.g., module, class).
        contained_id: ID of the contained node (e.g., function, method).

    Returns:
        A SemanticEdge representing containment.
    """
    return SemanticEdge(
        source_id=container_id,
        target_id=contained_id,
        edge_type=EdgeType.CONTAINS,
    )


def create_calls_edge(
    caller_id: str,
    callee_id: str,
    line_number: int,
    is_conditional: bool = False,
) -> SemanticEdge:
    """Factory function to create a CALLS edge.

    Args:
        caller_id: ID of the calling function.
        callee_id: ID of the called function.
        line_number: Line number where the call occurs.
        is_conditional: Whether the call is inside a conditional block.

    Returns:
        A SemanticEdge representing a function call.
    """
    return SemanticEdge(
        source_id=caller_id,
        target_id=callee_id,
        edge_type=EdgeType.CALLS,
        line_number=line_number,
        is_conditional=is_conditional,
    )


def create_imports_edge(
    importer_id: str,
    imported_id: str,
    line_number: int,
) -> SemanticEdge:
    """Factory function to create an IMPORTS edge.

    Args:
        importer_id: ID of the importing module.
        imported_id: ID of the imported module/name.
        line_number: Line number of the import statement.

    Returns:
        A SemanticEdge representing an import relationship.
    """
    return SemanticEdge(
        source_id=importer_id,
        target_id=imported_id,
        edge_type=EdgeType.IMPORTS,
        line_number=line_number,
    )


def create_inherits_edge(child_id: str, parent_id: str) -> SemanticEdge:
    """Factory function to create an INHERITS edge.

    Args:
        child_id: ID of the child class.
        parent_id: ID of the parent class.

    Returns:
        A SemanticEdge representing class inheritance.
    """
    return SemanticEdge(
        source_id=child_id,
        target_id=parent_id,
        edge_type=EdgeType.INHERITS,
    )
