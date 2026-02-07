"""Brutal unit tests for semantic graph edges.

Tests every code path, edge case, and validation rule for 100% coverage.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from codebase_intelligence.edges import (
    EDGE_TYPE_CATEGORIES,
    EdgeCategory,
    EdgeType,
    SemanticEdge,
    create_calls_edge,
    create_contains_edge,
    create_imports_edge,
    create_inherits_edge,
)


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_all_edge_types_exist(self) -> None:
        """Verify all 12 expected edge types are defined."""
        expected = {
            "CONTAINS",
            "DEFINED_IN",
            "IMPORTS",
            "CALLS",
            "USES_TYPE",
            "INHERITS",
            "IMPLEMENTS",
            "REFERENCES",
            "DECORATES",
            "INSTANTIATES",
            "DEPENDS_ON",
            "ROUTES_TO",
        }
        actual = {et.name for et in EdgeType}
        assert actual == expected

    def test_edge_type_values(self) -> None:
        """Verify edge type values are lowercase strings."""
        for et in EdgeType:
            assert et.value == et.name.lower()

    def test_edge_type_is_string_enum(self) -> None:
        """EdgeType should be usable as a string."""
        assert EdgeType.CONTAINS == "contains"
        assert EdgeType.CALLS == "calls"
        assert EdgeType.INHERITS == "inherits"

    def test_edge_type_count(self) -> None:
        """Exactly 12 edge types."""
        assert len(EdgeType) == 12


class TestEdgeCategory:
    """Tests for EdgeCategory enum."""

    def test_all_categories_exist(self) -> None:
        """Verify all 5 categories."""
        expected = {"STRUCTURAL", "DEPENDENCY", "INHERITANCE", "REFERENCE", "FRAMEWORK"}
        actual = {ec.name for ec in EdgeCategory}
        assert actual == expected

    def test_category_values(self) -> None:
        """Category values are lowercase."""
        for ec in EdgeCategory:
            assert ec.value == ec.name.lower()


class TestEdgeTypeCategoriesMapping:
    """Tests for EDGE_TYPE_CATEGORIES mapping."""

    def test_all_edge_types_mapped(self) -> None:
        """Every EdgeType must have a category mapping."""
        for et in EdgeType:
            assert et in EDGE_TYPE_CATEGORIES, f"{et} not in EDGE_TYPE_CATEGORIES"

    def test_structural_types(self) -> None:
        """Structural edges: CONTAINS, DEFINED_IN."""
        assert EDGE_TYPE_CATEGORIES[EdgeType.CONTAINS] == EdgeCategory.STRUCTURAL
        assert EDGE_TYPE_CATEGORIES[EdgeType.DEFINED_IN] == EdgeCategory.STRUCTURAL

    def test_dependency_types(self) -> None:
        """Dependency edges: IMPORTS, CALLS, USES_TYPE."""
        assert EDGE_TYPE_CATEGORIES[EdgeType.IMPORTS] == EdgeCategory.DEPENDENCY
        assert EDGE_TYPE_CATEGORIES[EdgeType.CALLS] == EdgeCategory.DEPENDENCY
        assert EDGE_TYPE_CATEGORIES[EdgeType.USES_TYPE] == EdgeCategory.DEPENDENCY

    def test_inheritance_types(self) -> None:
        """Inheritance edges: INHERITS, IMPLEMENTS."""
        assert EDGE_TYPE_CATEGORIES[EdgeType.INHERITS] == EdgeCategory.INHERITANCE
        assert EDGE_TYPE_CATEGORIES[EdgeType.IMPLEMENTS] == EdgeCategory.INHERITANCE

    def test_reference_types(self) -> None:
        """Reference edges: REFERENCES, DECORATES, INSTANTIATES."""
        assert EDGE_TYPE_CATEGORIES[EdgeType.REFERENCES] == EdgeCategory.REFERENCE
        assert EDGE_TYPE_CATEGORIES[EdgeType.DECORATES] == EdgeCategory.REFERENCE
        assert EDGE_TYPE_CATEGORIES[EdgeType.INSTANTIATES] == EdgeCategory.REFERENCE

    def test_framework_types(self) -> None:
        """Framework edges: DEPENDS_ON, ROUTES_TO."""
        assert EDGE_TYPE_CATEGORIES[EdgeType.DEPENDS_ON] == EdgeCategory.FRAMEWORK
        assert EDGE_TYPE_CATEGORIES[EdgeType.ROUTES_TO] == EdgeCategory.FRAMEWORK


class TestSemanticEdge:
    """Tests for SemanticEdge model."""

    def test_create_basic_edge(self) -> None:
        """Create a basic edge with required fields."""
        edge = SemanticEdge(
            source_id="node_a",
            target_id="node_b",
            edge_type=EdgeType.CALLS,
        )
        assert edge.source_id == "node_a"
        assert edge.target_id == "node_b"
        assert edge.edge_type == EdgeType.CALLS
        assert edge.weight == 1.0
        assert edge.metadata == {}
        assert edge.line_number is None
        assert edge.is_conditional is False
        assert len(edge.id) == 16

    def test_id_auto_generation(self) -> None:
        """ID is auto-generated from source, target, type, and line_number."""
        edge1 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        edge2 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        # Same properties → same ID
        assert edge1.id == edge2.id

    def test_different_edges_different_ids(self) -> None:
        """Different edge properties → different IDs."""
        edge1 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        edge2 = SemanticEdge(
            source_id="a",
            target_id="c",
            edge_type=EdgeType.CALLS,
        )
        assert edge1.id != edge2.id

    def test_different_types_different_ids(self) -> None:
        """Same source/target but different type → different IDs."""
        edge1 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        edge2 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.IMPORTS,
        )
        assert edge1.id != edge2.id

    def test_different_line_numbers_different_ids(self) -> None:
        """Same source/target/type but different line → different IDs."""
        edge1 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            line_number=10,
        )
        edge2 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            line_number=20,
        )
        assert edge1.id != edge2.id

    def test_custom_id_preserved(self) -> None:
        """Custom ID is preserved when provided."""
        edge = SemanticEdge(
            id="my_custom_id",
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        assert edge.id == "my_custom_id"

    def test_self_loop_rejected(self) -> None:
        """Self-loops are not allowed."""
        with pytest.raises(ValidationError, match="Self-loops"):
            SemanticEdge(
                source_id="same",
                target_id="same",
                edge_type=EdgeType.CALLS,
            )

    def test_empty_source_id_rejected(self) -> None:
        """Empty source_id is rejected."""
        with pytest.raises(ValidationError):
            SemanticEdge(
                source_id="",
                target_id="b",
                edge_type=EdgeType.CALLS,
            )

    def test_empty_target_id_rejected(self) -> None:
        """Empty target_id is rejected."""
        with pytest.raises(ValidationError):
            SemanticEdge(
                source_id="a",
                target_id="",
                edge_type=EdgeType.CALLS,
            )

    def test_weight_default(self) -> None:
        """Default weight is 1.0."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        assert edge.weight == 1.0

    def test_weight_custom(self) -> None:
        """Custom weight between 0.0 and 1.0."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            weight=0.5,
        )
        assert edge.weight == 0.5

    def test_weight_zero(self) -> None:
        """Weight can be 0.0."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            weight=0.0,
        )
        assert edge.weight == 0.0

    def test_weight_one(self) -> None:
        """Weight can be 1.0."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            weight=1.0,
        )
        assert edge.weight == 1.0

    def test_weight_too_low_rejected(self) -> None:
        """Weight below 0.0 is rejected."""
        with pytest.raises(ValidationError):
            SemanticEdge(
                source_id="a",
                target_id="b",
                edge_type=EdgeType.CALLS,
                weight=-0.1,
            )

    def test_weight_too_high_rejected(self) -> None:
        """Weight above 1.0 is rejected."""
        with pytest.raises(ValidationError):
            SemanticEdge(
                source_id="a",
                target_id="b",
                edge_type=EdgeType.CALLS,
                weight=1.1,
            )

    def test_metadata_default(self) -> None:
        """Default metadata is empty dict."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        assert edge.metadata == {}

    def test_metadata_custom(self) -> None:
        """Custom metadata preserved."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            metadata={"key": "val", "count": 42},
        )
        assert edge.metadata == {"key": "val", "count": 42}

    def test_line_number_valid(self) -> None:
        """Valid line numbers >= 1."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            line_number=42,
        )
        assert edge.line_number == 42

    def test_line_number_one(self) -> None:
        """Line number can be 1 (minimum)."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            line_number=1,
        )
        assert edge.line_number == 1

    def test_line_number_zero_rejected(self) -> None:
        """Line number 0 is rejected."""
        with pytest.raises(ValidationError):
            SemanticEdge(
                source_id="a",
                target_id="b",
                edge_type=EdgeType.CALLS,
                line_number=0,
            )

    def test_line_number_negative_rejected(self) -> None:
        """Negative line number is rejected."""
        with pytest.raises(ValidationError):
            SemanticEdge(
                source_id="a",
                target_id="b",
                edge_type=EdgeType.CALLS,
                line_number=-1,
            )

    def test_is_conditional_default_false(self) -> None:
        """is_conditional defaults to False."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        assert edge.is_conditional is False

    def test_is_conditional_true(self) -> None:
        """is_conditional can be set to True."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            is_conditional=True,
        )
        assert edge.is_conditional is True

    def test_edge_is_frozen(self) -> None:
        """Edge should be immutable."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        with pytest.raises(ValidationError):
            edge.source_id = "changed"  # type: ignore


class TestSemanticEdgeCategory:
    """Tests for edge category property and helper methods."""

    def test_category_structural_contains(self) -> None:
        """CONTAINS edge has STRUCTURAL category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CONTAINS,
        )
        assert edge.category == EdgeCategory.STRUCTURAL

    def test_category_structural_defined_in(self) -> None:
        """DEFINED_IN edge has STRUCTURAL category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.DEFINED_IN,
        )
        assert edge.category == EdgeCategory.STRUCTURAL

    def test_category_dependency_imports(self) -> None:
        """IMPORTS edge has DEPENDENCY category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.IMPORTS,
        )
        assert edge.category == EdgeCategory.DEPENDENCY

    def test_category_dependency_calls(self) -> None:
        """CALLS edge has DEPENDENCY category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        assert edge.category == EdgeCategory.DEPENDENCY

    def test_category_dependency_uses_type(self) -> None:
        """USES_TYPE edge has DEPENDENCY category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.USES_TYPE,
        )
        assert edge.category == EdgeCategory.DEPENDENCY

    def test_category_inheritance_inherits(self) -> None:
        """INHERITS edge has INHERITANCE category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.INHERITS,
        )
        assert edge.category == EdgeCategory.INHERITANCE

    def test_category_inheritance_implements(self) -> None:
        """IMPLEMENTS edge has INHERITANCE category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.IMPLEMENTS,
        )
        assert edge.category == EdgeCategory.INHERITANCE

    def test_category_reference_references(self) -> None:
        """REFERENCES edge has REFERENCE category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.REFERENCES,
        )
        assert edge.category == EdgeCategory.REFERENCE

    def test_category_reference_decorates(self) -> None:
        """DECORATES edge has REFERENCE category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.DECORATES,
        )
        assert edge.category == EdgeCategory.REFERENCE

    def test_category_reference_instantiates(self) -> None:
        """INSTANTIATES edge has REFERENCE category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.INSTANTIATES,
        )
        assert edge.category == EdgeCategory.REFERENCE

    def test_category_framework_depends_on(self) -> None:
        """DEPENDS_ON edge has FRAMEWORK category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.DEPENDS_ON,
        )
        assert edge.category == EdgeCategory.FRAMEWORK

    def test_category_framework_routes_to(self) -> None:
        """ROUTES_TO edge has FRAMEWORK category."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.ROUTES_TO,
        )
        assert edge.category == EdgeCategory.FRAMEWORK

    def test_is_structural_true(self) -> None:
        """is_structural() returns True for structural edges."""
        edge = SemanticEdge(
            source_id="a", target_id="b", edge_type=EdgeType.CONTAINS
        )
        assert edge.is_structural() is True

    def test_is_structural_false(self) -> None:
        """is_structural() returns False for non-structural edges."""
        edge = SemanticEdge(
            source_id="a", target_id="b", edge_type=EdgeType.CALLS
        )
        assert edge.is_structural() is False

    def test_is_dependency_true(self) -> None:
        """is_dependency() returns True for dependency edges."""
        edge = SemanticEdge(
            source_id="a", target_id="b", edge_type=EdgeType.IMPORTS
        )
        assert edge.is_dependency() is True

    def test_is_dependency_false(self) -> None:
        """is_dependency() returns False for non-dependency edges."""
        edge = SemanticEdge(
            source_id="a", target_id="b", edge_type=EdgeType.INHERITS
        )
        assert edge.is_dependency() is False

    def test_is_inheritance_true(self) -> None:
        """is_inheritance() returns True for inheritance edges."""
        edge = SemanticEdge(
            source_id="a", target_id="b", edge_type=EdgeType.INHERITS
        )
        assert edge.is_inheritance() is True

    def test_is_inheritance_false(self) -> None:
        """is_inheritance() returns False for non-inheritance edges."""
        edge = SemanticEdge(
            source_id="a", target_id="b", edge_type=EdgeType.CALLS
        )
        assert edge.is_inheritance() is False


class TestSemanticEdgeReversed:
    """Tests for reversed() method."""

    def test_reversed_swaps_source_target(self) -> None:
        """reversed() swaps source_id and target_id."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        rev = edge.reversed()
        assert rev.source_id == "b"
        assert rev.target_id == "a"

    def test_reversed_preserves_type(self) -> None:
        """reversed() preserves edge_type."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.INHERITS,
        )
        rev = edge.reversed()
        assert rev.edge_type == EdgeType.INHERITS

    def test_reversed_preserves_weight(self) -> None:
        """reversed() preserves weight."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            weight=0.75,
        )
        rev = edge.reversed()
        assert rev.weight == 0.75

    def test_reversed_preserves_metadata(self) -> None:
        """reversed() preserves metadata."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            metadata={"key": "val"},
        )
        rev = edge.reversed()
        assert rev.metadata == {"key": "val"}

    def test_reversed_preserves_line_number(self) -> None:
        """reversed() preserves line_number."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            line_number=42,
        )
        rev = edge.reversed()
        assert rev.line_number == 42

    def test_reversed_preserves_is_conditional(self) -> None:
        """reversed() preserves is_conditional."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            is_conditional=True,
        )
        rev = edge.reversed()
        assert rev.is_conditional is True

    def test_reversed_gets_new_id(self) -> None:
        """reversed() generates a new ID (different from original)."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        rev = edge.reversed()
        assert rev.id != edge.id


class TestSemanticEdgeWithWeight:
    """Tests for with_weight() method."""

    def test_with_weight_creates_new_edge(self) -> None:
        """with_weight() returns a new edge with updated weight."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            weight=1.0,
        )
        new_edge = edge.with_weight(0.5)
        assert new_edge.weight == 0.5
        assert edge.weight == 1.0  # Original unchanged

    def test_with_weight_preserves_id(self) -> None:
        """with_weight() preserves the original ID."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        new_edge = edge.with_weight(0.3)
        assert new_edge.id == edge.id

    def test_with_weight_preserves_all_fields(self) -> None:
        """with_weight() preserves all other fields."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            metadata={"key": "val"},
            line_number=10,
            is_conditional=True,
        )
        new_edge = edge.with_weight(0.7)
        assert new_edge.source_id == "a"
        assert new_edge.target_id == "b"
        assert new_edge.edge_type == EdgeType.CALLS
        assert new_edge.metadata == {"key": "val"}
        assert new_edge.line_number == 10
        assert new_edge.is_conditional is True

    def test_with_weight_zero(self) -> None:
        """with_weight(0.0) is valid."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        new_edge = edge.with_weight(0.0)
        assert new_edge.weight == 0.0

    def test_with_weight_invalid_rejected(self) -> None:
        """with_weight() rejects invalid weight values."""
        edge = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        with pytest.raises(ValidationError):
            edge.with_weight(1.5)


class TestSemanticEdgeDescribesSameRelationship:
    """Tests for describes_same_relationship() method."""

    def test_same_relationship_true(self) -> None:
        """Two edges with same source, target, type describe same relationship."""
        edge1 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            weight=0.5,
            line_number=10,
        )
        edge2 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
            weight=1.0,
            line_number=20,
        )
        assert edge1.describes_same_relationship(edge2)
        assert edge2.describes_same_relationship(edge1)

    def test_different_source_not_same(self) -> None:
        """Different source → not same relationship."""
        edge1 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        edge2 = SemanticEdge(
            source_id="c",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        assert not edge1.describes_same_relationship(edge2)

    def test_different_target_not_same(self) -> None:
        """Different target → not same relationship."""
        edge1 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        edge2 = SemanticEdge(
            source_id="a",
            target_id="c",
            edge_type=EdgeType.CALLS,
        )
        assert not edge1.describes_same_relationship(edge2)

    def test_different_type_not_same(self) -> None:
        """Different type → not same relationship."""
        edge1 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.CALLS,
        )
        edge2 = SemanticEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.IMPORTS,
        )
        assert not edge1.describes_same_relationship(edge2)


class TestCreateContainsEdge:
    """Tests for create_contains_edge factory."""

    def test_creates_contains_edge(self) -> None:
        """Factory creates a CONTAINS edge."""
        edge = create_contains_edge("module_id", "func_id")
        assert edge.source_id == "module_id"
        assert edge.target_id == "func_id"
        assert edge.edge_type == EdgeType.CONTAINS
        assert edge.weight == 1.0
        assert edge.is_conditional is False
        assert edge.line_number is None

    def test_self_loop_rejected(self) -> None:
        """Self-loop raises ValidationError."""
        with pytest.raises(ValidationError):
            create_contains_edge("same", "same")


class TestCreateCallsEdge:
    """Tests for create_calls_edge factory."""

    def test_creates_calls_edge(self) -> None:
        """Factory creates a CALLS edge."""
        edge = create_calls_edge("caller", "callee", 42)
        assert edge.source_id == "caller"
        assert edge.target_id == "callee"
        assert edge.edge_type == EdgeType.CALLS
        assert edge.line_number == 42
        assert edge.is_conditional is False

    def test_conditional_call(self) -> None:
        """Factory creates a conditional CALLS edge."""
        edge = create_calls_edge("caller", "callee", 10, is_conditional=True)
        assert edge.is_conditional is True

    def test_non_conditional_default(self) -> None:
        """is_conditional defaults to False."""
        edge = create_calls_edge("a", "b", 1)
        assert edge.is_conditional is False


class TestCreateImportsEdge:
    """Tests for create_imports_edge factory."""

    def test_creates_imports_edge(self) -> None:
        """Factory creates an IMPORTS edge."""
        edge = create_imports_edge("module_a", "module_b", 5)
        assert edge.source_id == "module_a"
        assert edge.target_id == "module_b"
        assert edge.edge_type == EdgeType.IMPORTS
        assert edge.line_number == 5
        assert edge.is_conditional is False


class TestCreateInheritsEdge:
    """Tests for create_inherits_edge factory."""

    def test_creates_inherits_edge(self) -> None:
        """Factory creates an INHERITS edge."""
        edge = create_inherits_edge("child", "parent")
        assert edge.source_id == "child"
        assert edge.target_id == "parent"
        assert edge.edge_type == EdgeType.INHERITS
        assert edge.line_number is None
        assert edge.weight == 1.0
