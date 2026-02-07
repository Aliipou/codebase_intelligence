"""Brutal unit tests for pattern extraction.

Tests every code path, edge case, and validation rule for 100% coverage.
"""

from __future__ import annotations

import pytest

from codebase_intelligence.edges import EdgeType, SemanticEdge
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    EndpointNode,
    FunctionNode,
    ModuleNode,
    NodeType,
    SemanticNode,
)
from codebase_intelligence.patterns import (
    AsyncPatternRule,
    ClassNamingRule,
    DependencyRule,
    FastAPIPatternRule,
    FunctionNamingRule,
    ModuleStructureRule,
    Pattern,
    PatternConfidence,
    PatternExtractor,
    PatternRule,
    PatternType,
    PydanticPatternRule,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _mod(name: str, fp: str = "", imports: list[str] | None = None) -> ModuleNode:
    fp = fp or f"{name}.py"
    return ModuleNode(name=name, file_path=fp, line_start=1, line_end=100, imports=imports or [])


def _cls(name: str, fp: str = "app.py", bases: list[str] | None = None,
         is_pydantic: bool = False, is_dataclass: bool = False) -> ClassNode:
    return ClassNode(
        name=name, file_path=fp, line_start=10, line_end=50,
        bases=bases or [], is_pydantic=is_pydantic, is_dataclass=is_dataclass,
    )


def _fn(name: str, fp: str = "app.py", is_async: bool = False) -> FunctionNode:
    return FunctionNode(name=name, file_path=fp, line_start=10, line_end=20, is_async=is_async)


def _endpoint(name: str, method: str = "GET", path: str = "/",
              response_model: str | None = None) -> EndpointNode:
    return EndpointNode(
        name=name, file_path="api.py", line_start=10, line_end=20,
        http_method=method, path=path, response_model=response_model,
    )


def _edge(src: str, tgt: str, etype: EdgeType = EdgeType.CONTAINS) -> SemanticEdge:
    return SemanticEdge(source_id=src, target_id=tgt, edge_type=etype)


# ── PatternType & PatternConfidence ───────────────────────────────────────


class TestPatternType:
    def test_all_values(self) -> None:
        expected = {"structural", "naming", "dependency", "framework", "behavioral"}
        actual = {pt.value for pt in PatternType}
        assert actual == expected


class TestPatternConfidence:
    def test_all_values(self) -> None:
        expected = {"high", "medium", "low"}
        actual = {pc.value for pc in PatternConfidence}
        assert actual == expected


# ── Pattern Dataclass ─────────────────────────────────────────────────────


class TestPattern:
    def test_create_basic_pattern(self) -> None:
        p = Pattern(name="test", pattern_type=PatternType.NAMING, description="A test")
        assert p.name == "test"
        assert p.pattern_type == PatternType.NAMING
        assert p.description == "A test"
        assert p.regex is None
        assert p.examples == ()
        assert p.confidence == PatternConfidence.MEDIUM
        assert p.occurrences == 0
        assert p.metadata is None

    def test_pattern_with_all_fields(self) -> None:
        p = Pattern(
            name="svc",
            pattern_type=PatternType.NAMING,
            description="Services",
            regex=r"^[A-Z].*Service$",
            examples=("UserService", "AuthService"),
            confidence=PatternConfidence.HIGH,
            occurrences=5,
            metadata={"key": "val"},
        )
        assert p.examples == ("UserService", "AuthService")
        assert p.metadata == {"key": "val"}

    def test_matches_with_regex(self) -> None:
        p = Pattern(name="t", pattern_type=PatternType.NAMING, description="",
                    regex=r"^[A-Z][a-zA-Z]*Service$")
        assert p.matches("UserService") is True
        assert p.matches("userservice") is False
        assert p.matches("") is False

    def test_matches_no_regex(self) -> None:
        p = Pattern(name="t", pattern_type=PatternType.NAMING, description="")
        assert p.matches("anything") is False

    def test_matches_invalid_regex(self) -> None:
        p = Pattern(name="t", pattern_type=PatternType.NAMING, description="",
                    regex="[invalid")
        assert p.matches("test") is False

    def test_with_updated_confidence(self) -> None:
        p = Pattern(
            name="test", pattern_type=PatternType.NAMING, description="Desc",
            regex=r"^test$", examples=("x",), confidence=PatternConfidence.LOW,
            occurrences=3, metadata={"a": 1},
        )
        p2 = p.with_updated_confidence(PatternConfidence.HIGH)
        assert p2.confidence == PatternConfidence.HIGH
        assert p2.name == p.name
        assert p2.regex == p.regex
        assert p2.examples == p.examples
        assert p2.occurrences == p.occurrences
        assert p2.metadata == p.metadata

    def test_pattern_is_frozen(self) -> None:
        p = Pattern(name="t", pattern_type=PatternType.NAMING, description="")
        with pytest.raises(AttributeError):
            p.name = "changed"  # type: ignore[misc]


# ── ClassNamingRule ───────────────────────────────────────────────────────


class TestClassNamingRule:
    def test_name(self) -> None:
        assert ClassNamingRule().name == "class_naming"

    def test_no_classes(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app"))
        assert ClassNamingRule().extract(g) == []

    def test_detects_service_suffix(self) -> None:
        g = SemanticGraph()
        for name in ["UserService", "AuthService", "PaymentService"]:
            g.add_node(_cls(name))
        patterns = ClassNamingRule().extract(g)
        names = [p.name for p in patterns]
        assert "class_suffix_service" in names

    def test_single_occurrence_not_extracted(self) -> None:
        """Need >= 2 occurrences to create a pattern."""
        g = SemanticGraph()
        g.add_node(_cls("UserService"))
        patterns = ClassNamingRule().extract(g)
        assert len(patterns) == 0

    def test_high_confidence(self) -> None:
        """5+ occurrences → HIGH confidence."""
        g = SemanticGraph()
        for i in range(6):
            g.add_node(_cls(f"Item{i}Service", fp=f"svc{i}.py"))
        patterns = ClassNamingRule().extract(g)
        svc = [p for p in patterns if "service" in p.name][0]
        assert svc.confidence == PatternConfidence.HIGH

    def test_medium_confidence(self) -> None:
        """3-4 occurrences → MEDIUM confidence."""
        g = SemanticGraph()
        for i in range(3):
            g.add_node(_cls(f"Item{i}Service", fp=f"svc{i}.py"))
        patterns = ClassNamingRule().extract(g)
        svc = [p for p in patterns if "service" in p.name][0]
        assert svc.confidence == PatternConfidence.MEDIUM

    def test_low_confidence(self) -> None:
        """2 occurrences → LOW confidence."""
        g = SemanticGraph()
        g.add_node(_cls("UserService"))
        g.add_node(_cls("AuthService", fp="auth.py"))
        patterns = ClassNamingRule().extract(g)
        svc = [p for p in patterns if "service" in p.name][0]
        assert svc.confidence == PatternConfidence.LOW

    def test_examples_limited_to_5(self) -> None:
        g = SemanticGraph()
        for i in range(8):
            g.add_node(_cls(f"Item{i}Service", fp=f"svc{i}.py"))
        patterns = ClassNamingRule().extract(g)
        svc = [p for p in patterns if "service" in p.name][0]
        assert len(svc.examples) <= 5

    def test_suffix_same_as_class_name_ignored(self) -> None:
        """A class named exactly 'Service' is not counted."""
        g = SemanticGraph()
        g.add_node(_cls("Service"))
        g.add_node(_cls("UserService", fp="u.py"))
        patterns = ClassNamingRule().extract(g)
        # Only 1 occurrence of Service suffix → not extracted
        assert len(patterns) == 0

    def test_multiple_suffixes(self) -> None:
        g = SemanticGraph()
        for name in ["UserService", "AuthService", "UserController", "AuthController"]:
            g.add_node(_cls(name, fp=f"{name}.py"))
        patterns = ClassNamingRule().extract(g)
        names = {p.name for p in patterns}
        assert "class_suffix_service" in names
        assert "class_suffix_controller" in names

    def test_non_class_nodes_ignored(self) -> None:
        """ClassNamingRule should only look at CLASS nodes."""
        g = SemanticGraph()
        # Add a SemanticNode with CLASS type but NOT a ClassNode
        node = SemanticNode(
            name="FakeService",
            node_type=NodeType.CLASS,
            file_path="fake.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(node)
        patterns = ClassNamingRule().extract(g)
        assert len(patterns) == 0


# ── FunctionNamingRule ────────────────────────────────────────────────────


class TestFunctionNamingRule:
    def test_name(self) -> None:
        assert FunctionNamingRule().name == "function_naming"

    def test_no_functions(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app"))
        assert FunctionNamingRule().extract(g) == []

    def test_detects_get_prefix(self) -> None:
        g = SemanticGraph()
        for name in ["get_user", "get_items"]:
            g.add_node(_fn(name, fp=f"{name}.py"))
        patterns = FunctionNamingRule().extract(g)
        names = [p.name for p in patterns]
        assert "function_prefix_get" in names

    def test_private_functions_skipped(self) -> None:
        g = SemanticGraph()
        for name in ["_get_user", "_get_items"]:
            g.add_node(_fn(name, fp=f"{name}.py"))
        patterns = FunctionNamingRule().extract(g)
        assert len(patterns) == 0

    def test_high_confidence_5_plus(self) -> None:
        g = SemanticGraph()
        for i in range(6):
            g.add_node(_fn(f"get_item{i}", fp=f"f{i}.py"))
        patterns = FunctionNamingRule().extract(g)
        assert patterns[0].confidence == PatternConfidence.HIGH

    def test_medium_confidence(self) -> None:
        g = SemanticGraph()
        for i in range(3):
            g.add_node(_fn(f"get_item{i}", fp=f"f{i}.py"))
        patterns = FunctionNamingRule().extract(g)
        assert patterns[0].confidence == PatternConfidence.MEDIUM

    def test_low_confidence(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("get_a"))
        g.add_node(_fn("get_b", fp="b.py"))
        patterns = FunctionNamingRule().extract(g)
        assert patterns[0].confidence == PatternConfidence.LOW

    def test_multiple_prefixes(self) -> None:
        g = SemanticGraph()
        for name in ["get_a", "get_b", "create_a", "create_b"]:
            g.add_node(_fn(name, fp=f"{name}.py"))
        patterns = FunctionNamingRule().extract(g)
        names = {p.name for p in patterns}
        assert "function_prefix_get" in names
        assert "function_prefix_create" in names

    def test_first_matching_prefix_wins(self) -> None:
        """A function matching multiple prefixes only counts for the first."""
        g = SemanticGraph()
        g.add_node(_fn("is_valid"))
        g.add_node(_fn("is_active", fp="b.py"))
        patterns = FunctionNamingRule().extract(g)
        names = [p.name for p in patterns]
        assert "function_prefix_is" in names

    def test_non_function_nodes_ignored(self) -> None:
        g = SemanticGraph()
        node = SemanticNode(
            name="get_something",
            node_type=NodeType.FUNCTION,
            file_path="f.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(node)
        patterns = FunctionNamingRule().extract(g)
        assert len(patterns) == 0


# ── ModuleStructureRule ───────────────────────────────────────────────────


class TestModuleStructureRule:
    def test_name(self) -> None:
        assert ModuleStructureRule().name == "module_structure"

    def test_no_modules(self) -> None:
        g = SemanticGraph()
        assert ModuleStructureRule().extract(g) == []

    def test_detects_common_dirs(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("user", "models/user.py"))
        patterns = ModuleStructureRule().extract(g)
        names = [p.name for p in patterns]
        assert "module_dir_models" in names

    def test_high_confidence_3_plus(self) -> None:
        g = SemanticGraph()
        for i in range(3):
            g.add_node(_mod(f"m{i}", f"services/m{i}.py"))
        patterns = ModuleStructureRule().extract(g)
        p = [p for p in patterns if "services" in p.name][0]
        assert p.confidence == PatternConfidence.HIGH

    def test_medium_confidence(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("m1", "services/m1.py"))
        patterns = ModuleStructureRule().extract(g)
        p = [p for p in patterns if "services" in p.name][0]
        assert p.confidence == PatternConfidence.MEDIUM

    def test_backslash_paths(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("user", "models\\user.py"))
        patterns = ModuleStructureRule().extract(g)
        names = [p.name for p in patterns]
        assert "module_dir_models" in names

    def test_non_module_nodes_ignored(self) -> None:
        g = SemanticGraph()
        node = SemanticNode(
            name="models",
            node_type=NodeType.MODULE,
            file_path="models/x.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(node)
        patterns = ModuleStructureRule().extract(g)
        assert len(patterns) == 0


# ── DependencyRule ────────────────────────────────────────────────────────


class TestDependencyRule:
    def test_name(self) -> None:
        assert DependencyRule().name == "dependency"

    def test_no_modules(self) -> None:
        g = SemanticGraph()
        assert DependencyRule().extract(g) == []

    def test_detects_common_imports(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("a", imports=["fastapi", "pydantic"]))
        g.add_node(_mod("b", fp="b.py", imports=["fastapi", "sqlalchemy"]))
        patterns = DependencyRule().extract(g)
        names = [p.name for p in patterns]
        assert "dependency_fastapi" in names

    def test_single_import_not_extracted(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("a", imports=["rare_lib"]))
        patterns = DependencyRule().extract(g)
        assert len(patterns) == 0

    def test_high_confidence_5_plus(self) -> None:
        g = SemanticGraph()
        for i in range(6):
            g.add_node(_mod(f"m{i}", fp=f"m{i}.py", imports=["fastapi"]))
        patterns = DependencyRule().extract(g)
        p = [p for p in patterns if "fastapi" in p.name][0]
        assert p.confidence == PatternConfidence.HIGH

    def test_medium_confidence(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("a", imports=["typing"]))
        g.add_node(_mod("b", fp="b.py", imports=["typing"]))
        patterns = DependencyRule().extract(g)
        p = [p for p in patterns if "typing" in p.name][0]
        assert p.confidence == PatternConfidence.MEDIUM

    def test_dotted_import_uses_top_level(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("a", imports=["os.path"]))
        g.add_node(_mod("b", fp="b.py", imports=["os.environ"]))
        patterns = DependencyRule().extract(g)
        names = [p.name for p in patterns]
        assert "dependency_os" in names

    def test_metadata_has_package(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("a", imports=["fastapi"]))
        g.add_node(_mod("b", fp="b.py", imports=["fastapi"]))
        patterns = DependencyRule().extract(g)
        p = [p for p in patterns if "fastapi" in p.name][0]
        assert p.metadata == {"package": "fastapi"}

    def test_non_module_nodes_ignored(self) -> None:
        g = SemanticGraph()
        node = SemanticNode(
            name="fake_mod",
            node_type=NodeType.MODULE,
            file_path="f.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(node)
        patterns = DependencyRule().extract(g)
        assert len(patterns) == 0


# ── FastAPIPatternRule ────────────────────────────────────────────────────


class TestFastAPIPatternRule:
    def test_name(self) -> None:
        assert FastAPIPatternRule().name == "fastapi"

    def test_not_fastapi_project(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app", imports=["flask"]))
        assert FastAPIPatternRule().extract(g) == []

    def test_fastapi_no_endpoints(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app", imports=["fastapi"]))
        patterns = FastAPIPatternRule().extract(g)
        assert len(patterns) == 0

    def test_fastapi_endpoint_distribution(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app", imports=["fastapi"]))
        g.add_node(_endpoint("get_users", "GET", "/users"))
        g.add_node(_endpoint("create_user", "POST", "/users"))
        patterns = FastAPIPatternRule().extract(g)
        names = [p.name for p in patterns]
        assert "fastapi_get_endpoints" in names
        assert "fastapi_post_endpoints" in names

    def test_response_model_high_ratio(self) -> None:
        """>=80% response model usage → HIGH confidence."""
        g = SemanticGraph()
        g.add_node(_mod("app", imports=["fastapi"]))
        for i in range(5):
            g.add_node(_endpoint(f"ep{i}", "GET", f"/ep{i}", response_model="Model"))
        patterns = FastAPIPatternRule().extract(g)
        rm = [p for p in patterns if "response_model" in p.name]
        assert len(rm) == 1
        assert rm[0].confidence == PatternConfidence.HIGH

    def test_response_model_medium_ratio(self) -> None:
        """50-79% response model usage → MEDIUM confidence."""
        g = SemanticGraph()
        g.add_node(_mod("app", imports=["fastapi"]))
        g.add_node(_endpoint("a", "GET", "/a", response_model="M"))
        g.add_node(_endpoint("b", "GET", "/b"))
        patterns = FastAPIPatternRule().extract(g)
        rm = [p for p in patterns if "response_model" in p.name]
        assert len(rm) == 1
        assert rm[0].confidence == PatternConfidence.MEDIUM

    def test_response_model_low_ratio_not_extracted(self) -> None:
        """<50% response model → not extracted."""
        g = SemanticGraph()
        g.add_node(_mod("app", imports=["fastapi"]))
        g.add_node(_endpoint("a", "GET", "/a"))
        g.add_node(_endpoint("b", "GET", "/b"))
        g.add_node(_endpoint("c", "GET", "/c", response_model="M"))
        patterns = FastAPIPatternRule().extract(g)
        rm = [p for p in patterns if "response_model" in p.name]
        assert len(rm) == 0

    def test_non_endpoint_nodes_ignored(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app", imports=["fastapi"]))
        node = SemanticNode(
            name="fake",
            node_type=NodeType.ENDPOINT,
            file_path="f.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(node)
        patterns = FastAPIPatternRule().extract(g)
        assert len(patterns) == 0

    def test_non_module_nodes_skipped_for_fastapi_check(self) -> None:
        g = SemanticGraph()
        node = SemanticNode(
            name="fake_mod",
            node_type=NodeType.MODULE,
            file_path="f.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(node)
        patterns = FastAPIPatternRule().extract(g)
        assert len(patterns) == 0


# ── PydanticPatternRule ───────────────────────────────────────────────────


class TestPydanticPatternRule:
    def test_name(self) -> None:
        assert PydanticPatternRule().name == "pydantic"

    def test_no_classes(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app"))
        assert PydanticPatternRule().extract(g) == []

    def test_pydantic_models(self) -> None:
        g = SemanticGraph()
        g.add_node(_cls("User", is_pydantic=True))
        g.add_node(_cls("Item", fp="item.py", is_pydantic=True))
        patterns = PydanticPatternRule().extract(g)
        names = [p.name for p in patterns]
        assert "pydantic_models" in names

    def test_dataclasses(self) -> None:
        g = SemanticGraph()
        g.add_node(_cls("Config", is_dataclass=True))
        patterns = PydanticPatternRule().extract(g)
        names = [p.name for p in patterns]
        assert "dataclasses" in names

    def test_examples_limited_to_5(self) -> None:
        g = SemanticGraph()
        for i in range(8):
            g.add_node(_cls(f"Model{i}", fp=f"m{i}.py", is_pydantic=True))
        patterns = PydanticPatternRule().extract(g)
        p = [p for p in patterns if "pydantic" in p.name][0]
        assert len(p.examples) <= 5

    def test_non_class_nodes_ignored(self) -> None:
        g = SemanticGraph()
        node = SemanticNode(
            name="Fake",
            node_type=NodeType.CLASS,
            file_path="f.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(node)
        patterns = PydanticPatternRule().extract(g)
        assert len(patterns) == 0


# ── AsyncPatternRule ──────────────────────────────────────────────────────


class TestAsyncPatternRule:
    def test_name(self) -> None:
        assert AsyncPatternRule().name == "async"

    def test_no_functions(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app"))
        assert AsyncPatternRule().extract(g) == []

    def test_all_sync(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("a"))
        g.add_node(_fn("b", fp="b.py"))
        patterns = AsyncPatternRule().extract(g)
        assert len(patterns) == 0  # No async functions

    def test_async_high_ratio(self) -> None:
        """>=50% async → HIGH confidence."""
        g = SemanticGraph()
        g.add_node(_fn("a", is_async=True))
        g.add_node(_fn("b", fp="b.py", is_async=True))
        patterns = AsyncPatternRule().extract(g)
        assert len(patterns) == 1
        assert patterns[0].confidence == PatternConfidence.HIGH

    def test_async_medium_ratio(self) -> None:
        """<50% async → MEDIUM confidence."""
        g = SemanticGraph()
        g.add_node(_fn("a", is_async=True))
        g.add_node(_fn("b", fp="b.py"))
        g.add_node(_fn("c", fp="c.py"))
        patterns = AsyncPatternRule().extract(g)
        assert len(patterns) == 1
        assert patterns[0].confidence == PatternConfidence.MEDIUM

    def test_metadata_has_ratio(self) -> None:
        g = SemanticGraph()
        g.add_node(_fn("a", is_async=True))
        g.add_node(_fn("b", fp="b.py"))
        patterns = AsyncPatternRule().extract(g)
        assert "ratio" in patterns[0].metadata

    def test_examples_limited_to_5(self) -> None:
        g = SemanticGraph()
        for i in range(8):
            g.add_node(_fn(f"async_fn{i}", fp=f"f{i}.py", is_async=True))
        patterns = AsyncPatternRule().extract(g)
        assert len(patterns[0].examples) <= 5

    def test_non_function_nodes_ignored(self) -> None:
        g = SemanticGraph()
        node = SemanticNode(
            name="fake_fn",
            node_type=NodeType.FUNCTION,
            file_path="f.py",
            line_start=1,
            line_end=10,
        )
        g.add_node(node)
        patterns = AsyncPatternRule().extract(g)
        assert len(patterns) == 0


# ── PatternExtractor ──────────────────────────────────────────────────────


class TestPatternExtractor:
    def test_extract_all_empty_graph(self) -> None:
        g = SemanticGraph()
        extractor = PatternExtractor(g)
        patterns = extractor.extract_all()
        assert patterns == []

    def test_extract_all_populated(self) -> None:
        g = SemanticGraph()
        g.add_node(_mod("app", imports=["fastapi"]))
        g.add_node(_cls("UserService"))
        g.add_node(_cls("AuthService", fp="auth.py"))
        g.add_node(_fn("get_user"))
        g.add_node(_fn("get_item", fp="item.py"))
        extractor = PatternExtractor(g)
        patterns = extractor.extract_all()
        assert len(patterns) > 0

    def test_extract_by_type(self) -> None:
        g = SemanticGraph()
        g.add_node(_cls("UserService"))
        g.add_node(_cls("AuthService", fp="auth.py"))
        extractor = PatternExtractor(g)
        naming = extractor.extract_by_type(PatternType.NAMING)
        for p in naming:
            assert p.pattern_type == PatternType.NAMING

    def test_extract_high_confidence(self) -> None:
        g = SemanticGraph()
        for i in range(6):
            g.add_node(_cls(f"X{i}Service", fp=f"s{i}.py"))
        extractor = PatternExtractor(g)
        high = extractor.extract_high_confidence()
        for p in high:
            assert p.confidence == PatternConfidence.HIGH

    def test_custom_rules(self) -> None:
        """Custom rules override defaults."""

        class DummyRule(PatternRule):
            @property
            def name(self) -> str:
                return "dummy"

            def extract(self, graph: SemanticGraph) -> list[Pattern]:
                return [Pattern(name="dummy", pattern_type=PatternType.BEHAVIORAL, description="test")]

        g = SemanticGraph()
        extractor = PatternExtractor(g, rules=[DummyRule()])
        patterns = extractor.extract_all()
        assert len(patterns) == 1
        assert patterns[0].name == "dummy"

    def test_add_rule(self) -> None:
        class ExtraRule(PatternRule):
            @property
            def name(self) -> str:
                return "extra"

            def extract(self, graph: SemanticGraph) -> list[Pattern]:
                return [Pattern(name="extra", pattern_type=PatternType.BEHAVIORAL, description="extra")]

        g = SemanticGraph()
        extractor = PatternExtractor(g)
        original_count = len(extractor.get_rule_names())
        extractor.add_rule(ExtraRule())
        assert len(extractor.get_rule_names()) == original_count + 1

    def test_get_rule_names_defaults(self) -> None:
        g = SemanticGraph()
        extractor = PatternExtractor(g)
        names = extractor.get_rule_names()
        assert "class_naming" in names
        assert "function_naming" in names
        assert "module_structure" in names
        assert "dependency" in names
        assert "fastapi" in names
        assert "pydantic" in names
        assert "async" in names
        assert len(names) == 7


# ── Branch Partial Coverage ──────────────────────────────────────────────


class TestBranchPartials:
    """Target remaining branch partials for 100% branch coverage."""

    def test_function_no_common_prefix(self) -> None:
        """299->291: Function name doesn't match any common prefix."""
        g = SemanticGraph()
        # "compute" doesn't start with get_, set_, is_, has_, create_, delete_, update_, find_
        g.add_node(_fn("compute_data", fp="a.py"))
        g.add_node(_fn("analyze_data", fp="b.py"))
        patterns = FunctionNamingRule().extract(g)
        # No patterns since no common prefixes matched
        assert len(patterns) == 0

    def test_function_prefix_single_occurrence(self) -> None:
        """310->309: Prefix appears only once — skipped (count < 2)."""
        g = SemanticGraph()
        g.add_node(_fn("get_user", fp="a.py"))
        # Only one "get_" function — doesn't meet threshold of 2
        patterns = FunctionNamingRule().extract(g)
        assert all("get" not in p.name for p in patterns)

    def test_module_path_no_common_dir(self) -> None:
        """376->375: Directory part not in common_dirs set."""
        g = SemanticGraph()
        g.add_node(_mod("app", fp="random_dir/app.py"))
        patterns = ModuleStructureRule().extract(g)
        # "random_dir" not in common_dirs, so no pattern
        assert len(patterns) == 0
