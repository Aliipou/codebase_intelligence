"""Pattern extraction and representation.

Patterns capture recurring structural and behavioral conventions in a codebase.
They are extracted from the semantic graph and used to generate constraints.

Types of Patterns:
    - Structural: How code is organized (module layout, class hierarchies)
    - Naming: Naming conventions for different entity types
    - Dependency: Common import and call patterns
    - Framework: Framework-specific patterns (FastAPI routes, Pydantic models)

Usage:
    >>> extractor = PatternExtractor(graph)
    >>> patterns = extractor.extract_all()
    >>> for pattern in patterns:
    ...     print(f"{pattern.name}: {pattern.description}")
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from codebase_intelligence.edges import EdgeType
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    EndpointNode,
    FunctionNode,
    ModuleNode,
    NodeType,
)


class PatternType(str, Enum):
    """Classification of pattern types."""

    STRUCTURAL = "structural"
    NAMING = "naming"
    DEPENDENCY = "dependency"
    FRAMEWORK = "framework"
    BEHAVIORAL = "behavioral"


class PatternConfidence(str, Enum):
    """Confidence level of an extracted pattern.

    HIGH: Pattern appears in >80% of applicable cases
    MEDIUM: Pattern appears in 50-80% of applicable cases
    LOW: Pattern appears in 20-50% of applicable cases
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Pattern:
    """Represents an extracted code pattern.

    A pattern captures a recurring convention or structure observed
    in the codebase. Patterns are immutable once created.

    Attributes:
        name: Short identifier for the pattern.
        pattern_type: Classification of the pattern.
        description: Human-readable description of what the pattern represents.
        regex: Optional regex pattern for matching (used in naming patterns).
        examples: Concrete examples of this pattern from the codebase.
        confidence: How consistently this pattern appears.
        occurrences: Number of times this pattern was observed.
        metadata: Additional pattern-specific data.

    Examples:
        >>> pattern = Pattern(
        ...     name="service_class_naming",
        ...     pattern_type=PatternType.NAMING,
        ...     description="Service classes end with 'Service'",
        ...     regex=r"^[A-Z][a-zA-Z]*Service$",
        ...     examples=["UserService", "AuthService", "PaymentService"],
        ...     confidence=PatternConfidence.HIGH,
        ...     occurrences=12,
        ... )
    """

    name: str
    pattern_type: PatternType
    description: str
    regex: str | None = None
    examples: tuple[str, ...] = ()
    confidence: PatternConfidence = PatternConfidence.MEDIUM
    occurrences: int = 0
    metadata: dict[str, Any] | None = None

    def matches(self, text: str) -> bool:
        """Check if text matches this pattern's regex.

        Args:
            text: Text to match against the pattern.

        Returns:
            True if text matches, False otherwise.
            Always returns False if pattern has no regex.
        """
        if self.regex is None:
            return False
        try:
            return bool(re.match(self.regex, text))
        except re.error:
            return False

    def with_updated_confidence(
        self,
        new_confidence: PatternConfidence,
    ) -> "Pattern":
        """Create a copy with updated confidence level.

        Args:
            new_confidence: The new confidence level.

        Returns:
            A new Pattern instance with updated confidence.
        """
        return Pattern(
            name=self.name,
            pattern_type=self.pattern_type,
            description=self.description,
            regex=self.regex,
            examples=self.examples,
            confidence=new_confidence,
            occurrences=self.occurrences,
            metadata=self.metadata,
        )


class PatternRule(ABC):
    """Abstract base class for pattern extraction rules.

    Each rule knows how to extract a specific type of pattern
    from a semantic graph.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this rule."""
        ...

    @abstractmethod
    def extract(self, graph: SemanticGraph) -> list[Pattern]:
        """Extract patterns from the graph.

        Args:
            graph: The semantic graph to analyze.

        Returns:
            List of extracted patterns.
        """
        ...


class ClassNamingRule(PatternRule):
    """Extracts naming patterns for classes.

    Identifies common suffixes like Service, Controller, Repository,
    Handler, Manager, Factory, etc.
    """

    @property
    def name(self) -> str:
        return "class_naming"

    def extract(self, graph: SemanticGraph) -> list[Pattern]:
        """Extract class naming patterns."""
        patterns: list[Pattern] = []
        suffix_counter: Counter[str] = Counter()
        suffix_examples: dict[str, list[str]] = {}

        common_suffixes = [
            "Service",
            "Controller",
            "Repository",
            "Handler",
            "Manager",
            "Factory",
            "Builder",
            "Adapter",
            "Provider",
            "Validator",
            "Middleware",
            "Router",
            "Model",
            "Schema",
            "Exception",
            "Error",
            "Config",
            "Settings",
        ]

        for node in graph.get_nodes(NodeType.CLASS):
            if not isinstance(node, ClassNode):
                continue

            for suffix in common_suffixes:
                if node.name.endswith(suffix) and node.name != suffix:
                    suffix_counter[suffix] += 1
                    if suffix not in suffix_examples:
                        suffix_examples[suffix] = []
                    if len(suffix_examples[suffix]) < 5:
                        suffix_examples[suffix].append(node.name)

        # Create patterns for suffixes that appear multiple times
        for suffix, count in suffix_counter.items():
            if count >= 2:
                confidence = self._calculate_confidence(count)
                pattern = Pattern(
                    name=f"class_suffix_{suffix.lower()}",
                    pattern_type=PatternType.NAMING,
                    description=f"Classes ending with '{suffix}' follow a naming convention",
                    regex=rf"^[A-Z][a-zA-Z0-9]*{suffix}$",
                    examples=tuple(suffix_examples.get(suffix, [])),
                    confidence=confidence,
                    occurrences=count,
                )
                patterns.append(pattern)

        return patterns

    def _calculate_confidence(self, count: int) -> PatternConfidence:
        """Calculate confidence based on occurrence count."""
        if count >= 5:
            return PatternConfidence.HIGH
        elif count >= 3:
            return PatternConfidence.MEDIUM
        return PatternConfidence.LOW


class FunctionNamingRule(PatternRule):
    """Extracts naming patterns for functions.

    Identifies prefixes like get_, set_, is_, has_, create_, update_,
    delete_, validate_, process_, handle_, etc.
    """

    @property
    def name(self) -> str:
        return "function_naming"

    def extract(self, graph: SemanticGraph) -> list[Pattern]:
        """Extract function naming patterns."""
        patterns: list[Pattern] = []
        prefix_counter: Counter[str] = Counter()
        prefix_examples: dict[str, list[str]] = {}

        common_prefixes = [
            "get_",
            "set_",
            "is_",
            "has_",
            "can_",
            "should_",
            "create_",
            "update_",
            "delete_",
            "remove_",
            "add_",
            "validate_",
            "check_",
            "process_",
            "handle_",
            "parse_",
            "build_",
            "make_",
            "fetch_",
            "load_",
            "save_",
            "send_",
            "receive_",
            "calculate_",
            "compute_",
            "generate_",
            "convert_",
            "transform_",
        ]

        for node in graph.get_nodes(NodeType.FUNCTION):
            if not isinstance(node, FunctionNode):
                continue

            # Skip private/dunder methods
            if node.name.startswith("_"):
                continue

            for prefix in common_prefixes:
                if node.name.startswith(prefix):
                    prefix_counter[prefix] += 1
                    if prefix not in prefix_examples:
                        prefix_examples[prefix] = []
                    if len(prefix_examples[prefix]) < 5:
                        prefix_examples[prefix].append(node.name)
                    break

        # Create patterns for prefixes that appear multiple times
        for prefix, count in prefix_counter.items():
            if count >= 2:
                confidence = self._calculate_confidence(count)
                clean_prefix = prefix.rstrip("_")
                pattern = Pattern(
                    name=f"function_prefix_{clean_prefix}",
                    pattern_type=PatternType.NAMING,
                    description=f"Functions starting with '{prefix}' follow a naming convention",
                    regex=rf"^{prefix}[a-z][a-z0-9_]*$",
                    examples=tuple(prefix_examples.get(prefix, [])),
                    confidence=confidence,
                    occurrences=count,
                )
                patterns.append(pattern)

        return patterns

    def _calculate_confidence(self, count: int) -> PatternConfidence:
        """Calculate confidence based on occurrence count."""
        if count >= 5:
            return PatternConfidence.HIGH
        elif count >= 3:
            return PatternConfidence.MEDIUM
        return PatternConfidence.LOW


class ModuleStructureRule(PatternRule):
    """Extracts module organization patterns.

    Identifies common module naming and organization patterns like
    models/, services/, api/, etc.
    """

    @property
    def name(self) -> str:
        return "module_structure"

    def extract(self, graph: SemanticGraph) -> list[Pattern]:
        """Extract module structure patterns."""
        patterns: list[Pattern] = []
        dir_counter: Counter[str] = Counter()

        common_dirs = [
            "models",
            "schemas",
            "services",
            "controllers",
            "api",
            "routes",
            "handlers",
            "utils",
            "helpers",
            "core",
            "config",
            "tests",
            "middleware",
            "exceptions",
            "dependencies",
        ]

        for node in graph.get_nodes(NodeType.MODULE):
            if not isinstance(node, ModuleNode):
                continue

            # Extract directory name from path
            parts = node.file_path.replace("\\", "/").split("/")
            for part in parts[:-1]:  # Exclude filename
                if part in common_dirs:
                    dir_counter[part] += 1

        # Create patterns for directory structures
        for dir_name, count in dir_counter.items():
            pattern = Pattern(
                name=f"module_dir_{dir_name}",
                pattern_type=PatternType.STRUCTURAL,
                description=f"Modules organized in '{dir_name}/' directory",
                occurrences=count,
                confidence=PatternConfidence.HIGH if count >= 3 else PatternConfidence.MEDIUM,
            )
            patterns.append(pattern)

        return patterns


class DependencyRule(PatternRule):
    """Extracts dependency patterns.

    Identifies common import patterns and module dependencies.
    """

    @property
    def name(self) -> str:
        return "dependency"

    def extract(self, graph: SemanticGraph) -> list[Pattern]:
        """Extract dependency patterns."""
        patterns: list[Pattern] = []

        # Analyze import patterns
        import_counter: Counter[str] = Counter()

        for node in graph.get_nodes(NodeType.MODULE):
            if not isinstance(node, ModuleNode):
                continue

            for imp in node.imports:
                # Get top-level package
                top_level = imp.split(".")[0]
                import_counter[top_level] += 1

        # Create patterns for common dependencies
        for package, count in import_counter.most_common(10):
            if count >= 2:
                pattern = Pattern(
                    name=f"dependency_{package}",
                    pattern_type=PatternType.DEPENDENCY,
                    description=f"Common dependency on '{package}'",
                    occurrences=count,
                    confidence=PatternConfidence.HIGH if count >= 5 else PatternConfidence.MEDIUM,
                    metadata={"package": package},
                )
                patterns.append(pattern)

        return patterns


class FastAPIPatternRule(PatternRule):
    """Extracts FastAPI-specific patterns.

    Identifies endpoint patterns, dependency injection usage,
    response models, etc.
    """

    @property
    def name(self) -> str:
        return "fastapi"

    def extract(self, graph: SemanticGraph) -> list[Pattern]:
        """Extract FastAPI patterns."""
        patterns: list[Pattern] = []

        # Check if this is a FastAPI project
        is_fastapi = False
        for node in graph.get_nodes(NodeType.MODULE):
            if isinstance(node, ModuleNode) and "fastapi" in node.imports:
                is_fastapi = True
                break

        if not is_fastapi:
            return patterns

        # Analyze endpoints
        endpoint_methods: Counter[str] = Counter()
        endpoints_with_response_model = 0
        total_endpoints = 0

        for node in graph.get_nodes(NodeType.ENDPOINT):
            if isinstance(node, EndpointNode):
                total_endpoints += 1
                endpoint_methods[node.http_method] += 1
                if node.response_model:
                    endpoints_with_response_model += 1

        if total_endpoints > 0:
            # Pattern: HTTP method distribution
            for method, count in endpoint_methods.items():
                pattern = Pattern(
                    name=f"fastapi_{method.lower()}_endpoints",
                    pattern_type=PatternType.FRAMEWORK,
                    description=f"FastAPI {method} endpoints",
                    occurrences=count,
                    confidence=PatternConfidence.HIGH,
                )
                patterns.append(pattern)

            # Pattern: Response model usage
            response_model_ratio = endpoints_with_response_model / total_endpoints
            if response_model_ratio >= 0.5:
                pattern = Pattern(
                    name="fastapi_response_models",
                    pattern_type=PatternType.FRAMEWORK,
                    description="Endpoints use response_model for type safety",
                    occurrences=endpoints_with_response_model,
                    confidence=(
                        PatternConfidence.HIGH
                        if response_model_ratio >= 0.8
                        else PatternConfidence.MEDIUM
                    ),
                    metadata={"ratio": response_model_ratio},
                )
                patterns.append(pattern)

        return patterns


class PydanticPatternRule(PatternRule):
    """Extracts Pydantic usage patterns.

    Identifies model patterns, validation usage, etc.
    """

    @property
    def name(self) -> str:
        return "pydantic"

    def extract(self, graph: SemanticGraph) -> list[Pattern]:
        """Extract Pydantic patterns."""
        patterns: list[Pattern] = []

        pydantic_models: list[str] = []
        dataclasses: list[str] = []

        for node in graph.get_nodes(NodeType.CLASS):
            if isinstance(node, ClassNode):
                if node.is_pydantic:
                    pydantic_models.append(node.name)
                if node.is_dataclass:
                    dataclasses.append(node.name)

        if pydantic_models:
            pattern = Pattern(
                name="pydantic_models",
                pattern_type=PatternType.FRAMEWORK,
                description="Uses Pydantic BaseModel for data validation",
                examples=tuple(pydantic_models[:5]),
                occurrences=len(pydantic_models),
                confidence=PatternConfidence.HIGH,
            )
            patterns.append(pattern)

        if dataclasses:
            pattern = Pattern(
                name="dataclasses",
                pattern_type=PatternType.FRAMEWORK,
                description="Uses @dataclass decorator",
                examples=tuple(dataclasses[:5]),
                occurrences=len(dataclasses),
                confidence=PatternConfidence.HIGH,
            )
            patterns.append(pattern)

        return patterns


class AsyncPatternRule(PatternRule):
    """Extracts async/await usage patterns."""

    @property
    def name(self) -> str:
        return "async"

    def extract(self, graph: SemanticGraph) -> list[Pattern]:
        """Extract async patterns."""
        patterns: list[Pattern] = []

        async_functions: list[str] = []
        sync_functions: list[str] = []

        for node in graph.get_nodes(NodeType.FUNCTION):
            if isinstance(node, FunctionNode):
                if node.is_async:
                    async_functions.append(node.name)
                else:
                    sync_functions.append(node.name)

        total = len(async_functions) + len(sync_functions)
        if total > 0 and async_functions:
            async_ratio = len(async_functions) / total
            pattern = Pattern(
                name="async_codebase",
                pattern_type=PatternType.BEHAVIORAL,
                description=f"Async functions ({len(async_functions)} of {total})",
                examples=tuple(async_functions[:5]),
                occurrences=len(async_functions),
                confidence=(
                    PatternConfidence.HIGH if async_ratio >= 0.5 else PatternConfidence.MEDIUM
                ),
                metadata={"ratio": async_ratio},
            )
            patterns.append(pattern)

        return patterns


class PatternExtractor:
    """Extracts patterns from a semantic graph.

    The extractor runs a set of pattern rules against the graph
    and collects all identified patterns.

    Attributes:
        _graph: The semantic graph to analyze.
        _rules: List of pattern extraction rules.

    Examples:
        >>> graph = parser.parse_directory("myproject/")
        >>> extractor = PatternExtractor(graph)
        >>> patterns = extractor.extract_all()
        >>> naming_patterns = extractor.extract_by_type(PatternType.NAMING)
    """

    DEFAULT_RULES: list[type[PatternRule]] = [
        ClassNamingRule,
        FunctionNamingRule,
        ModuleStructureRule,
        DependencyRule,
        FastAPIPatternRule,
        PydanticPatternRule,
        AsyncPatternRule,
    ]

    def __init__(
        self,
        graph: SemanticGraph,
        rules: Sequence[PatternRule] | None = None,
    ) -> None:
        """Initialize the pattern extractor.

        Args:
            graph: The semantic graph to analyze.
            rules: Custom pattern rules to use. If None, uses default rules.
        """
        self._graph = graph
        if rules is not None:
            self._rules = list(rules)
        else:
            self._rules = [rule_class() for rule_class in self.DEFAULT_RULES]

    def add_rule(self, rule: PatternRule) -> None:
        """Add a custom pattern rule.

        Args:
            rule: The pattern rule to add.
        """
        self._rules.append(rule)

    def extract_all(self) -> list[Pattern]:
        """Extract all patterns using all registered rules.

        Returns:
            List of all extracted patterns.
        """
        patterns: list[Pattern] = []
        for rule in self._rules:
            patterns.extend(rule.extract(self._graph))
        return patterns

    def extract_by_type(self, pattern_type: PatternType) -> list[Pattern]:
        """Extract patterns of a specific type.

        Args:
            pattern_type: The type of patterns to extract.

        Returns:
            List of patterns matching the specified type.
        """
        all_patterns = self.extract_all()
        return [p for p in all_patterns if p.pattern_type == pattern_type]

    def extract_high_confidence(self) -> list[Pattern]:
        """Extract only high-confidence patterns.

        Returns:
            List of patterns with HIGH confidence level.
        """
        all_patterns = self.extract_all()
        return [p for p in all_patterns if p.confidence == PatternConfidence.HIGH]

    def get_rule_names(self) -> list[str]:
        """Get names of all registered rules.

        Returns:
            List of rule names.
        """
        return [rule.name for rule in self._rules]
