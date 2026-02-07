"""Semantic graph node definitions.

Nodes represent code entities: modules, classes, functions, etc.
Each node has a unique identifier, type, and metadata.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class NodeType(str, Enum):
    """Types of nodes in the semantic graph."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    PARAMETER = "parameter"
    DECORATOR = "decorator"
    ENDPOINT = "endpoint"


class SemanticNode(BaseModel):
    """Base class for all semantic graph nodes.

    Attributes:
        id: Unique identifier for the node.
        name: Human-readable name of the code entity.
        node_type: Type classification of the node.
        file_path: Path to the source file containing this entity.
        line_start: Starting line number in the source file.
        line_end: Ending line number in the source file.
        metadata: Additional type-specific metadata.
    """

    id: str = Field(default="")
    name: str = Field(min_length=1)
    node_type: NodeType
    file_path: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    def model_post_init(self, __context: Any) -> None:
        """Generate ID after initialization if not provided."""
        if not self.id:
            object.__setattr__(self, "id", self._generate_id())

    def _generate_id(self) -> str:
        """Generate a unique ID based on node properties."""
        unique_str = f"{self.node_type.value}:{self.file_path}:{self.name}:{self.line_start}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

    @field_validator("line_end")
    @classmethod
    def validate_line_range(cls, v: int, info: Any) -> int:
        """Ensure line_end >= line_start."""
        if "line_start" in info.data and v < info.data["line_start"]:
            raise ValueError("line_end must be >= line_start")
        return v

    def overlaps_with(self, other: SemanticNode) -> bool:
        """Check if this node's line range overlaps with another."""
        if self.file_path != other.file_path:
            return False
        return not (self.line_end < other.line_start or self.line_start > other.line_end)

    def contains(self, other: SemanticNode) -> bool:
        """Check if this node fully contains another node."""
        if self.file_path != other.file_path:
            return False
        return self.line_start <= other.line_start and self.line_end >= other.line_end

    def qualified_name(self) -> str:
        """Return fully qualified name including file path."""
        return f"{self.file_path}::{self.name}"


class ModuleNode(SemanticNode):
    """Represents a Python module (file).

    Attributes:
        docstring: Module-level docstring if present.
        is_package: Whether this module is a package (__init__.py).
        imports: List of import statements in the module.
    """

    node_type: NodeType = Field(default=NodeType.MODULE, frozen=True)
    docstring: str | None = None
    is_package: bool = False
    imports: list[str] = Field(default_factory=list)

    @field_validator("file_path")
    @classmethod
    def validate_python_file(cls, v: str) -> str:
        """Ensure file path ends with .py."""
        if not v.endswith(".py"):
            raise ValueError("Module file_path must end with .py")
        return v


class ClassNode(SemanticNode):
    """Represents a Python class definition.

    Attributes:
        docstring: Class docstring if present.
        bases: List of base class names.
        is_dataclass: Whether class is decorated with @dataclass.
        is_pydantic: Whether class inherits from pydantic BaseModel.
    """

    node_type: NodeType = Field(default=NodeType.CLASS, frozen=True)
    docstring: str | None = None
    bases: list[str] = Field(default_factory=list)
    is_dataclass: bool = False
    is_pydantic: bool = False


class FunctionNode(SemanticNode):
    """Represents a Python function or method.

    Attributes:
        docstring: Function docstring if present.
        parameters: List of parameter names.
        return_type: Return type annotation if present.
        is_async: Whether function is async.
        is_generator: Whether function is a generator.
        decorators: List of decorator names.
        complexity: Cyclomatic complexity score.
    """

    node_type: NodeType = Field(default=NodeType.FUNCTION, frozen=True)
    docstring: str | None = None
    parameters: list[str] = Field(default_factory=list)
    return_type: str | None = None
    is_async: bool = False
    is_generator: bool = False
    decorators: list[str] = Field(default_factory=list)
    complexity: int = Field(default=1, ge=1)

    def is_method(self) -> bool:
        """Check if this function is a method (has self/cls parameter)."""
        return len(self.parameters) > 0 and self.parameters[0] in ("self", "cls")

    def is_private(self) -> bool:
        """Check if function is private (starts with underscore)."""
        return self.name.startswith("_") and not self.name.startswith("__")

    def is_dunder(self) -> bool:
        """Check if function is a dunder method."""
        return self.name.startswith("__") and self.name.endswith("__")


class VariableNode(SemanticNode):
    """Represents a variable assignment.

    Attributes:
        type_annotation: Type annotation if present.
        is_constant: Whether variable is a constant (UPPER_CASE).
        scope: Scope of the variable (module, class, local).
    """

    node_type: NodeType = Field(default=NodeType.VARIABLE, frozen=True)
    type_annotation: str | None = None
    is_constant: bool = False
    scope: str = "local"

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: str) -> str:
        """Ensure scope is valid."""
        valid_scopes = {"module", "class", "local"}
        if v not in valid_scopes:
            raise ValueError(f"scope must be one of {valid_scopes}")
        return v


class ImportNode(SemanticNode):
    """Represents an import statement.

    Attributes:
        module: The module being imported.
        alias: Import alias if using 'as'.
        is_from_import: Whether this is a 'from x import y' style.
        imported_names: Names imported in 'from' style imports.
    """

    node_type: NodeType = Field(default=NodeType.IMPORT, frozen=True)
    module: str = Field(min_length=1)
    alias: str | None = None
    is_from_import: bool = False
    imported_names: list[str] = Field(default_factory=list)


class DecoratorNode(SemanticNode):
    """Represents a decorator usage.

    Attributes:
        decorator_name: Name of the decorator.
        arguments: Arguments passed to the decorator.
        target_node_id: ID of the decorated node.
    """

    node_type: NodeType = Field(default=NodeType.DECORATOR, frozen=True)
    decorator_name: str = Field(min_length=1)
    arguments: list[str] = Field(default_factory=list)
    target_node_id: str | None = None


class EndpointNode(SemanticNode):
    """Represents a FastAPI/Flask endpoint.

    Attributes:
        http_method: HTTP method (GET, POST, etc.).
        path: URL path pattern.
        response_model: Response model class name if specified.
        dependencies: List of dependency injection names.
    """

    node_type: NodeType = Field(default=NodeType.ENDPOINT, frozen=True)
    http_method: str = Field(min_length=1)
    path: str = Field(min_length=1)
    response_model: str | None = None
    dependencies: list[str] = Field(default_factory=list)

    @field_validator("http_method")
    @classmethod
    def validate_http_method(cls, v: str) -> str:
        """Ensure HTTP method is valid."""
        valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}
        upper_v = v.upper()
        if upper_v not in valid_methods:
            raise ValueError(f"http_method must be one of {valid_methods}")
        return upper_v
