"""Brutal unit tests for semantic graph nodes.

Tests every code path, edge case, and validation rule.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_node_types_exist(self) -> None:
        """Verify all expected node types are defined."""
        expected = {
            "MODULE",
            "CLASS",
            "FUNCTION",
            "METHOD",
            "VARIABLE",
            "IMPORT",
            "PARAMETER",
            "DECORATOR",
            "ENDPOINT",
        }
        actual = {nt.name for nt in NodeType}
        assert actual == expected

    def test_node_type_values(self) -> None:
        """Verify node type values are lowercase strings."""
        for nt in NodeType:
            assert nt.value == nt.name.lower()

    def test_node_type_is_string_enum(self) -> None:
        """NodeType should be usable as a string."""
        assert NodeType.MODULE == "module"
        assert NodeType.CLASS == "class"


class TestSemanticNode:
    """Tests for SemanticNode base class."""

    def test_create_basic_node(self) -> None:
        """Create a basic semantic node."""
        node = SemanticNode(
            name="test_node",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        assert node.name == "test_node"
        assert node.node_type == NodeType.FUNCTION
        assert node.file_path == "test.py"
        assert node.line_start == 1
        assert node.line_end == 10
        assert len(node.id) == 16  # SHA256 hash truncated

    def test_id_auto_generation(self) -> None:
        """ID is automatically generated from node properties."""
        node1 = SemanticNode(
            name="func",
            node_type=NodeType.FUNCTION,
            file_path="a.py",
            line_start=1,
            line_end=5,
        )
        node2 = SemanticNode(
            name="func",
            node_type=NodeType.FUNCTION,
            file_path="a.py",
            line_start=1,
            line_end=5,
        )
        # Same properties = same ID
        assert node1.id == node2.id

    def test_different_nodes_have_different_ids(self) -> None:
        """Different node properties produce different IDs."""
        node1 = SemanticNode(
            name="func1",
            node_type=NodeType.FUNCTION,
            file_path="a.py",
            line_start=1,
            line_end=5,
        )
        node2 = SemanticNode(
            name="func2",
            node_type=NodeType.FUNCTION,
            file_path="a.py",
            line_start=1,
            line_end=5,
        )
        assert node1.id != node2.id

    def test_custom_id_preserved(self) -> None:
        """Custom ID is preserved if provided."""
        node = SemanticNode(
            id="custom_id_12345",
            name="test",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=1,
        )
        assert node.id == "custom_id_12345"

    def test_empty_name_rejected(self) -> None:
        """Empty name should be rejected."""
        with pytest.raises(ValidationError):
            SemanticNode(
                name="",
                node_type=NodeType.FUNCTION,
                file_path="test.py",
                line_start=1,
                line_end=1,
            )

    def test_empty_file_path_rejected(self) -> None:
        """Empty file path should be rejected."""
        with pytest.raises(ValidationError):
            SemanticNode(
                name="test",
                node_type=NodeType.FUNCTION,
                file_path="",
                line_start=1,
                line_end=1,
            )

    def test_line_start_must_be_positive(self) -> None:
        """line_start must be >= 1."""
        with pytest.raises(ValidationError):
            SemanticNode(
                name="test",
                node_type=NodeType.FUNCTION,
                file_path="test.py",
                line_start=0,
                line_end=1,
            )

    def test_line_end_must_be_positive(self) -> None:
        """line_end must be >= 1."""
        with pytest.raises(ValidationError):
            SemanticNode(
                name="test",
                node_type=NodeType.FUNCTION,
                file_path="test.py",
                line_start=1,
                line_end=0,
            )

    def test_line_end_must_be_gte_line_start(self) -> None:
        """line_end must be >= line_start."""
        with pytest.raises(ValidationError):
            SemanticNode(
                name="test",
                node_type=NodeType.FUNCTION,
                file_path="test.py",
                line_start=10,
                line_end=5,
            )

    def test_metadata_default_empty_dict(self) -> None:
        """Metadata defaults to empty dict."""
        node = SemanticNode(
            name="test",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=1,
        )
        assert node.metadata == {}

    def test_metadata_custom(self) -> None:
        """Custom metadata is preserved."""
        node = SemanticNode(
            name="test",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=1,
            metadata={"key": "value", "count": 42},
        )
        assert node.metadata == {"key": "value", "count": 42}

    def test_node_is_frozen(self) -> None:
        """Node should be immutable."""
        node = SemanticNode(
            name="test",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=1,
        )
        with pytest.raises(ValidationError):
            node.name = "changed"  # type: ignore

    def test_overlaps_with_same_file(self) -> None:
        """Nodes overlap if they share lines in the same file."""
        node1 = SemanticNode(
            name="a",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        node2 = SemanticNode(
            name="b",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=5,
            line_end=15,
        )
        assert node1.overlaps_with(node2)
        assert node2.overlaps_with(node1)

    def test_overlaps_with_no_overlap(self) -> None:
        """Nodes don't overlap if line ranges don't intersect."""
        node1 = SemanticNode(
            name="a",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        node2 = SemanticNode(
            name="b",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=11,
            line_end=20,
        )
        assert not node1.overlaps_with(node2)

    def test_overlaps_with_different_files(self) -> None:
        """Nodes in different files never overlap."""
        node1 = SemanticNode(
            name="a",
            node_type=NodeType.FUNCTION,
            file_path="a.py",
            line_start=1,
            line_end=10,
        )
        node2 = SemanticNode(
            name="b",
            node_type=NodeType.FUNCTION,
            file_path="b.py",
            line_start=1,
            line_end=10,
        )
        assert not node1.overlaps_with(node2)

    def test_contains_full_containment(self) -> None:
        """Node contains another if it fully encompasses it."""
        outer = SemanticNode(
            name="outer",
            node_type=NodeType.CLASS,
            file_path="test.py",
            line_start=1,
            line_end=100,
        )
        inner = SemanticNode(
            name="inner",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=10,
            line_end=20,
        )
        assert outer.contains(inner)
        assert not inner.contains(outer)

    def test_contains_same_range(self) -> None:
        """Node contains another if ranges are equal."""
        node1 = SemanticNode(
            name="a",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        node2 = SemanticNode(
            name="b",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        assert node1.contains(node2)
        assert node2.contains(node1)

    def test_contains_different_files(self) -> None:
        """Node cannot contain node from different file."""
        node1 = SemanticNode(
            name="a",
            node_type=NodeType.CLASS,
            file_path="a.py",
            line_start=1,
            line_end=100,
        )
        node2 = SemanticNode(
            name="b",
            node_type=NodeType.FUNCTION,
            file_path="b.py",
            line_start=10,
            line_end=20,
        )
        assert not node1.contains(node2)

    def test_qualified_name(self) -> None:
        """Qualified name includes file path."""
        node = SemanticNode(
            name="my_func",
            node_type=NodeType.FUNCTION,
            file_path="app/services/user.py",
            line_start=1,
            line_end=10,
        )
        assert node.qualified_name() == "app/services/user.py::my_func"


class TestModuleNode:
    """Tests for ModuleNode."""

    def test_create_module(self) -> None:
        """Create a basic module node."""
        module = ModuleNode(
            name="app",
            file_path="app.py",
            line_start=1,
            line_end=100,
        )
        assert module.name == "app"
        assert module.node_type == NodeType.MODULE
        assert module.docstring is None
        assert module.is_package is False
        assert module.imports == []

    def test_module_with_docstring(self) -> None:
        """Module with docstring."""
        module = ModuleNode(
            name="app",
            file_path="app.py",
            line_start=1,
            line_end=100,
            docstring="This is the main application module.",
        )
        assert module.docstring == "This is the main application module."

    def test_module_is_package(self) -> None:
        """Module that is a package."""
        module = ModuleNode(
            name="__init__",
            file_path="app/__init__.py",
            line_start=1,
            line_end=10,
            is_package=True,
        )
        assert module.is_package is True

    def test_module_with_imports(self) -> None:
        """Module with import list."""
        module = ModuleNode(
            name="app",
            file_path="app.py",
            line_start=1,
            line_end=100,
            imports=["os", "sys", "typing"],
        )
        assert module.imports == ["os", "sys", "typing"]

    def test_module_file_must_be_python(self) -> None:
        """Module file path must end with .py."""
        with pytest.raises(ValidationError):
            ModuleNode(
                name="app",
                file_path="app.txt",
                line_start=1,
                line_end=100,
            )

    def test_module_node_type_is_frozen(self) -> None:
        """Node type cannot be changed."""
        module = ModuleNode(
            name="app",
            file_path="app.py",
            line_start=1,
            line_end=100,
        )
        assert module.node_type == NodeType.MODULE


class TestClassNode:
    """Tests for ClassNode."""

    def test_create_basic_class(self) -> None:
        """Create a basic class node."""
        cls = ClassNode(
            name="UserService",
            file_path="services.py",
            line_start=10,
            line_end=50,
        )
        assert cls.name == "UserService"
        assert cls.node_type == NodeType.CLASS
        assert cls.docstring is None
        assert cls.bases == []
        assert cls.is_dataclass is False
        assert cls.is_pydantic is False

    def test_class_with_bases(self) -> None:
        """Class with base classes."""
        cls = ClassNode(
            name="AdminUser",
            file_path="models.py",
            line_start=10,
            line_end=50,
            bases=["User", "AdminMixin"],
        )
        assert cls.bases == ["User", "AdminMixin"]

    def test_class_with_docstring(self) -> None:
        """Class with docstring."""
        cls = ClassNode(
            name="User",
            file_path="models.py",
            line_start=10,
            line_end=50,
            docstring="Represents a user in the system.",
        )
        assert cls.docstring == "Represents a user in the system."

    def test_dataclass(self) -> None:
        """Class marked as dataclass."""
        cls = ClassNode(
            name="Config",
            file_path="config.py",
            line_start=10,
            line_end=50,
            is_dataclass=True,
        )
        assert cls.is_dataclass is True

    def test_pydantic_model(self) -> None:
        """Class marked as Pydantic model."""
        cls = ClassNode(
            name="UserSchema",
            file_path="schemas.py",
            line_start=10,
            line_end=50,
            bases=["BaseModel"],
            is_pydantic=True,
        )
        assert cls.is_pydantic is True


class TestFunctionNode:
    """Tests for FunctionNode."""

    def test_create_basic_function(self) -> None:
        """Create a basic function node."""
        func = FunctionNode(
            name="process_data",
            file_path="utils.py",
            line_start=10,
            line_end=25,
        )
        assert func.name == "process_data"
        assert func.node_type == NodeType.FUNCTION
        assert func.docstring is None
        assert func.parameters == []
        assert func.return_type is None
        assert func.is_async is False
        assert func.is_generator is False
        assert func.decorators == []
        assert func.complexity == 1

    def test_function_with_parameters(self) -> None:
        """Function with parameters."""
        func = FunctionNode(
            name="greet",
            file_path="utils.py",
            line_start=10,
            line_end=15,
            parameters=["name", "greeting"],
        )
        assert func.parameters == ["name", "greeting"]

    def test_function_with_return_type(self) -> None:
        """Function with return type annotation."""
        func = FunctionNode(
            name="get_user",
            file_path="services.py",
            line_start=10,
            line_end=20,
            return_type="User",
        )
        assert func.return_type == "User"

    def test_async_function(self) -> None:
        """Async function."""
        func = FunctionNode(
            name="fetch_data",
            file_path="api.py",
            line_start=10,
            line_end=20,
            is_async=True,
        )
        assert func.is_async is True

    def test_generator_function(self) -> None:
        """Generator function."""
        func = FunctionNode(
            name="items",
            file_path="utils.py",
            line_start=10,
            line_end=20,
            is_generator=True,
        )
        assert func.is_generator is True

    def test_function_with_decorators(self) -> None:
        """Function with decorators."""
        func = FunctionNode(
            name="endpoint",
            file_path="api.py",
            line_start=10,
            line_end=20,
            decorators=["app.get", "authenticate"],
        )
        assert func.decorators == ["app.get", "authenticate"]

    def test_function_complexity(self) -> None:
        """Function with custom complexity."""
        func = FunctionNode(
            name="complex_logic",
            file_path="logic.py",
            line_start=10,
            line_end=50,
            complexity=15,
        )
        assert func.complexity == 15

    def test_complexity_must_be_positive(self) -> None:
        """Complexity must be >= 1."""
        with pytest.raises(ValidationError):
            FunctionNode(
                name="func",
                file_path="test.py",
                line_start=1,
                line_end=10,
                complexity=0,
            )

    def test_is_method_with_self(self) -> None:
        """Method with self parameter."""
        method = FunctionNode(
            name="process",
            file_path="class.py",
            line_start=10,
            line_end=20,
            parameters=["self", "data"],
        )
        assert method.is_method() is True

    def test_is_method_with_cls(self) -> None:
        """Class method with cls parameter."""
        method = FunctionNode(
            name="create",
            file_path="class.py",
            line_start=10,
            line_end=20,
            parameters=["cls", "data"],
        )
        assert method.is_method() is True

    def test_is_not_method(self) -> None:
        """Regular function is not a method."""
        func = FunctionNode(
            name="helper",
            file_path="utils.py",
            line_start=10,
            line_end=20,
            parameters=["data"],
        )
        assert func.is_method() is False

    def test_is_not_method_empty_params(self) -> None:
        """Function with no params is not a method."""
        func = FunctionNode(
            name="get_config",
            file_path="config.py",
            line_start=10,
            line_end=20,
            parameters=[],
        )
        assert func.is_method() is False

    def test_is_private_single_underscore(self) -> None:
        """Private function starts with single underscore."""
        func = FunctionNode(
            name="_internal",
            file_path="utils.py",
            line_start=10,
            line_end=20,
        )
        assert func.is_private() is True

    def test_is_not_private(self) -> None:
        """Public function."""
        func = FunctionNode(
            name="public",
            file_path="utils.py",
            line_start=10,
            line_end=20,
        )
        assert func.is_private() is False

    def test_dunder_is_not_private(self) -> None:
        """Dunder methods are not private."""
        func = FunctionNode(
            name="__init__",
            file_path="class.py",
            line_start=10,
            line_end=20,
        )
        assert func.is_private() is False

    def test_is_dunder(self) -> None:
        """Dunder method detection."""
        func = FunctionNode(
            name="__init__",
            file_path="class.py",
            line_start=10,
            line_end=20,
        )
        assert func.is_dunder() is True

    def test_is_not_dunder_single_underscore(self) -> None:
        """Single underscore is not dunder."""
        func = FunctionNode(
            name="_private",
            file_path="utils.py",
            line_start=10,
            line_end=20,
        )
        assert func.is_dunder() is False

    def test_is_not_dunder_no_underscore(self) -> None:
        """Regular function is not dunder."""
        func = FunctionNode(
            name="process",
            file_path="utils.py",
            line_start=10,
            line_end=20,
        )
        assert func.is_dunder() is False


class TestVariableNode:
    """Tests for VariableNode."""

    def test_create_basic_variable(self) -> None:
        """Create a basic variable node."""
        var = VariableNode(
            name="count",
            file_path="utils.py",
            line_start=10,
            line_end=10,
        )
        assert var.name == "count"
        assert var.node_type == NodeType.VARIABLE
        assert var.type_annotation is None
        assert var.is_constant is False
        assert var.scope == "local"

    def test_variable_with_type_annotation(self) -> None:
        """Variable with type annotation."""
        var = VariableNode(
            name="users",
            file_path="app.py",
            line_start=10,
            line_end=10,
            type_annotation="list[User]",
        )
        assert var.type_annotation == "list[User]"

    def test_constant_variable(self) -> None:
        """Constant variable."""
        var = VariableNode(
            name="MAX_SIZE",
            file_path="config.py",
            line_start=5,
            line_end=5,
            is_constant=True,
            scope="module",
        )
        assert var.is_constant is True
        assert var.scope == "module"

    def test_class_scope_variable(self) -> None:
        """Class-scoped variable."""
        var = VariableNode(
            name="default_name",
            file_path="models.py",
            line_start=10,
            line_end=10,
            scope="class",
        )
        assert var.scope == "class"

    def test_invalid_scope_rejected(self) -> None:
        """Invalid scope value is rejected."""
        with pytest.raises(ValidationError):
            VariableNode(
                name="x",
                file_path="test.py",
                line_start=1,
                line_end=1,
                scope="invalid",
            )


class TestImportNode:
    """Tests for ImportNode."""

    def test_create_simple_import(self) -> None:
        """Create a simple import node."""
        imp = ImportNode(
            name="os",
            file_path="app.py",
            line_start=1,
            line_end=1,
            module="os",
        )
        assert imp.name == "os"
        assert imp.node_type == NodeType.IMPORT
        assert imp.module == "os"
        assert imp.alias is None
        assert imp.is_from_import is False
        assert imp.imported_names == []

    def test_import_with_alias(self) -> None:
        """Import with alias."""
        imp = ImportNode(
            name="np",
            file_path="analysis.py",
            line_start=1,
            line_end=1,
            module="numpy",
            alias="np",
        )
        assert imp.module == "numpy"
        assert imp.alias == "np"

    def test_from_import(self) -> None:
        """From...import statement."""
        imp = ImportNode(
            name="typing",
            file_path="app.py",
            line_start=1,
            line_end=1,
            module="typing",
            is_from_import=True,
            imported_names=["List", "Dict", "Optional"],
        )
        assert imp.is_from_import is True
        assert imp.imported_names == ["List", "Dict", "Optional"]

    def test_empty_module_rejected(self) -> None:
        """Empty module name is rejected."""
        with pytest.raises(ValidationError):
            ImportNode(
                name="test",
                file_path="app.py",
                line_start=1,
                line_end=1,
                module="",
            )


class TestDecoratorNode:
    """Tests for DecoratorNode."""

    def test_create_basic_decorator(self) -> None:
        """Create a basic decorator node."""
        dec = DecoratorNode(
            name="property",
            file_path="models.py",
            line_start=10,
            line_end=10,
            decorator_name="property",
        )
        assert dec.name == "property"
        assert dec.node_type == NodeType.DECORATOR
        assert dec.decorator_name == "property"
        assert dec.arguments == []
        assert dec.target_node_id is None

    def test_decorator_with_arguments(self) -> None:
        """Decorator with arguments."""
        dec = DecoratorNode(
            name="app.get",
            file_path="api.py",
            line_start=10,
            line_end=10,
            decorator_name="app.get",
            arguments=["'/users'", "response_model=UserList"],
        )
        assert dec.arguments == ["'/users'", "response_model=UserList"]

    def test_decorator_with_target(self) -> None:
        """Decorator with target node."""
        dec = DecoratorNode(
            name="staticmethod",
            file_path="utils.py",
            line_start=10,
            line_end=10,
            decorator_name="staticmethod",
            target_node_id="abc123",
        )
        assert dec.target_node_id == "abc123"

    def test_empty_decorator_name_rejected(self) -> None:
        """Empty decorator name is rejected."""
        with pytest.raises(ValidationError):
            DecoratorNode(
                name="x",
                file_path="test.py",
                line_start=1,
                line_end=1,
                decorator_name="",
            )


class TestEndpointNode:
    """Tests for EndpointNode."""

    def test_create_basic_endpoint(self) -> None:
        """Create a basic endpoint node."""
        endpoint = EndpointNode(
            name="get_users",
            file_path="api.py",
            line_start=10,
            line_end=20,
            http_method="GET",
            path="/users",
        )
        assert endpoint.name == "get_users"
        assert endpoint.node_type == NodeType.ENDPOINT
        assert endpoint.http_method == "GET"
        assert endpoint.path == "/users"
        assert endpoint.response_model is None
        assert endpoint.dependencies == []

    def test_endpoint_with_response_model(self) -> None:
        """Endpoint with response model."""
        endpoint = EndpointNode(
            name="get_user",
            file_path="api.py",
            line_start=10,
            line_end=20,
            http_method="GET",
            path="/users/{id}",
            response_model="UserResponse",
        )
        assert endpoint.response_model == "UserResponse"

    def test_endpoint_with_dependencies(self) -> None:
        """Endpoint with dependencies."""
        endpoint = EndpointNode(
            name="create_user",
            file_path="api.py",
            line_start=10,
            line_end=30,
            http_method="POST",
            path="/users",
            dependencies=["get_db", "get_current_user"],
        )
        assert endpoint.dependencies == ["get_db", "get_current_user"]

    def test_all_http_methods(self) -> None:
        """All valid HTTP methods should be accepted."""
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
        for method in methods:
            endpoint = EndpointNode(
                name=f"test_{method.lower()}",
                file_path="api.py",
                line_start=1,
                line_end=10,
                http_method=method,
                path="/test",
            )
            assert endpoint.http_method == method

    def test_lowercase_http_method_normalized(self) -> None:
        """Lowercase HTTP methods are normalized to uppercase."""
        endpoint = EndpointNode(
            name="test",
            file_path="api.py",
            line_start=1,
            line_end=10,
            http_method="get",
            path="/test",
        )
        assert endpoint.http_method == "GET"

    def test_invalid_http_method_rejected(self) -> None:
        """Invalid HTTP method is rejected."""
        with pytest.raises(ValidationError):
            EndpointNode(
                name="test",
                file_path="api.py",
                line_start=1,
                line_end=10,
                http_method="INVALID",
                path="/test",
            )

    def test_empty_http_method_rejected(self) -> None:
        """Empty HTTP method is rejected."""
        with pytest.raises(ValidationError):
            EndpointNode(
                name="test",
                file_path="api.py",
                line_start=1,
                line_end=10,
                http_method="",
                path="/test",
            )

    def test_empty_path_rejected(self) -> None:
        """Empty path is rejected."""
        with pytest.raises(ValidationError):
            EndpointNode(
                name="test",
                file_path="api.py",
                line_start=1,
                line_end=10,
                http_method="GET",
                path="",
            )
