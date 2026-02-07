"""Brutal unit tests for AST parser.

Tests every code path, edge case, and validation rule for 100% coverage.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from codebase_intelligence.edges import EdgeType
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    DecoratorNode,
    EndpointNode,
    FunctionNode,
    ImportNode,
    ModuleNode,
    NodeType,
    VariableNode,
)
from codebase_intelligence.parser import (
    ASTParser,
    CallExtractor,
    ComplexityCalculator,
    ParseError,
)


# ── ParseError ────────────────────────────────────────────────────────────


class TestParseError:
    """Tests for ParseError exception."""

    def test_basic_message(self) -> None:
        e = ParseError("bad syntax")
        assert str(e) == "bad syntax"
        assert e.message == "bad syntax"
        assert e.file_path is None
        assert e.line is None

    def test_with_file_path(self) -> None:
        e = ParseError("bad syntax", file_path="foo.py")
        assert "in foo.py" in str(e)

    def test_with_line(self) -> None:
        e = ParseError("bad syntax", line=42)
        assert "at line 42" in str(e)

    def test_with_file_and_line(self) -> None:
        e = ParseError("bad syntax", file_path="foo.py", line=10)
        msg = str(e)
        assert "in foo.py" in msg
        assert "at line 10" in msg


# ── ComplexityCalculator ──────────────────────────────────────────────────


class TestComplexityCalculator:
    """Tests for cyclomatic complexity calculation."""

    def _calc(self, source: str) -> int:
        import ast

        tree = ast.parse(source)
        func = tree.body[0]
        calc = ComplexityCalculator()
        return calc.calculate(func)

    def test_empty_function(self) -> None:
        """Empty function has complexity 1."""
        assert self._calc("def f(): pass") == 1

    def test_if_statement(self) -> None:
        assert self._calc("def f():\n if x: pass") == 2

    def test_if_elif(self) -> None:
        assert self._calc("def f():\n if x: pass\n elif y: pass") == 3

    def test_for_loop(self) -> None:
        assert self._calc("def f():\n for x in y: pass") == 2

    def test_while_loop(self) -> None:
        assert self._calc("def f():\n while x: pass") == 2

    def test_except_handler(self) -> None:
        assert self._calc("def f():\n try: pass\n except: pass") == 2

    def test_boolean_and(self) -> None:
        assert self._calc("def f():\n if x and y: pass") == 3

    def test_boolean_or(self) -> None:
        assert self._calc("def f():\n if x or y: pass") == 3

    def test_boolean_triple(self) -> None:
        """x and y and z → 2 additional branches."""
        assert self._calc("def f():\n if x and y and z: pass") == 4

    def test_ternary(self) -> None:
        assert self._calc("def f():\n a = x if c else y") == 2

    def test_assert(self) -> None:
        assert self._calc("def f():\n assert x") == 2

    def test_comprehension_with_if(self) -> None:
        assert self._calc("def f():\n [x for x in y if x]") == 2

    def test_comprehension_multiple_ifs(self) -> None:
        assert self._calc("def f():\n [x for x in y if x if z]") == 3

    def test_combined_complexity(self) -> None:
        src = "def f():\n if x:\n  for i in r:\n   if y and z: pass"
        assert self._calc(src) == 5  # 1 + if + for + if + and


# ── CallExtractor ─────────────────────────────────────────────────────────


class TestCallExtractor:
    """Tests for function call extraction."""

    def _extract(self, source: str) -> list[tuple[str, int, bool]]:
        import ast

        tree = ast.parse(source)
        func = tree.body[0]
        extractor = CallExtractor()
        return extractor.extract(func)

    def test_simple_call(self) -> None:
        calls = self._extract("def f():\n foo()")
        assert len(calls) == 1
        assert calls[0][0] == "foo"
        assert calls[0][2] is False  # not conditional

    def test_attribute_call(self) -> None:
        calls = self._extract("def f():\n obj.method()")
        assert calls[0][0] == "obj.method"

    def test_chained_call(self) -> None:
        calls = self._extract("def f():\n a.b.c()")
        assert calls[0][0] == "a.b.c"

    def test_conditional_call_in_if(self) -> None:
        calls = self._extract("def f():\n if x:\n  foo()")
        assert calls[0][2] is True  # conditional

    def test_conditional_call_in_for(self) -> None:
        calls = self._extract("def f():\n for x in y:\n  foo()")
        assert calls[0][2] is True

    def test_conditional_call_in_while(self) -> None:
        calls = self._extract("def f():\n while x:\n  foo()")
        assert calls[0][2] is True

    def test_non_conditional_after_conditional(self) -> None:
        """Call after if block is NOT conditional."""
        calls = self._extract("def f():\n if x:\n  foo()\n bar()")
        foo_calls = [c for c in calls if c[0] == "foo"]
        bar_calls = [c for c in calls if c[0] == "bar"]
        assert foo_calls[0][2] is True
        assert bar_calls[0][2] is False

    def test_no_calls(self) -> None:
        calls = self._extract("def f():\n x = 1")
        assert len(calls) == 0

    def test_call_with_non_name_non_attribute(self) -> None:
        """Calls like (lambda: ...)() have no extractable name."""
        calls = self._extract("def f():\n (lambda: 1)()")
        # lambda call has no name — filtered out
        named = [c for c in calls if c[0] is not None]
        # The lambda itself doesn't produce a name
        assert all(c[0] is not None for c in calls) or len(calls) == 0

    def test_multiple_calls(self) -> None:
        calls = self._extract("def f():\n a()\n b()\n c()")
        names = [c[0] for c in calls]
        assert "a" in names
        assert "b" in names
        assert "c" in names


# ── ASTParser.parse_source ────────────────────────────────────────────────


class TestASTParserParseSource:
    """Tests for parse_source method."""

    def test_empty_module(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source("", "empty.py")
        modules = list(graph.get_nodes(NodeType.MODULE))
        assert len(modules) == 1
        assert modules[0].name == "empty"

    def test_module_with_docstring(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source('"""My module."""', "mod.py")
        modules = list(graph.get_nodes(NodeType.MODULE))
        assert modules[0].docstring == "My module."

    def test_module_is_package(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source("", "pkg/__init__.py")
        modules = list(graph.get_nodes(NodeType.MODULE))
        assert modules[0].is_package is True

    def test_module_not_package(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source("", "app.py")
        modules = list(graph.get_nodes(NodeType.MODULE))
        assert modules[0].is_package is False

    def test_syntax_error_raises_parse_error(self) -> None:
        parser = ASTParser()
        with pytest.raises(ParseError, match="Syntax error"):
            parser.parse_source("def f(:", "bad.py")

    def test_simple_function(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source("def hello():\n    pass", "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 1
        assert funcs[0].name == "hello"

    def test_function_with_params_and_return(self) -> None:
        parser = ASTParser()
        src = "def greet(name: str, age: int) -> str:\n    return name"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        f = funcs[0]
        assert f.parameters == ["name", "age"]
        assert f.return_type == "str"

    def test_async_function(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source("async def fetch():\n    pass", "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].is_async is True

    def test_generator_function(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source("def gen():\n    yield 1", "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].is_generator is True

    def test_yield_from_is_generator(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source("def gen():\n    yield from [1]", "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].is_generator is True

    def test_non_generator(self) -> None:
        parser = ASTParser()
        graph = parser.parse_source("def f():\n    return 1", "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].is_generator is False

    def test_function_with_docstring(self) -> None:
        parser = ASTParser()
        src = 'def f():\n    """Doc."""\n    pass'
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].docstring == "Doc."

    def test_function_decorators(self) -> None:
        parser = ASTParser()
        src = "@staticmethod\ndef f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert "staticmethod" in funcs[0].decorators

    def test_function_complexity(self) -> None:
        parser = ASTParser()
        src = "def f():\n    if x:\n        for i in r:\n            pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].complexity >= 3

    def test_simple_class(self) -> None:
        parser = ASTParser()
        src = "class MyClass:\n    pass"
        graph = parser.parse_source(src, "app.py")
        classes = list(graph.get_nodes(NodeType.CLASS))
        assert len(classes) == 1
        assert classes[0].name == "MyClass"

    def test_class_with_bases(self) -> None:
        parser = ASTParser()
        src = "class Child(Parent):\n    pass"
        graph = parser.parse_source(src, "app.py")
        classes = list(graph.get_nodes(NodeType.CLASS))
        assert "Parent" in classes[0].bases

    def test_class_with_attribute_base(self) -> None:
        parser = ASTParser()
        src = "class Model(db.Base):\n    pass"
        graph = parser.parse_source(src, "app.py")
        classes = list(graph.get_nodes(NodeType.CLASS))
        assert "db.Base" in classes[0].bases

    def test_class_with_docstring(self) -> None:
        parser = ASTParser()
        src = 'class C:\n    """Class doc."""\n    pass'
        graph = parser.parse_source(src, "app.py")
        classes = list(graph.get_nodes(NodeType.CLASS))
        assert classes[0].docstring == "Class doc."

    def test_dataclass_detected(self) -> None:
        parser = ASTParser()
        src = "from dataclasses import dataclass\n@dataclass\nclass Config:\n    x: int = 1"
        graph = parser.parse_source(src, "app.py")
        classes = list(graph.get_nodes(NodeType.CLASS))
        assert classes[0].is_dataclass is True

    def test_pydantic_model_detected(self) -> None:
        parser = ASTParser()
        src = "class UserSchema(BaseModel):\n    name: str"
        graph = parser.parse_source(src, "app.py")
        classes = list(graph.get_nodes(NodeType.CLASS))
        assert classes[0].is_pydantic is True

    def test_base_settings_is_pydantic(self) -> None:
        parser = ASTParser()
        src = "class Settings(BaseSettings):\n    debug: bool"
        graph = parser.parse_source(src, "app.py")
        classes = list(graph.get_nodes(NodeType.CLASS))
        assert classes[0].is_pydantic is True

    def test_class_method(self) -> None:
        parser = ASTParser()
        src = "class C:\n    def method(self):\n        pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 1
        assert funcs[0].name == "method"

    def test_containment_edges(self) -> None:
        parser = ASTParser()
        src = "class C:\n    def m(self):\n        pass"
        graph = parser.parse_source(src, "app.py")
        contains = list(graph.get_edges(EdgeType.CONTAINS))
        assert len(contains) >= 2  # module→class, class→method

    def test_inheritance_edge(self) -> None:
        """Inheritance edge when base class is defined in same source."""
        parser = ASTParser()
        src = "class Base:\n    pass\nclass Child(Base):\n    pass"
        graph = parser.parse_source(src, "app.py")
        inherits = list(graph.get_edges(EdgeType.INHERITS))
        assert len(inherits) == 1

    def test_call_edges(self) -> None:
        parser = ASTParser()
        src = "def a():\n    pass\ndef b():\n    a()"
        graph = parser.parse_source(src, "app.py")
        calls = list(graph.get_edges(EdgeType.CALLS))
        assert len(calls) >= 1

    def test_import_statement(self) -> None:
        parser = ASTParser()
        src = "import os"
        graph = parser.parse_source(src, "app.py")
        imports = list(graph.get_nodes(NodeType.IMPORT))
        assert len(imports) == 1
        assert imports[0].module == "os"

    def test_import_with_alias(self) -> None:
        parser = ASTParser()
        src = "import numpy as np"
        graph = parser.parse_source(src, "app.py")
        imports = list(graph.get_nodes(NodeType.IMPORT))
        assert imports[0].name == "np"
        assert imports[0].alias == "np"
        assert imports[0].module == "numpy"

    def test_from_import(self) -> None:
        parser = ASTParser()
        src = "from os import path"
        graph = parser.parse_source(src, "app.py")
        imports = list(graph.get_nodes(NodeType.IMPORT))
        assert len(imports) == 1
        assert imports[0].is_from_import is True
        assert imports[0].module == "os"
        assert "path" in imports[0].imported_names

    def test_from_import_with_module(self) -> None:
        """from os.path import join."""
        parser = ASTParser()
        src = "from os.path import join, exists"
        graph = parser.parse_source(src, "app.py")
        imports = list(graph.get_nodes(NodeType.IMPORT))
        assert len(imports) == 1
        assert imports[0].is_from_import is True
        assert imports[0].module == "os.path"
        assert "join" in imports[0].imported_names
        assert "exists" in imports[0].imported_names

    def test_module_imports_list(self) -> None:
        parser = ASTParser()
        src = "import os\nimport sys\nfrom typing import List"
        graph = parser.parse_source(src, "app.py")
        modules = list(graph.get_nodes(NodeType.MODULE))
        imports = modules[0].imports
        assert "os" in imports
        assert "sys" in imports
        assert "typing" in imports

    def test_variable_assignment(self) -> None:
        parser = ASTParser()
        src = "x = 42"
        graph = parser.parse_source(src, "app.py")
        vars_ = list(graph.get_nodes(NodeType.VARIABLE))
        assert len(vars_) == 1
        assert vars_[0].name == "x"
        assert vars_[0].scope == "module"

    def test_annotated_variable(self) -> None:
        parser = ASTParser()
        src = "count: int = 0"
        graph = parser.parse_source(src, "app.py")
        vars_ = list(graph.get_nodes(NodeType.VARIABLE))
        assert vars_[0].type_annotation == "int"

    def test_constant_variable(self) -> None:
        parser = ASTParser()
        src = "MAX_SIZE = 100"
        graph = parser.parse_source(src, "app.py")
        vars_ = list(graph.get_nodes(NodeType.VARIABLE))
        assert vars_[0].is_constant is True

    def test_class_scope_variable(self) -> None:
        parser = ASTParser()
        src = "class C:\n    x = 10"
        graph = parser.parse_source(src, "app.py")
        vars_ = list(graph.get_nodes(NodeType.VARIABLE))
        assert vars_[0].scope == "class"

    def test_multiple_assignment_targets_skipped(self) -> None:
        """Assignments with multiple targets (a = b = 1) are skipped."""
        parser = ASTParser()
        src = "a = b = 1"
        graph = parser.parse_source(src, "app.py")
        # a = b = 1 has 2 targets, so it's skipped
        vars_ = list(graph.get_nodes(NodeType.VARIABLE))
        assert len(vars_) == 0

    def test_tuple_assignment_skipped(self) -> None:
        """Tuple assignments (a, b = 1, 2) are skipped."""
        parser = ASTParser()
        src = "a, b = 1, 2"
        graph = parser.parse_source(src, "app.py")
        vars_ = list(graph.get_nodes(NodeType.VARIABLE))
        assert len(vars_) == 0

    def test_annotated_assignment_non_name_skipped(self) -> None:
        """Annotated assignment to non-Name target skipped."""
        parser = ASTParser()
        src = "class C:\n    self.x: int = 1"
        # self.x is an Attribute, not a Name — should be skipped
        graph = parser.parse_source(src, "app.py")
        vars_ = list(graph.get_nodes(NodeType.VARIABLE))
        assert len(vars_) == 0

    def test_decorator_node_created(self) -> None:
        parser = ASTParser()
        src = "@property\ndef x(self):\n    return 1"
        graph = parser.parse_source(src, "app.py")
        decs = list(graph.get_nodes(NodeType.DECORATOR))
        assert len(decs) >= 1
        assert decs[0].decorator_name == "property"

    def test_decorator_with_arguments(self) -> None:
        parser = ASTParser()
        src = "@app.get('/users', response_model=UserList)\ndef get_users():\n    pass"
        graph = parser.parse_source(src, "app.py")
        decs = list(graph.get_nodes(NodeType.DECORATOR))
        assert len(decs) >= 1
        assert "'/users'" in decs[0].arguments

    def test_decorator_call_no_args(self) -> None:
        parser = ASTParser()
        src = "@cache()\ndef f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        decs = list(graph.get_nodes(NodeType.DECORATOR))
        assert decs[0].decorator_name == "cache"

    def test_decorates_edge_created(self) -> None:
        parser = ASTParser()
        src = "@staticmethod\ndef f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        decorates = list(graph.get_edges(EdgeType.DECORATES))
        assert len(decorates) >= 1

    def test_class_decorator(self) -> None:
        parser = ASTParser()
        src = "@dataclass\nclass Config:\n    x: int = 1"
        graph = parser.parse_source(src, "app.py")
        decs = list(graph.get_nodes(NodeType.DECORATOR))
        assert len(decs) >= 1

    def test_nested_function(self) -> None:
        parser = ASTParser()
        src = "class C:\n    def outer(self):\n        pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 1


# ── FastAPI Endpoint Detection ────────────────────────────────────────────


class TestASTParserEndpoints:
    """Tests for FastAPI/Flask endpoint detection."""

    def test_fastapi_get_endpoint(self) -> None:
        parser = ASTParser()
        src = "@app.get('/users')\ndef get_users():\n    pass"
        graph = parser.parse_source(src, "app.py")
        endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
        assert len(endpoints) == 1
        assert endpoints[0].http_method == "GET"
        assert endpoints[0].path == "/users"

    def test_fastapi_post_endpoint(self) -> None:
        parser = ASTParser()
        src = "@router.post('/items')\ndef create_item():\n    pass"
        graph = parser.parse_source(src, "app.py")
        endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
        assert len(endpoints) == 1
        assert endpoints[0].http_method == "POST"

    def test_all_http_methods(self) -> None:
        parser = ASTParser()
        methods = ["get", "post", "put", "delete", "patch", "options", "head"]
        for method in methods:
            src = f"@app.{method}('/')\ndef handler():\n    pass"
            graph = parser.parse_source(src, f"{method}.py")
            endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
            assert len(endpoints) == 1
            assert endpoints[0].http_method == method.upper()

    def test_endpoint_with_response_model(self) -> None:
        parser = ASTParser()
        src = "@app.get('/users', response_model=UserList)\ndef get_users():\n    pass"
        graph = parser.parse_source(src, "app.py")
        endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
        assert endpoints[0].response_model == "UserList"

    def test_endpoint_default_path(self) -> None:
        """Endpoint without path arg defaults to '/'."""
        parser = ASTParser()
        src = "@app.get()\ndef root():\n    pass"
        graph = parser.parse_source(src, "app.py")
        endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
        assert endpoints[0].path == "/"

    def test_endpoint_routes_to_edge(self) -> None:
        parser = ASTParser()
        src = "@app.get('/test')\ndef handler():\n    pass"
        graph = parser.parse_source(src, "app.py")
        routes = list(graph.get_edges(EdgeType.ROUTES_TO))
        assert len(routes) >= 1

    def test_non_endpoint_decorator_ignored(self) -> None:
        parser = ASTParser()
        src = "@app.middleware('http')\ndef middleware_func():\n    pass"
        graph = parser.parse_source(src, "app.py")
        endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
        assert len(endpoints) == 0


# ── ASTParser.parse_file ──────────────────────────────────────────────────


class TestASTParserParseFile:
    """Tests for parse_file method."""

    def test_parse_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass", encoding="utf-8")
        parser = ASTParser()
        graph = parser.parse_file(str(f))
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 1

    def test_file_not_found(self) -> None:
        parser = ASTParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent.py")

    def test_not_python_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        parser = ASTParser()
        with pytest.raises(ParseError, match="Not a Python file"):
            parser.parse_file(str(f))

    def test_unicode_decode_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_bytes(b"\xff\xfe\x00\x80invalid")
        parser = ASTParser()
        with pytest.raises(ParseError, match="Failed to read"):
            parser.parse_file(str(f))

    def test_parse_file_with_path_object(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1", encoding="utf-8")
        parser = ASTParser()
        graph = parser.parse_file(f)
        assert len(list(graph.get_nodes())) >= 1


# ── ASTParser.parse_directory ─────────────────────────────────────────────


class TestASTParserParseDirectory:
    """Tests for parse_directory method."""

    def test_parse_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
        (tmp_path / "b.py").write_text("def b(): pass", encoding="utf-8")
        parser = ASTParser()
        graph = parser.parse_directory(str(tmp_path))
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 2

    def test_parse_directory_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
        (sub / "b.py").write_text("def b(): pass", encoding="utf-8")
        parser = ASTParser()
        graph = parser.parse_directory(str(tmp_path), recursive=True)
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 2

    def test_parse_directory_non_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
        (sub / "b.py").write_text("def b(): pass", encoding="utf-8")
        parser = ASTParser()
        graph = parser.parse_directory(str(tmp_path), recursive=False)
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 1  # Only a.py

    def test_parse_directory_not_a_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        parser = ASTParser()
        with pytest.raises(NotADirectoryError):
            parser.parse_directory(str(f))

    def test_parse_empty_directory(self, tmp_path: Path) -> None:
        parser = ASTParser()
        graph = parser.parse_directory(str(tmp_path))
        assert len(list(graph.get_nodes())) == 0

    def test_parse_directory_skips_bad_files(self, tmp_path: Path) -> None:
        (tmp_path / "good.py").write_text("def f(): pass", encoding="utf-8")
        (tmp_path / "bad.py").write_text("def f(:", encoding="utf-8")
        parser = ASTParser()
        graph = parser.parse_directory(str(tmp_path))
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) >= 1  # At least the good file parsed

    def test_parse_directory_skips_encoding_errors(self, tmp_path: Path) -> None:
        (tmp_path / "good.py").write_text("def f(): pass", encoding="utf-8")
        (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00\x80invalid")
        parser = ASTParser()
        graph = parser.parse_directory(str(tmp_path))
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) >= 1


# ── Call Resolution ───────────────────────────────────────────────────────


class TestCallResolution:
    """Tests for _resolve_pending_calls."""

    def test_direct_call_resolved(self) -> None:
        parser = ASTParser()
        src = "def target():\n    pass\ndef caller():\n    target()"
        graph = parser.parse_source(src, "app.py")
        calls = list(graph.get_edges(EdgeType.CALLS))
        assert len(calls) >= 1

    def test_method_call_resolved(self) -> None:
        """obj.method() resolves to method if defined."""
        parser = ASTParser()
        src = (
            "def method():\n    pass\n"
            "def caller():\n    obj.method()"
        )
        graph = parser.parse_source(src, "app.py")
        calls = list(graph.get_edges(EdgeType.CALLS))
        assert len(calls) >= 1

    def test_unresolved_call_no_edge(self) -> None:
        """Call to undefined function creates no edge."""
        parser = ASTParser()
        src = "def caller():\n    unknown_function()"
        graph = parser.parse_source(src, "app.py")
        calls = list(graph.get_edges(EdgeType.CALLS))
        assert len(calls) == 0

    def test_recursive_call_no_self_edge(self) -> None:
        """Recursive call (calling itself) should not create self-loop edge."""
        parser = ASTParser()
        src = "def f():\n    f()"
        graph = parser.parse_source(src, "app.py")
        calls = list(graph.get_edges(EdgeType.CALLS))
        # Self-loop edges are forbidden, so no CALLS edge created
        assert len(calls) == 0


# ── Decorator Name Extraction ─────────────────────────────────────────────


class TestDecoratorExtraction:
    """Tests for _get_decorator_name edge cases."""

    def test_simple_decorator(self) -> None:
        parser = ASTParser()
        src = "@property\ndef x(self):\n    return 1"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert "property" in funcs[0].decorators

    def test_attribute_decorator(self) -> None:
        parser = ASTParser()
        src = "@app.route\ndef f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert "app.route" in funcs[0].decorators

    def test_call_decorator(self) -> None:
        parser = ASTParser()
        src = "@app.get('/')\ndef f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert "app.get" in funcs[0].decorators

    def test_decorator_arguments_extraction(self) -> None:
        parser = ASTParser()
        src = "@app.get('/users', response_model=UserList)\ndef f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        decs = list(graph.get_nodes(NodeType.DECORATOR))
        args = decs[0].arguments
        assert "'/users'" in args
        assert "response_model=UserList" in args

    def test_decorator_with_non_constant_arg(self) -> None:
        parser = ASTParser()
        src = "@dec(some_var)\ndef f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        decs = list(graph.get_nodes(NodeType.DECORATOR))
        assert len(decs[0].arguments) >= 1


# ── Annotation String Extraction ──────────────────────────────────────────


class TestAnnotationExtraction:
    """Tests for _get_annotation_string."""

    def test_simple_annotation(self) -> None:
        parser = ASTParser()
        src = "def f(x: int) -> str:\n    pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].return_type == "str"

    def test_complex_annotation(self) -> None:
        parser = ASTParser()
        src = "def f() -> list[dict[str, int]]:\n    pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert "list" in funcs[0].return_type

    def test_no_return_annotation(self) -> None:
        parser = ASTParser()
        src = "def f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].return_type is None


# ── End Line Calculation ──────────────────────────────────────────────────


class TestEndLineCalculation:
    """Tests for _get_end_line."""

    def test_end_line_matches_ast(self) -> None:
        parser = ASTParser()
        src = "def f():\n    x = 1\n    y = 2\n    return x + y"
        graph = parser.parse_source(src, "app.py")
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert funcs[0].line_end >= 4


# ── Multi-file Cross-reference ────────────────────────────────────────────


class TestMultiFileParsing:
    """Tests for cross-file resolution in parse_directory."""

    def test_cross_file_call_resolution(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def helper():\n    pass", encoding="utf-8")
        (tmp_path / "b.py").write_text("def main():\n    helper()", encoding="utf-8")
        parser = ASTParser()
        graph = parser.parse_directory(str(tmp_path))
        calls = list(graph.get_edges(EdgeType.CALLS))
        assert len(calls) >= 1

    def test_qualified_name_resolution(self) -> None:
        parser = ASTParser()
        src = "class C:\n    def method(self):\n        pass"
        graph = parser.parse_source(src, "app.py")
        # Both "method" and "C.method" should resolve
        funcs = list(graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 1


# ── Internal Method Edge Cases ────────────────────────────────────────────


class TestParserInternalEdgeCases:
    """Tests for hard-to-reach internal method paths."""

    def test_get_decorator_name_unknown(self) -> None:
        """Decorator that is not Name, Attribute, or Call returns 'unknown'."""
        import ast

        parser = ASTParser()
        parser._current_file = "test.py"
        # Create a Constant node as decorator (not a valid decorator type)
        node = ast.Constant(value=42, lineno=1, col_offset=0)
        result = parser._get_decorator_name(node)
        assert result == "unknown"

    def test_get_annotation_string_none(self) -> None:
        """_get_annotation_string with None returns None."""
        parser = ASTParser()
        assert parser._get_annotation_string(None) is None

    def test_get_annotation_string_exception(self) -> None:
        """_get_annotation_string returns None on unparse failure."""
        import ast
        from unittest.mock import patch

        parser = ASTParser()
        node = ast.Name(id="int", ctx=ast.Load())
        with patch("ast.unparse", side_effect=ValueError("mock")):
            result = parser._get_annotation_string(node)
        assert result is None

    def test_get_end_line_fallback(self) -> None:
        """_get_end_line uses fallback when end_lineno is missing."""
        import ast

        parser = ASTParser()
        # Create an AST node without end_lineno, but with child nodes
        # that have higher line numbers
        parent = ast.Module(body=[], type_ignores=[])
        parent.lineno = 5  # type: ignore[attr-defined]
        parent.end_lineno = None
        # Add a child with a higher line number
        child = ast.Constant(value=1)
        child.lineno = 10  # type: ignore[attr-defined]
        parent.body.append(child)  # type: ignore[arg-type]
        result = parser._get_end_line(parent)
        assert result == 10  # Should find child's line as max

    def test_get_end_line_no_lineno(self) -> None:
        """_get_end_line uses fallback=1 when no lineno at all."""
        import ast

        parser = ASTParser()
        node = ast.AST()
        # No lineno attribute → getattr default is 1
        if hasattr(node, "end_lineno"):
            del node.end_lineno
        result = parser._get_end_line(node)
        assert result >= 1

    def test_get_attribute_name_non_name_root(self) -> None:
        """_get_attribute_name where root is not a Name node."""
        import ast

        parser = ASTParser()
        # Create attr chain ending in a Call (not Name)
        root_call = ast.Call(
            func=ast.Name(id="foo", ctx=ast.Load()),
            args=[],
            keywords=[],
        )
        attr = ast.Attribute(value=root_call, attr="bar", ctx=ast.Load())
        result = parser._get_attribute_name(attr)
        # Only "bar" extracted since root is Call, not Name
        assert result == "bar"

    def test_decorator_keyword_without_arg(self) -> None:
        """Decorator with **kwargs keyword (keyword.arg is None)."""
        parser = ASTParser()
        src = "@dec(**config)\ndef f():\n    pass"
        graph = parser.parse_source(src, "app.py")
        decs = list(graph.get_nodes(NodeType.DECORATOR))
        # **config has keyword.arg = None, so it's skipped
        assert len(decs) >= 1


# ── Branch Partial Coverage ──────────────────────────────────────────────


class TestBranchPartials:
    """Target remaining branch partials for 100% branch coverage."""

    def test_call_name_non_name_base(self) -> None:
        """242->245: Call chain base is not ast.Name (e.g., subscript)."""
        parser = ASTParser()
        # Call must be inside a function so CallExtractor processes it
        src = "def foo():\n    data[0].method()\n"
        graph = parser.parse_source(src, "app.py")
        # The call chain base is a Subscript, not Name → _get_call_name returns None
        # So no CALLS edge is created for data[0].method()
        calls_edges = [e for e in graph.get_edges(EdgeType.CALLS)]
        assert len(calls_edges) == 0

    def test_base_class_not_name_or_attribute(self) -> None:
        """480->477: Base class is neither Name nor Attribute (e.g., function call)."""
        parser = ASTParser()
        src = "class C(make_base()): pass\n"
        graph = parser.parse_source(src, "app.py")
        classes = [n for n in graph.get_nodes(NodeType.CLASS)]
        assert len(classes) == 1
        # make_base() is a Call, not Name or Attribute, so bases is empty
        assert classes[0].bases == []

    def test_endpoint_bare_decorator(self) -> None:
        """742->747: Endpoint decorator without Call (bare @app.get)."""
        parser = ASTParser()
        # @app.get without parentheses — ast.Attribute, not ast.Call
        src = "import app\n@app.get\ndef index(): pass\n"
        graph = parser.parse_source(src, "app.py")
        endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
        assert len(endpoints) == 1
        assert endpoints[0].path == "/"  # default path since no args

    def test_endpoint_non_constant_arg(self) -> None:
        """737->741: Endpoint decorator arg is not a Constant."""
        parser = ASTParser()
        src = "import app\nROUTE = '/api'\n@app.get(ROUTE)\ndef index(): pass\n"
        graph = parser.parse_source(src, "app.py")
        endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
        assert len(endpoints) == 1
        # ROUTE is a Name, not Constant, so path stays default "/"
        assert endpoints[0].path == "/"

    def test_endpoint_keyword_not_response_model(self) -> None:
        """744->743: Endpoint keyword that's not response_model."""
        parser = ASTParser()
        src = (
            "import app\n"
            "@app.get('/items', status_code=200, tags=['items'])\n"
            "def get_items(): pass\n"
        )
        graph = parser.parse_source(src, "app.py")
        endpoints = list(graph.get_nodes(NodeType.ENDPOINT))
        assert len(endpoints) == 1
        # Keywords exist but none is response_model
        assert endpoints[0].response_model is None

    def test_from_import_no_module(self) -> None:
        """456->451: from . import x where node.module is None."""
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = []
        parser._name_to_node_id = {}
        parser._pending_calls = []
        # Build tree with from . import and extract import names
        tree = _ast.parse("from . import utils\n")
        result = parser._extract_import_names(tree)
        # Relative import with no module → node.module is None → skipped
        assert result == []

    def test_class_empty_scope_stack(self) -> None:
        """510->515: _process_class with empty scope stack."""
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = []
        parser._name_to_node_id = {}
        parser._pending_calls = []
        node = _ast.parse("class Foo:\n    pass\n").body[0]
        parser._process_class(node)
        classes = list(parser._graph.get_nodes(NodeType.CLASS))
        assert len(classes) == 1
        assert classes[0].name == "Foo"
        # No containment edge since scope stack is empty
        edges = list(parser._graph.get_edges(EdgeType.CONTAINS))
        assert len(edges) == 0

    def test_function_empty_scope_stack(self) -> None:
        """575->580, 584->589: _process_function with empty scope stack."""
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = []
        parser._name_to_node_id = {}
        parser._pending_calls = []
        node = _ast.parse("def bar(): pass\n").body[0]
        parser._process_function(node)
        funcs = list(parser._graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 1
        assert funcs[0].name == "bar"
        # No containment edge since scope stack is empty
        edges = list(parser._graph.get_edges(EdgeType.CONTAINS))
        assert len(edges) == 0

    def test_function_scope_stack_dangling_id(self) -> None:
        """parent_node is None when scope stack ID not in graph."""
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = ["nonexistent-id"]
        parser._name_to_node_id = {}
        parser._pending_calls = []
        node = _ast.parse("def baz(): pass\n").body[0]
        parser._process_function(node)
        funcs = list(parser._graph.get_nodes(NodeType.FUNCTION))
        assert len(funcs) == 1
        # parent_node is None → no containment edge, just simple name
        assert "baz" in parser._name_to_node_id
        edges = list(parser._graph.get_edges(EdgeType.CONTAINS))
        assert len(edges) == 0

    def test_import_empty_scope_stack(self) -> None:
        """615->601: _process_import with empty scope stack."""
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = []
        parser._name_to_node_id = {}
        parser._pending_calls = []
        node = _ast.parse("import os\n").body[0]
        parser._process_import(node)
        imports = list(parser._graph.get_nodes(NodeType.IMPORT))
        assert len(imports) == 1
        assert imports[0].name == "os"
        edges = list(parser._graph.get_edges(EdgeType.CONTAINS))
        assert len(edges) == 0

    def test_import_from_empty_scope_stack(self) -> None:
        """637->exit: _process_import_from with empty scope stack."""
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = []
        parser._name_to_node_id = {}
        parser._pending_calls = []
        node = _ast.parse("from os import path\n").body[0]
        parser._process_import_from(node)
        imports = list(parser._graph.get_nodes(NodeType.IMPORT))
        assert len(imports) == 1
        edges = list(parser._graph.get_edges(EdgeType.CONTAINS))
        assert len(edges) == 0

    def test_assignment_empty_scope_stack(self) -> None:
        """660->668, 682->exit: _process_assignment with empty scope stack."""
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = []
        parser._name_to_node_id = {}
        parser._pending_calls = []
        node = _ast.parse("x = 42\n").body[0]
        parser._process_assignment(node)
        variables = list(parser._graph.get_nodes(NodeType.VARIABLE))
        assert len(variables) == 1
        assert variables[0].name == "x"
        assert variables[0].scope == "local"  # default when no scope stack
        edges = list(parser._graph.get_edges(EdgeType.CONTAINS))
        assert len(edges) == 0

    def test_assignment_scope_stack_dangling_id(self) -> None:
        """parent is None when scope stack ID not in graph."""
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = ["nonexistent-id"]
        parser._name_to_node_id = {}
        parser._pending_calls = []
        node = _ast.parse("y = 10\n").body[0]
        parser._process_assignment(node)
        variables = list(parser._graph.get_nodes(NodeType.VARIABLE))
        assert len(variables) == 1
        assert variables[0].scope == "local"  # parent is None, stays local
        edges = list(parser._graph.get_edges(EdgeType.CONTAINS))
        assert len(edges) == 0  # No containment edge since parent not found

    def test_variable_inside_function_local_scope(self) -> None:
        """665->668: Variable with parent as FunctionNode → scope is 'local'.

        The parser doesn't recurse into function bodies for assignments,
        so we test via direct _process_assignment call with function parent.
        """
        import ast as _ast

        parser = ASTParser()
        parser._graph = SemanticGraph()
        parser._current_file = "test.py"
        parser._scope_stack = []
        parser._name_to_node_id = {}
        parser._pending_calls = []
        # Add a FunctionNode as the parent scope
        func_node = FunctionNode(
            name="func",
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        parser._graph.add_node(func_node)
        parser._scope_stack.append(func_node.id)
        node = _ast.parse("x = 42\n").body[0]
        parser._process_assignment(node)
        variables = list(parser._graph.get_nodes(NodeType.VARIABLE))
        assert len(variables) == 1
        assert variables[0].name == "x"
        assert variables[0].scope == "local"
