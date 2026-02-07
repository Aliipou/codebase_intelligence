"""AST Parser for extracting semantic information from Python source code.

This module provides the ASTParser class which analyzes Python source files
and extracts semantic nodes (modules, classes, functions) and edges
(relationships like contains, calls, imports).

The parser uses Python's built-in ast module for parsing and performs
multiple passes to:
1. Extract all node definitions
2. Resolve relationships between nodes
3. Build the semantic graph

Usage:
    >>> parser = ASTParser()
    >>> graph = parser.parse_file("path/to/module.py")
    >>> print(graph.get_stats())

    >>> # Parse multiple files
    >>> graph = parser.parse_directory("path/to/package/")
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from codebase_intelligence.edges import (
    EdgeType,
    SemanticEdge,
    create_calls_edge,
    create_contains_edge,
    create_imports_edge,
    create_inherits_edge,
)
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    DecoratorNode,
    EndpointNode,
    FunctionNode,
    ImportNode,
    ModuleNode,
    VariableNode,
)


class ParseError(Exception):
    """Raised when parsing fails.

    Attributes:
        file_path: Path to the file that failed to parse.
        line: Line number where the error occurred (if known).
        message: Description of the error.
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        line: int | None = None,
    ) -> None:
        """Initialize ParseError.

        Args:
            message: Description of the error.
            file_path: Path to the file that failed to parse.
            line: Line number where the error occurred.
        """
        self.file_path = file_path
        self.line = line
        self.message = message
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with location information."""
        parts = []
        if self.file_path:
            parts.append(f"in {self.file_path}")
        if self.line:
            parts.append(f"at line {self.line}")
        location = " ".join(parts)
        if location:
            return f"{self.message} ({location})"
        return self.message


class ComplexityCalculator(ast.NodeVisitor):
    """Calculates cyclomatic complexity of a function.

    Cyclomatic complexity measures the number of independent paths
    through a function. Higher complexity indicates more branches
    and potential for bugs.

    Complexity increases for:
        - if/elif statements
        - for/while loops
        - except handlers
        - boolean operators (and/or)
        - conditional expressions
        - assert statements
        - comprehensions with conditions

    Usage:
        >>> calculator = ComplexityCalculator()
        >>> complexity = calculator.calculate(function_ast_node)
    """

    def __init__(self) -> None:
        """Initialize the complexity calculator."""
        self._complexity = 1  # Base complexity

    def calculate(self, node: ast.AST) -> int:
        """Calculate complexity for an AST node.

        Args:
            node: The AST node (usually a FunctionDef) to analyze.

        Returns:
            The cyclomatic complexity score (minimum 1).
        """
        self._complexity = 1
        self.visit(node)
        return self._complexity

    def visit_If(self, node: ast.If) -> None:
        """Count if statements."""
        self._complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Count for loops."""
        self._complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Count while loops."""
        self._complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Count except handlers."""
        self._complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Count boolean operators (and/or add branches)."""
        self._complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Count conditional expressions (ternary)."""
        self._complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Count assert statements."""
        self._complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        """Count comprehension conditions."""
        self._complexity += len(node.ifs)
        self.generic_visit(node)


class CallExtractor(ast.NodeVisitor):
    """Extracts function call information from AST nodes.

    Identifies all function/method calls within a given AST node
    and records their names and line numbers.

    Usage:
        >>> extractor = CallExtractor()
        >>> calls = extractor.extract(function_ast_node)
        >>> for name, line in calls:
        ...     print(f"Call to {name} at line {line}")
    """

    def __init__(self) -> None:
        """Initialize the call extractor."""
        self._calls: list[tuple[str, int, bool]] = []
        self._in_conditional = False

    def extract(self, node: ast.AST) -> list[tuple[str, int, bool]]:
        """Extract all calls from an AST node.

        Args:
            node: The AST node to analyze.

        Returns:
            List of tuples: (function_name, line_number, is_conditional).
        """
        self._calls = []
        self._in_conditional = False
        self.visit(node)
        return self._calls

    def visit_If(self, node: ast.If) -> None:
        """Track conditional context for if statements."""
        old_conditional = self._in_conditional
        self._in_conditional = True
        self.generic_visit(node)
        self._in_conditional = old_conditional

    def visit_For(self, node: ast.For) -> None:
        """Track conditional context for for loops."""
        old_conditional = self._in_conditional
        self._in_conditional = True
        self.generic_visit(node)
        self._in_conditional = old_conditional

    def visit_While(self, node: ast.While) -> None:
        """Track conditional context for while loops."""
        old_conditional = self._in_conditional
        self._in_conditional = True
        self.generic_visit(node)
        self._in_conditional = old_conditional

    def visit_Call(self, node: ast.Call) -> None:
        """Extract function call information."""
        name = self._get_call_name(node.func)
        if name:
            self._calls.append((name, node.lineno, self._in_conditional))
        self.generic_visit(node)

    def _get_call_name(self, node: ast.expr) -> str | None:
        """Extract the name from a call expression.

        Handles simple names (func()), attribute access (obj.method()),
        and chained calls (a.b.c()).
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Get the full dotted name
            parts = []
            current: ast.expr = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None


class ASTParser:
    """Parses Python source code and builds a semantic graph.

    The parser performs multi-pass analysis:
    1. Parse source into AST
    2. Extract all definitions (modules, classes, functions)
    3. Extract relationships (imports, calls, inheritance)
    4. Build and return the semantic graph

    Attributes:
        _graph: The semantic graph being built.
        _current_file: Path to the current file being parsed.
        _scope_stack: Stack of current scope IDs for nested definitions.
        _name_to_node_id: Mapping from names to node IDs for resolution.
        _complexity_calculator: Reusable complexity calculator instance.
        _call_extractor: Reusable call extractor instance.

    Examples:
        >>> parser = ASTParser()

        >>> # Parse a single file
        >>> graph = parser.parse_file("app/main.py")

        >>> # Parse a directory
        >>> graph = parser.parse_directory("app/")

        >>> # Parse source code directly
        >>> source = '''
        ... def hello(name: str) -> str:
        ...     return f"Hello, {name}!"
        ... '''
        >>> graph = parser.parse_source(source, "example.py")
    """

    def __init__(self) -> None:
        """Initialize the AST parser."""
        self._graph: SemanticGraph = SemanticGraph()
        self._current_file: str = ""
        self._scope_stack: list[str] = []
        self._name_to_node_id: dict[str, str] = {}
        self._complexity_calculator = ComplexityCalculator()
        self._call_extractor = CallExtractor()
        self._pending_calls: list[tuple[str, str, int, bool]] = []

    def parse_file(self, file_path: str | Path) -> SemanticGraph:
        """Parse a Python file and return its semantic graph.

        Args:
            file_path: Path to the Python file to parse.

        Returns:
            SemanticGraph containing the parsed code structure.

        Raises:
            ParseError: If the file cannot be read or parsed.
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.suffix == ".py":
            raise ParseError("Not a Python file", str(file_path))

        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ParseError(f"Failed to read file: {e}", str(file_path)) from e

        return self.parse_source(source, str(path))

    def parse_source(self, source: str, file_path: str) -> SemanticGraph:
        """Parse Python source code and return its semantic graph.

        Args:
            source: Python source code as a string.
            file_path: Path to associate with this source (for error messages).

        Returns:
            SemanticGraph containing the parsed code structure.

        Raises:
            ParseError: If the source code has syntax errors.
        """
        self._graph = SemanticGraph()
        self._current_file = file_path
        self._scope_stack = []
        self._name_to_node_id = {}
        self._pending_calls = []

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            raise ParseError(
                f"Syntax error: {e.msg}",
                file_path,
                e.lineno,
            ) from e

        # First pass: extract all definitions
        self._extract_module(tree, source)

        # Second pass: resolve relationships
        self._resolve_pending_calls()

        return self._graph

    def parse_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
    ) -> SemanticGraph:
        """Parse all Python files in a directory.

        Args:
            directory_path: Path to the directory to parse.
            recursive: Whether to recursively parse subdirectories.

        Returns:
            SemanticGraph containing all parsed code structures.

        Raises:
            ParseError: If any file fails to parse.
            NotADirectoryError: If the path is not a directory.
        """
        path = Path(directory_path)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        # Reset graph for fresh parse
        self._graph = SemanticGraph()
        self._name_to_node_id = {}
        self._pending_calls = []

        # Find all Python files
        pattern = "**/*.py" if recursive else "*.py"
        py_files = list(path.glob(pattern))

        if not py_files:
            return self._graph

        # Parse each file
        for py_file in py_files:
            self._current_file = str(py_file)
            self._scope_stack = []

            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
                self._extract_module(tree, source)
            except (SyntaxError, UnicodeDecodeError) as e:
                # Skip files that can't be parsed but continue with others
                continue

        # Resolve all pending calls after all files are parsed
        self._resolve_pending_calls()

        return self._graph

    def _extract_module(self, tree: ast.Module, source: str) -> None:
        """Extract module-level information from AST.

        Creates a ModuleNode and processes all top-level definitions.
        """
        lines = source.split("\n")
        line_end = len(lines) if lines else 1

        # Get module docstring
        docstring = ast.get_docstring(tree)

        # Determine if this is a package
        is_package = self._current_file.endswith("__init__.py")

        # Extract import statements
        imports = self._extract_import_names(tree)

        # Create module node
        module_name = Path(self._current_file).stem
        module_node = ModuleNode(
            name=module_name,
            file_path=self._current_file,
            line_start=1,
            line_end=line_end,
            docstring=docstring,
            is_package=is_package,
            imports=imports,
        )

        self._graph.add_node(module_node)
        self._name_to_node_id[module_name] = module_node.id
        self._scope_stack.append(module_node.id)

        # Process all top-level statements
        for node in ast.iter_child_nodes(tree):
            self._process_node(node)

        self._scope_stack.pop()

    def _extract_import_names(self, tree: ast.Module) -> list[str]:
        """Extract all imported module names from a module."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def _process_node(self, node: ast.AST) -> None:
        """Process an AST node and extract semantic information."""
        if isinstance(node, ast.ClassDef):
            self._process_class(node)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            self._process_function(node)
        elif isinstance(node, ast.Import):
            self._process_import(node)
        elif isinstance(node, ast.ImportFrom):
            self._process_import_from(node)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            self._process_assignment(node)

    def _process_class(self, node: ast.ClassDef) -> None:
        """Process a class definition."""
        # Extract base class names
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))

        # Check for special class types
        is_dataclass = any(
            self._get_decorator_name(d) == "dataclass" for d in node.decorator_list
        )
        is_pydantic = "BaseModel" in bases or "BaseSettings" in bases

        # Get docstring
        docstring = ast.get_docstring(node)

        # Calculate line range
        line_end = self._get_end_line(node)

        class_node = ClassNode(
            name=node.name,
            file_path=self._current_file,
            line_start=node.lineno,
            line_end=line_end,
            docstring=docstring,
            bases=bases,
            is_dataclass=is_dataclass,
            is_pydantic=is_pydantic,
        )

        self._graph.add_node(class_node)
        self._name_to_node_id[node.name] = class_node.id

        # Add containment edge from current scope
        if self._scope_stack:
            edge = create_contains_edge(self._scope_stack[-1], class_node.id)
            self._graph.add_edge(edge)

        # Add inheritance edges
        for base_name in bases:
            if base_name in self._name_to_node_id:
                edge = create_inherits_edge(class_node.id, self._name_to_node_id[base_name])
                self._graph.add_edge(edge)

        # Process class body
        self._scope_stack.append(class_node.id)
        for child in node.body:
            self._process_node(child)
        self._scope_stack.pop()

        # Process decorators
        self._process_decorators(node.decorator_list, class_node.id)

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Process a function or method definition."""
        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]

        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._get_annotation_string(node.returns)

        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Check for async
        is_async = isinstance(node, ast.AsyncFunctionDef)

        # Check for generator
        is_generator = self._is_generator(node)

        # Calculate complexity
        complexity = self._complexity_calculator.calculate(node)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Calculate line range
        line_end = self._get_end_line(node)

        func_node = FunctionNode(
            name=node.name,
            file_path=self._current_file,
            line_start=node.lineno,
            line_end=line_end,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            is_generator=is_generator,
            decorators=decorators,
            complexity=complexity,
        )

        self._graph.add_node(func_node)

        # Build qualified name and containment edge
        parent_node = None
        if self._scope_stack:
            parent_node = self._graph.get_node(self._scope_stack[-1])

        qualified_name = node.name
        if parent_node:
            qualified_name = f"{parent_node.name}.{node.name}"
            edge = create_contains_edge(parent_node.id, func_node.id)
            self._graph.add_edge(edge)

        self._name_to_node_id[qualified_name] = func_node.id
        self._name_to_node_id[node.name] = func_node.id

        # Extract function calls (defer resolution)
        calls = self._call_extractor.extract(node)
        for call_name, line_num, is_conditional in calls:
            self._pending_calls.append((func_node.id, call_name, line_num, is_conditional))

        # Check for FastAPI endpoint decorators
        self._check_for_endpoint(node, func_node.id, decorators)

        # Process decorators
        self._process_decorators(node.decorator_list, func_node.id)

    def _process_import(self, node: ast.Import) -> None:
        """Process an import statement."""
        for alias in node.names:
            import_node = ImportNode(
                name=alias.asname or alias.name,
                file_path=self._current_file,
                line_start=node.lineno,
                line_end=node.lineno,
                module=alias.name,
                alias=alias.asname,
                is_from_import=False,
            )

            self._graph.add_node(import_node)

            # Add containment edge
            if self._scope_stack:
                edge = create_contains_edge(self._scope_stack[-1], import_node.id)
                self._graph.add_edge(edge)

    def _process_import_from(self, node: ast.ImportFrom) -> None:
        """Process a from...import statement."""
        module = node.module or ""
        imported_names = [alias.name for alias in node.names]

        import_node = ImportNode(
            name=module or ".",
            file_path=self._current_file,
            line_start=node.lineno,
            line_end=node.lineno,
            module=module,
            is_from_import=True,
            imported_names=imported_names,
        )

        self._graph.add_node(import_node)

        # Add containment edge
        if self._scope_stack:
            edge = create_contains_edge(self._scope_stack[-1], import_node.id)
            self._graph.add_edge(edge)

    def _process_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
        """Process a variable assignment."""
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                return
            name = node.target.id
            type_annotation = self._get_annotation_string(node.annotation)
        else:
            # Regular assignment - only handle simple name targets
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return
            name = node.targets[0].id
            type_annotation = None

        # Determine if constant (UPPER_CASE)
        is_constant = name.isupper()

        # Determine scope and parent
        parent = None
        if self._scope_stack:
            parent = self._graph.get_node(self._scope_stack[-1])

        scope = "local"
        if parent:
            if parent.node_type.value == "module":
                scope = "module"
            elif parent.node_type.value == "class":
                scope = "class"

        var_node = VariableNode(
            name=name,
            file_path=self._current_file,
            line_start=node.lineno,
            line_end=node.lineno,
            type_annotation=type_annotation,
            is_constant=is_constant,
            scope=scope,
        )

        self._graph.add_node(var_node)
        self._name_to_node_id[name] = var_node.id

        # Add containment edge
        if parent:
            edge = create_contains_edge(parent.id, var_node.id)
            self._graph.add_edge(edge)

    def _process_decorators(
        self,
        decorators: list[ast.expr],
        target_id: str,
    ) -> None:
        """Process decorator nodes and create edges."""
        for decorator in decorators:
            decorator_name = self._get_decorator_name(decorator)
            arguments = self._get_decorator_arguments(decorator)

            dec_node = DecoratorNode(
                name=decorator_name,
                file_path=self._current_file,
                line_start=decorator.lineno,
                line_end=decorator.lineno,
                decorator_name=decorator_name,
                arguments=arguments,
                target_node_id=target_id,
            )

            self._graph.add_node(dec_node)

            # Add decorates edge
            edge = SemanticEdge(
                source_id=dec_node.id,
                target_id=target_id,
                edge_type=EdgeType.DECORATES,
                line_number=decorator.lineno,
            )
            self._graph.add_edge(edge)

    def _check_for_endpoint(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        func_id: str,
        decorators: list[str],
    ) -> None:
        """Check if function is a FastAPI/Flask endpoint and create EndpointNode."""
        http_methods = {"get", "post", "put", "delete", "patch", "options", "head"}

        for i, dec in enumerate(node.decorator_list):
            dec_name = self._get_decorator_name(dec).lower()

            # Check for router.get, app.post, etc.
            parts = dec_name.split(".")
            method = parts[-1] if parts else ""

            if method in http_methods:
                # Extract path from decorator arguments
                path = "/"
                if isinstance(dec, ast.Call) and dec.args:
                    if isinstance(dec.args[0], ast.Constant):
                        path = str(dec.args[0].value)

                # Extract response_model if present
                response_model = None
                if isinstance(dec, ast.Call):
                    for keyword in dec.keywords:
                        if keyword.arg == "response_model":
                            response_model = self._get_annotation_string(keyword.value)

                endpoint_node = EndpointNode(
                    name=node.name,
                    file_path=self._current_file,
                    line_start=node.lineno,
                    line_end=self._get_end_line(node),
                    http_method=method.upper(),
                    path=path,
                    response_model=response_model,
                )

                self._graph.add_node(endpoint_node)

                # Add routes_to edge from endpoint to function
                edge = SemanticEdge(
                    source_id=endpoint_node.id,
                    target_id=func_id,
                    edge_type=EdgeType.ROUTES_TO,
                )
                self._graph.add_edge(edge)

    def _resolve_pending_calls(self) -> None:
        """Resolve pending function calls to edges."""
        for caller_id, call_name, line_num, is_conditional in self._pending_calls:
            # Try to find the target function
            # Handle method calls (obj.method -> just look for method)
            parts = call_name.split(".")
            simple_name = parts[-1]

            target_id = None
            if call_name in self._name_to_node_id:
                target_id = self._name_to_node_id[call_name]
            elif simple_name in self._name_to_node_id:
                target_id = self._name_to_node_id[simple_name]

            if target_id and target_id != caller_id:
                edge = create_calls_edge(caller_id, target_id, line_num, is_conditional)
                self._graph.add_edge(edge)

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract the name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_attribute_name(decorator)
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return "unknown"

    def _get_decorator_arguments(self, decorator: ast.expr) -> list[str]:
        """Extract arguments from a decorator call."""
        if not isinstance(decorator, ast.Call):
            return []

        args = []
        for arg in decorator.args:
            if isinstance(arg, ast.Constant):
                args.append(repr(arg.value))
            else:
                args.append(ast.unparse(arg))

        for keyword in decorator.keywords:
            if keyword.arg:
                args.append(f"{keyword.arg}={ast.unparse(keyword.value)}")

        return args

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full dotted name from an Attribute node."""
        parts = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _get_annotation_string(self, annotation: ast.expr | None) -> str | None:
        """Convert an annotation AST node to a string."""
        if annotation is None:
            return None
        try:
            return ast.unparse(annotation)
        except Exception:
            return None

    def _get_end_line(self, node: ast.AST) -> int:
        """Get the end line number of an AST node."""
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            return node.end_lineno
        # Fallback: find the maximum line number in the subtree
        max_line = getattr(node, "lineno", 1)
        for child in ast.walk(node):
            child_line = getattr(child, "lineno", 0)
            if child_line > max_line:
                max_line = child_line
        return max_line

    def _is_generator(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if a function is a generator."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False
