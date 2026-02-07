# Codebase Intelligence

**Constraint-aware code generation through semantic graph analysis.**

Codebase Intelligence generates **project-conformant code**, not generic code.

It parses Python repositories into a semantic graph, extracts structural and behavioral patterns, compiles them into enforceable constraints, and uses those constraints to **guide and validate LLM-based code generation** through an autonomous agent loop.

Generated code is accepted **only if it satisfies the project's implicit rules**.

---

## Motivation

LLMs produce syntactically valid code but consistently fail at respecting **project-specific invariants**:

- Architectural boundary violations
- Inconsistent naming and structure
- Broken domain assumptions
- Subtle regressions that pass local checks

> **LLMs do not understand how a specific codebase works.**

Codebase Intelligence makes those rules explicit, enforces them, and self-corrects when violations are detected.

---

## Core Concept

```
1. Parse repository   -->  Semantic Graph
2. Detect patterns    -->  Constraints
3. Agent observes     -->  Plan
4. LLM generates      -->  Source code
5. Validate output    -->  Pass / Violations
6. Diagnose & refine  -->  Retry with feedback
7. Accept or reject   -->  Merge-ready code
```

The LLM generates code.
The system decides whether that code is allowed to exist.
The agent refines until it gets it right.

---

## Architecture

```
Source Code
    |
    v
[Parser] --> [Semantic Graph] --> [Pattern Extractor] --> [Constraints]
                                                              |
                                                              v
                                                        [Constraint DSL]
                                                              |
                                                              v
                                              +---> [Prompt Compiler]
                                              |           |
                                              |           v
                                              |     [LLM Provider]
                                              |           |
                                              |           v
                                              |     [Validator]
                                              |           |
                                              |           v
                                              |    Pass? --+--> [Generated Code]
                                              |           |
                                              |           No
                                              |           |
                                              |           v
                                              |   [Feedback Engine]
                                              |           |
                                              +-----------+
                                                (retry loop)
```

---

## Module Overview

| Module | Responsibility |
|--------|---------------|
| `nodes.py` | 9 semantic node types: Module, Class, Function, Variable, Import, Decorator, Endpoint, Method, Parameter |
| `edges.py` | 12 edge types across 5 relationship categories with factory helpers |
| `graph.py` | `SemanticGraph`: directed graph with traversal, BFS path search, cycle detection, NetworkX export |
| `parser.py` | Multi-pass AST parser: complexity analysis, call extraction, endpoint detection |
| `patterns.py` | Pattern detection: naming, structure, dependencies, FastAPI, Pydantic, async conventions |
| `constraints.py` | Constraint definitions: Naming, Usage, Boundary, Error format; confidence and break cost scoring |
| `dsl.py` | JSON-based constraint serialization, deserialization, and merging |
| `compiler.py` | Prompt compiler: graph slicing and constraint-aware prompt generation |
| `llm.py` | Abstract LLM provider interface with deterministic testing stub |
| `validator.py` | Multi-layer validation: syntax, constraints, consistency, linting, testing |
| `pipeline.py` | End-to-end orchestration with retry logic |
| `feedback.py` | Feedback engine: violation diagnosis, categorization, escalating refinement prompts |
| `agent.py` | Code agent: observe, plan, act, validate, refine loop with history tracking |

---

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

Requirements: Python >= 3.10, pydantic >= 2.0, networkx >= 3.0

---

## Quick Start

### Parse a repository

```python
from codebase_intelligence import ASTParser

parser = ASTParser()
graph = parser.parse_directory("path/to/repo")

stats = graph.get_stats()
print(f"Nodes: {stats.node_count}, Edges: {stats.edge_count}")
```

### Extract patterns and compile constraints

```python
from codebase_intelligence import PatternExtractor, ConstraintCompiler

extractor = PatternExtractor(graph)
patterns = extractor.extract_all()

compiler = ConstraintCompiler()
constraint_set = compiler.compile_to_set(
    patterns,
    name="project_rules",
    description="Auto-detected constraints",
)
```

### Persist constraints using the DSL

```python
from codebase_intelligence import ConstraintDSL

dsl = ConstraintDSL()
dsl.save(constraint_set, "constraints.json")

loaded = dsl.load("constraints.json")
```

### Validate generated code

```python
from codebase_intelligence import CodeValidator

validator = CodeValidator()
result = validator.validate(
    source="def get_user():\n    pass\n",
    file_path="app/routes/users.py",
    constraints=constraint_set,
    original_graph=graph,
)

print(f"Valid: {result.is_valid}")
for violation in result.violations:
    print(violation.format_message())
```

### Run the pipeline

```python
from codebase_intelligence import Pipeline, PipelineConfig, StubLLMProvider

provider = StubLLMProvider(
    responses=["def get_user():\n    return User()\n"]
)

pipeline = Pipeline(
    llm=provider,
    config=PipelineConfig(max_retries=3),
)

graph, constraints = pipeline.analyze_repo("path/to/repo")
result = pipeline.generate("Add a get_user endpoint", graph, constraints)

print(f"Valid: {result.is_valid}, Attempts: {result.attempts}")
print(result.source)
```

### Run the agent (autonomous loop)

```python
from codebase_intelligence import CodeAgent, AgentConfig, StubLLMProvider

provider = StubLLMProvider(
    responses=["def get_user():\n    pass\n"]
)

agent = CodeAgent(
    llm=provider,
    config=AgentConfig(max_attempts=5),
)

result = agent.run(
    request="Add a get_user endpoint",
    graph=graph,
    constraints=constraint_set,
    file_path="app/routes/users.py",
)

print(f"Valid: {result.is_valid}, Attempts: {result.attempts}")
if not result.is_valid:
    for diagnosis in result.diagnoses:
        print(f"  [{diagnosis.category.value}] {diagnosis.suggestion}")
```

### Use the feedback engine directly

```python
from codebase_intelligence import FeedbackEngine, RefinementContext

engine = FeedbackEngine()
diagnoses = engine.diagnose(result.violations, constraint_set)

for d in diagnoses:
    print(f"[{d.category.value}] {d.root_cause} -> {d.suggestion}")

context = RefinementContext(
    original_request="Add endpoint",
    violations=result.violations,
    diagnoses=diagnoses,
    attempt=2,
    max_attempts=5,
)
prompt = engine.build_refinement(context)
```

---

## Constraint Types

### NamingConstraint

Enforces naming conventions using regex patterns.

```python
NamingConstraint(
    name="snake_case_functions",
    description="Functions must use snake_case",
    pattern=r"^[a-z_][a-z0-9_]*$",
    node_types=[NodeType.FUNCTION],
    severity=ConstraintSeverity.ERROR,
    confidence=0.95,
    break_cost=2.0,
)
```

### MustUseConstraint

Requires specific constructs on nodes (docstrings, decorators, type hints).

```python
MustUseConstraint(
    name="require_docstrings",
    description="Public functions must have docstrings",
    requirement="docstring",
    node_types=[NodeType.FUNCTION],
    exclude_private=True,
    exclude_dunder=True,
)
```

### MustNotCrossConstraint

Prevents imports across architectural boundaries.

```python
MustNotCrossConstraint(
    name="service_boundary",
    description="Services cannot import controllers",
    source_pattern=r".*/services/.*",
    forbidden_targets=[r".*/controllers/.*"],
)
```

### ErrorFormatConstraint

Enforces exception structure and inheritance.

```python
ErrorFormatConstraint(
    name="exception_naming",
    description="Exceptions must end with Error and extend BaseError",
    exception_pattern=r"^[A-Z].*Error$",
    required_bases=["BaseError"],
)
```

All constraints support `confidence` (0.0-1.0) and `break_cost` (penalty weight) for scoring.

---

## Validation Layers

The `CodeValidator` enforces five layers:

1. **Static** -- Syntax and parse validation
2. **Constraint** -- Enforced project rules
3. **Consistency** -- Conflict detection with existing code
4. **Linting** -- Ruff or equivalent
5. **Testing** -- Existing test suite execution

---

## Agent Loop

The `CodeAgent` implements a self-correcting generation cycle:

1. **Observe** -- Analyze the semantic graph and summarize constraints
2. **Plan** -- Determine generation strategy and select relevant context
3. **Act** -- Compile a prompt and call the LLM
4. **Validate** -- Check output against all five validation layers
5. **Refine** -- Diagnose violations, escalate feedback, and retry

Escalation levels increase across retries:

| Attempt | Level | Behavior |
|---------|-------|----------|
| 1st | Hint | Gentle suggestion to fix issues |
| Middle | Explicit | Direct instructions with root cause analysis |
| Final | Rewrite | Full rewrite instruction with all rules inlined |

---

## Feedback Engine

The `FeedbackEngine` categorizes violations into four types:

| Category | Description |
|----------|-------------|
| `NAMING` | Naming convention violations |
| `STRUCTURAL` | Missing required constructs |
| `BOUNDARY` | Forbidden cross-module dependencies |
| `ERROR_FORMAT` | Exception class naming or inheritance issues |

Each violation is diagnosed with a root cause, suggestion, and confidence score.

---

## Constraint DSL (JSON)

Constraints are portable and mergeable:

```json
{
  "name": "project_rules",
  "description": "Coding constraints",
  "version": "1.0.0",
  "constraints": [
    {
      "type": "NamingConstraint",
      "name": "snake_case_functions",
      "pattern": "^[a-z_][a-z0-9_]*$",
      "node_types": ["function"],
      "severity": "error",
      "confidence": 0.95,
      "break_cost": 2.0
    }
  ]
}
```

Later constraint sets override earlier ones on name collision.

---

## LLM Provider Interface

The system is model-agnostic:

```python
class MyProvider(LLMProvider):
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...
```

A stub provider is included for deterministic, offline testing.

---

## Testing

```bash
python -m pytest tests/ -v
```

- 1052 tests
- 100% line and branch coverage
- ~7 seconds runtime

---

## Project Structure

```
codebase_intelligence/
├── src/codebase_intelligence/
│   ├── nodes.py
│   ├── edges.py
│   ├── graph.py
│   ├── parser.py
│   ├── patterns.py
│   ├── constraints.py
│   ├── dsl.py
│   ├── compiler.py
│   ├── llm.py
│   ├── validator.py
│   ├── pipeline.py
│   ├── feedback.py
│   └── agent.py
├── tests/
│   ├── test_nodes.py
│   ├── test_edges.py
│   ├── test_graph.py
│   ├── test_parser.py
│   ├── test_patterns.py
│   ├── test_constraints.py
│   ├── test_dsl.py
│   ├── test_compiler.py
│   ├── test_llm.py
│   ├── test_validator.py
│   ├── test_pipeline.py
│   ├── test_feedback.py
│   └── test_agent.py
├── pyproject.toml
└── README.md
```

---

## License

MIT
