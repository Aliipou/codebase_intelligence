<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

# codebase-intelligence

**Constraint-aware code generation: semantic graph + LLM pipeline.**

</div>

## What This Is

A Python library that builds a semantic model of a codebase and uses it to constrain LLM code generation. It is **not** a CLI tool or a vector-search product — it is a library you embed in your own tooling.

The core loop:

1. Parse a Python repo into a `SemanticGraph` (nodes = modules/classes/functions, edges = imports/calls/containment)
2. Extract structural patterns and compile them into a `ConstraintSet`
3. Compile a task description + constraints into a grounded LLM prompt
4. Call an LLM provider (or the included `StubLLMProvider` for tests)
5. Validate the generated code against the constraints; retry up to `max_retries` on violation

## What Is Implemented

| Component | Status |
|-----------|--------|
| `SemanticGraph` — directed graph with NetworkX | Done |
| `ASTParser` — Python AST → graph nodes/edges | Done |
| `PatternExtractor` — naming, error format, layer patterns | Done |
| `ConstraintSet` / constraint DSL (JSON serialization) | Done |
| `ConstraintCompiler` — turns graph + patterns into constraints | Done |
| `PromptCompiler` — injects constraints into LLM prompt | Done |
| `Pipeline` — full analyze → generate → validate loop | Done |
| `FeedbackEngine` — violation diagnosis + refinement context | Done |
| `CodeAgent` — multi-step agent over the pipeline | Done |
| HTTP API / CLI | **Not yet built** |
| Embedding / vector search | **Not yet built** |
| Go / TypeScript parser | **Not yet built** |

## Install

```bash
pip install codebase-intelligence
```

## Usage

```python
from codebase_intelligence import Pipeline, StubLLMProvider

# Point at a real LLM provider (OpenAI, Anthropic, etc.)
# StubLLMProvider is for tests only.
provider = StubLLMProvider(responses=["def hello(): return 'hello'"])
pipeline = Pipeline(llm=provider)

graph, constraints = pipeline.analyze_repo("path/to/your/repo")
result = pipeline.generate("Add a hello function", graph, constraints)

print(result.is_valid)        # True/False
print(result.generated_code)  # the output
print(result.violations)      # ConstraintViolation list if invalid
```

## Constraint DSL

Constraints are expressed in JSON and can be saved/loaded:

```python
from codebase_intelligence import ConstraintDSL, ConstraintSet, MustNotCrossConstraint

cs = ConstraintSet(name="my-project", constraints=[
    MustNotCrossConstraint(from_layer="api", to_layer="db"),
])
ConstraintDSL.save(cs, "constraints.json")

# Later:
cs2 = ConstraintDSL.load("constraints.json")
```

## Architecture

```
Repo on disk
    |
    v
[ASTParser]           Python AST → SemanticGraph nodes + edges
    |
    v
[PatternExtractor]    Structural patterns from graph
    |
    v
[ConstraintCompiler]  Patterns → ConstraintSet
    |
    v
[PromptCompiler]      Task description + constraints → LLM prompt
    |
    v
[LLMProvider]         Any provider implementing LLMProvider ABC
    |
    v
[CodeValidator]       Lint + test run + constraint check
    |
    v (on violation, up to max_retries)
[FeedbackEngine]      Diagnose violation → RefinementContext → retry
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

100% branch coverage enforced.

## License

MIT
