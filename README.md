<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&amp;logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

# Codebase Intelligence

**AI-powered codebase analysis with semantic search, dependency mapping, and automated documentation.**

</div>

## What It Does

Point it at a codebase and it builds a semantic understanding of what the code does, how the parts relate to each other, and what the documentation gaps are.

## Capabilities

**Semantic Search**
Ask questions in natural language. The system embeds all code into a vector store and retrieves semantically relevant functions, classes, and modules — not just keyword matches.

```bash
codebase-intel search "where is authentication handled?"
# Returns: auth/middleware.py:AuthMiddleware, auth/jwt.py:verify_token, ...
```

**Dependency Analysis**
Builds a complete dependency graph of your codebase. Identifies circular dependencies, unused modules, and high-coupling hotspots.

**Automated Documentation**
Generates docstrings for undocumented functions using the surrounding code as context. Reviews existing docs for accuracy drift.

**Change Impact Analysis**
Given a diff, predicts which other parts of the codebase are likely affected. Reduces surprise breakages during refactors.

## Quick Start

```bash
pip install codebase-intelligence
codebase-intel index ./my-project
codebase-intel search "rate limiting logic"
codebase-intel deps --show-circular
codebase-intel docs --generate-missing
```

## Architecture

```
Source Code
     |
     v
[Parser]        AST extraction for Python, Go, TypeScript
     |
     v
[Embedder]      Code chunks embedded via OpenAI / local model
     |
     v
[Vector Store]  ChromaDB for semantic retrieval
     |
     v
[Graph Store]   NetworkX for dependency relationships
     |
     v
[CLI / API]     Query interface
```

## License

MIT
