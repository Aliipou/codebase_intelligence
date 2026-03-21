# Contributing to codebase_intelligence

## Setup

```bash
git clone https://github.com/Aliipou/codebase_intelligence.git
cd codebase_intelligence
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --cov=codebase_intelligence
```

## Adding a New Language Parser

1. Create `parsers/your_language.py` implementing the `BaseParser` interface
2. Register the parser in `parsers/__init__.py`
3. Add test fixtures in `tests/fixtures/your_language/`
4. Write tests covering common patterns in that language

## Commit Messages

`feat:`, `fix:`, `docs:`, `test:`, `chore:`
