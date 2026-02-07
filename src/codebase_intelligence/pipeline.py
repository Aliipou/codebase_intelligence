"""End-to-end pipeline for constraint-aware code generation.

Orchestrates the full workflow:
1. Analyze repository → build semantic graph
2. Extract patterns → compile constraints
3. Accept task → compile prompt
4. Call LLM → validate output
5. Retry on validation failure

Usage:
    >>> from codebase_intelligence.pipeline import Pipeline, PipelineConfig
    >>> from codebase_intelligence.llm import StubLLMProvider
    >>>
    >>> provider = StubLLMProvider(responses=["def hello(): pass"])
    >>> pipeline = Pipeline(llm=provider)
    >>> graph, constraints = pipeline.analyze_repo("path/to/repo")
    >>> result = pipeline.generate("Add a hello function", graph, constraints)
    >>> print(result.is_valid)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codebase_intelligence.compiler import CompiledPrompt, PromptCompiler
from codebase_intelligence.constraints import ConstraintSet
from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.llm import (
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MessageRole,
)
from codebase_intelligence.parser import ASTParser
from codebase_intelligence.patterns import PatternExtractor
from codebase_intelligence.constraints import ConstraintCompiler
from codebase_intelligence.validator import CodeValidator, ValidationResult


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the generation pipeline.

    Attributes:
        max_retries: Maximum generation attempts on validation failure.
        temperature: LLM sampling temperature.
        max_tokens: Max tokens for LLM response.
        validate_output: Whether to validate generated code.
        relevant_files: Files to focus context on (None = all).
    """

    max_retries: int = 3
    temperature: float = 0.2
    max_tokens: int = 4096
    validate_output: bool = True
    relevant_files: list[str] | None = None


@dataclass
class GenerationResult:
    """Result of a code generation pipeline run.

    Attributes:
        source: The generated source code.
        validation: Validation result (if validation enabled).
        attempts: Number of generation attempts.
        prompt: The compiled prompt used.
        llm_response: Raw LLM response from the final attempt.
        is_valid: Whether the generated code passed validation.
    """

    source: str = ""
    validation: ValidationResult | None = None
    attempts: int = 0
    prompt: CompiledPrompt | None = None
    llm_response: LLMResponse | None = None
    is_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "source": self.source,
            "is_valid": self.is_valid,
            "attempts": self.attempts,
            "validation": self.validation.to_dict() if self.validation else None,
        }


class PipelineError(Exception):
    """Raised when the pipeline encounters an unrecoverable error."""


class Pipeline:
    """End-to-end code generation pipeline.

    Orchestrates: parse → patterns → constraints → prompt → LLM → validate.

    Examples:
        >>> pipeline = Pipeline(llm=provider)
        >>> graph, constraints = pipeline.analyze_repo("./my_project")
        >>> result = pipeline.generate(
        ...     task="Add a health check endpoint",
        ...     graph=graph,
        ...     constraints=constraints,
        ... )
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            llm: LLM provider for code generation.
            config: Pipeline configuration.
        """
        self._llm = llm
        self._config = config or PipelineConfig()
        self._parser = ASTParser()
        self._compiler = PromptCompiler()
        self._validator = CodeValidator()
        self._constraint_compiler = ConstraintCompiler()

    @property
    def config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._config

    def analyze_repo(
        self,
        path: str | Path,
        recursive: bool = True,
    ) -> tuple[SemanticGraph, ConstraintSet]:
        """Analyze a repository and extract constraints.

        Args:
            path: Path to the repository root.
            recursive: Whether to parse subdirectories.

        Returns:
            Tuple of (semantic graph, compiled constraint set).

        Raises:
            PipelineError: If the repository cannot be parsed.
        """
        path = Path(path)

        if not path.is_dir():
            raise PipelineError(f"Not a directory: {path}")

        try:
            graph = self._parser.parse_directory(str(path), recursive=recursive)
        except Exception as e:
            raise PipelineError(f"Failed to parse repository: {e}") from e

        extractor = PatternExtractor(graph)
        patterns = extractor.extract_all()
        constraints = self._constraint_compiler.compile_to_set(
            patterns,
            name=f"{path.name}_constraints",
            description=f"Auto-extracted constraints for {path.name}",
        )

        return graph, constraints

    def generate(
        self,
        task: str,
        graph: SemanticGraph,
        constraints: ConstraintSet,
        relevant_files: list[str] | None = None,
    ) -> GenerationResult:
        """Generate code for a task with constraint enforcement.

        Args:
            task: Natural language task description.
            graph: Semantic graph of the codebase.
            constraints: Active constraints.
            relevant_files: Files to focus context on.

        Returns:
            GenerationResult with generated code and validation.
        """
        result = GenerationResult()
        files = relevant_files or self._config.relevant_files

        # Compile prompt
        prompt = self._compiler.compile(task, graph, constraints, files)
        result.prompt = prompt

        # Build LLM request
        request = self._build_request(prompt)

        # Generation loop with retries
        max_attempts = self._config.max_retries if self._config.validate_output else 1

        for attempt in range(1, max_attempts + 1):
            result.attempts = attempt

            try:
                response = self._llm.complete(request)
            except LLMError as e:
                if not e.retryable or attempt == max_attempts:
                    result.source = ""
                    result.is_valid = False
                    return result
                continue

            result.source = response.content
            result.llm_response = response

            if not self._config.validate_output:
                result.is_valid = True
                return result

            # Validate
            file_path = self._infer_file_path(task, files)
            validation = self._validator.validate(
                source=response.content,
                file_path=file_path,
                constraints=constraints,
                original_graph=graph,
            )
            result.validation = validation

            if validation.is_valid:
                result.is_valid = True
                return result

            # Retry with violation feedback
            if attempt < max_attempts:
                request = self._build_retry_request(prompt, response.content, validation)

        return result

    def _build_request(self, prompt: CompiledPrompt) -> LLMRequest:
        """Build an LLM request from a compiled prompt."""
        messages = (
            LLMMessage(role=MessageRole.SYSTEM, content=prompt.system_message()),
            LLMMessage(role=MessageRole.USER, content=prompt.user_message()),
        )
        return LLMRequest(
            messages=messages,
            max_tokens=prompt.max_tokens,
            temperature=self._config.temperature,
        )

    def _build_retry_request(
        self,
        prompt: CompiledPrompt,
        previous_output: str,
        validation: ValidationResult,
    ) -> LLMRequest:
        """Build a retry request with violation feedback."""
        violation_text = "\n".join(
            f"- {v.format_message()}" for v in validation.violations
        )
        feedback = (
            f"Your previous output had constraint violations:\n"
            f"{violation_text}\n\n"
            f"Fix ALL violations and regenerate the code."
        )

        messages = (
            LLMMessage(role=MessageRole.SYSTEM, content=prompt.system_message()),
            LLMMessage(role=MessageRole.USER, content=prompt.user_message()),
            LLMMessage(role=MessageRole.ASSISTANT, content=previous_output),
            LLMMessage(role=MessageRole.USER, content=feedback),
        )

        return LLMRequest(
            messages=messages,
            max_tokens=prompt.max_tokens,
            temperature=self._config.temperature,
        )

    def _infer_file_path(
        self,
        task: str,
        relevant_files: list[str] | None,
    ) -> str:
        """Infer the output file path from context."""
        if relevant_files and len(relevant_files) == 1:
            return relevant_files[0]
        return "generated.py"
