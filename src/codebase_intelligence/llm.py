"""Abstract LLM interface for code generation.

Provides a model-agnostic abstraction over chat-completion LLMs.
The system is designed so that swapping one LLM for another changes
nothing about the constraint enforcement or validation logic.

Implementations must subclass LLMProvider and implement the
complete() method.

Usage:
    >>> class MyProvider(LLMProvider):
    ...     def complete(self, request: LLMRequest) -> LLMResponse:
    ...         # Call your LLM API here
    ...         ...
    >>>
    >>> provider = MyProvider(model="gpt-4", max_context=128000)
    >>> response = provider.complete(request)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class LLMMessage:
    """A single message in a conversation.

    Attributes:
        role: Who sent the message.
        content: The message text.
    """

    role: MessageRole
    content: str


@dataclass(frozen=True)
class TokenUsage:
    """Token usage statistics for an LLM call.

    Attributes:
        prompt_tokens: Tokens used by the prompt.
        completion_tokens: Tokens used by the completion.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens


class FinishReason(str, Enum):
    """Reason the LLM stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"


@dataclass(frozen=True)
class LLMRequest:
    """A request to an LLM provider.

    Attributes:
        messages: Conversation messages.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic).
        stop_sequences: Sequences that stop generation.
    """

    messages: tuple[LLMMessage, ...] = ()
    max_tokens: int = 4096
    temperature: float = 0.2
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMResponse:
    """A response from an LLM provider.

    Attributes:
        content: The generated text.
        model: Model identifier that produced this response.
        usage: Token usage statistics.
        finish_reason: Why generation stopped.
    """

    content: str
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: FinishReason = FinishReason.STOP


class LLMError(Exception):
    """Raised when an LLM call fails.

    Attributes:
        message: Error description.
        retryable: Whether the error is transient and retryable.
    """

    def __init__(self, message: str, retryable: bool = False) -> None:
        self.retryable = retryable
        super().__init__(message)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclass this and implement complete() to integrate any
    chat-completion LLM.

    Attributes:
        _model_name: Identifier for the LLM model.
        _max_context_tokens: Maximum context window size.
        _chars_per_token: Character-to-token estimation ratio.

    Examples:
        >>> class OpenAIProvider(LLMProvider):
        ...     def complete(self, request):
        ...         # Call OpenAI API
        ...         return LLMResponse(content="...", model=self.model_name)
    """

    def __init__(
        self,
        model_name: str,
        max_context_tokens: int,
        chars_per_token: int = 4,
    ) -> None:
        """Initialize the LLM provider.

        Args:
            model_name: Identifier for the model.
            max_context_tokens: Maximum context window size in tokens.
            chars_per_token: Rough character-to-token ratio.
        """
        self._model_name = model_name
        self._max_context_tokens = max_context_tokens
        self._chars_per_token = chars_per_token

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    @property
    def max_context_tokens(self) -> int:
        """Return the maximum context window size."""
        return self._max_context_tokens

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to the LLM.

        Args:
            request: The completion request.

        Returns:
            The LLM response.

        Raises:
            LLMError: If the request fails.
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count (minimum 1).
        """
        return max(1, len(text) // self._chars_per_token)

    def fits_context(self, request: LLMRequest) -> bool:
        """Check if a request fits within the context window.

        Args:
            request: The request to check.

        Returns:
            True if the request fits.
        """
        total = sum(self.estimate_tokens(m.content) for m in request.messages)
        return total + request.max_tokens <= self._max_context_tokens


class StubLLMProvider(LLMProvider):
    """A stub LLM provider for testing.

    Returns pre-configured responses without making any API calls.

    Examples:
        >>> provider = StubLLMProvider(responses=["def hello(): pass"])
        >>> response = provider.complete(request)
        >>> print(response.content)
        def hello(): pass
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        model_name: str = "stub-model",
        max_context_tokens: int = 128000,
        error: LLMError | None = None,
    ) -> None:
        """Initialize stub provider.

        Args:
            responses: Pre-configured responses to return in order.
            model_name: Model name to report.
            max_context_tokens: Reported context window size.
            error: If set, complete() will raise this error.
        """
        super().__init__(model_name, max_context_tokens)
        self._responses = list(responses or ["# generated code"])
        self._call_count = 0
        self._error = error
        self._requests: list[LLMRequest] = []

    @property
    def call_count(self) -> int:
        """Number of times complete() was called."""
        return self._call_count

    @property
    def requests(self) -> list[LLMRequest]:
        """All requests received."""
        return self._requests

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next pre-configured response.

        Args:
            request: The completion request (recorded but not sent).

        Returns:
            Pre-configured response.

        Raises:
            LLMError: If error was configured.
        """
        self._requests.append(request)
        self._call_count += 1

        if self._error:
            raise self._error

        # Cycle through responses
        idx = (self._call_count - 1) % len(self._responses)
        content = self._responses[idx]

        prompt_tokens = sum(self.estimate_tokens(m.content) for m in request.messages)

        return LLMResponse(
            content=content,
            model=self._model_name,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=self.estimate_tokens(content),
            ),
            finish_reason=FinishReason.STOP,
        )
