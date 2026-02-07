"""Tests for LLM interface abstraction."""

from __future__ import annotations

import pytest

from codebase_intelligence.llm import (
    FinishReason,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MessageRole,
    StubLLMProvider,
    TokenUsage,
)


# -- Helpers ----------------------------------------------------------------


class ConcreteLLMProvider(LLMProvider):
    """Minimal concrete subclass for testing the abstract base class."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="test", model=self.model_name)


# -- MessageRole enum ------------------------------------------------------


class TestMessageRole:
    def test_all_values(self) -> None:
        assert {r.value for r in MessageRole} == {"system", "user", "assistant"}

    def test_system(self) -> None:
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.SYSTEM.value == "system"

    def test_user(self) -> None:
        assert MessageRole.USER == "user"
        assert MessageRole.USER.value == "user"

    def test_assistant(self) -> None:
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_is_str_subclass(self) -> None:
        assert isinstance(MessageRole.SYSTEM, str)


# -- LLMMessage -------------------------------------------------------------


class TestLLMMessage:
    def test_system_message(self) -> None:
        msg = LLMMessage(role=MessageRole.SYSTEM, content="You are helpful.")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful."

    def test_user_message(self) -> None:
        msg = LLMMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_assistant_message(self) -> None:
        msg = LLMMessage(role=MessageRole.ASSISTANT, content="Hi there")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there"

    def test_frozen(self) -> None:
        msg = LLMMessage(role=MessageRole.USER, content="test")
        with pytest.raises(AttributeError):
            msg.content = "changed"  # type: ignore[misc]

    def test_empty_content(self) -> None:
        msg = LLMMessage(role=MessageRole.USER, content="")
        assert msg.content == ""


# -- TokenUsage --------------------------------------------------------------


class TestTokenUsage:
    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_total_tokens_with_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.total_tokens == 0

    def test_total_tokens_with_values(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_total_tokens_prompt_only(self) -> None:
        usage = TokenUsage(prompt_tokens=42)
        assert usage.total_tokens == 42

    def test_total_tokens_completion_only(self) -> None:
        usage = TokenUsage(completion_tokens=33)
        assert usage.total_tokens == 33

    def test_frozen(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
        with pytest.raises(AttributeError):
            usage.prompt_tokens = 99  # type: ignore[misc]

    def test_custom_values(self) -> None:
        usage = TokenUsage(prompt_tokens=500, completion_tokens=200)
        assert usage.prompt_tokens == 500
        assert usage.completion_tokens == 200


# -- FinishReason enum -------------------------------------------------------


class TestFinishReason:
    def test_all_values(self) -> None:
        assert {r.value for r in FinishReason} == {"stop", "length", "error"}

    def test_stop(self) -> None:
        assert FinishReason.STOP == "stop"
        assert FinishReason.STOP.value == "stop"

    def test_length(self) -> None:
        assert FinishReason.LENGTH == "length"
        assert FinishReason.LENGTH.value == "length"

    def test_error(self) -> None:
        assert FinishReason.ERROR == "error"
        assert FinishReason.ERROR.value == "error"

    def test_is_str_subclass(self) -> None:
        assert isinstance(FinishReason.STOP, str)


# -- LLMRequest ---------------------------------------------------------------


class TestLLMRequest:
    def test_defaults(self) -> None:
        req = LLMRequest()
        assert req.messages == ()
        assert req.max_tokens == 4096
        assert req.temperature == 0.2
        assert req.stop_sequences == ()

    def test_custom_values(self) -> None:
        msgs = (
            LLMMessage(role=MessageRole.SYSTEM, content="Be concise."),
            LLMMessage(role=MessageRole.USER, content="Hi"),
        )
        req = LLMRequest(
            messages=msgs,
            max_tokens=1024,
            temperature=0.8,
            stop_sequences=("###", "END"),
        )
        assert len(req.messages) == 2
        assert req.messages[0].role == MessageRole.SYSTEM
        assert req.messages[1].content == "Hi"
        assert req.max_tokens == 1024
        assert req.temperature == 0.8
        assert req.stop_sequences == ("###", "END")

    def test_empty_messages(self) -> None:
        req = LLMRequest(messages=())
        assert req.messages == ()
        assert len(req.messages) == 0

    def test_frozen(self) -> None:
        req = LLMRequest()
        with pytest.raises(AttributeError):
            req.max_tokens = 100  # type: ignore[misc]


# -- LLMResponse --------------------------------------------------------------


class TestLLMResponse:
    def test_defaults(self) -> None:
        resp = LLMResponse(content="hello")
        assert resp.content == "hello"
        assert resp.model == ""
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0
        assert resp.finish_reason == FinishReason.STOP

    def test_custom_values(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
        resp = LLMResponse(
            content="generated code",
            model="gpt-4",
            usage=usage,
            finish_reason=FinishReason.LENGTH,
        )
        assert resp.content == "generated code"
        assert resp.model == "gpt-4"
        assert resp.usage.total_tokens == 30
        assert resp.finish_reason == FinishReason.LENGTH

    def test_error_finish_reason(self) -> None:
        resp = LLMResponse(
            content="",
            model="test-model",
            finish_reason=FinishReason.ERROR,
        )
        assert resp.finish_reason == FinishReason.ERROR

    def test_frozen(self) -> None:
        resp = LLMResponse(content="test")
        with pytest.raises(AttributeError):
            resp.content = "changed"  # type: ignore[misc]


# -- LLMError ------------------------------------------------------------------


class TestLLMError:
    def test_default_not_retryable(self) -> None:
        err = LLMError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.retryable is False

    def test_retryable(self) -> None:
        err = LLMError("Rate limited", retryable=True)
        assert str(err) == "Rate limited"
        assert err.retryable is True

    def test_not_retryable_explicit(self) -> None:
        err = LLMError("Bad request", retryable=False)
        assert err.retryable is False

    def test_is_exception(self) -> None:
        err = LLMError("fail")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(LLMError, match="boom"):
            raise LLMError("boom")

    def test_message_access_via_args(self) -> None:
        err = LLMError("test message")
        assert err.args[0] == "test message"


# -- LLMProvider (abstract, tested via ConcreteLLMProvider) ------------------


class TestLLMProvider:
    def test_model_name_property(self) -> None:
        provider = ConcreteLLMProvider(model_name="gpt-4", max_context_tokens=128000)
        assert provider.model_name == "gpt-4"

    def test_max_context_tokens_property(self) -> None:
        provider = ConcreteLLMProvider(model_name="gpt-4", max_context_tokens=128000)
        assert provider.max_context_tokens == 128000

    def test_default_chars_per_token(self) -> None:
        provider = ConcreteLLMProvider(model_name="m", max_context_tokens=1000)
        # Default chars_per_token is 4, so 8 chars -> 2 tokens
        assert provider.estimate_tokens("12345678") == 2

    def test_custom_chars_per_token(self) -> None:
        provider = ConcreteLLMProvider(
            model_name="m", max_context_tokens=1000, chars_per_token=2,
        )
        assert provider.estimate_tokens("12345678") == 4

    def test_estimate_tokens_empty_string(self) -> None:
        provider = ConcreteLLMProvider(model_name="m", max_context_tokens=1000)
        # Empty string: len("") // 4 = 0, max(1, 0) = 1
        assert provider.estimate_tokens("") == 1

    def test_estimate_tokens_short_string(self) -> None:
        provider = ConcreteLLMProvider(model_name="m", max_context_tokens=1000)
        # "ab" -> len=2, 2 // 4 = 0, max(1, 0) = 1
        assert provider.estimate_tokens("ab") == 1

    def test_estimate_tokens_exact_multiple(self) -> None:
        provider = ConcreteLLMProvider(model_name="m", max_context_tokens=1000)
        # 12 chars / 4 = 3 tokens
        assert provider.estimate_tokens("a" * 12) == 3

    def test_estimate_tokens_large_text(self) -> None:
        provider = ConcreteLLMProvider(model_name="m", max_context_tokens=1000)
        # 400 chars / 4 = 100 tokens
        assert provider.estimate_tokens("x" * 400) == 100

    def test_fits_context_fits(self) -> None:
        provider = ConcreteLLMProvider(model_name="m", max_context_tokens=10000)
        msg = LLMMessage(role=MessageRole.USER, content="Hello world")
        req = LLMRequest(messages=(msg,), max_tokens=100)
        assert provider.fits_context(req) is True

    def test_fits_context_does_not_fit(self) -> None:
        provider = ConcreteLLMProvider(model_name="m", max_context_tokens=10)
        # "a" * 40 => 40 // 4 = 10 tokens; 10 + 4096 (default max_tokens) > 10
        msg = LLMMessage(role=MessageRole.USER, content="a" * 40)
        req = LLMRequest(messages=(msg,))
        assert provider.fits_context(req) is False

    def test_fits_context_exact_boundary(self) -> None:
        provider = ConcreteLLMProvider(
            model_name="m", max_context_tokens=200, chars_per_token=4,
        )
        # 400 chars / 4 = 100 tokens; 100 + 100 max_tokens = 200 == context limit
        msg = LLMMessage(role=MessageRole.USER, content="a" * 400)
        req = LLMRequest(messages=(msg,), max_tokens=100)
        assert provider.fits_context(req) is True

    def test_fits_context_just_over(self) -> None:
        provider = ConcreteLLMProvider(
            model_name="m", max_context_tokens=200, chars_per_token=4,
        )
        # 404 chars / 4 = 101 tokens; 101 + 100 = 201 > 200
        msg = LLMMessage(role=MessageRole.USER, content="a" * 404)
        req = LLMRequest(messages=(msg,), max_tokens=100)
        assert provider.fits_context(req) is False

    def test_fits_context_empty_messages(self) -> None:
        provider = ConcreteLLMProvider(model_name="m", max_context_tokens=10000)
        req = LLMRequest(messages=(), max_tokens=100)
        # 0 + 100 = 100 <= 10000
        assert provider.fits_context(req) is True

    def test_fits_context_multiple_messages(self) -> None:
        provider = ConcreteLLMProvider(
            model_name="m", max_context_tokens=100, chars_per_token=4,
        )
        msgs = (
            LLMMessage(role=MessageRole.SYSTEM, content="a" * 100),  # 25 tokens
            LLMMessage(role=MessageRole.USER, content="b" * 100),    # 25 tokens
        )
        # 50 tokens + 40 max_tokens = 90 <= 100
        req = LLMRequest(messages=msgs, max_tokens=40)
        assert provider.fits_context(req) is True

    def test_complete_returns_response(self) -> None:
        provider = ConcreteLLMProvider(model_name="test-model", max_context_tokens=1000)
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="hi"),),
        )
        resp = provider.complete(req)
        assert resp.content == "test"
        assert resp.model == "test-model"

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider(model_name="m", max_context_tokens=100)  # type: ignore[abstract]


# -- StubLLMProvider -----------------------------------------------------------


class TestStubLLMProvider:
    def test_default_response(self) -> None:
        provider = StubLLMProvider()
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="hello"),),
        )
        resp = provider.complete(req)
        assert resp.content == "# generated code"
        assert resp.model == "stub-model"
        assert resp.finish_reason == FinishReason.STOP

    def test_default_model_name(self) -> None:
        provider = StubLLMProvider()
        assert provider.model_name == "stub-model"

    def test_default_max_context_tokens(self) -> None:
        provider = StubLLMProvider()
        assert provider.max_context_tokens == 128000

    def test_custom_responses(self) -> None:
        provider = StubLLMProvider(responses=["def foo(): pass", "def bar(): pass"])
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="code"),),
        )
        resp1 = provider.complete(req)
        resp2 = provider.complete(req)
        assert resp1.content == "def foo(): pass"
        assert resp2.content == "def bar(): pass"

    def test_response_cycling(self) -> None:
        provider = StubLLMProvider(responses=["first", "second"])
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="x"),),
        )
        r1 = provider.complete(req)
        r2 = provider.complete(req)
        r3 = provider.complete(req)  # Should cycle back to "first"
        r4 = provider.complete(req)  # Should cycle back to "second"
        assert r1.content == "first"
        assert r2.content == "second"
        assert r3.content == "first"
        assert r4.content == "second"

    def test_call_count(self) -> None:
        provider = StubLLMProvider()
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="hi"),),
        )
        assert provider.call_count == 0
        provider.complete(req)
        assert provider.call_count == 1
        provider.complete(req)
        assert provider.call_count == 2

    def test_requests_recording(self) -> None:
        provider = StubLLMProvider()
        req1 = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="first"),),
        )
        req2 = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="second"),),
            max_tokens=512,
        )
        provider.complete(req1)
        provider.complete(req2)
        assert len(provider.requests) == 2
        assert provider.requests[0] is req1
        assert provider.requests[1] is req2

    def test_error_raising(self) -> None:
        error = LLMError("API down", retryable=True)
        provider = StubLLMProvider(error=error)
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="hi"),),
        )
        with pytest.raises(LLMError, match="API down") as exc_info:
            provider.complete(req)
        assert exc_info.value.retryable is True

    def test_error_still_records_request_and_increments_count(self) -> None:
        error = LLMError("fail")
        provider = StubLLMProvider(error=error)
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="hi"),),
        )
        with pytest.raises(LLMError):
            provider.complete(req)
        assert provider.call_count == 1
        assert len(provider.requests) == 1
        assert provider.requests[0] is req

    def test_token_estimation_in_response(self) -> None:
        provider = StubLLMProvider(responses=["abcdefghijklmnop"])  # 16 chars
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="a" * 20),),
        )
        resp = provider.complete(req)
        # prompt_tokens: 20 // 4 = 5
        assert resp.usage.prompt_tokens == 5
        # completion_tokens: 16 // 4 = 4
        assert resp.usage.completion_tokens == 4
        assert resp.usage.total_tokens == 9

    def test_token_estimation_multiple_messages(self) -> None:
        provider = StubLLMProvider(responses=["ok"])
        msgs = (
            LLMMessage(role=MessageRole.SYSTEM, content="a" * 40),   # 10 tokens
            LLMMessage(role=MessageRole.USER, content="b" * 20),     # 5 tokens
        )
        req = LLMRequest(messages=msgs)
        resp = provider.complete(req)
        assert resp.usage.prompt_tokens == 15

    def test_token_estimation_empty_prompt(self) -> None:
        provider = StubLLMProvider(responses=["result"])
        req = LLMRequest(messages=())
        resp = provider.complete(req)
        assert resp.usage.prompt_tokens == 0

    def test_custom_model_name(self) -> None:
        provider = StubLLMProvider(model_name="custom-model")
        assert provider.model_name == "custom-model"
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="hi"),),
        )
        resp = provider.complete(req)
        assert resp.model == "custom-model"

    def test_custom_max_context_tokens(self) -> None:
        provider = StubLLMProvider(max_context_tokens=4096)
        assert provider.max_context_tokens == 4096

    def test_single_response_cycles(self) -> None:
        provider = StubLLMProvider(responses=["only one"])
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="x"),),
        )
        r1 = provider.complete(req)
        r2 = provider.complete(req)
        assert r1.content == "only one"
        assert r2.content == "only one"

    def test_none_responses_uses_default(self) -> None:
        provider = StubLLMProvider(responses=None)
        req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="x"),),
        )
        resp = provider.complete(req)
        assert resp.content == "# generated code"

    def test_fits_context_via_stub(self) -> None:
        provider = StubLLMProvider(max_context_tokens=100)
        small_req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="hi"),),
            max_tokens=10,
        )
        assert provider.fits_context(small_req) is True

        big_req = LLMRequest(
            messages=(LLMMessage(role=MessageRole.USER, content="x" * 10000),),
            max_tokens=10,
        )
        assert provider.fits_context(big_req) is False

    def test_estimate_tokens_via_stub(self) -> None:
        provider = StubLLMProvider()
        assert provider.estimate_tokens("") == 1
        assert provider.estimate_tokens("abcd") == 1
        assert provider.estimate_tokens("a" * 8) == 2
