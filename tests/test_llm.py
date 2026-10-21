from types import SimpleNamespace

import pytest

from judgecal.llm import Cache, LLMClient, QuotaExhausted, is_retryable


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeCompletion:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return response(out)


MSGS = [{"role": "user", "content": "hi"}]


def make_client(tmp_path, outcomes, **kw):
    fake = FakeCompletion(outcomes)
    sleeps = []
    client = LLMClient(Cache(tmp_path / "c.sqlite"), completion_fn=fake, sleep=sleeps.append, **kw)
    return client, fake, sleeps


def test_cache_prevents_second_call(tmp_path):
    client, fake, _ = make_client(tmp_path, ["[[A]]"])
    assert client.complete("m", MSGS).cached is False
    second = client.complete("m", MSGS)
    assert second.text == "[[A]]" and second.cached is True
    assert len(fake.calls) == 1


def test_cache_persists_across_clients(tmp_path):
    client, _, _ = make_client(tmp_path, ["[[B]]"])
    client.complete("m", MSGS)
    client.cache.close()
    client2, fake2, _ = make_client(tmp_path, [])
    assert client2.complete("m", MSGS).text == "[[B]]"
    assert fake2.calls == []


def test_cache_key_depends_on_model_and_messages(tmp_path):
    client, fake, _ = make_client(tmp_path, ["x", "y", "z"])
    client.complete("m1", MSGS)
    client.complete("m2", MSGS)
    client.complete("m1", [{"role": "user", "content": "other"}])
    assert len(fake.calls) == 3


def test_retries_with_backoff_then_succeeds(tmp_path):
    client, fake, sleeps = make_client(tmp_path, [RateLimitError(), RateLimitError(), "[[A]]"], base_delay=1.0)
    assert client.complete("m", MSGS).text == "[[A]]"
    assert len(fake.calls) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.9  # exponential, with jitter
    assert client.stats["retries"] == 2


def test_gives_up_after_max_retries(tmp_path):
    client, fake, _ = make_client(tmp_path, [RateLimitError()] * 3, max_retries=2)
    with pytest.raises(RateLimitError):
        client.complete("m", MSGS)
    assert len(fake.calls) == 3


def test_failed_calls_are_not_cached(tmp_path):
    client, fake, _ = make_client(tmp_path, [RateLimitError(), "[[B]]"], max_retries=0)
    with pytest.raises(RateLimitError):
        client.complete("m", MSGS)
    assert client.complete("m", MSGS).text == "[[B]]"


def test_non_retryable_raises_immediately(tmp_path):
    client, fake, sleeps = make_client(tmp_path, [AuthenticationError()])
    with pytest.raises(AuthenticationError):
        client.complete("m", MSGS)
    assert len(fake.calls) == 1 and sleeps == []


def test_daily_quota_is_not_retried(tmp_path):
    exc = RateLimitError('{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}')
    client, fake, sleeps = make_client(tmp_path, [exc])
    with pytest.raises(QuotaExhausted, match="PerDay"):
        client.complete("m", MSGS)
    assert len(fake.calls) == 1 and sleeps == []


def test_is_retryable_by_name():
    ServiceUnavailableError = type("ServiceUnavailableError", (Exception,), {})
    assert is_retryable(ServiceUnavailableError())
    assert not is_retryable(ValueError())
