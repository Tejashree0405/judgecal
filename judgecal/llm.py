"""litellm wrapper with a SQLite disk cache, retries with backoff, and optional rate limiting."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
RETRYABLE_NAMES = {
    "RateLimitError",
    "ServiceUnavailableError",
    "InternalServerError",
    "APIConnectionError",
    "Timeout",
    "APITimeoutError",
}


class QuotaExhausted(Exception):
    """A per-day quota was hit; retrying within this run cannot succeed."""


def daily_quota_id(exc: Exception) -> str | None:
    m = re.search(r'"quotaId":\s*"([^"]*PerDay[^"]*)"', str(exc))
    return m.group(1) if m else None


def is_retryable(exc: Exception) -> bool:
    if daily_quota_id(exc):
        return False
    if type(exc).__name__ in RETRYABLE_NAMES:
        return True
    return getattr(exc, "status_code", None) in RETRYABLE_STATUS


class Cache:
    """Thread-safe key -> response text store."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT, response TEXT, created REAL)"
            )
            self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, model: str, response: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?)", (key, model, response, time.time())
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def cache_key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class CallResult:
    text: str
    cached: bool


def _litellm_completion(**kwargs):
    import litellm

    litellm.suppress_debug_info = True
    return litellm.completion(**kwargs)


class LLMClient:
    def __init__(
        self,
        cache: Cache,
        temperature: float = 0.0,
        max_retries: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        timeout: float = 180.0,
        rpm: float | None = None,
        completion_fn: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache = cache
        self.params = {"temperature": temperature}
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self._completion = completion_fn or _litellm_completion
        self._sleep = sleep
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._next_slot = 0.0
        self._rate_lock = threading.Lock()
        self.stats = {"cache_hits": 0, "api_calls": 0, "retries": 0}
        self._stats_lock = threading.Lock()

    def _bump(self, name: str) -> None:
        with self._stats_lock:
            self.stats[name] += 1

    def _wait_for_slot(self) -> None:
        if not self._min_interval:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            self._sleep(wait)

    def complete(self, model: str, messages: list[dict]) -> CallResult:
        key = cache_key(model, messages, self.params)
        hit = self.cache.get(key)
        if hit is not None:
            self._bump("cache_hits")
            return CallResult(hit, cached=True)

        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            try:
                self._bump("api_calls")
                resp = self._completion(
                    model=model, messages=messages, timeout=self.timeout, num_retries=0, **self.params
                )
                text = resp.choices[0].message.content or ""
                break
            except Exception as exc:
                quota = daily_quota_id(exc)
                if quota:
                    raise QuotaExhausted(f"{model}: daily quota exhausted ({quota})") from exc
                if not is_retryable(exc) or attempt == self.max_retries:
                    raise
                self._bump("retries")
                delay = min(self.max_delay, self.base_delay * 2**attempt)
                self._sleep(delay * (0.5 + random.random() / 2))

        self.cache.put(key, model, text)
        return CallResult(text, cached=False)
