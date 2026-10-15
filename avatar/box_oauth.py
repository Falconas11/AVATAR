"""Server-side Box OAuth2 refresh with mandatory durable token persistence.

Initial authorization is performed separately using Box's OAuth authorization-code flow.
Store the initial access/refresh token pair in a durable encrypted secret store.
Never expose tokens in API responses, logs, GitHub, or browser JavaScript.

The supplied token_store must provide:
    load() -> dict containing access_token, refresh_token, expires_at (Unix seconds)
    save(tokens: dict) -> None, durably persisting the *entire* new token pair
A distributed deployment also needs an inter-process lock or a transactional store.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol

TOKEN_URL = "https://api.box.com/oauth2/token"


class BoxOAuthError(RuntimeError):
    pass


class TokenStore(Protocol):
    def load(self) -> dict: ...
    def save(self, tokens: dict) -> None: ...


class BoxOAuthClient:
    def __init__(self, token_store: TokenStore, *, client_id: str | None = None,
                 client_secret: str | None = None, timeout: int = 20):
        self.store = token_store
        self.client_id = client_id or os.environ.get("BOX_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("BOX_CLIENT_SECRET")
        self.timeout = timeout
        self._lock = threading.Lock()
        if not self.client_id or not self.client_secret:
            raise BoxOAuthError("BOX_CLIENT_ID and BOX_CLIENT_SECRET are required")

    def _refresh(self, current: dict) -> dict:
        refresh_token = current.get("refresh_token")
        if not refresh_token:
            raise BoxOAuthError("No refresh token; complete Box OAuth authorization first")
        body = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }).encode("utf-8")
        request = urllib.request.Request(
            TOKEN_URL, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            return {
                "access_token": result["access_token"],
                "refresh_token": result["refresh_token"],
                "expires_at": time.time() + int(result["expires_in"]),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, OSError,
                ValueError, KeyError, TypeError) as exc:
            raise BoxOAuthError("Box token refresh failed") from exc

    def access_token(self) -> str:
        # Transactional PostgreSQL stores serialize refreshes across workers.
        atomic = getattr(self.store, "get_or_refresh", None)
        if callable(atomic):
            try:
                return atomic(self._refresh, now=time.time())
            except BoxOAuthError:
                raise
            except Exception as exc:
                raise BoxOAuthError("Box token storage or refresh failed") from exc

        # Fallback is appropriate only for a single process, not Render workers.
        with self._lock:
            current = self.store.load()
            if (current.get("access_token") and
                    float(current.get("expires_at", 0)) > time.time() + 120):
                return current["access_token"]
            new_tokens = self._refresh(current)
            try:
                self.store.save(new_tokens)
            except Exception as exc:
                raise BoxOAuthError("Box token persistence failed") from exc
            return new_tokens["access_token"]
