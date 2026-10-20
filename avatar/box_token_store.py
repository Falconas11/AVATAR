"""Encrypted, transactional Box OAuth token storage for Render PostgreSQL.

Requires DATABASE_URL, BOX_TOKEN_ENCRYPTION_KEY (Fernet key), psycopg and
cryptography. Initialize schema explicitly before first use.

Refresh coordination holds a PostgreSQL advisory transaction lock across the
refresh request and token rotation. Use a *single* Box OAuth app per store key.
Never log tokens, connection URLs, or decrypted payloads.
"""
from __future__ import annotations

import json
import os
from typing import Callable

import psycopg
from cryptography.fernet import Fernet, InvalidToken


class BoxTokenStoreError(RuntimeError):
    pass


class PostgresBoxTokenStore:
    LOCK_ID = 4628713901

    def __init__(self, *, database_url: str | None = None,
                 encryption_key: str | None = None):
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        key = encryption_key or os.environ.get("BOX_TOKEN_ENCRYPTION_KEY")
        if not self.database_url or not key:
            raise BoxTokenStoreError("Database URL and token encryption key required")
        try:
            self.cipher = Fernet(key.encode("ascii"))
        except (ValueError, TypeError) as exc:
            raise BoxTokenStoreError("Invalid Fernet encryption key") from exc

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=10)

    def initialize(self) -> None:
        """Create the token table. Run explicitly during setup, not at import."""
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS box_oauth_tokens ("
                " id INTEGER PRIMARY KEY CHECK (id = 1),"
                " encrypted_payload BYTEA NOT NULL,"
                " updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )

    def _decrypt(self, payload: bytes) -> dict:
        try:
            value = json.loads(self.cipher.decrypt(bytes(payload)))
            if not isinstance(value, dict):
                raise ValueError("Invalid token payload")
            return value
        except (InvalidToken, ValueError, TypeError) as exc:
            raise BoxTokenStoreError("Unable to decrypt stored Box credentials") from exc

    def _encrypt(self, value: dict) -> bytes:
        if not isinstance(value, dict) or not all(
            value.get(k) for k in ("access_token", "refresh_token", "expires_at")
        ):
            raise BoxTokenStoreError("Incomplete token pair")
        return self.cipher.encrypt(json.dumps(value).encode("utf-8"))

    def load(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT encrypted_payload FROM box_oauth_tokens WHERE id = 1"
            ).fetchone()
        return self._decrypt(row[0]) if row else {}

    def save(self, tokens: dict) -> None:
        """For initial seeding only; do not use for concurrent token refresh."""
        payload = self._encrypt(tokens)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO box_oauth_tokens (id, encrypted_payload) VALUES (1, %s)"
                " ON CONFLICT (id) DO UPDATE SET encrypted_payload = EXCLUDED.encrypted_payload,"
                " updated_at = now()",
                (payload,),
            )

    def get_or_refresh(self, refresh: Callable[[dict], dict],
                       *, now: float, margin_seconds: int = 120) -> str:
        """Serialize refreshes across workers and atomically persist rotated pair.

        The transaction stays open during Box HTTP refresh. This prevents a second
        worker from using a refresh token already consumed by the first worker.
        """
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (self.LOCK_ID,))
            row = conn.execute(
                "SELECT encrypted_payload FROM box_oauth_tokens WHERE id = 1 FOR UPDATE"
            ).fetchone()
            current = self._decrypt(row[0]) if row else {}
            if (current.get("access_token") and
                    float(current.get("expires_at", 0)) > now + margin_seconds):
                return current["access_token"]
            if not current.get("refresh_token"):
                raise BoxTokenStoreError("Box OAuth initial authorization required")
            new_tokens = refresh(current)
            payload = self._encrypt(new_tokens)
            conn.execute(
                "UPDATE box_oauth_tokens SET encrypted_payload = %s,"
                " updated_at = now() WHERE id = 1",
                (payload,),
            )
            # Commit happens on successful context exit, before returning token.
            token = new_tokens["access_token"]
        return token
