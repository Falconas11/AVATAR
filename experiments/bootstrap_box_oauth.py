"""One-time local Box OAuth authorization and PostgreSQL bootstrap.

Run from the repository root on a trusted computer. The OAuth callback only
listens on 127.0.0.1 and validates the random state parameter. Use the exact
redirect URI registered in the Box Developer Console:
http://127.0.0.1:8765/callback

Requires BOX_CLIENT_ID, BOX_CLIENT_SECRET, BOX_TOKEN_ENCRYPTION_KEY and
DATABASE_URL (the Render PostgreSQL *external* URL for local use).
No credentials are printed. Do not use this script in the public web service.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from avatar.box_token_store import PostgresBoxTokenStore

REDIRECT_URI = "http://127.0.0.1:8765/callback"
AUTHORIZE_URL = "https://account.box.com/api/oauth2/authorize"
TOKEN_URL = "https://api.box.com/oauth2/token"


def main() -> None:
    client_id = os.environ.get("BOX_CLIENT_ID")
    client_secret = os.environ.get("BOX_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("Set BOX_CLIENT_ID and BOX_CLIENT_SECRET locally.")
    # Fail early if database and encryption config are unavailable.
    store = PostgresBoxTokenStore()
    state = secrets.token_urlsafe(32)
    result: dict[str, str] = {}
    event = threading.Event()

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            if parsed.path != "/callback" or params.get("state") != [state]:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid OAuth callback.")
                return
            if "error" in params:
                result["error"] = "Box authorization was denied or failed."
            elif len(params.get("code", [])) == 1:
                result["code"] = params["code"][0]
            else:
                result["error"] = "Missing authorization code."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Authorization received. Return to your terminal.")
            event.set()

        def log_message(self, format, *args):
            # Do not log OAuth authorization codes or query strings.
            pass

    try:
        server = HTTPServer(("127.0.0.1", 8765), Callback)
    except OSError as exc:
        raise SystemExit("Cannot bind localhost port 8765; close other listeners.") from exc
    server.timeout = 1
    auth_url = AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "state": state,
    })
    print("Opening Box authorization in your browser...")
    print("If your browser does not open, check your default browser settings.")
    webbrowser.open(auth_url)
    deadline = time.monotonic() + 180
    try:
        while not event.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if not event.is_set():
        raise SystemExit("Authorization timed out after 3 minutes.")
    if "error" in result:
        raise SystemExit(result["error"])

    body = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": result["code"],
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
    }).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.load(response)
        tokens = {
            "access_token": payload["access_token"],
            "refresh_token": payload["refresh_token"],
            "expires_at": time.time() + int(payload["expires_in"]),
        }
        store.initialize()
        store.save(tokens)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            "OAuth exchange or PostgreSQL persistence failed. "
            "Check local configuration; re-authorize before retrying."
        ) from None
    print("Box OAuth token pair encrypted and saved in PostgreSQL.")
    print("No tokens or authorization codes were displayed.")


if __name__ == "__main__":
    main()
