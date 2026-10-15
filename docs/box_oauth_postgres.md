# AVATAR Box OAuth setup (development only until IRB approval)

The token store is implemented but **not activated in the production API**.
Do not collect or store participant images until IRB and institutional data
handling approval, including the Box folder's required classification.

## Render environment

- DATABASE_URL: Render Postgres **internal** URL, already configured.
- BOX_CLIENT_ID: Box OAuth app client ID.
- BOX_CLIENT_SECRET: Box OAuth app client secret.
- BOX_TOKEN_ENCRYPTION_KEY: generate once with
  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
  and store only in Render secrets. Back it up in an approved secret manager:
  losing it makes the stored refresh token unrecoverable.
- BOX_STUDY_FOLDER_ID: approved Box folder ID.

Do **not** set AVATAR_BOX_STORAGE_ENABLED=true yet. Never store credentials
in Git, browser JavaScript, logs, or chat.

## Initialize table

Use a temporary local administrative script with a secure DB connection or
a controlled one-off Render job:

```python
from avatar.box_token_store import PostgresBoxTokenStore
PostgresBoxTokenStore().initialize()
```

Render internal DATABASE_URL works only from an eligible Render service in
the same region. Do not paste the URL in a terminal transcript or chat.

## Initial OAuth authorization (not yet automated)

The Box **Developer Token** is not a refresh token. Complete the Box
authorization-code grant using the Box app's registered redirect URI,
exchange the one-time code for an access/refresh pair on the server, and
seed the store with `store.save({"access_token": ..., "refresh_token": ...,
"expires_at": time.time() + expires_in})` using a secure one-time setup
process. No OAuth callback or one-time seeding command has been deployed yet.

## Refresh semantics

`BoxOAuthClient(PostgresBoxTokenStore()).access_token()` locks the DB row,
refreshes before expiry, and commits the *rotated* refresh token before
returning the new access token. Do not deploy multiple independently seeded
databases for the same Box OAuth authorization.

**Operational caveat:** If Box accepts a refresh but the database write or
commit fails, the old refresh token may already be invalid. Recovery requires
re-authorizing the Box OAuth app. Monitor failures without logging tokens.

## API integration

The upload helper `avatar.box_integration.store_pair` exists but the
`api/main.py` route is not yet wired. Pair uploads are not atomic: if the
second upload fails, the first remains and needs reconciliation. Add explicit
feature and IRB gating and a recovery plan before enabling uploads.
