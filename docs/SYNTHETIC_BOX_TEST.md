# AVATAR synthetic-only end-to-end test

This workflow is **only for generated/synthetic portraits**, never actual participant photos.
The IRB collection approval flag must remain `false` until approval is obtained.

## 1. Local configuration

Add these entries to the local, Git-ignored project-root `.env`:

```dotenv
AVATAR_ENV=local
AVATAR_BOX_SYNTHETIC_TEST=true
AVATAR_BOX_STORAGE_ENABLED=true
AVATAR_IRB_COLLECTION_APPROVED=false
BOX_STUDY_FOLDER_ID=426895667507
```

Keep the already working `DATABASE_URL` (Render **external** PostgreSQL URL),
`BOX_TOKEN_ENCRYPTION_KEY`, `BOX_CLIENT_ID`, `BOX_CLIENT_SECRET`,
and `OPENAI_API_KEY` in the same file. Never commit or share credentials.

If python-dotenv is missing: `python -m pip install python-dotenv`.
The local API reads `.env` automatically. This is disabled in Render.

## 2. Run locally

From the project root, start the API in one PowerShell terminal:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
python -m http.server 5500 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:5500` (not the production site). The frontend
automatically uses the local API on localhost/127.0.0.1.

## 3. Verify

1. Upload **only a fully synthetic** portrait; choose ages and Generate.
2. Open browser Developer Tools > Network > `generate`.
3. Verify Request URL is `http://127.0.0.1:8000/api/generate`.
4. Response must contain `storage_status: "stored"` and a `P-...` participant ID.
5. Verify Box test folder has `P-..._original.*` and `P-..._aged.png`.
6. Click Feedback; confirm the Google Form pre-fills the same participant ID.
7. Submit the form **manually** if testing survey responses, and verify the linked sheet.
   Any existing required image-upload questions in Google Forms must be removed
   or made optional by the form owner first.

`storage_status: "disabled"` means the storage gate was not enabled; check the
local environment, loopback address, and folder ID. `"failed"` means Box
upload was attempted but not confirmed; inspect the local API terminal logs.

Run local unit tests: `python -m pytest tests/test_synthetic_storage_gate.py tests/test_box_pair_cleanup.py -q`.

**Do not use this testing mode for human subjects or expose the local API to the network.**
A test-mode flag cannot prove that a photo is synthetic.
