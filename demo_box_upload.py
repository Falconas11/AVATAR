"""AVATAR synthetic-image paired upload demo.

Run from the AVATAR repository root:
    python demo_box_upload.py

Required environment variables:
    DATABASE_URL (external Render PostgreSQL URL when running locally)
    BOX_TOKEN_ENCRYPTION_KEY
    BOX_CLIENT_ID
    BOX_CLIENT_SECRET
    BOX_STUDY_FOLDER_ID

Uses only synthetic test images. Do not upload participant images before
IRB approval and institutional data-handling authorization.
"""

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path

from avatar.box_oauth import BoxOAuthClient
from avatar.box_token_store import PostgresBoxTokenStore
from avatar.box_integration import store_pair


def main() -> None:
    original = Path("demo_original.jpg")
    aged = Path("demo_aged.png")

    if not original.is_file() or not aged.is_file():
        raise SystemExit("Missing demo_original.jpg or demo_aged.png")

    print("Both demo images found.")
    provider = BoxOAuthClient(PostgresBoxTokenStore())
    print("Box OAuth provider initialized.")
    print("Uploading image pair to Box...")

    result = store_pair(
        original=original.read_bytes(),
        aged=aged.read_bytes(),
        original_content_type="image/jpeg",
        aged_content_type="image/png",
        token_provider=provider,
    )

    print()
    print("======================================")
    print("Box paired upload successful!")
    print("======================================")
    print("Participant ID:", result.participant_id)
    print("Original file ID:", result.original.file_id)
    print("Aged file ID:", result.aged.file_id)
    print("======================================")


if __name__ == "__main__":
    main()
