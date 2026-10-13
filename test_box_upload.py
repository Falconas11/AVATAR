
import os
import json
from pathlib import Path

import requests

TOKEN = os.environ["BOX_ACCESS_TOKEN"]
FOLDER_ID = os.environ["BOX_STUDY_FOLDER_ID"]

file_path = Path("avatar-box-test.txt")

attributes = {
    "name": file_path.name,
    "parent": {"id": FOLDER_ID}
}

with file_path.open("rb") as f:
    response = requests.post(
        "https://upload.box.com/api/2.0/files/content",
        headers={
            "Authorization": f"Bearer {TOKEN}"
        },
        data={
            "attributes": json.dumps(attributes)
        },
        files={
            "file": (file_path.name, f, "text/plain")
        },
        timeout=30
    )

print("HTTP status:", response.status_code)

if response.ok:
    result = response.json()
    for entry in result.get("entries", []):
        print("Uploaded:", entry["name"])
        print("Box File ID:", entry["id"])
else:
    print("Box error:", response.text)
    raise SystemExit(1)
