# AVATAR pilot research: Box storage integration

Status: **development only — not IRB-approved, not deployed, not connected to the web app.**

## Research requirements to reconcile

The draft consent describes a 35-minute session, investigator-device image capture, alternative lifestyle scenarios, a post-interaction survey, and restricted-access UIUC Box storage for original photographs, aged outputs and survey responses. The IRB draft also mentions processing through Render and OpenAI. The consent currently omits that third-party processing, and promises email delivery of an aged image. The PI/IRB team must reconcile these before recruitment. Face photographs remain identifiable even when assigned random study codes.

**Do not collect real participant photographs until IRB approval, institutional Box approval, and end-to-end verification of the disclosed data flow.**

## Added module

`avatar/box_storage.py` is a backend-only helper. It does not change the existing image pipeline or create a public endpoint.

- `new_participant_id()`: cryptographically random `P-<32 hex>` identifier.
- `upload_study_image(image, participant_id=..., image_kind="original"|"aged", content_type="image/jpeg"|"image/png"|"image/webp")`.
- Uploads using Box Files API multipart upload, with file names `P-..._original.jpg` or `P-..._aged.png`.
- Returns only the Box file ID and study code, never a shareable URL.
- Validates file size (10 MiB max), MIME type, and ID format. Caller must independently verify actual image format, authorization, consent and file size before invoking.
- Requires a restricted-access study folder, and a server-side token.

## Environment variables (server only)

- `BOX_STUDY_FOLDER_ID`: numeric ID of the PI-approved restricted Box folder.
- `BOX_ACCESS_TOKEN`: short-lived token **for development testing only**. A production deployment requires a PI/IT-approved Box application and secure token renewal.

Never store either token or real participant images in GitHub, browser local storage, build artifacts, or Render logs.

## Example for a synthetic image only

```python
from avatar.box_storage import new_participant_id, upload_study_image
participant_id = new_participant_id()
result = upload_study_image(
    synthetic_png_bytes,
    participant_id=participant_id,
    image_kind="original",
    content_type="image/png",
)
print(result.file_id)
```

## Before connecting to production

1. Obtain PI and UIUC IT confirmation that the Box application has access to the **High Risk Data Folder**, and determine the permitted token flow.
2. Update IRB and consent to describe Render, OpenAI, Box, retention, potential logs, email delivery, and any temporary processing. Confirm the real data flow matches the approved protocol.
3. Add authenticated, consent-gated backend orchestration for capture, upload of original and generated images, and survey linking. Do not accept client-supplied IDs as authorization.
4. Add image signature checks, server-side request-size limits, rate limiting, retry/idempotency strategy, partial-upload cleanup and deletion procedures.
5. Verify Box folder access controls, audit logging, retention/deletion policy and secure token renewal.
6. Run only synthetic-image tests before approval; verify failure behavior, not just happy path.

## Pilot research outcomes

Primary pilot measures: perceived realism, emotional response, usability, interpretation and reflective experience. Future-self connection and empathy are exploratory unless a validated instrument and suitable study design are used. Avoid representing generated faces as medically accurate personal predictions.
