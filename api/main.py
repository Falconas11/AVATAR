from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from avatar.image_renderer import render_portrait
from avatar.knowledge_engine_v02 import KnowledgeEngine
from avatar.prompt_builder import build_aging_prompt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

NODES_PATH = ROOT_DIR / "data" / "nodes_v02.csv"
EDGES_PATH = ROOT_DIR / "data" / "edges_v02.csv"

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="AVATAR API",
    version="1.0.0",
    description=(
        "Biomedical knowledge-guided facial aging inference "
        "and portrait synthesis API."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load Knowledge Engine once during startup
# ---------------------------------------------------------

try:
    engine = KnowledgeEngine(
        nodes_path=NODES_PATH,
        edges_path=EDGES_PATH,
    )
except Exception as exc:
    engine = None
    engine_startup_error = str(exc)
else:
    engine_startup_error = None


# ---------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "project": "AVATAR",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy" if engine is not None else "degraded",
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "nodes_file_found": NODES_PATH.exists(),
        "edges_file_found": EDGES_PATH.exists(),
        "knowledge_engine_loaded": engine is not None,
        "knowledge_engine_error": engine_startup_error,
    }


# ---------------------------------------------------------
# Main generation endpoint
# ---------------------------------------------------------

@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    profile_json: str = Form(...),
) -> dict[str, Any]:
    print("AVATAR: request received", flush=True)

    if engine is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Knowledge Engine failed to load: "
                f"{engine_startup_error}"
            ),
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on Render.",
        )

    profile = parse_profile(profile_json)
    validate_profile(profile)

    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Image must be JPEG, PNG, or WebP.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The image exceeds the 10 MB limit.",
        )

    try:
        print("AVATAR: running Knowledge Engine", flush=True)

        knowledge_result = engine.infer(profile)

        print("AVATAR: building aging prompt", flush=True)

        prompt, prompt_plan = build_aging_prompt(
            knowledge_result,
            target_age=int(profile["target_age"]),
            target_age_source="profile",
            minimum_score=0.30,
            max_features=12,
            include_mechanism_context=True,
        )

        file_extension = SUPPORTED_IMAGE_TYPES[image.content_type]

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            input_path = temporary_path / f"input_portrait{file_extension}"
            output_path = temporary_path / "aged_portrait.png"

            input_path.write_bytes(image_bytes)

            print("AVATAR: calling image renderer", flush=True)

            rendered_result = render_portrait(
                image_path=input_path,
                prompt=prompt,
                output_path=output_path,
                model="gpt-image-2",
                quality="low",
                size="1024x1024",
                output_format="png",
            )

            saved_path, usage = normalize_renderer_result(
                rendered_result,
                output_path,
            )

            if not saved_path.exists():
                raise RuntimeError(
                    f"Renderer did not create output image: {saved_path}"
                )

            encoded_image = base64.b64encode(
                saved_path.read_bytes()
            ).decode("utf-8")

        print("AVATAR: generation completed", flush=True)

        return {
            "success": True,
            "image": {
                "mime_type": "image/png",
                "base64": encoded_image,
            },
            "profile": profile,
            "aging_representation": knowledge_result.get(
                "aging_representation",
                {},
            ),
            "features": knowledge_result.get("features", []),
            "mechanisms": knowledge_result.get("mechanisms", {}),
            "factor_activations": knowledge_result.get(
                "factor_activations",
                {},
            ),
            "prompt": prompt,
            "prompt_plan": prompt_plan,
            "usage": usage,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"AVATAR generation error: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"AVATAR generation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def parse_profile(profile_json: str) -> dict[str, Any]:
    try:
        profile = json.loads(profile_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"profile_json is invalid: {exc.msg}",
        ) from exc

    if not isinstance(profile, dict):
        raise HTTPException(
            status_code=400,
            detail="profile_json must contain a JSON object.",
        )

    return profile


def validate_profile(profile: dict[str, Any]) -> None:
    required_fields = {
        "age",
        "target_age",
        "sex",
        "ethnicity",
        "factors",
    }

    missing_fields = required_fields - profile.keys()

    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Missing profile fields: {sorted(missing_fields)}",
        )

    try:
        current_age = int(profile["age"])
        target_age = int(profile["target_age"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="age and target_age must be integers.",
        ) from exc

    if not 1 <= current_age <= 120:
        raise HTTPException(
            status_code=400,
            detail="age must be between 1 and 120.",
        )

    if not 1 <= target_age <= 120:
        raise HTTPException(
            status_code=400,
            detail="target_age must be between 1 and 120.",
        )

    if target_age <= current_age:
        raise HTTPException(
            status_code=400,
            detail="target_age must be greater than age.",
        )

    if not isinstance(profile["factors"], dict):
        raise HTTPException(
            status_code=400,
            detail="factors must be a JSON object.",
        )

    normalized_factors: dict[str, float] = {}

    for factor_name, factor_value in profile["factors"].items():
        try:
            numeric_value = float(factor_value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Factor '{factor_name}' must be numeric.",
            ) from exc

        if not 0.0 <= numeric_value <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Factor '{factor_name}' must be "
                    "between 0.0 and 1.0."
                ),
            )

        normalized_factors[factor_name] = numeric_value

    profile["age"] = current_age
    profile["target_age"] = target_age
    profile["factors"] = normalized_factors


def normalize_renderer_result(
    rendered_result: Any,
    default_output_path: Path,
) -> tuple[Path, Any]:
    """
    Supports either:

        render_portrait(...) -> (saved_path, usage)

    or:

        render_portrait(...) -> saved_path
    """

    if (
        isinstance(rendered_result, tuple)
        and len(rendered_result) == 2
    ):
        saved_path, usage = rendered_result
        return Path(saved_path), usage

    if rendered_result is None:
        return default_output_path, {}

    return Path(rendered_result), {}