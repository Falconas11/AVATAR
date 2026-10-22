"""Age-stage knowledge snapshots, without image generation."""
from __future__ import annotations

from avatar.explainability import build_feature_traces


def build_timeline(engine, profile: dict, *, stages: int = 4) -> dict:
    current = int(profile["age"])
    target = int(profile["target_age"])
    if target <= current:
        raise ValueError("Target age must exceed current age")
    if type(stages) is not int or stages < 1 or stages > 12:
        raise ValueError("Stages must be between 1 and 12")
    ages = sorted({max(current + 1, current + round((target - current) * i / stages)) for i in range(1, stages + 1)})
    snapshots = []
    for age in ages:
        result = engine.infer({**profile, "target_age": age})
        snapshots.append({
            "age": age,
            "aging_representation": result["aging_representation"],
            "feature_traces": build_feature_traces(engine, result),
        })
    return {
        "mode": "knowledge_only",
        "current_age": current,
        "target_age": target,
        "snapshots": snapshots,
        "disclaimer": "Illustrative knowledge-engine outputs, not a personalized prediction.",
    }
