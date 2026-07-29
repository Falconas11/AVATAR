from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILDER_VERSION = "1.2.1"


@dataclass(frozen=True)
class PromptFeature:
    feature_id: str
    label: str
    score: float
    region: str
    category: str
    intensity: str
    phrase: str
    priority: str


FEATURE_PHRASES: dict[str, str] = {
    "nodular_elastosis": "coarse, weathered, photoaged skin texture",
    "crow_feet": "crow's feet",
    "facial_wrinkles": "facial wrinkles",
    "forehead_wrinkles": "horizontal forehead wrinkles",
    "lower_face_sagging": "lower-face skin laxity and sagging",
    "deep_nasolabial_folds": "deep nasolabial folds",
    "nasolabial_folds": "nasolabial folds",
    "jowls": "jowls along the jawline",
    "spider_veins": "very subtle superficial vascular visibility",
    "sunspots": "sunspots and age spots",
    "uneven_skin_tone": "uneven skin tone",
    "dull_skin_tone": "dull skin tone",
    "skin_dullness": "dull skin tone",
    "enlarged_pores": "enlarged pores",
    "under_eye_circles": "under-eye circles",
    "dark_circles": "under-eye circles",
    "eye_bags": "under-eye bags",
    "under_eye_bags": "under-eye bags",
    "dry_skin": "dry, slightly coarse skin texture",
    "hyperpigmentation": "areas of hyperpigmentation",
    "broken_capillaries": "small broken capillaries",
    "facial_puffiness": "mild facial puffiness",
    "facial_redness": "mild facial redness",
    "cheek_volume_loss": "reduced cheek volume",
    "skin_laxity": "reduced skin firmness",
    "hair_whitening": "natural gray hair",
    "nasal_tip_ptosis": "a subtly lowered nasal tip consistent with normal aging",
    "increased_nasal_length": "a slight apparent increase in nasal length",
    "earlobe_elongation": "slightly elongated earlobes",
    "earlobe_ptosis": "mild age-related earlobe sagging",
}


REDUNDANCY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("deep_nasolabial_folds", "nasolabial_folds"),
    ("crow_feet", "facial_wrinkles"),
    ("forehead_wrinkles", "facial_wrinkles"),
    ("sunspots", "hyperpigmentation"),
    ("spider_veins", "broken_capillaries"),
    ("under_eye_circles", "dark_circles"),
    ("under_eye_bags", "eye_bags"),
    ("dull_skin_tone", "skin_dullness"),
)


REGION_ALIASES: dict[str, str] = {
    "": "entire face",
    "whole face": "entire face",
    "whole_face": "entire face",
    "entireface": "entire face",
    "entire_face": "entire face",
    "eye corner": "eye corners",
    "eye_corners": "eye corners",
    "under eye": "under-eye",
    "under_eye": "under-eye",
    "lower_face": "lower face",
    "lower cheek jaw": "lower cheek and jaw",
    "lower_cheek_jaw": "lower cheek and jaw",
    "nasolabial": "nasolabial region",
    "ear": "ears",
    "earlobes": "earlobes",
    "ear lobe": "earlobes",
    "ear lobes": "earlobes",
}


REGION_ORDER: tuple[str, ...] = (
    "forehead",
    "eye corners",
    "under-eye",
    "cheek",
    "cheekbones, nose, forehead",
    "nasolabial region",
    "lower face",
    "lower cheek and jaw",
    "nose",
    "earlobes",
    "ears",
    "nose and neck",
    "entire face",
)


MECHANISM_VISIBLE_EFFECTS: dict[str, str] = {
    "collagen_degradation": "reduced dermal support and more visible wrinkling",
    "elastic_fiber_damage": "reduced elasticity and mild skin laxity",
    "oxidative_stress": "duller, less even skin texture",
    "skin_barrier_impairment": "dryness and slightly coarse skin texture",
    "mmp_activity": "gradual collagen loss and reduced skin firmness",
    "melanin_dysregulation": "uneven pigmentation and localized age spots",
    "microvascular_damage": "subtle visible capillaries and uneven redness",
    "nasal_support_weakening": "subtle age-related nasal-tip descent and apparent nasal elongation",
    "auricular_tissue_laxity": "subtle earlobe elongation and mild earlobe sagging",
}


MORPHOLOGICAL_FEATURES: frozenset[str] = frozenset({
    "nasal_tip_ptosis",
    "increased_nasal_length",
    "earlobe_elongation",
    "earlobe_ptosis",
})


def morphology_intensity(feature_id: str, score: float) -> str:
    """Keep structural aging visible but anatomically conservative."""
    if feature_id not in MORPHOLOGICAL_FEATURES:
        return intensity_label(score)
    if score >= 0.85:
        return "mild to moderate"
    if score >= 0.40:
        return "subtle"
    return "very subtle"


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature score must be numeric, got {value!r}") from exc
    return max(0.0, min(1.0, score))


def parse_age(value: Any, field_name: str) -> int | None:
    if value is None or value == "":
        return None

    try:
        age = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc

    if not 0 <= age <= 120:
        raise ValueError(f"{field_name} must be between 0 and 120, got {age}")

    return age


def normalize_region(region: Any) -> str:
    raw = str(region or "").strip().lower()
    normalized = " ".join(raw.replace("-", " ").split())
    return REGION_ALIASES.get(raw, REGION_ALIASES.get(normalized, raw or "entire face"))


def intensity_label(score: float) -> str:
    if score >= 0.85:
        return "pronounced"
    if score >= 0.60:
        return "moderate to strong"
    if score >= 0.40:
        return "moderate"
    if score >= 0.25:
        return "subtle"
    return "very subtle"


def priority_label(score: float) -> str:
    if score >= 0.60:
        return "primary"
    if score >= 0.40:
        return "secondary"
    return "supporting"


def build_age_anchor(target_age: int) -> str:
    if target_age >= 70:
        age_class = "an older adult"
        contrast = "not middle-aged"
    elif target_age >= 60:
        age_class = "an older adult"
        contrast = "not merely middle-aged"
    elif target_age >= 50:
        age_class = "a mature adult"
        contrast = "not a young adult"
    elif target_age >= 40:
        age_class = "a middle-aged adult"
        contrast = "not a young adult"
    elif target_age >= 30:
        age_class = "an adult in their thirties"
        contrast = "not a teenager or very young adult"
    else:
        age_class = f"an adult of approximately {target_age}"
        contrast = "consistent with the requested age"

    return (
        f"AGE ANCHOR: The subject must be immediately perceived as approximately "
        f"{target_age} years old and as {age_class} before fine skin details are "
        f"noticed. The overall age impression should be unmistakable, "
        f"{contrast}. The personalized changes below refine this age impression "
        f"rather than replace it."
    )


def build_age_typical_features(target_age: int) -> list[str]:
    if target_age >= 70:
        return [
            (
                "predominantly dark hair with approximately 35 to 45 percent "
                "naturally distributed gray hairs, somewhat denser around the temples "
                "and frontal hairline, while preserving the original hairstyle, "
                "hairline, and overall hair density"
            ),
            "moderate generalized skin laxity, especially in the cheeks and jawline",
            "moderate age-related facial volume loss in the mid-face",
            "clearly deepened nasolabial folds",
            "visible but natural neck aging where the neck is shown",
        ]

    if target_age >= 60:
        return [
            (
                "predominantly black hair with approximately 20 to 30 percent naturally "
                "distributed gray hairs, concentrated mainly around the temples and "
                "frontal hairline while remaining sparsely scattered throughout the "
                "rest of the hair; preserve the original hairstyle, hairline, and "
                "overall hair density, with black hair remaining the clearly dominant hair color. Gray hairs should appear as individual strands rather than large continuous patches."
            ),
            "mild to moderate generalized skin laxity, especially in the cheeks and jawline",
            "mild age-related facial volume loss in the mid-face",
            "naturally deepened nasolabial folds",
            "subtle neck lines and mild neck laxity where the neck is visible",
        ]

    if target_age >= 50:
        return [
            (
                "predominantly black hair with approximately 10 to 20 percent "
                "naturally distributed gray hairs, especially near the temples, while "
                "preserving the original hairstyle, hairline, and overall hair density"
            ),
            "mild skin laxity in the cheeks and lower face",
            "slight mid-face volume loss",
            "moderately deepened nasolabial folds",
        ]

    if target_age >= 40:
        return [
            (
                "predominantly black hair with only a few scattered gray strands, "
                "mainly near the temples, without changing the hairstyle, hairline, "
                "or hair density"
            ),
        ]
    if target_age >= 30:
        return [
            (
                "the original natural hair color with no required graying, while "
                "preserving the original hairstyle, hairline, and overall hair density"
            ),
            "subtle early adult skin texture maturation without pronounced wrinkles",
            "very faint expression lines around the eyes or forehead only where naturally appropriate",
            "slight under-eye definition while preserving a healthy and age-appropriate appearance",
        ]

    return []


def format_age_typical_features(features: list[str]) -> str:
    if not features:
        return ""

    lines = "\n".join(f"- Add {feature}." for feature in features)
    return (
        "AGE-TYPICAL FEATURES: Establish the normal overall appearance expected "
        "at the target age before applying the personalized regional details.\n"
        f"{lines}"
    )


def readable_feature_name(feature_id: str) -> str:
    return feature_id.replace("_", " ").strip()


def feature_phrase(feature_id: str, label: str) -> str:
    phrase = FEATURE_PHRASES.get(feature_id)
    if phrase:
        return phrase
    if label:
        return label.lower()
    return readable_feature_name(feature_id)


def validate_engine_output(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise TypeError("Knowledge Engine output must be a JSON object.")

    representation = data.get("aging_representation")
    if not isinstance(representation, dict):
        raise ValueError(
            "Input must be the complete Knowledge Engine output containing "
            "'aging_representation'."
        )

    profile = data.get("profile", {})
    if profile is not None and not isinstance(profile, dict):
        raise ValueError("'profile' must be a JSON object when present.")

    features = data.get("features")
    if features is not None and not isinstance(features, dict):
        raise ValueError("'features' must be a JSON object when present.")


def resolve_target_age(
    data: dict[str, Any],
    cli_target_age: int | None,
) -> tuple[int, str]:
    profile = data.get("profile", {})

    if cli_target_age is not None:
        target_age = parse_age(cli_target_age, "--target-age")
        assert target_age is not None
        return target_age, "command_line"

    target_age = parse_age(profile.get("target_age"), "profile.target_age")
    if target_age is None:
        raise ValueError(
            "Target age is missing. Add 'target_age' to configs/profile.json "
            "or provide --target-age."
        )

    return target_age, "profile"


def select_nonredundant_features(
    data: dict[str, Any],
    *,
    minimum_score: float,
    max_features: int,
) -> list[PromptFeature]:
    representation = data["aging_representation"]
    feature_metadata = data.get("features", {})

    candidates: dict[str, PromptFeature] = {}

    for feature_id, raw_score in representation.items():
        score = clamp_score(raw_score)
        if score < minimum_score:
            continue

        metadata = feature_metadata.get(feature_id, {})
        if not isinstance(metadata, dict):
            metadata = {}

        if metadata.get("renderable") is False:
            continue
        if metadata.get("aging_related") is False:
            continue

        label = str(metadata.get("label") or readable_feature_name(feature_id))
        region = normalize_region(metadata.get("region"))
        category = str(metadata.get("category") or "").strip()

        candidates[feature_id] = PromptFeature(
            feature_id=feature_id,
            label=label,
            score=score,
            region=region,
            category=category,
            intensity=morphology_intensity(feature_id, score),
            phrase=feature_phrase(feature_id, label),
            priority=priority_label(score),
        )

    suppressed: set[str] = set()

    for group in REDUNDANCY_GROUPS:
        present = [feature_id for feature_id in group if feature_id in candidates]
        if len(present) <= 1:
            continue

        keeper = present[0]
        for feature_id in present[1:]:
            if candidates[feature_id].score > candidates[keeper].score + 0.10:
                keeper = feature_id

        for feature_id in present:
            if feature_id != keeper:
                suppressed.add(feature_id)

    selected = [
        feature
        for feature_id, feature in candidates.items()
        if feature_id not in suppressed
    ]
    selected.sort(key=lambda item: (-item.score, item.feature_id))
    return selected[:max_features]


def group_features_by_region(
    features: list[PromptFeature],
) -> dict[str, list[PromptFeature]]:
    grouped: dict[str, list[PromptFeature]] = defaultdict(list)

    for feature in features:
        grouped[feature.region].append(feature)

    for region_features in grouped.values():
        region_features.sort(key=lambda item: (-item.score, item.feature_id))

    region_rank = {region: index for index, region in enumerate(REGION_ORDER)}

    return dict(
        sorted(
            grouped.items(),
            key=lambda item: (
                region_rank.get(item[0], len(REGION_ORDER)),
                -max(feature.score for feature in item[1]),
                item[0],
            ),
        )
    )


def join_items(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_mechanism_summary(
    data: dict[str, Any],
    *,
    minimum_score: float = 0.60,
    max_mechanisms: int = 4,
) -> list[dict[str, Any]]:
    mechanisms = data.get("mechanisms", {})
    if not isinstance(mechanisms, dict):
        return []

    selected: list[dict[str, Any]] = []

    for mechanism_id, metadata in mechanisms.items():
        if not isinstance(metadata, dict):
            continue

        score = clamp_score(metadata.get("score", 0.0))
        if score < minimum_score:
            continue

        raw_label = str(metadata.get("label") or readable_feature_name(mechanism_id))
        visible_effect = MECHANISM_VISIBLE_EFFECTS.get(mechanism_id, raw_label.lower())

        selected.append(
            {
                "id": mechanism_id,
                "label": raw_label,
                "visible_effect": visible_effect,
                "score": score,
                "region": normalize_region(metadata.get("region")),
            }
        )

    selected.sort(key=lambda item: (-item["score"], item["id"]))
    return selected[:max_mechanisms]


def build_prompt_plan(
    data: dict[str, Any],
    *,
    target_age: int,
    target_age_source: str,
    minimum_score: float,
    max_features: int,
) -> dict[str, Any]:
    validate_engine_output(data)

    selected = select_nonredundant_features(
        data,
        minimum_score=minimum_score,
        max_features=max_features,
    )
    grouped = group_features_by_region(selected)
    mechanisms = build_mechanism_summary(data)

    profile = data.get("profile", {})
    current_age = parse_age(profile.get("age"), "profile.age")

    return {
        "source_profile": {
            "age": current_age,
            "target_age": parse_age(profile.get("target_age"), "profile.target_age"),
            "sex": profile.get("sex"),
            "ethnicity": profile.get("ethnicity"),
        },
        "target_age": target_age,
        "target_age_source": target_age_source,
        "age_typical_features": build_age_typical_features(target_age),
        "selected_features": [asdict(feature) for feature in selected],
        "regional_groups": {
            region: [asdict(feature) for feature in features]
            for region, features in grouped.items()
        },
        "active_mechanisms": mechanisms,
        "builder_metadata": {
            "version": BUILDER_VERSION,
            "minimum_score": minimum_score,
            "max_features": max_features,
            "strategy": (
                "profile_target_age_resolution_age_anchor_age_typical_features_"
                "salt_and_pepper_hair_regional_pore_control_vascular_visibility_"
                "intrinsic_aging_consistency_perceptual_priority_feature_metadata_filtering_region_"
                "normalization_redundancy_suppression_priority_tiering_"
                "morphological_aging_intensity_caps_anatomy_preservation"
            ),
        },
    }


def render_regional_instruction(
    region: str,
    features: list[PromptFeature],
) -> str:
    feature_ids = {feature.feature_id for feature in features}

    morphology = [
        feature for feature in features
        if feature.feature_id in MORPHOLOGICAL_FEATURES
    ]
    remaining_after_morphology = [
        feature for feature in features
        if feature.feature_id not in MORPHOLOGICAL_FEATURES
    ]

    if morphology:
        changes = [feature.phrase for feature in morphology]
        sentences = [
            f"In the {region}, introduce only subtle, anatomically plausible "
            f"age-related structural change: {join_items(changes)}. Preserve the "
            "person's recognizable baseline anatomy and avoid exaggerated growth, "
            "cartoonish drooping, or surgical-looking deformation."
        ]
        if remaining_after_morphology:
            other_changes = [
                f"{feature.intensity} {feature.phrase}"
                for feature in remaining_after_morphology
            ]
            sentences.append(f"Also add {join_items(other_changes)}.")
        return " ".join(sentences)

    # Enlarged pores should be localized rather than spread uniformly over the face.
    if "enlarged_pores" in feature_ids:
        remaining = [
            feature for feature in features
            if feature.feature_id != "enlarged_pores"
        ]

        sentences = [
            (
                "In the nose, medial cheeks, and chin, add moderate enlarged pores. "
                "Maintain relatively smoother skin on the forehead and lateral cheeks."
            )
        ]

        if remaining:
            changes = [
                f"{feature.intensity} {feature.phrase}"
                for feature in remaining
            ]
            sentences.append(
                "Across the remaining facial skin, add only "
                f"{join_items(changes)}."
            )

        return " ".join(sentences)

    # Avoid contradictory intensity phrases such as "moderate very subtle".
    if "spider_veins" in feature_ids:
        remaining = [
            feature for feature in features
            if feature.feature_id != "spider_veins"
        ]

        sentences = [
            (
                f"In localized areas of the {region}, add very subtle superficial "
                "vascular visibility that appears natural and non-pathological."
            )
        ]

        if remaining:
            changes = [
                f"{feature.intensity} {feature.phrase}"
                for feature in remaining
            ]
            sentences.append(f"Also add {join_items(changes)}.")

        return " ".join(sentences)

    changes = [
        f"{feature.intensity} {feature.phrase}"
        for feature in features
    ]
    return f"In the {region}, add {join_items(changes)}."


def build_aging_prompt(
    data: dict[str, Any],
    *,
    target_age: int,
    target_age_source: str,
    minimum_score: float = 0.30,
    max_features: int = 12,
    include_mechanism_context: bool = True,
) -> tuple[str, dict[str, Any]]:
    plan = build_prompt_plan(
        data,
        target_age=target_age,
        target_age_source=target_age_source,
        minimum_score=minimum_score,
        max_features=max_features,
    )

    selected = [PromptFeature(**feature) for feature in plan["selected_features"]]
    grouped = group_features_by_region(selected)

    primary = [feature for feature in selected if feature.priority == "primary"]
    secondary = [feature for feature in selected if feature.priority == "secondary"]
    supporting = [feature for feature in selected if feature.priority == "supporting"]

    prompt_sections: list[str] = []

    prompt_sections.append(
        "TASK: Edit the provided portrait of the same person. "
        "Do not generate a different individual."
    )

    prompt_sections.append(
        f"TARGET APPEARANCE: Edit the portrait so that the subject appears "
        f"approximately {target_age} years old through realistic, gradual facial aging."
    )

    prompt_sections.append(build_age_anchor(target_age))

    age_typical_features = build_age_typical_features(target_age)
    if age_typical_features:
        prompt_sections.append(
            format_age_typical_features(age_typical_features)
        )

    prompt_sections.append(
        "IDENTITY PRESERVATION: Preserve the subject's identity, facial bone "
        "structure, ethnicity, sex presentation, eye shape, recognizable baseline "
        "nose and ear anatomy, mouth shape, hairstyle, expression, pose, camera "
        "angle, lighting, clothing, "
        "background, and image composition."
    )

    if primary:
        focus_regions = join_items(
            list(dict.fromkeys(feature.region for feature in primary))
        )
        prompt_sections.append(
            "PRIMARY AGING IMPRESSION: The strongest overall impression of age "
            f"should come from changes in the {focus_regions}. These regions should "
            "establish the perceived age before smaller surface details are noticed. "
            "Exact changes are specified below by region."
        )

    if secondary:
        secondary_text = join_items(
            [f"{feature.phrase} ({feature.region})" for feature in secondary]
        )
        prompt_sections.append(
            "SECONDARY AGING CHANGES: Add these with lower visual emphasis: "
            f"{secondary_text}."
        )

    if supporting:
        supporting_text = join_items(
            [f"{feature.phrase} ({feature.region})" for feature in supporting]
        )
        prompt_sections.append(
            "SUPPORTING DETAILS: Keep these subtle and subordinate to the main "
            f"aging changes: {supporting_text}."
        )

    if grouped:
        regional_lines = [
            render_regional_instruction(region, features)
            for region, features in grouped.items()
        ]
        prompt_sections.append("REGIONAL INSTRUCTIONS:\n- " + "\n- ".join(regional_lines))

    if include_mechanism_context:
        personalized_clause = ""
        if plan["active_mechanisms"]:
            visible_effects = join_items(
                [
                    str(mechanism["visible_effect"])
                    for mechanism in plan["active_mechanisms"]
                ]
            )
            personalized_clause = (
                " Also preserve the personalized aging tendency toward "
                f"{visible_effects}."
            )

        prompt_sections.append(
            "BIOLOGICAL CONSISTENCY: The visible appearance should reflect normal "
            "intrinsic aging, including gradual collagen loss, reduced skin "
            "elasticity, age-related facial soft-tissue descent, natural facial "
            "volume redistribution, and subtle remodeling of nasal and auricular "
            f"support structures.{personalized_clause} Express these mechanisms "
            "only through realistic visible facial and skin changes. Do not introduce "
            "dermatological disease, medical annotations, pathological lesions, or "
            "anatomical overlays."
        )

    prompt_sections.append(
        "REALISM: Keep the edits anatomically plausible, region-specific, "
        "photorealistic, naturally asymmetric, and integrated with the original "
        "skin texture. Use continuous transitions rather than pasted-on marks."
    )

    prompt_sections.append(
        "NEGATIVE CONSTRAINTS: Do not alter identity or make broad, unrelated changes "
        "to facial proportions. Permit only the subtle age-related nasal or earlobe "
        "changes explicitly requested above. Avoid duplicated facial features, extra "
        "eyes or nostrils, warped anatomy, "
        "extreme asymmetry, caricature-like aging, uniform wrinkle overlays, "
        "plastic skin, heavy makeup, illustration, text, or watermarks. "
        "Do not create baldness, an altered hairline, or reduced hair density. "
        "Do not make the hair predominantly gray, white, or silver. Maintain black "
        "as the dominant hair color, with only the age-appropriate proportion of "
        "naturally scattered gray hairs specified above. Avoid uniformly white hair, "
        "large solid gray patches, or exaggerated temple whitening."
    )

    return "\n\n".join(prompt_sections), plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AVATAR Knowledge Engine output into a structured, "
            "regional image-edit prompt."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "aging_representation.json",
        help="Complete Knowledge Engine output JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "aging_prompt.txt",
        help="Generated text prompt.",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=ROOT / "outputs" / "aging_prompt_plan.json",
        help="Structured prompt plan for debugging and future renderers.",
    )
    parser.add_argument(
        "--target-age",
        type=int,
        default=None,
        help="Optional target age. Overrides profile.target_age when supplied.",
    )
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=0.30,
        help="Ignore aging features below this score.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=12,
        help="Maximum number of nonredundant aging features.",
    )
    parser.add_argument(
        "--no-mechanism-context",
        action="store_true",
        help="Exclude mechanism-derived biological consistency language.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    if not 0.0 <= args.minimum_score <= 1.0:
        raise ValueError("--minimum-score must be between 0 and 1.")

    if args.max_features <= 0:
        raise ValueError("--max-features must be greater than zero.")

    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    validate_engine_output(data)
    target_age, target_age_source = resolve_target_age(data, args.target_age)

    prompt, plan = build_aging_prompt(
        data,
        target_age=target_age,
        target_age_source=target_age_source,
        minimum_score=args.minimum_score,
        max_features=args.max_features,
        include_mechanism_context=not args.no_mechanism_context,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(prompt + "\n", encoding="utf-8")
    args.plan_output.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    current_age = plan["source_profile"]["age"]

    print(f"AVATAR Prompt Builder v{BUILDER_VERSION} completed.")
    if current_age is not None:
        print(f"Current age: {current_age}")
    print(f"Target age:  {target_age} (source: {target_age_source})")
    print(f"Prompt saved to: {args.output}")
    print(f"Prompt plan saved to: {args.plan_output}")
    print(f"Selected {len(plan['selected_features'])} nonredundant aging features.")
    print("\nGenerated prompt:\n")
    print(prompt)


if __name__ == "__main__":
    main()