from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# from avatar.knowledge_engine import KnowledgeEngine
from avatar.knowledge_engine_v03 import KnowledgeEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AVATAR Knowledge Engine."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs" / "profile.json",
        help="Path to the input profile JSON.",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        # default=ROOT / "data" / "nodes.csv",
        default=ROOT / "data" / "nodes_v03.csv",
        help="Path to nodes.csv.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        # default=ROOT / "data" / "edges.csv",
        default=ROOT / "data" / "edges_v03.csv",
        help="Path to weighted edges.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "aging_representation.json",
        help="Path for the output JSON.",
    )
    return parser.parse_args()


def load_profile(profile_path: Path) -> dict:
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    with profile_path.open("r", encoding="utf-8") as file:
        profile = json.load(file)

    if not isinstance(profile, dict):
        raise ValueError("The profile JSON must contain a JSON object.")

    return profile


def print_summary(result: dict) -> None:
    representation = result.get("aging_representation", {})
    profile = result.get("profile", {})

    current_age = profile.get("age")
    target_age = profile.get("target_age")

    print("\nAVATAR Knowledge Engine completed.")

    if current_age is not None:
        print(f"Current age: {current_age}")

    if target_age is not None:
        print(f"Target age:  {target_age}")

    print(f"Generated {len(representation)} renderable aging features.")

    if representation:
        print("\nTop aging features:")
        for feature_id, score in list(representation.items())[:10]:
            print(f"  {feature_id:<30} {score:.3f}")

def main() -> None:
    args = parse_args()

    for path, label in (
        (args.nodes, "nodes"),
        (args.edges, "edges"),
        (args.profile, "profile"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    profile = load_profile(args.profile)

    engine = KnowledgeEngine(
        nodes_path=args.nodes,
        edges_path=args.edges,
    )

    result = engine.infer(profile)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_summary(result)
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()