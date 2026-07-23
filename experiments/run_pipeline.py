from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN_AVATAR = ROOT / "avatar" / "run_avatar.py"
PROMPT_BUILDER = ROOT / "avatar" / "prompt_builder.py"


def run_module(name: str, command: Sequence[str]) -> None:
    print(f"\n=== {name} ===")
    print("Running:", subprocess.list2cmdline(list(command)))

    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the AVATAR Knowledge Engine and Prompt Builder sequentially "
            "while keeping both modules independently executable."
        )
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs" / "profile.json",
        help="Profile JSON passed to experiments/run_avatar.py.",
    )
    parser.add_argument(
        "--representation-output",
        type=Path,
        default=ROOT / "outputs" / "aging_representation.json",
        help="Knowledge Engine output JSON.",
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=ROOT / "outputs" / "aging_prompt.txt",
        help="Generated aging prompt.",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=ROOT / "outputs" / "aging_prompt_plan.json",
        help="Generated structured prompt plan.",
    )
    parser.add_argument(
        "--target-age",
        type=int,
        default=None,
        help=(
            "Optional target-age override for Prompt Builder. "
            "When omitted, profile.target_age is used."
        ),
    )
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=0.30,
        help="Minimum feature score passed to Prompt Builder.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=12,
        help="Maximum number of nonredundant features.",
    )
    parser.add_argument(
        "--no-mechanism-context",
        action="store_true",
        help="Exclude biological mechanism context from the final prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for required_file in (RUN_AVATAR, PROMPT_BUILDER):
        if not required_file.exists():
            raise FileNotFoundError(f"Required module not found: {required_file}")

    knowledge_command = [
        sys.executable,
        str(RUN_AVATAR),
        "--profile",
        str(args.profile),
        "--output",
        str(args.representation_output),
    ]

    prompt_command = [
        sys.executable,
        str(PROMPT_BUILDER),
        "--input",
        str(args.representation_output),
        "--output",
        str(args.prompt_output),
        "--plan-output",
        str(args.plan_output),
        "--minimum-score",
        str(args.minimum_score),
        "--max-features",
        str(args.max_features),
    ]

    if args.target_age is not None:
        prompt_command.extend(["--target-age", str(args.target_age)])

    if args.no_mechanism_context:
        prompt_command.append("--no-mechanism-context")

    run_module("AVATAR Knowledge Engine", knowledge_command)

    if not args.representation_output.exists():
        raise FileNotFoundError(
            "Knowledge Engine finished without producing the expected output: "
            f"{args.representation_output}"
        )

    run_module("AVATAR Prompt Builder", prompt_command)

    print("\nAVATAR pipeline completed.")
    print(f"Representation: {args.representation_output}")
    print(f"Prompt:         {args.prompt_output}")
    print(f"Prompt plan:    {args.plan_output}")


if __name__ == "__main__":
    main()