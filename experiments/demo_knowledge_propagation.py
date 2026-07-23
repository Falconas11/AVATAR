from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from avatar.knowledge_engine_v02 import KnowledgeEngine


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a counterfactual smoking-propagation demo."
    )
    parser.add_argument("--profile", type=Path, default=root / "configs" / "profile.json")
    parser.add_argument("--nodes", type=Path, default=root / "data" / "nodes_v02.csv")
    parser.add_argument("--edges", type=Path, default=root / "data" / "edges_v02.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "smoking_demo")
    return parser.parse_args()


def enumerate_paths(engine: KnowledgeEngine, source: str, target: str, max_depth: int = 6):
    adjacency = defaultdict(list)
    for edge in engine.edges:
        adjacency[edge.source].append(edge)

    found = []
    stack = [(source, [source], [])]
    while stack:
        node, node_path, edge_path = stack.pop()
        if len(edge_path) >= max_depth:
            continue
        for edge in adjacency.get(node, []):
            if edge.target in node_path:
                continue
            new_nodes = node_path + [edge.target]
            new_edges = edge_path + [edge]
            if edge.target == target:
                found.append((new_nodes, new_edges))
            else:
                stack.append((edge.target, new_nodes, new_edges))
    return found


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.profile.open("r", encoding="utf-8") as f:
        baseline_profile = json.load(f)

    smoking_profile = json.loads(json.dumps(baseline_profile))
    smoking_profile.setdefault("factors", {})["smoking"] = 1.0

    engine = KnowledgeEngine(args.nodes, args.edges)
    baseline_result = engine.infer(baseline_profile)
    smoking_result = engine.infer(smoking_profile)

    baseline_repr = baseline_result["aging_representation"]
    smoking_repr = smoking_result["aging_representation"]

    rows = []
    for feature_id in sorted(set(baseline_repr) | set(smoking_repr)):
        before = float(baseline_repr.get(feature_id, 0.0))
        after = float(smoking_repr.get(feature_id, 0.0))
        delta = after - before
        if delta > 1e-9:
            rows.append({
                "feature_id": feature_id,
                "feature": engine.nodes[feature_id].get("label", feature_id),
                "baseline_score": round(before, 4),
                "smoking_score": round(after, 4),
                "change_due_to_smoking": round(delta, 4),
            })

    rows.sort(key=lambda row: (-row["change_due_to_smoking"], row["feature"]))

    propagation_paths = {}
    for row in rows:
        paths = enumerate_paths(engine, "smoking", row["feature_id"])
        propagation_paths[row["feature_id"]] = [
            {
                "nodes": node_path,
                "relations": [edge.relation for edge in edge_path],
                "weights": [edge.weight for edge in edge_path],
                "evidence": [edge.evidence for edge in edge_path],
            }
            for node_path, edge_path in paths
        ]

    output = {
        "experiment": "Counterfactual smoking knowledge propagation",
        "method": "All profile values are held constant while smoking is changed to 1.0.",
        "baseline_profile": baseline_profile,
        "modified_profile": smoking_profile,
        "affected_features": rows,
        "propagation_paths": propagation_paths,
        "engine_metadata": smoking_result["metadata"],
    }

    (args.output_dir / "smoking_propagation_demo.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (args.output_dir / "smoking_feature_changes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "feature_id",
            "feature",
            "baseline_score",
            "smoking_score",
            "change_due_to_smoking",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("Smoking propagation demo completed.")
    print(f"Affected renderable features: {len(rows)}")
    print("\nLargest changes:")
    for row in rows[:10]:
        print(
            f"  {row['feature']:<28} "
            f"{row['baseline_score']:.4f} -> {row['smoking_score']:.4f} "
            f"(delta {row['change_due_to_smoking']:+.4f})"
        )
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()