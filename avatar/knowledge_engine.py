from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    weight: float
    region: str = ""
    age_min: int | None = None
    age_max: int | None = None
    sex_condition: str = ""
    ethnicity_condition: str = ""
    evidence: str = ""
    notes: str = ""


class KnowledgeEngine:
    """Rule-based graph propagation engine for AVATAR v0.1.

    v0.1 assumptions:
    - All graph weights may initially be 1.0.
    - Lifestyle inputs are numeric activations in [0, 1].
    - Demographic factors are binary activations inferred from profile fields.
    - Edges are filtered by age, sex, and ethnicity conditions.
    - Feature scores are normalized by the total eligible incoming weight,
      preventing features with more incoming edges from automatically exceeding 1.
    """

    def __init__(self, nodes_path: str | Path, edges_path: str | Path) -> None:
        self.nodes_path = Path(nodes_path)
        self.edges_path = Path(edges_path)
        self.nodes = self._load_nodes(self.nodes_path)
        self.edges = self._load_edges(self.edges_path)
        self._validate_graph()

    @staticmethod
    def _load_nodes(path: Path) -> dict[str, dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {row["id"]: row for row in csv.DictReader(handle)}

    @staticmethod
    def _parse_optional_int(value: str | None) -> int | None:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))

    @classmethod
    def _load_edges(cls, path: Path) -> list[Edge]:
        edges: list[Edge] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                edges.append(
                    Edge(
                        source=row["source"],
                        target=row["target"],
                        relation=row["relation"],
                        weight=float(row.get("weight", 1.0)),
                        region=row.get("region", ""),
                        age_min=cls._parse_optional_int(row.get("age_min")),
                        age_max=cls._parse_optional_int(row.get("age_max")),
                        sex_condition=row.get("sex_condition", "").strip().lower(),
                        ethnicity_condition=row.get(
                            "ethnicity_condition", ""
                        ).strip().lower(),
                        evidence=row.get("evidence", ""),
                        notes=row.get("notes", ""),
                    )
                )
        return edges

    def _validate_graph(self) -> None:
        missing: list[tuple[str, str]] = []
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                missing.append((edge.source, edge.target))
            if edge.weight < 0:
                raise ValueError(
                    f"Negative edge weight is not supported: "
                    f"{edge.source} -> {edge.target}"
                )
        if missing:
            raise ValueError(f"Edges reference missing nodes: {missing[:10]}")

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value or "").strip().lower()

    def build_factor_activations(self, profile: dict[str, Any]) -> dict[str, float]:
        """Convert a user profile into graph factor activations."""
        activations: dict[str, float] = {}

        # Direct factor activations: {"smoking": 1.0, "uv_exposure": 0.7, ...}
        factors = profile.get("factors", {})
        if not isinstance(factors, dict):
            raise TypeError("profile['factors'] must be an object/dictionary.")

        for factor_id, value in factors.items():
            if factor_id not in self.nodes:
                raise KeyError(f"Unknown factor node: {factor_id}")
            if self.nodes[factor_id].get("layer") != "factor":
                raise ValueError(f"{factor_id} is not a factor node.")
            activations[factor_id] = self._clamp(float(value))

        # Demographic factors are derived from profile metadata.
        sex = self._normalize_text(profile.get("sex"))
        if sex in {"male", "man", "men"}:
            activations["men"] = 1.0
        elif sex in {"female", "woman", "women"}:
            activations["women"] = 1.0

        ethnicity_aliases = {
            "asian": "asian",
            "caucasian": "caucasian",
            "white": "caucasian",
            "african american": "african_american",
            "black": "african_american",
            "hispanic": "hispanic_latino",
            "latino": "hispanic_latino",
            "hispanic/latino": "hispanic_latino",
        }
        ethnicity = self._normalize_text(profile.get("ethnicity"))
        if ethnicity in ethnicity_aliases:
            activations[ethnicity_aliases[ethnicity]] = 1.0

        return activations

    def _edge_is_eligible(self, edge: Edge, profile: dict[str, Any]) -> bool:
        age = profile.get("age")
        if age is not None:
            age = int(age)
            if edge.age_min is not None and age < edge.age_min:
                return False
            if edge.age_max is not None and age > edge.age_max:
                return False

        sex = self._normalize_text(profile.get("sex"))
        if edge.sex_condition:
            allowed = {
                "male": {"male", "man", "men"},
                "female": {"female", "woman", "women"},
            }.get(edge.sex_condition, {edge.sex_condition})
            if sex not in allowed:
                return False

        ethnicity = self._normalize_text(profile.get("ethnicity"))
        if edge.ethnicity_condition:
            normalized_condition = edge.ethnicity_condition.replace("_", " ")
            if ethnicity not in {
                edge.ethnicity_condition,
                normalized_condition,
            }:
                aliases = {
                    "white": "caucasian",
                    "black": "african american",
                    "hispanic": "hispanic/latino",
                    "latino": "hispanic/latino",
                }
                if aliases.get(ethnicity, ethnicity) != normalized_condition:
                    return False

        return True

    def infer(self, profile: dict[str, Any]) -> dict[str, Any]:
        activations = self.build_factor_activations(profile)

        raw_scores: dict[str, float] = {}
        eligible_weight_sum: dict[str, float] = {}
        contributions: dict[str, list[dict[str, Any]]] = {}

        for edge in self.edges:
            if edge.source not in activations:
                continue
            if not self._edge_is_eligible(edge, profile):
                continue

            source_activation = activations[edge.source]
            contribution = source_activation * edge.weight

            raw_scores[edge.target] = raw_scores.get(edge.target, 0.0) + contribution
            eligible_weight_sum[edge.target] = (
                eligible_weight_sum.get(edge.target, 0.0) + edge.weight
            )
            contributions.setdefault(edge.target, []).append(
                {
                    "source": edge.source,
                    "source_activation": round(source_activation, 4),
                    "relation": edge.relation,
                    "weight": round(edge.weight, 4),
                    "contribution": round(contribution, 4),
                    "evidence": edge.evidence,
                }
            )

        features: dict[str, dict[str, Any]] = {}
        for target, raw_score in raw_scores.items():
            denominator = eligible_weight_sum[target]
            normalized_score = raw_score / denominator if denominator else 0.0
            node = self.nodes[target]
            features[target] = {
                "label": node.get("label", target),
                "score": round(self._clamp(normalized_score), 4),
                "raw_score": round(raw_score, 4),
                "category": node.get("category", ""),
                "region": node.get("region", ""),
                "renderable": node.get("renderable", "").lower() == "true",
                "aging_related": node.get("aging_related", "").lower() == "true",
                "contributions": sorted(
                    contributions[target],
                    key=lambda item: item["contribution"],
                    reverse=True,
                ),
            }

        ranked = dict(
            sorted(
                features.items(),
                key=lambda item: (-item[1]["score"], item[0]),
            )
        )

        aging_representation = {
            feature_id: data["score"]
            for feature_id, data in ranked.items()
            if data["renderable"] and data["aging_related"]
        }

        return {
            "profile": profile,
            "factor_activations": activations,
            "aging_representation": aging_representation,
            "features": ranked,
            "metadata": {
                "engine_version": "0.1.0",
                "normalization": "weighted_mean_over_eligible_incoming_edges",
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AVATAR Knowledge Engine.")
    parser.add_argument("--nodes", required=True, help="Path to nodes.csv")
    parser.add_argument("--edges", required=True, help="Path to weighted edges.csv")
    parser.add_argument("--profile", required=True, help="Path to profile JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    with Path(args.profile).open("r", encoding="utf-8") as handle:
        profile = json.load(handle)

    engine = KnowledgeEngine(args.nodes, args.edges)
    result = engine.infer(profile)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved AVATAR aging representation to {output_path}")


if __name__ == "__main__":
    main()