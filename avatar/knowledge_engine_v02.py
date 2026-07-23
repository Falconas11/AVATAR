from __future__ import annotations

import csv
from collections import defaultdict, deque
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
    """AVATAR v0.2 multi-hop knowledge propagation engine.

    Supported paths include:
        factor -> mechanism -> feature
        factor -> mechanism -> mechanism -> feature
        factor -> feature  (legacy/direct evidence)

    Each node is computed once in topological order. Incoming active signals are
    aggregated with a weighted mean, preserving the v0.1 score range [0, 1].
    """

    def __init__(self, nodes_path: str | Path, edges_path: str | Path) -> None:
        self.nodes_path = Path(nodes_path)
        self.edges_path = Path(edges_path)
        self.nodes = self._load_nodes(self.nodes_path)
        self.edges = self._load_edges(self.edges_path)
        self._validate_graph()
        self.topological_order = self._topological_sort()

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
        result: list[Edge] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                result.append(
                    Edge(
                        source=row["source"],
                        target=row["target"],
                        relation=row.get("relation", ""),
                        weight=float(row.get("weight") or 1.0),
                        region=row.get("region", ""),
                        age_min=cls._parse_optional_int(row.get("age_min")),
                        age_max=cls._parse_optional_int(row.get("age_max")),
                        sex_condition=(row.get("sex_condition") or "").strip().lower(),
                        ethnicity_condition=(row.get("ethnicity_condition") or "").strip().lower(),
                        evidence=row.get("evidence", ""),
                        notes=row.get("notes", ""),
                    )
                )
        return result

    def _validate_graph(self) -> None:
        missing: list[tuple[str, str]] = []
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                missing.append((edge.source, edge.target))
            if edge.weight < 0:
                raise ValueError(f"Negative edge weight: {edge.source} -> {edge.target}")
        if missing:
            raise ValueError(f"Edges reference missing nodes: {missing[:10]}")

    def _topological_sort(self) -> list[str]:
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1

        queue = deque(sorted(n for n, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for target in adjacency[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)

        if len(order) != len(self.nodes):
            cyclic = sorted(n for n, degree in indegree.items() if degree > 0)
            raise ValueError(f"Knowledge graph contains a cycle involving: {cyclic[:10]}")
        return order

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value or "").strip().lower()

    def build_factor_activations(self, profile: dict[str, Any]) -> dict[str, float]:
        activations: dict[str, float] = {}
        factors = profile.get("factors", {})
        if not isinstance(factors, dict):
            raise TypeError("profile['factors'] must be an object/dictionary.")

        for factor_id, value in factors.items():
            if factor_id not in self.nodes:
                raise KeyError(f"Unknown factor node: {factor_id}")
            if self.nodes[factor_id].get("layer") != "factor":
                raise ValueError(f"{factor_id} is not a factor node.")
            activations[factor_id] = self._clamp(float(value))

        sex = self._normalize_text(profile.get("sex"))
        if sex in {"male", "man", "men"} and "men" in self.nodes:
            activations["men"] = 1.0
        elif sex in {"female", "woman", "women"} and "women" in self.nodes:
            activations["women"] = 1.0

        aliases = {
            "asian": "asian", "caucasian": "caucasian", "white": "caucasian",
            "african american": "african_american", "black": "african_american",
            "hispanic": "hispanic_latino", "latino": "hispanic_latino",
            "hispanic/latino": "hispanic_latino",
        }
        ethnicity = self._normalize_text(profile.get("ethnicity"))
        node_id = aliases.get(ethnicity)
        if node_id in self.nodes:
            activations[node_id] = 1.0
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
            condition = edge.ethnicity_condition.replace("_", " ")
            aliases = {
                "white": "caucasian", "black": "african american",
                "hispanic": "hispanic/latino", "latino": "hispanic/latino",
            }
            if aliases.get(ethnicity, ethnicity) != condition:
                return False
        return True

    def infer(self, profile: dict[str, Any]) -> dict[str, Any]:
        activations = self.build_factor_activations(profile)
        incoming: dict[str, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            if self._edge_is_eligible(edge, profile):
                incoming[edge.target].append(edge)

        contributions: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for target in self.topological_order:
            # Explicit profile values remain authoritative.
            if target in activations:
                continue

            active_edges = [e for e in incoming[target] if e.source in activations]
            if not active_edges:
                continue

            numerator = 0.0
            denominator = 0.0
            for edge in active_edges:
                source_activation = activations[edge.source]
                contribution = source_activation * edge.weight
                numerator += contribution
                denominator += edge.weight
                contributions[target].append({
                    "source": edge.source,
                    "source_layer": self.nodes[edge.source].get("layer", ""),
                    "source_activation": round(source_activation, 4),
                    "relation": edge.relation,
                    "weight": round(edge.weight, 4),
                    "contribution": round(contribution, 4),
                    "evidence": edge.evidence,
                })

            if denominator > 0:
                activations[target] = self._clamp(numerator / denominator)

        mechanisms: dict[str, dict[str, Any]] = {}
        features: dict[str, dict[str, Any]] = {}
        for node_id, score in activations.items():
            node = self.nodes[node_id]
            layer = node.get("layer", "")
            data = {
                "label": node.get("label", node_id),
                "score": round(score, 4),
                "category": node.get("category", ""),
                "region": node.get("region", ""),
                "contributions": sorted(
                    contributions.get(node_id, []),
                    key=lambda item: item["contribution"],
                    reverse=True,
                ),
            }
            if layer == "mechanism":
                mechanisms[node_id] = data
            elif layer == "feature":
                data["renderable"] = node.get("renderable", "").lower() == "true"
                data["aging_related"] = node.get("aging_related", "").lower() == "true"
                features[node_id] = data

        ranked_features = dict(sorted(features.items(), key=lambda x: (-x[1]["score"], x[0])))
        aging_representation = {
            node_id: data["score"] for node_id, data in ranked_features.items()
            if data["renderable"] and data["aging_related"]
        }

        return {
            "profile": profile,
            "factor_activations": {
                k: round(v, 4) for k, v in activations.items()
                if self.nodes[k].get("layer") == "factor"
            },
            "mechanisms": dict(sorted(mechanisms.items(), key=lambda x: (-x[1]["score"], x[0]))),
            "aging_representation": aging_representation,
            "features": ranked_features,
            "metadata": {
                "engine_version": "0.2.0",
                "propagation": "topological_multi_hop_weighted_mean",
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
        }
