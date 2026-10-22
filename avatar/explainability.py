"""Evidence-aware explanation paths for AVATAR aging features.

This module describes the knowledge engine's recorded decisions; it does not
claim to explain image-model internals or establish individual causation.
"""
from __future__ import annotations

from typing import Any


def _provisional(evidence: str, notes: str) -> bool:
    value = f"{evidence} {notes}".lower()
    return any(word in value for word in ("provisional", "hypothesis", "unverified", "citation needed"))


def _missing_citation(value: str) -> bool:
    return value.strip().lower() in {"", "n/a", "na", "none", "unknown", "tbd", "citation needed"} or _provisional(value, "")


def build_feature_traces(engine: Any, inference: dict[str, Any], *, max_paths: int = 5, max_hops: int = 4) -> dict[str, Any]:
    """Return bounded, evidence-labelled active graph paths for renderable features.

    Paths follow *actual recorded contributions* rather than all theoretical edges.
    A path is 'documented' only when every edge has a non-provisional citation.
    """
    if max_paths < 1 or max_hops < 1:
        raise ValueError("max_paths and max_hops must be positive")

    features = inference.get("features", {})
    mechanisms = inference.get("mechanisms", {})
    factors = inference.get("factor_activations", {})
    nodes = engine.nodes
    edge_index = {}
    for edge in engine.edges:
        edge_index.setdefault((edge.source, edge.target, edge.relation), []).append(edge)

    def contributions(node_id: str) -> list[dict[str, Any]]:
        return (features.get(node_id) or mechanisms.get(node_id) or {}).get("contributions", [])

    def walk(node_id: str, seen: frozenset[str], remaining: int) -> list[list[dict[str, Any]]]:
        if node_id in seen:
            return []
        if node_id in factors:
            return [[]] if factors[node_id] > 0 else []
        if remaining == 0:
            return []
        paths: list[list[dict[str, Any]]] = []
        for c in contributions(node_id):
            source = c["source"]
            if c.get("contribution", 0) <= 0:
                continue
            candidates = edge_index.get((source, node_id, c["relation"]), [])
            recorded_index = c.get("edge_index")
            if recorded_index is not None:
                if type(recorded_index) is not int or not 0 <= recorded_index < len(engine.edges):
                    continue
                edge = engine.edges[recorded_index]
                if edge not in candidates:
                    continue
            elif len(candidates) == 1:
                edge = candidates[0]
            else:
                # Ambiguous legacy records cannot identify the evidence used.
                continue
            step = {
                "source": source,
                "target": node_id,
                "relation": edge.relation,
                "weight": edge.weight,
                "contribution": c["contribution"],
                "evidence": edge.evidence.strip(),
                "notes": edge.notes.strip(),
                "provisional": _provisional(edge.evidence, edge.notes),
                "missing_citation": _missing_citation(edge.evidence),
            }
            for prefix in walk(source, seen | {node_id}, remaining - 1):
                paths.append(prefix + [step])
        return paths

    result: dict[str, Any] = {}
    for feature_id in inference.get("aging_representation", {}):
        feature = features.get(feature_id, {})
        paths = walk(feature_id, frozenset(), max_hops)
        # Prefer documented paths, then higher-contribution paths.
        paths.sort(key=lambda p: (
            not all(not _missing_citation(s["evidence"]) and not s["provisional"] for s in p),
            -min((s["contribution"] for s in p), default=0),
            len(p),
        ))
        explanations = []
        for steps in paths[:max_paths]:
            documented = all(not _missing_citation(s["evidence"]) and not s["provisional"] for s in steps)
            explanations.append({
                "nodes": [steps[0]["source"]] + [s["target"] for s in steps],
                "steps": steps,
                "evidence_status": "documented" if documented else "incomplete_or_provisional",
                "citations": list(dict.fromkeys(s["evidence"] for s in steps if not _missing_citation(s["evidence"]))),
                "missing_citation_steps": [i for i, s in enumerate(steps) if _missing_citation(s["evidence"])],
            })
        result[feature_id] = {
            "label": feature.get("label", nodes.get(feature_id, {}).get("label", feature_id)),
            "region": feature.get("region", ""),
            "score": inference["aging_representation"][feature_id],
            "evidence_status": "documented" if explanations and all(p["evidence_status"] == "documented" for p in explanations) else "incomplete_or_provisional",
            "paths": explanations,
            "disclaimer": "General evidence and engine decision trace; not a personal medical prediction or explanation of image-model internals.",
        }
    return result
