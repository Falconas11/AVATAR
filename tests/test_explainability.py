import unittest
from dataclasses import dataclass
from avatar.explainability import build_feature_traces


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    evidence: str = ""
    notes: str = ""


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.engine = type("Engine", (), {
            "nodes": {"uv": {}, "damage": {}, "wrinkles": {}},
            "edges": [
                Edge("uv", "damage", "activates", evidence="Paper A"),
                Edge("damage", "wrinkles", "contributes_to", evidence="Paper B"),
            ],
        })()
        self.inference = {
            "factor_activations": {"uv": 0.8},
            "mechanisms": {"damage": {"contributions": [{"source": "uv", "relation": "activates", "contribution": 0.8}]}},
            "features": {"wrinkles": {"label": "Wrinkles", "region": "eyes", "score": 0.8,
                "contributions": [{"source": "damage", "relation": "contributes_to", "contribution": 0.8}]}},
            "aging_representation": {"wrinkles": 0.8},
        }

    def test_documented_multihop(self):
        trace = build_feature_traces(self.engine, self.inference)["wrinkles"]
        self.assertEqual(trace["paths"][0]["nodes"], ["uv", "damage", "wrinkles"])
        self.assertEqual(trace["paths"][0]["evidence_status"], "documented")

    def test_provisional_flag(self):
        self.engine.edges[1].notes = "Provisional pathway hypothesis"
        trace = build_feature_traces(self.engine, self.inference)["wrinkles"]
        self.assertEqual(trace["paths"][0]["evidence_status"], "incomplete_or_provisional")

    def test_no_fabricated_path(self):
        self.inference["mechanisms"]["damage"]["contributions"] = []
        self.assertEqual(build_feature_traces(self.engine, self.inference)["wrinkles"]["paths"], [])


if __name__ == "__main__":
    unittest.main()
