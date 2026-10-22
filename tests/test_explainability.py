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


def test_real_engine_traces_and_missing_citations():
    from pathlib import Path
    from avatar.knowledge_engine_v03 import KnowledgeEngine
    root = Path(__file__).resolve().parents[1]
    engine = KnowledgeEngine(root/'data/nodes_v03.csv', root/'data/edges_v03.csv')
    result = engine.infer({"age": 20, "target_age": 65, "factors": {"uv_exposure": .7}})
    traces = build_feature_traces(engine, result)
    assert traces
    assert any(step['provisional'] for trace in traces.values() for path in trace['paths'] for step in path['steps'])
    assert all('Provisional pathway hypothesis' not in path['citations'] for trace in traces.values() for path in trace['paths'])


def test_missing_citation_and_zero_contribution():
    fixture = TraceTests(); fixture.setUp()
    fixture.engine.edges[1].evidence = 'N/A'
    path = build_feature_traces(fixture.engine, fixture.inference)['wrinkles']['paths'][0]
    assert path['missing_citation_steps'] == [1]
    assert path['evidence_status'] == 'incomplete_or_provisional'
    fixture.inference['features']['wrinkles']['contributions'][0]['contribution'] = 0
    assert build_feature_traces(fixture.engine, fixture.inference)['wrinkles']['paths'] == []


def test_duplicate_edges_do_not_fabricate_citation():
    fixture = TraceTests(); fixture.setUp()
    fixture.engine.edges.append(Edge('damage', 'wrinkles', 'contributes_to', evidence='Wrong paper'))
    assert build_feature_traces(fixture.engine, fixture.inference)['wrinkles']['paths'] == []
    fixture.inference['features']['wrinkles']['contributions'][0]['edge_index'] = 1
    path = build_feature_traces(fixture.engine, fixture.inference, max_hops=2)['wrinkles']['paths'][0]
    assert path['citations'] == ['Paper A', 'Paper B']
