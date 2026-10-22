import unittest
from avatar.timeline import build_timeline


class FakeEngine:
    nodes = {"aging": {}, "wrinkles": {}}
    edges = []

    def infer(self, profile):
        age = profile["target_age"]
        return {
            "aging_representation": {"wrinkles": age / 120},
            "features": {"wrinkles": {"label": "Wrinkles", "region": "face", "score": age / 120, "contributions": []}},
            "mechanisms": {},
            "factor_activations": {},
        }


class TimelineTests(unittest.TestCase):
    def test_stages_are_monotonic_and_end_at_target(self):
        result = build_timeline(FakeEngine(), {"age": 20, "target_age": 65, "factors": {}}, stages=4)
        ages = [item["age"] for item in result["snapshots"]]
        self.assertEqual(ages, sorted(set(ages)))
        self.assertEqual(ages[-1], 65)
        self.assertEqual(result["mode"], "knowledge_only")

    def test_reject_backward_progression(self):
        with self.assertRaises(ValueError):
            build_timeline(FakeEngine(), {"age": 65, "target_age": 20})

    def test_no_image_fields(self):
        result = build_timeline(FakeEngine(), {"age": 20, "target_age": 65})
        self.assertTrue(all("image" not in item for item in result["snapshots"]))


if __name__ == "__main__":
    unittest.main()


def test_short_interval_never_includes_current_age():
    result = build_timeline(FakeEngine(), {"age": 64, "target_age": 65}, stages=12)
    assert [s['age'] for s in result['snapshots']] == [65]


def test_preview_does_not_mutate_profile():
    profile = {"age": 20, "target_age": 65, "factors": {"smoking": .5}}
    import copy
    before = copy.deepcopy(profile)
    build_timeline(FakeEngine(), profile)
    assert profile == before
