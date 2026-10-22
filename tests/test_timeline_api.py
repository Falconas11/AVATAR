"""Offline knowledge API tests: no participant photos or external services."""
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
import api.main as api

PROFILE = {"age": 20, "target_age": 65, "sex": "male", "ethnicity": "Asian", "factors": {"smoking": .4}}

@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Any call to image generation or storage is a regression.
    for name in ("render_portrait", "store_pair", "get_box_token_provider"):
        monkeypatch.setattr(api, name, Mock(side_effect=AssertionError(name + " called")))
    return TestClient(api.app)


def test_knowledge_only_and_real_age_dependent_traces(client):
    response = client.post('/api/timeline', json={"profile": PROFILE, "stages": 4})
    assert response.status_code == 200
    data = response.json()
    assert data['mode'] == 'knowledge_only'
    assert data['snapshots'][-1]['age'] == 65
    young, old = data['snapshots'][0], data['snapshots'][-1]
    assert young['aging_representation'] != old['aging_representation']
    for snapshot in data['snapshots']:
        assert 'image' not in snapshot and 'participant_id' not in snapshot
        for name, trace in snapshot['feature_traces'].items():
            assert trace['score'] == snapshot['aging_representation'][name]
            for path in trace['paths']:
                assert all(step['contribution'] > 0 for step in path['steps'])

@pytest.mark.parametrize('update', [
    {"age": True}, {"age": 20.5}, {"target_age": 20}, {"target_age": 121},
    {"factors": {"unknown": .5}}, {"factors": {"smoking": -1}},
    {"factors": []}, {"age": None}, {"target_age": "Infinity"},
])
def test_invalid_profiles(client, update):
    assert client.post('/api/timeline', json={"profile": {**PROFILE, **update}}).status_code == 400

@pytest.mark.parametrize('stages', [0, 13, True, 1.5, '4'])
def test_invalid_stages(client, stages):
    assert client.post('/api/timeline', json={"profile": PROFILE, "stages": stages}).status_code == 400


def test_missing_profile(client):
    assert client.post('/api/timeline', json={}).status_code == 400


def test_engine_unavailable(client, monkeypatch):
    monkeypatch.setattr(api, 'engine', None)
    assert client.post('/api/timeline', json={"profile": PROFILE}).status_code == 503

@pytest.mark.parametrize('enabled, approved, expected', [
    ('false', 'false', 0), ('true', 'false', 0), ('true', 'true', 1),
])
def test_generation_storage_irb_gates_unchanged(monkeypatch, enabled, approved, expected):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-not-a-credential')
    monkeypatch.setenv('AVATAR_BOX_STORAGE_ENABLED', enabled)
    monkeypatch.setenv('AVATAR_IRB_COLLECTION_APPROVED', approved)
    monkeypatch.delenv('AVATAR_BOX_SYNTHETIC_TEST', raising=False)
    def render(**kwargs):
        kwargs['output_path'].write_bytes(b'synthetic-image-test-bytes')
        return kwargs['output_path'], {}
    monkeypatch.setattr(api, 'render_portrait', render)
    monkeypatch.setattr(api, 'get_box_token_provider', Mock(return_value=object()))
    stored = Mock(return_value=type('Stored', (), {'participant_id': 'P-' + 'a'*32})())
    monkeypatch.setattr(api, 'store_pair', stored)
    response = TestClient(api.app).post('/api/generate', data={'profile_json': __import__('json').dumps(PROFILE)}, files={'image': ('synthetic.png', b'synthetic-input', 'image/png')})
    assert response.status_code == 200
    assert stored.call_count == expected
    assert response.json()['storage_status'] == ('stored' if expected else 'disabled')
