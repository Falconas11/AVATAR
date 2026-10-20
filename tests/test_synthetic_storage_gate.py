"""Unit tests for local-only synthetic Box upload gate."""
from types import SimpleNamespace

from api.main import synthetic_box_test_allowed


def request_from(host):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def configure(monkeypatch):
    monkeypatch.setenv("AVATAR_BOX_SYNTHETIC_TEST", "true")
    monkeypatch.setenv("AVATAR_ENV", "local")
    monkeypatch.setenv("BOX_STUDY_FOLDER_ID", "426895667507")
    monkeypatch.delenv("RENDER", raising=False)


def test_loopback_synthetic_mode(monkeypatch):
    configure(monkeypatch)
    assert synthetic_box_test_allowed(request_from("127.0.0.1"))
    assert synthetic_box_test_allowed(request_from("::1"))


def test_remote_clients_rejected(monkeypatch):
    configure(monkeypatch)
    assert not synthetic_box_test_allowed(request_from("192.168.1.10"))


def test_production_rejected(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv("RENDER", "true")
    assert not synthetic_box_test_allowed(request_from("127.0.0.1"))


def test_wrong_folder_rejected(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv("BOX_STUDY_FOLDER_ID", "999")
    assert not synthetic_box_test_allowed(request_from("127.0.0.1"))


def test_explicit_opt_in_required(monkeypatch):
    configure(monkeypatch)
    monkeypatch.delenv("AVATAR_BOX_SYNTHETIC_TEST")
    assert not synthetic_box_test_allowed(request_from("127.0.0.1"))
