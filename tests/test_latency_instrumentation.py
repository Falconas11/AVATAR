import ast
from pathlib import Path


def test_profiled_modules_parse():
    for path in (Path("api/main.py"), Path("avatar/image_renderer.py")):
        ast.parse(path.read_text(encoding="utf-8"))


def test_api_contains_pipeline_markers():
    text = Path("api/main.py").read_text(encoding="utf-8")
    for marker in (
        "Knowledge Engine:",
        "Prompt builder:",
        "Renderer total:",
        "Response image encoding:",
        "TOTAL generation:",
    ):
        assert marker in text


def test_renderer_contains_openai_marker():
    text = Path("avatar/image_renderer.py").read_text(encoding="utf-8")
    assert "OpenAI image edit:" in text
    assert "time.perf_counter()" in text
