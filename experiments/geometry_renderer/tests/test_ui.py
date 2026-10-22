from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]/'app.py'


def test_streamlit_sample_sliders_cache_overlay_reset_and_clear():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert 'Select a local portrait' in app.info[0].value
    app.checkbox(key='use_sample').check().run()
    assert not app.exception and not app.error
    session = app.session_state['demo_session']
    prepared = session.prepared
    assert prepared is not None and session.detection_count == 1
    app.slider(key='target_age').set_value(85).run()
    app.slider(key='nose').set_value(.9).run()
    app.checkbox(key='show_mesh').check().run()
    assert not app.exception and not app.error
    assert session.prepared is prepared and session.detection_count == 1
    assert any('not a scientifically calibrated prediction' in item.value for item in app.warning)
    assert any('detection passes for this image: 1' in item.value for item in app.caption)
    # Streamlit generates button keys for these callbacks.
    next(button for button in app.button if button.label == 'Reset deformation').click().run()
    assert app.slider(key='target_age').value == app.slider(key='current_age').value
    assert app.slider(key='nose').value == 0
    assert session.prepared is prepared and session.detection_count == 1
    app.slider(key='current_age').set_value(90).run()
    assert not app.exception and not app.error
    assert app.slider(key='target_age').value >= 90
    next(button for button in app.button if button.label == 'Clear image from this session').click().run()
    assert not app.exception and not app.error
    assert session.prepared is None
    assert not app.checkbox(key='use_sample').value
