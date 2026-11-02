from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]/'app.py'


def test_streamlit_sample_sliders_cache_overlay_reset_and_clear():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert 'Select a local portrait' in app.info[0].value
    app.checkbox(key='use_sample').check().run()
    assert not app.exception and not app.error
    session = app.session_state['demo_session']
    prepared = session.prepared
    assert prepared is not None and session.detection_count == 1
    assert app.checkbox(key='focus_face').value
    assert app.slider(key='nose').value == .65
    assert session.segmentation_count == 1
    assert session.proposal_count == 1
    app.selectbox(key='skin_quality').select('Fast (1 window)').run()
    proposal_count=session.proposal_count
    app.selectbox(key='lifestyle_scenario').select('UV exposure').run()
    assert app.slider(key='uv_modifier').value==1.
    assert app.slider(key='smoking_modifier').value==0.
    app.checkbox(key='compare_scenarios').check().run()
    assert any('Same-age lifestyle comparison' in heading.value for heading in app.subheader)
    count=session.render_count
    app.selectbox(key='lifestyle_reference').select('smoking').run()
    assert session.render_count==count  # both compositions are cached
    app.selectbox(key='lifestyle_scenario').select('Smoking').run()
    assert session._render_result.plan['metadata']['lifestyle_values']=={'smoking':1.}
    assert app.slider(key='smoking_modifier').value==1.
    assert app.slider(key='uv_modifier').value==0.
    app.selectbox(key='lifestyle_scenario').select('Custom').run()
    app.slider(key='alcohol_modifier').set_value(.5).run()
    assert session._render_result.plan['metadata']['lifestyle_values']['alcohol']==.5
    assert session.proposal_count==proposal_count
    app.checkbox(key='compare_scenarios').uncheck().run()
    app.selectbox(key='comparison_view').select('Before / after wipe').run()
    assert not app.exception and not app.error
    app.slider(key='wipe_position').set_value(30).run()
    app.selectbox(key='comparison_view').select('Pixel difference ×4').run()
    assert any('diagnostic view' in heading.value for heading in app.subheader)
    app.checkbox(key='focus_face').uncheck().run()
    app.selectbox(key='comparison_view').select('Side by side').run()
    app.checkbox(key='focus_face').check().run()
    assert session.prepared is prepared and session.detection_count == 1
    app.slider(key='target_age').set_value(85).run()
    app.slider(key='nose').set_value(.9).run()
    app.checkbox(key='show_mesh').check().run()
    assert not app.exception and not app.error
    assert session.prepared is prepared and session.detection_count == 1
    assert any('not a calibrated prediction' in item.value for item in app.warning)
    assert any('source detection passes: 1' in item.value for item in app.caption)
    # Streamlit generates button keys for these callbacks.
    next(button for button in app.button if button.label == 'Reset aging').click().run()
    assert app.slider(key='target_age').value == app.slider(key='current_age').value
    assert session._render_result.metrics['skin_changed_pixels'] == 0
    assert session.prepared is prepared and session.detection_count == 1
    app.slider(key='current_age').set_value(90).run()
    assert not app.exception and not app.error
    assert app.slider(key='target_age').value >= 90
    next(button for button in app.button if button.label == 'Clear image from this session').click().run()
    assert not app.exception and not app.error
    assert session.prepared is None and not session._render_cache
    assert not app.checkbox(key='use_sample').value
