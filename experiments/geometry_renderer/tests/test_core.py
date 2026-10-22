import numpy as np
import pytest
from experiments.geometry_renderer.core import (REGIONS, displacement_field, inverse_maps,
    overlay, prepare_face, render, safe_destination, signed_areas)


def field(prepared, **kwargs):
    return displacement_field(prepared.points, prepared.face_count,
        **({'current_age': 20, 'target_age': 80, 'nose': .5, 'cheeks': .5, 'jowls': .5} | kwargs))

@pytest.mark.parametrize('kwargs', [{'nose': 0, 'cheeks': 0, 'jowls': 0}, {'target_age': 20}])
def test_zero_identity_is_pixel_exact(synthetic, kwargs):
    options = dict(current_age=20, target_age=80, nose=.5, cheeks=.5, jowls=.5) | kwargs
    result = render(synthetic, **options)
    np.testing.assert_array_equal(result.image, synthetic.image)
    np.testing.assert_array_equal(result.points, synthetic.points)

@pytest.mark.parametrize('region', list(REGIONS))
def test_authored_strength_and_age_monotonicity(synthetic, region):
    options = dict(nose=0, cheeks=0, jowls=0)
    low = field(synthetic, **(options | {region: .25}))
    high = field(synthetic, **(options | {region: .5}))
    np.testing.assert_allclose(high, 2*low)
    assert high[:, 1].max() > 0
    assert np.all(high[:, 0] == 0)
    young = field(synthetic, target_age=50, **(options | {region: .5}))
    np.testing.assert_allclose(high, 2*young)
    assert np.all(high[synthetic.face_count:] == 0)


def test_compact_falloff(synthetic):
    delta = field(synthetic, nose=1, cheeks=0, jowls=0)
    assert delta[REGIONS['nose'][0], 1] > 0
    assert delta[REGIONS['jowls'][0], 1] == 0

@pytest.mark.parametrize('strength', [0, .25, .5, 1])
def test_bounded_noninverted_triangles_and_pinned_border(synthetic, strength):
    dest, scale = safe_destination(synthetic, field(synthetic, nose=strength, cheeks=strength, jowls=strength))
    assert 0 <= scale <= 1
    assert (signed_areas(dest, synthetic.triangles) > 0).all()
    assert (signed_areas(dest, synthetic.triangles) >= .1*signed_areas(synthetic.points, synthetic.triangles)).all()
    assert (dest >= 0).all() and (dest <= 159).all()
    np.testing.assert_array_equal(dest[468:], synthetic.points[468:])


def test_safety_limiter_reduces_dangerous_field(synthetic):
    delta = field(synthetic)*1000
    dest, scale = safe_destination(synthetic, delta)
    assert scale < 1
    assert (signed_areas(dest, synthetic.triangles) > 0).all()
    assert (dest >= 0).all() and (dest <= 159).all()


def test_inverse_map_complete_and_shared_edges(synthetic):
    # A coordinate-ramp texture reveals holes or bad inverse interpolation.
    x, y = inverse_maps(synthetic.points, synthetic.points, synthetic.triangles, 160, 160)
    expected_y, expected_x = np.indices((160, 160))
    np.testing.assert_allclose(x, expected_x, atol=1e-4)
    np.testing.assert_allclose(y, expected_y, atol=1e-4)
    result = render(synthetic, current_age=20, target_age=80, nose=1, cheeks=1, jowls=1)
    assert result.image.shape == synthetic.image.shape
    assert np.any(result.image != synthetic.image)
    for index in (0, -1):
        np.testing.assert_array_equal(result.image[index], synthetic.image[index])
        np.testing.assert_array_equal(result.image[:, index], synthetic.image[:, index])


def test_render_and_overlay_do_not_mutate_prepared(synthetic):
    before = (synthetic.image.copy(), synthetic.points.copy(), synthetic.triangles.copy())
    result = render(synthetic, current_age=20, target_age=80)
    painted = overlay(result.image, result.points, synthetic.triangles)
    assert np.any(painted != result.image)
    for actual, expected in zip((synthetic.image, synthetic.points, synthetic.triangles), before):
        np.testing.assert_array_equal(actual, expected)

@pytest.mark.parametrize('kwargs', [{'nose': -1}, {'nose': float('nan')}, {'jowls': 2},
    {'current_age': 17}, {'target_age': 101}, {'target_age': 19}, {'target_age': 20.5}])
def test_invalid_controls(synthetic, kwargs):
    with pytest.raises(ValueError):
        field(synthetic, **kwargs)


def test_bad_mesh_rejected(synthetic):
    points = synthetic.points[:468].copy()
    points[1] = points[0]
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_face(synthetic.image, points)
