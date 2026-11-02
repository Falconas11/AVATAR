"""Visibility regressions using synthetic pixels and the licensed NASA fixture."""
import numpy as np
from experiments.geometry_renderer.comparison import (applied_shifts, crop_view,
    difference_view, face_crop_bounds, wipe_view)
from experiments.geometry_renderer.core import DEFAULT_STRENGTH, render, signed_areas


def test_default_demo_has_visible_bounded_displacements(licensed_prepared):
    result = render(licensed_prepared, current_age=30, target_age=65)
    shifts = applied_shifts(licensed_prepared, result.points)
    face_height = np.ptp(licensed_prepared.points[:468,1])
    # Test an observable demo, not just "one pixel changed". These thresholds
    # are software visibility requirements, not biological calibration claims.
    assert DEFAULT_STRENGTH == .7
    assert shifts['nose'] > 1
    assert .02*face_height < shifts['maximum'] < .1*face_height
    assert result.safety_scale == 1
    assert (signed_areas(result.points, licensed_prepared.triangles) > 0).all()
    bounds = face_crop_bounds(licensed_prepared)
    original = crop_view(licensed_prepared.image, bounds)
    warped = crop_view(result.image, bounds)
    assert original.shape == warped.shape
    assert original.shape[0] < licensed_prepared.image.shape[0]/2
    assert np.abs(original.astype(float)-warped).mean() > 1


def test_extreme_controls_on_real_mesh_remain_valid(licensed_prepared):
    result = render(licensed_prepared, current_age=18, target_age=100, nose=1, cheeks=1, jowls=1)
    areas = signed_areas(result.points, licensed_prepared.triangles)
    assert (areas >= .1*signed_areas(licensed_prepared.points,licensed_prepared.triangles)).all()
    np.testing.assert_array_equal(result.points[468:],licensed_prepared.points[468:])


def test_inspection_views_do_not_change_source_pixels(synthetic):
    result = render(synthetic, current_age=20, target_age=80)
    bounds = face_crop_bounds(synthetic)
    original = crop_view(synthetic.image, bounds)
    warped = crop_view(result.image, bounds)
    before = original.copy()
    np.testing.assert_array_equal(wipe_view(original,warped,0),warped)
    np.testing.assert_array_equal(wipe_view(original,warped,100),original)
    split = round(original.shape[1]/2)
    wipe = wipe_view(original,warped,50)
    np.testing.assert_array_equal(wipe[:,:split],original[:,:split])
    np.testing.assert_array_equal(wipe[:,split+1:],warped[:,split+1:])
    assert not difference_view(original,original).any()
    assert difference_view(original,warped).any()
    np.testing.assert_array_equal(original,before)


def test_reset_displacement_metrics_are_zero(synthetic):
    result = render(synthetic, current_age=30, target_age=30)
    assert all(value == 0 for value in applied_shifts(synthetic,result.points).values())
