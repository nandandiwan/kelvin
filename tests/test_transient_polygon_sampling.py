"""Layer colors must not sample a concave gap or a neighboring z layer."""

import numpy as np
import pytest

gdstk = pytest.importorskip("gdstk")
pytest.importorskip("dolfinx")

from cases.render_transient_layers import _dof_indices_per_volume


def _region(footprint, z0=0.0, z1=0.01):
    return {"name": "test_layer", "z_min_um": z0, "z_max_um": z1,
            "volumes": [{"footprint_xy_um": footprint}]}


def test_concave_polygon_excludes_bbox_gap_and_neighboring_layers():
    region = _region([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    points = np.array([[0.5, 0.5, 0.005], [1.5, 1.5, 0.005],
                       [0.5, 0.5, 0.04], [0.5, 0.5, -0.04]])
    selected = _dof_indices_per_volume([region], points)
    assert len(selected) == 1
    np.testing.assert_array_equal(selected[0][1], [0])


def test_polygon_boundaries_allow_roundoff_but_not_physical_padding():
    region = _region([(0, 0), (1, 0), (1, 1), (0, 1)])
    points = np.array([[1, 0.5, 0.01], [1 + 5e-10, 0.5, 0.01 + 5e-10],
                       [1 + 1e-6, 0.5, 0.005], [0.5, 0.5, 0.01 + 1e-6]])
    selected = _dof_indices_per_volume([region], points)
    np.testing.assert_array_equal(selected[0][1], [0, 1])


def test_hole_in_gds_polygon_is_not_a_temperature_sample():
    outer = gdstk.rectangle((0, 0), (3, 3))
    hole = gdstk.rectangle((1, 1), (2, 2))
    ring = gdstk.boolean([outer], [hole], "not", precision=1e-6)
    assert len(ring) == 1
    region = _region(ring[0].points.tolist())
    selected = _dof_indices_per_volume([region], np.array([[0.5, 0.5, 0.005], [1.5, 1.5, 0.005]]))
    np.testing.assert_array_equal(selected[0][1], [0])


@pytest.mark.parametrize("point", [[1.5, 1.5, 0.005], [0.5, 0.5, 0.1]])
def test_missing_in_prism_samples_fail_instead_of_using_nearest_node(point):
    region = _region([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    with pytest.raises(ValueError, match="No temperature nodes lie in polygon prism"):
        _dof_indices_per_volume([region], np.array([point]))
