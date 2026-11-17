import pytest

from gi.lightmap import (
    texel_density,
    uv_overlap,
    dilate_lightmap,
    cosine_hemisphere,
    cluster_lights
)


def test_density_4m_512():
    assert texel_density(4.0, 512) == pytest.approx(128.0)

def test_density_2m_256():
    assert texel_density(2.0, 256) == pytest.approx(128.0)

def test_density_10m_1024():
    assert texel_density(10.0, 1024) == pytest.approx(102.4)

def test_uv_overlap_one_pair():
    assert uv_overlap([(0.0, 0.0, 1.0, 1.0), (0.5, 0.5, 1.5, 1.5)]) == [(0, 1)]

def test_uv_disjoint():
    assert uv_overlap([(0.0, 0.0, 0.5, 0.5), (0.6, 0.6, 1.0, 1.0)]) == []

def test_uv_touching_is_not_overlap():
    assert uv_overlap([(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)]) == []

def test_dilate_one_pass():
    assert dilate_lightmap([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3, 3, 1)[1] == 1.0

def test_dilate_two_passes():
    assert dilate_lightmap([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3, 3, 2)[0] == 1.0

def test_dilate_keeps_center():
    assert dilate_lightmap([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3, 3, 2)[4] == 1.0

def test_cosine_is_unit():
    assert sum(c * c for c in cosine_hemisphere((0.0, 0.0, 1.0), 0, 16)) == pytest.approx(1.0)

def test_cosine_positive_z():
    assert cosine_hemisphere((0.0, 0.0, 1.0), 3, 16)[2] > 0.0

def test_cosine_negative_z():
    assert cosine_hemisphere((0.0, 0.0, -1.0), 3, 16)[2] < 0.0

def test_cluster_counts():
    assert cluster_lights([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], [(0.0, 0.0, 0.0, 5.0, 5.0, 5.0), (5.0, 0.0, 0.0, 15.0, 5.0, 5.0)]) == [1, 1]

def test_cluster_excludes_far():
    assert cluster_lights([(20.0, 0.0, 0.0)], [(0.0, 0.0, 0.0, 5.0, 5.0, 5.0), (5.0, 0.0, 0.0, 15.0, 5.0, 5.0)]) == [0, 0]

def test_cluster_no_lights():
    assert cluster_lights([], [(0.0, 0.0, 0.0, 5.0, 5.0, 5.0), (5.0, 0.0, 0.0, 15.0, 5.0, 5.0)]) == [0, 0]
