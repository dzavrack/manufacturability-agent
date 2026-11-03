"""
Unit tests for checks.py on simple shapes with known answers.

Run with:  pytest
These use generated meshes (a cube, a thin plate), so the expected values
can be worked out by hand rather than trusted from the code under test.
"""
import os
import tempfile

import pytest
import trimesh

from checks import (
    check_bounding_box,
    check_overhangs,
    check_wall_thickness,
    check_watertight,
    load_part_mm,
)


def box(x, y, z):
    """Axis-aligned box sitting on the build plate (z from 0 to z)."""
    m = trimesh.creation.box(extents=(x, y, z))
    m.apply_translation((0, 0, z / 2))
    return m


def test_cube_is_watertight():
    assert check_watertight(box(20, 20, 20))["pass"]


def test_mesh_with_a_hole_is_not_watertight():
    m = box(20, 20, 20)
    m.update_faces([i for i in range(len(m.faces)) if i != 0])  # delete one triangle
    assert not check_watertight(m)["pass"]


def test_thick_cube_passes_wall_thickness():
    r = check_wall_thickness(box(20, 20, 20), min_thickness_mm=0.8)
    assert r["pass"]
    assert r["detail"]["measured_min_mm"] == pytest.approx(20, abs=0.1)


def test_thin_plate_fails_wall_thickness_at_its_real_thickness():
    r = check_wall_thickness(box(20, 20, 0.4), min_thickness_mm=0.8)
    assert not r["pass"]
    # the thinnest reading should be the plate's actual 0.4 mm, not a ray artifact
    assert r["detail"]["measured_min_mm"] == pytest.approx(0.4, abs=0.01)


def test_only_downward_faces_are_flagged_as_overhangs():
    # For a cube, the only downward-facing surface is the bottom face (1/6 of the area).
    # Note: this check does not yet exclude faces resting on the build plate,
    # so the bottom face is counted. See "Known limitations" in the README.
    r = check_overhangs(box(20, 20, 20), build_direction=(0, 0, 1))
    assert r["detail"]["flagged_area_mm2"] == pytest.approx(400, rel=1e-6)
    assert r["detail"]["flagged_area_fraction"] == pytest.approx(1 / 6, rel=1e-6)


def test_vertical_walls_never_need_support():
    # Lying the cube on its side changes nothing: still exactly one face points down.
    r = check_overhangs(box(20, 20, 20), build_direction=(1, 0, 0))
    assert r["detail"]["flagged_area_fraction"] == pytest.approx(1 / 6, rel=1e-6)


def test_bounding_box_fit():
    assert check_bounding_box(box(100, 100, 100), max_bed_mm=(220, 220, 250))["pass"]
    assert not check_bounding_box(box(300, 50, 50), max_bed_mm=(220, 220, 250))["pass"]


def test_inch_stl_is_converted_to_mm():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "one_inch_cube.stl")
        trimesh.creation.box(extents=(1, 1, 1)).export(path)
        mesh = load_part_mm(path, assume_units="in")
    assert mesh.bounding_box.extents == pytest.approx([25.4, 25.4, 25.4])


def test_unknown_units_are_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cube.stl")
        trimesh.creation.box(extents=(1, 1, 1)).export(path)
        with pytest.raises(ValueError):
            load_part_mm(path, assume_units="furlongs")


@pytest.mark.parametrize("part", ["parts/thread_saver_25_20.stl", "parts/thread_saver_375_16.stl"])
def test_sample_parts_are_valid_solids(part):
    mesh = load_part_mm(part, assume_units="in")
    assert check_watertight(mesh)["pass"]
