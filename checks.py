"""
Deterministic manufacturability checks for FDM/SLA 3D-printed parts.

Design principle (this is the important part to be able to explain):
    These functions do NOT use an LLM. They are plain geometry math.
    Claude's job later is to decide *which* of these to call, in what
    order, and to turn the numeric output into a plain-language verdict.
    The physics/geometry stays deterministic and auditable; the language
    and orchestration is where the LLM adds value.

    The agent (agent.py) calls these real, deterministic tools rather
    than "reasoning" its way to a wall thickness number, which would be
    both unreliable and unauditable.
"""
from __future__ import annotations

import numpy as np
import trimesh


INCH_TO_MM = 25.4


def load_part_mm(path: str, assume_units: str = "in") -> trimesh.Trimesh:
    """Load an STL and normalize it to millimeters.

    STL files carry no unit metadata -- this is a real, common failure
    mode in manufacturability pipelines. We make the assumption explicit
    and overridable instead of silently guessing.
    """
    mesh = trimesh.load(path)
    if assume_units == "in":
        mesh.apply_scale(INCH_TO_MM)
    elif assume_units != "mm":
        raise ValueError(f"unsupported units: {assume_units}")
    return mesh


def check_watertight(mesh: trimesh.Trimesh) -> dict:
    """Prerequisite check: is this a valid, closed solid?

    Ray-casting (used by the wall-thickness check) and slicers both
    require a mesh where every edge borders exactly two triangles and
    normals are consistently oriented. If this fails, every other check
    downstream is unreliable and should not be trusted.
    """
    ok = bool(mesh.is_watertight and mesh.is_winding_consistent)
    return {
        "check": "watertight",
        "pass": ok,
        "detail": {
            "is_watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
        },
        "explanation": (
            "Mesh is a valid closed solid."
            if ok
            else "Mesh has holes or inconsistent normals; geometry checks below may be unreliable."
        ),
    }


def check_wall_thickness(
    mesh: trimesh.Trimesh,
    min_thickness_mm: float = 0.8,
    sample_count: int = 2000,
    seed: int = 0,
    grazing_alignment_threshold: float = 0.7,
) -> dict:
    """Estimate local wall thickness by ray-casting into the solid.

    Method: sample points on the surface, cast a ray from each point
    along its *inward* normal (i.e. -normal), and measure the distance
    to the first surface it hits on the far side. For a thin wall, that
    distance is the wall thickness at that point. This is the same
    technique FDM slicers (e.g. Cura's "thin wall" warning) use.

    min_thickness_mm=0.8 is a common conservative floor for FDM with a
    0.4mm nozzle (roughly 2 perimeter lines); it's a parameter, not a
    law of physics, and a real tool would set it per printer/material.

    Grazing-hit filter: on geometry with oblique surfaces close together
    (e.g. thread flanks), a ray can skim past a nearby face at a shallow
    angle instead of exiting cleanly through the true opposing wall,
    producing a near-zero "thickness" that's a ray-casting artifact, not
    a real defect. A genuine through-wall hit exits a face whose outward
    normal is roughly *parallel* to the ray's direction of travel; a
    grazing hit's normal is closer to *perpendicular* to it. We keep
    only hits where that alignment (dot product) exceeds
    grazing_alignment_threshold, confirmed empirically on this part:
    filtering dropped the false "thin" rate from 22% to under 1% and
    moved the measured minimum from 0.001mm to a plausible 0.34mm.
    """
    points, face_idx = trimesh.sample.sample_surface(mesh, sample_count)
    normals = mesh.face_normals[face_idx]

    # nudge the ray origin slightly inside the surface to avoid
    # immediately re-hitting the same triangle it started on
    eps = 1e-4
    origins = points - normals * eps
    directions = -normals

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=False
    )

    hit_normals = mesh.face_normals[index_tri]
    hit_directions = directions[index_ray]
    alignment = np.einsum("ij,ij->i", hit_normals, hit_directions)
    clean = alignment > grazing_alignment_threshold

    thicknesses = np.linalg.norm(locations[clean] - origins[index_ray][clean], axis=1)
    thin_mask = thicknesses < min_thickness_mm
    thin_points = points[index_ray][clean][thin_mask]

    result = {
        "check": "wall_thickness",
        "pass": bool(thin_mask.sum() == 0),
        "detail": {
            "min_thickness_mm": min_thickness_mm,
            "samples_cast": int(len(origins)),
            "samples_hit": int(len(locations)),
            "samples_kept_after_grazing_filter": int(clean.sum()),
            "measured_min_mm": float(thicknesses.min()) if len(thicknesses) else None,
            "measured_median_mm": float(np.median(thicknesses)) if len(thicknesses) else None,
            "thin_sample_count": int(thin_mask.sum()),
            "thin_sample_fraction": float(thin_mask.sum() / len(thicknesses)) if len(thicknesses) else None,
            "example_thin_locations_mm": thin_points[:5].tolist(),
        },
    }
    if result["pass"]:
        result["explanation"] = (
            f"No walls thinner than {min_thickness_mm}mm detected "
            f"(measured min {result['detail']['measured_min_mm']:.3f}mm)."
        )
    else:
        result["explanation"] = (
            f"{thin_mask.sum()} of {len(thicknesses)} sampled points are thinner than "
            f"{min_thickness_mm}mm (min found: {result['detail']['measured_min_mm']:.3f}mm). "
            "These walls are at risk of not printing cleanly or snapping."
        )
    return result


def check_overhangs(
    mesh: trimesh.Trimesh,
    build_direction: tuple = (0, 0, 1),
    max_overhang_deg: float = 45.0,
) -> dict:
    """Flag faces that will need print supports.

    Method: for every triangle, compute the angle between its outward
    normal and the build (up) direction. A downward-facing face whose
    normal is more than `max_overhang_deg` away from straight-down is
    printing into thin air and needs support material -- this is the
    standard rule of thumb FDM slicers use (45 degrees from vertical is
    the typical default for supportless overhangs).

    We only count faces that face *downward* (negative Z component)
    since upward-facing overhangs aren't a support problem.
    """
    build_dir = np.array(build_direction, dtype=float)
    build_dir /= np.linalg.norm(build_dir)

    normals = mesh.face_normals
    areas = mesh.area_faces

    # angle between each face normal and straight-down (-build_dir):
    #   0deg   = normal points straight down  -> horizontal ceiling, worst overhang
    #   90deg  = normal points sideways       -> vertical wall, no support needed
    #   >90deg = normal points upward         -> not a downward-facing surface at all
    cos_angle = -(normals @ build_dir)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_from_down_deg = np.degrees(np.arccos(cos_angle))
    downward_facing = angle_from_down_deg < 90.0

    # printed-surface angle measured from vertical = 90 - angle_from_down.
    # the common slicer rule of thumb needs support when that exceeds
    # max_overhang_deg (typically 45), which is equivalent to:
    critical = downward_facing & (angle_from_down_deg < (90.0 - max_overhang_deg))

    flagged_area = float(areas[critical].sum())
    total_area = float(areas.sum())

    result = {
        "check": "overhangs",
        "pass": bool(flagged_area == 0.0),
        "detail": {
            "max_overhang_deg": max_overhang_deg,
            "build_direction": build_direction,
            "flagged_face_count": int(critical.sum()),
            "flagged_area_mm2": flagged_area,
            "flagged_area_fraction": flagged_area / total_area if total_area else None,
        },
    }
    if result["pass"]:
        result["explanation"] = "No overhangs beyond the support-free threshold detected."
    else:
        result["explanation"] = (
            f"{critical.sum()} triangles ({flagged_area:.2f}mm^2, "
            f"{result['detail']['flagged_area_fraction']*100:.1f}% of surface area) "
            f"exceed the {max_overhang_deg}-degree support-free overhang limit for build "
            f"direction {build_direction}. These regions will need support material or reorientation."
        )
    return result


def check_bounding_box(mesh: trimesh.Trimesh, max_bed_mm: tuple = (220, 220, 250)) -> dict:
    """Sanity check: does the part fit on a typical consumer FDM bed?"""
    extents = mesh.bounding_box.extents
    fits = bool(all(e <= b for e, b in zip(extents, max_bed_mm)))
    return {
        "check": "bounding_box",
        "pass": fits,
        "detail": {
            "extents_mm": extents.tolist(),
            "bed_mm": max_bed_mm,
        },
        "explanation": (
            f"Part ({extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm) fits a "
            f"{max_bed_mm[0]}x{max_bed_mm[1]}x{max_bed_mm[2]}mm bed."
            if fits
            else f"Part ({extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm) exceeds "
            f"a {max_bed_mm[0]}x{max_bed_mm[1]}x{max_bed_mm[2]}mm bed and won't fit without scaling or splitting."
        ),
    }
