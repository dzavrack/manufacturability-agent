"""
Diagnose whether the very small wall-thickness readings are real or a
ray-casting artifact on the helical thread geometry.
"""
import numpy as np
from checks import load_part_mm

mesh = load_part_mm("parts/thread_saver_25_20.stl", assume_units="in")

rng = np.random.default_rng(0)
import trimesh
points, face_idx = trimesh.sample.sample_surface(mesh, 2000)
normals = mesh.face_normals[face_idx]
eps = 1e-4
origins = points - normals * eps
directions = -normals

locations, index_ray, index_tri = mesh.ray.intersects_location(
    ray_origins=origins, ray_directions=directions, multiple_hits=False
)
thicknesses = np.linalg.norm(locations - origins[index_ray], axis=1)
hit_normals = mesh.face_normals[index_tri]
ray_dirs_for_hits = directions[index_ray]

# for a genuine "exit through the opposing wall" hit, the hit face's
# outward normal should be roughly PARALLEL to the ray direction
# (imagine a ray crossing a thin flat plate: it enters one face and
# exits the far face, whose outward normal points the same way the
# ray was traveling). a grazing/artifact hit -- skimming an oblique
# thread flank -- will have a normal closer to PERPENDICULAR to the
# ray direction.
alignment = np.einsum("ij,ij->i", hit_normals, ray_dirs_for_hits)

thin_mask = thicknesses < 0.8
print(f"thin samples: {thin_mask.sum()} / {len(thicknesses)}")
print(f"alignment (dot of hit-normal, ray-dir) for THIN hits: "
      f"mean={alignment[thin_mask].mean():.3f}, median={np.median(alignment[thin_mask]):.3f}")
print(f"alignment for all hits: mean={alignment.mean():.3f}, median={np.median(alignment):.3f}")
print(f"alignment for NON-thin hits: mean={alignment[~thin_mask].mean():.3f}")

# a stricter true-hit filter: only count it as a real wall-thickness
# reading if the ray exits roughly straight through (alignment > 0.7,
# i.e. within ~45deg of parallel)
clean_mask = alignment > 0.7
print(f"\nwith grazing-hit filter (alignment > 0.7): {clean_mask.sum()} / {len(thicknesses)} kept")
clean_thicknesses = thicknesses[clean_mask]
if len(clean_thicknesses):
    print(f"min={clean_thicknesses.min():.3f}mm  median={np.median(clean_thicknesses):.3f}mm")
    still_thin = (clean_thicknesses < 0.8).sum()
    print(f"still under 0.8mm after filtering: {still_thin} / {len(clean_thicknesses)}")
