"""
Step 1: just look at the raw mesh data. No manufacturability logic yet.

This is the equivalent of opening a part in a CAD viewer and checking
"is this file even valid" before doing any real analysis on it.
"""
import sys
import trimesh


def inspect(path: str) -> None:
    mesh = trimesh.load(path)

    print(f"\n=== {path} ===")
    print(f"vertices:        {len(mesh.vertices)}")
    print(f"triangles:       {len(mesh.faces)}")
    print(f"watertight:      {mesh.is_watertight}")   # every edge shared by exactly 2 triangles
    print(f"winding_consistent: {mesh.is_winding_consistent}")  # normals all point outward consistently
    print(f"volume (mm^3):   {mesh.volume:.2f}")
    print(f"surface area (mm^2): {mesh.area:.2f}")
    print(f"bounding box (mm): {mesh.bounding_box.extents}")
    print(f"is_convex:       {mesh.is_convex}")


if __name__ == "__main__":
    for f in ["parts/thread_saver_25_20.stl", "parts/thread_saver_375_16.stl"]:
        inspect(f)
