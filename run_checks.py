import json
from checks import load_part_mm, check_watertight, check_wall_thickness, check_overhangs, check_bounding_box

for path in ["parts/thread_saver_25_20.stl", "parts/thread_saver_375_16.stl"]:
    print(f"\n========== {path} ==========")
    mesh = load_part_mm(path, assume_units="in")
    for check in [check_watertight, check_wall_thickness, check_overhangs, check_bounding_box]:
        result = check(mesh)
        print(json.dumps(result, indent=2))
