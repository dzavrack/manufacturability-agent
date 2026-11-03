# Manufacturability Agent

[![Tests](https://github.com/dzavrack/manufacturability-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/dzavrack/manufacturability-agent/actions/workflows/tests.yml)

**An AI agent that reviews 3D-printable parts using real geometry analysis rather than guesswork.**

You give it an STL and a question like *"Can I print this on a Prusa MK4 with a 0.4 mm nozzle?"*. Claude decides which checks to run and with what parameters, the Python code does the measurements, and Claude writes a verdict that cites the measured numbers.

```
$ python agent.py parts/thread_saver_25_20.stl --question "Can I print this in PETG on an Ender 3?"
```

**A real run** (the 1/4"-20 thread saver in `parts/`). Claude chose these checks and parameters itself:

```
  -> check_watertight({})
     Mesh is a valid closed solid.
  -> check_wall_thickness({"min_thickness_mm": 0.8})
     6 of 1187 sampled points are thinner than 0.8mm (min found: 0.363mm).
  -> check_overhangs({"build_direction": [0, 0, 1], "max_overhang_deg": 45})
     2105 triangles (143.02mm^2, 36.3% of surface area) exceed the 45-degree limit
  -> check_bounding_box({"max_bed_mm": [220, 220, 250]})
     Part (6.3 x 6.3 x 12.4 mm) fits a 220x220x250mm bed.
  -> check_overhangs({"build_direction": [0, 0, -1], "max_overhang_deg": 45})
     1823 triangles (143.55mm^2, 36.4% of surface area) exceed the 45-degree limit
  -> check_overhangs({"build_direction": [1, 0, 0], "max_overhang_deg": 45})
     860 triangles (23.41mm^2, 5.9% of surface area) exceed the 45-degree limit
```

Its verdict, condensed:

> **VERDICT: PRINTABLE WITH CHANGES.** The part is watertight and fits the bed. The thin-wall readings (6 of 1,187 samples, 0.5%) cluster near the thread start and are probably thread artifacts rather than real thin walls.
>
> | Build direction | Area needing support |
> |---|---|
> | Upright, Z up (as modeled) | 36.3% |
> | Flipped, Z down | 36.4% |
> | On its side, X up | 5.9% |
>
> **Print it on its side**, which cuts the overhang area by about 84%. Supports on 36% of a 6 mm threaded part would be hard to remove and would probably damage the threads. Use 0.1 mm layers, and make a test print to check thread fit.

Every number in the verdict comes from one of the tool calls above. The full log is saved with `--report`.

## Why it's built this way

LLMs are good at deciding *what* to check and at explaining results to people. They are not reliable at producing numbers. If you ask a model "how thick is the thinnest wall?", it will guess. This project keeps the two jobs separate:

| Layer | Who does it | File |
|---|---|---|
| Geometry and physics (wall thickness, overhangs, fit) | Deterministic Python (`trimesh`, `numpy`) | `checks.py` |
| Picking checks and parameters, comparing orientations, writing the verdict | Claude, through tool use | `agent.py` |

Every tool call and its raw result is logged (`--report report.json`), so each claim in the verdict can be traced back to a measurement.

## The checks

- **Watertight / manifold.** Every other check depends on a closed solid with consistent normals, so this runs first. If it fails, the agent says the other results can't be trusted.
- **Wall thickness.** The code samples about 2,000 points on the surface, casts a ray inward from each one, and measures the distance to the opposite wall. This is the same idea slicers use for thin-wall warnings.
- **Overhangs.** It flags downward-facing surfaces steeper than the support-free limit (45° by default) for a given build direction. The agent tries other orientations before it recommends supports.
- **Build volume.** It checks whether the part fits the printer's bed.

STL files carry no unit information, so the loader makes the unit assumption explicit (`--units in|mm`) instead of guessing silently. A wrong guess here is a common real-world failure.

## Engineering note: a false positive in the wall-thickness check

The first version of the wall-thickness check reported **22% of the surface as dangerously thin (minimum 0.001 mm)** on a threaded part that prints fine. `diagnose_thickness.py` traced the cause. On helical thread flanks, rays skim along nearby oblique faces at shallow angles instead of passing straight through the opposite wall.

The fix uses the hit geometry itself. A genuine exit through the far wall hits a face whose outward normal is roughly *parallel* to the ray, while a grazing hit is closer to *perpendicular*. Discarding hits with alignment ≤ 0.7 (about 45° off) **dropped the false-thin rate from 22% to under 1%** and moved the measured minimum from 0.001 mm to a plausible 0.34 mm.

## Tests

`test_geometry.py` checks each function against shapes whose answers are known by hand: a 20 mm cube, a 0.4 mm plate, a 1-inch cube exported as an STL, and a mesh with a hole in it. GitHub Actions runs the tests, plus every check on the sample parts, on each push.

## Known limitations

- **Faces resting on the build plate are counted as overhangs.** The overhang check flags every downward-facing surface, including the bottom face that sits on the bed. That inflates the number for flat-bottomed parts; on a cube it reports 1/6 of the surface. The fix is to ignore faces lying within a layer height of the part's lowest point. It's the next thing on the list.
- **Wall thickness is sampled, not exhaustive.** About 2,000 random surface points are checked, so a very small thin feature can be missed. The sample count is a parameter.
- **Only FDM rules so far.** The thresholds (0.8 mm walls, 45° overhangs) are FDM rules of thumb and can be changed per call.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Anthropic API key in .env

python agent.py parts/thread_saver_25_20.stl                  # agent review
python agent.py parts/thread_saver_375_16.stl --report r.json # also save the audit log
python run_checks.py                                          # raw checks only, no AI
pytest                                                        # unit tests
python inspect_part.py                                        # mesh stats
```

The sample parts in `parts/` are thread savers for 1/4"-20 and 3/8"-16 threads, exported in inches.

## Files

```
agent.py               Claude tool-use loop: tool schemas, dispatch, audit log
checks.py              deterministic geometry checks (no AI)
run_checks.py          run every check on the sample parts and print raw JSON
inspect_part.py        basic mesh stats (vertices, volume, watertightness)
diagnose_thickness.py  the investigation behind the grazing-hit filter
test_geometry.py       unit tests on shapes with known answers
parts/                 sample STL files
```

## Possible next steps

- Exclude bed-contact faces from the overhang check (see Known limitations)
- Material-aware rules (PETG and ABS bridging and overhang limits differ from PLA)
- Choosing an orientation automatically by trading support area against print height
- Checks for CNC machining (tool access, internal corner radii), not only additive
- STEP file input so the agent works on CAD files directly
