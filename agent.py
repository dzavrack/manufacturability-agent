"""
Manufacturability agent: Claude orchestrates deterministic geometry checks.

Usage:
    python agent.py parts/thread_saver_25_20.stl
    python agent.py parts/thread_saver_25_20.stl --question "Can I print this on a Prusa MK4 in PETG?"
    python agent.py parts/my_part.stl --units mm --report report.json

How it works:
    1. The STL is loaded once and normalized to millimeters (checks.py).
    2. Claude is given the user's question plus a set of *tools*, one per
       check in checks.py, each described with a JSON schema.
    3. Claude decides which checks to run and with which parameters
       (nozzle-appropriate wall thickness, bed size for the named printer,
       alternative build orientations, ...). This script executes each
       request with the real geometry code and returns the numbers.
    4. Claude repeats until it has enough evidence, then writes a verdict
       that cites the measured values.

    Every number in the final answer comes from a tool call, and every
    tool call is printed (and optionally saved with --report) so the
    reasoning is auditable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import anthropic

try:  # optional: lets you keep ANTHROPIC_API_KEY in a .env file
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        pass

from checks import (
    check_bounding_box,
    check_overhangs,
    check_wall_thickness,
    check_watertight,
    load_part_mm,
)

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TURNS = 12

SYSTEM_PROMPT = """You are a manufacturing engineer reviewing a part for 3D printing (FDM unless the user says otherwise).

You have tools that run real geometry analysis on the user's part. Rules:
- Run check_watertight first. If it fails, say the remaining results are unreliable.
- Never estimate or invent a measurement. Every number you state must come from a tool result.
- Choose parameters that match the user's situation (nozzle size -> minimum wall, printer -> bed size).
  If they don't say, assume a 0.4 mm nozzle and a 220 x 220 x 250 mm bed and say so.
- If overhangs fail, try at least one alternative build orientation before recommending supports.
- Be aware that thin-wall readings on threads and other fine detail can be sampling artifacts;
  note the thin-sample fraction, not just the minimum.

Finish with:
VERDICT: PRINTABLE, PRINTABLE WITH CHANGES, or NOT PRINTABLE
then a short list of the evidence (with numbers) and concrete recommendations."""

TOOLS = [
    {
        "name": "check_watertight",
        "description": "Check whether the mesh is a closed, consistently oriented solid. "
        "Prerequisite for every other check.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_wall_thickness",
        "description": "Estimate local wall thickness by ray-casting inward from ~2000 surface samples. "
        "Returns the measured minimum/median thickness and how many samples fall under the limit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_thickness_mm": {
                    "type": "number",
                    "description": "Thinnest acceptable wall. Typically 2x nozzle diameter (0.8 for a 0.4 mm nozzle).",
                }
            },
            "required": ["min_thickness_mm"],
        },
    },
    {
        "name": "check_overhangs",
        "description": "Find downward-facing surfaces steeper than the support-free limit for a given build direction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "build_direction": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Unit vector pointing 'up' on the printer, in part coordinates. [0,0,1] = as modeled.",
                },
                "max_overhang_deg": {
                    "type": "number",
                    "description": "Support-free overhang limit from vertical; 45 is the usual FDM default.",
                },
            },
            "required": ["build_direction"],
        },
    },
    {
        "name": "check_bounding_box",
        "description": "Check whether the part fits inside a printer's build volume.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_bed_mm": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Build volume [x, y, z] in mm.",
                }
            },
            "required": ["max_bed_mm"],
        },
    },
]


def run_tool(mesh, name: str, args: dict) -> dict:
    """Dispatch one tool request from Claude to the real geometry code."""
    if name == "check_watertight":
        return check_watertight(mesh)
    if name == "check_wall_thickness":
        return check_wall_thickness(mesh, min_thickness_mm=float(args["min_thickness_mm"]))
    if name == "check_overhangs":
        return check_overhangs(
            mesh,
            build_direction=tuple(float(v) for v in args["build_direction"]),
            max_overhang_deg=float(args.get("max_overhang_deg", 45.0)),
        )
    if name == "check_bounding_box":
        return check_bounding_box(mesh, max_bed_mm=tuple(float(v) for v in args["max_bed_mm"]))
    raise ValueError(f"unknown tool: {name}")


def review_part(path: str, question: str, units: str = "in", model: str = DEFAULT_MODEL) -> tuple[str, list]:
    """Run the agent loop. Returns (final answer text, audit log of tool calls)."""
    mesh = load_part_mm(path, assume_units=units)
    client = anthropic.Anthropic()
    audit_log = []

    messages = [
        {
            "role": "user",
            "content": f"Part file: {os.path.basename(path)} (loaded assuming {units}, converted to mm).\n\n{question}",
        }
    ]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(block.text for block in response.content if block.type == "text")
            return answer, audit_log

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  -> {block.name}({json.dumps(block.input)})", file=sys.stderr)
            try:
                result = run_tool(mesh, block.name, block.input)
                is_error = False
            except Exception as exc:  # report the failure back to Claude instead of crashing
                result = {"error": str(exc)}
                is_error = True
            print(f"     {result.get('explanation', result.get('error'))}", file=sys.stderr)
            audit_log.append({"tool": block.name, "input": block.input, "result": result})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: the agent hit the maximum number of tool-use turns.", audit_log


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask Claude whether a part is printable, backed by real geometry checks.")
    parser.add_argument("stl", help="path to an STL file")
    parser.add_argument(
        "--question",
        default="Is this part ready to 3D print on a typical FDM printer? What would you change?",
    )
    parser.add_argument("--units", choices=["in", "mm"], default="in", help="units the STL was exported in")
    parser.add_argument("--model", default=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--report", help="optional path to save the full tool-call audit log as JSON")
    args = parser.parse_args()

    print(f"Reviewing {args.stl} ...", file=sys.stderr)
    answer, audit_log = review_part(args.stl, args.question, units=args.units, model=args.model)
    print("\n" + answer)

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"part": args.stl, "question": args.question, "tool_calls": audit_log, "answer": answer}, f, indent=2)
        print(f"\nAudit log saved to {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
