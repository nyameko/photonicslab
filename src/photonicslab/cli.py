"""Local lesson runner. Production submission is delegated to the platform."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import run_polarization_lab


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="photonicslab")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    lab = commands.add_parser("polarization", help="Run the half-wave-plate teaching experiment")
    lab.add_argument("--angle-deg", type=float, default=22.5)
    lab.add_argument("--samples", type=int, default=64)
    lab.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        summary, directory = run_polarization_lab(
            args.output,
            angle=float(np.deg2rad(args.angle_deg)),
            samples=args.samples,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"summary": summary, "directory": str(directory)}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
