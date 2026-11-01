"""Adapter example using the actual quantum-workflows RunArtifacts contract.

Install quantum-workflows in the runner environment, then run this file from that
repository. This does not register a qw command or grant a platform entitlement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from quantum_workflows.artifacts import RunArtifacts

from photonicslab import __version__
from photonicslab.artifacts import source_metadata
from photonicslab.experiments import polarization_experiment, polarization_parameters


def run(output: Path, *, angle: float = np.pi / 8, samples: int = 64) -> tuple[dict, Path]:
    artifacts = RunArtifacts(
        "photonics-polarization",
        output,
        polarization_parameters(angle=angle, samples=samples),
        "numpy-cpu",
    )
    # Preserve the canonical runner's source/software metadata and add the science package.
    artifacts.manifest["science_source"] = source_metadata()
    artifacts.manifest["software"]["packages"]["photonicslab"] = __version__
    try:
        summary, arrays, trace = polarization_experiment(angle=angle, samples=samples)
        artifacts.write_json("observations.json", trace, data=True)
        artifacts.write_json(
            "screen.json",
            {
                "x_m": arrays["x_m"].tolist(),
                "y_m": arrays["y_m"].tolist(),
                "intensity_w_m2": arrays["intensity_w_m2"].tolist(),
            },
            data=True,
        )
        if not summary["passed"]:
            raise RuntimeError("polarization prediction failed")
        artifacts.finish(summary)
        return summary, artifacts.directory
    except Exception as error:
        artifacts.fail(error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    summary, directory = run(args.output)
    print(f"Transmission: {summary['transmission']:.6f}; results: {directory}")
