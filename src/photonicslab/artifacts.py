"""Local lesson artifacts. Production workflow manifests remain owned by quantum-workflows."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .experiments import polarization_experiment, polarization_parameters


def source_metadata() -> dict:
    root = Path(__file__).resolve().parents[2]
    commit, dirty = None, None
    if (root / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                ).stdout.strip()
            )
        except (OSError, subprocess.SubprocessError):
            commit, dirty = None, None
    return {
        "repository": "https://github.com/nyameko/photonicslab",
        "commit": commit,
        "dirty": dirty,
        "version": __version__,
    }


def _json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def run_polarization_lab(
    output: Path,
    *,
    angle: float = np.pi / 8,
    samples: int = 64,
) -> tuple[dict, Path]:
    """Write a unique run with inputs, result hashes and honest success/failure status.

    Does not collect environment variables, credentials, hostnames or user identity.
    No scheduler or production service is contacted.
    """
    parameters = polarization_parameters(angle=angle, samples=samples)
    started = time.perf_counter()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex
    directory = Path(output) / "photonics-polarization" / f"{stamp}-{run_id}"
    data = directory / "data"
    data.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "photonicslab.lesson-run/1",
        "experiment_id": run_id,
        "workflow": "photonics-polarization",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "source": source_metadata(),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "runtime": {"executor": "numpy-cpu"},
        "parameters": parameters,
        "artifacts": [],
    }
    manifest_path = directory / "manifest.json"
    _json(manifest_path, manifest)
    try:
        summary, arrays, trace = polarization_experiment(angle=angle, samples=samples)
        _json(directory / "summary.json", summary)
        _json(data / "observations.json", trace)
        np.savez_compressed(data / "field.npz", **arrays)
        for path in sorted([directory / "summary.json", *data.iterdir()]):
            manifest["artifacts"].append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
        manifest["status"] = "completed" if summary["passed"] else "failed"
    except Exception as error:
        manifest["status"] = "failed"
        # Avoid persisting exception messages which may contain paths or credentials.
        manifest["error"] = {"type": type(error).__name__}
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["elapsed_seconds"] = time.perf_counter() - started
        _json(manifest_path, manifest)
    return summary, directory
