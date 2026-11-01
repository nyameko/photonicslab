"""Optional cross-repository checks; run explicitly in a workflow runner environment."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("quantum_workflows.artifacts")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/quantum_workflows_polarization.py"
spec = importlib.util.spec_from_file_location("polarization_adapter", EXAMPLE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_canonical_manifest_and_hashes(tmp_path):
    summary, directory = adapter.run(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert summary["passed"]
    assert manifest["schema_version"] == "1.0"
    assert manifest["status"] == "completed"
    assert manifest["runtime"]["executor"] == "numpy-cpu"
    assert manifest["source"]["repository"].endswith("/quantum-workflows")
    assert manifest["science_source"]["repository"].endswith("/photonicslab")
    assert manifest["software"]["packages"]["photonicslab"] == adapter.__version__
    for artifact in manifest["artifacts"]:
        assert (
            hashlib.sha256((directory / artifact["path"]).read_bytes()).hexdigest()
            == artifact["sha256"]
        )


def test_canonical_failure_record(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(adapter, "polarization_experiment", broken)
    with pytest.raises(RuntimeError, match="injected"):
        adapter.run(tmp_path)
    manifest_path = next(tmp_path.glob("*/*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
