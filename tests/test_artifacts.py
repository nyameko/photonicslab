import hashlib
import json

import numpy as np
import pytest

from photonicslab import artifacts
from photonicslab.cli import main
from photonicslab.experiments import polarization_experiment


def test_complete_run_records_inputs_hashes_and_repeatable_science(tmp_path):
    first, path1 = artifacts.run_polarization_lab(tmp_path)
    second, path2 = artifacts.run_polarization_lab(tmp_path)
    assert path1 != path2
    assert first == second
    assert first["transmission"] == pytest.approx(0.5)
    manifest = json.loads((path1 / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["runtime"] == {"executor": "numpy-cpu"}
    assert manifest["parameters"]["source"]["vacuum_wavelength_m"] == 632.8e-9
    assert manifest["source"]["repository"].endswith("/photonicslab")
    assert len(manifest["artifacts"]) == 3
    for item in manifest["artifacts"]:
        payload = (path1 / item["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
        assert len(payload) == item["bytes"]
    with np.load(path1 / "data/field.npz", allow_pickle=False) as arrays:
        power = arrays["intensity_w_m2"].sum() * (4e-3 / 64) ** 2
        assert power == pytest.approx(first["output_power_w"])
        assert arrays["amplitude"].shape == (2, 64, 64)


def test_solver_failure_persists_failed_status_without_message(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("private-value-must-not-be-recorded")

    monkeypatch.setattr(artifacts, "polarization_experiment", broken)
    with pytest.raises(RuntimeError):
        artifacts.run_polarization_lab(tmp_path)
    path = next(tmp_path.glob("*/*/manifest.json"))
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == {"type": "RuntimeError"}
    assert "private-value" not in path.read_text()


def test_failed_scientific_acceptance_is_not_completed(tmp_path, monkeypatch):
    def failing(**kwargs):
        summary, arrays, trace = polarization_experiment(**kwargs)
        summary["passed"] = False
        return summary, arrays, trace

    monkeypatch.setattr(artifacts, "polarization_experiment", failing)
    _, path = artifacts.run_polarization_lab(tmp_path)
    assert json.loads((path / "manifest.json").read_text())["status"] == "failed"


def test_cli_and_invalid_input(tmp_path, capsys):
    assert main(["polarization", "--angle-deg", "45", "--output", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["transmission"] < 1e-28
    with pytest.raises(SystemExit) as error:
        main(["polarization", "--samples", "1", "--output", str(tmp_path)])
    assert error.value.code == 2


def test_nonfinite_angle_rejected_before_creating_run(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        artifacts.run_polarization_lab(tmp_path, angle=np.nan)
    assert not list(tmp_path.iterdir())
