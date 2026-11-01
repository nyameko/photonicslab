"""Execute the teaching notebook in a fresh kernel; retain outputs outside the source."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    lesson = root / "labs/01-polarization/lesson.ipynb"
    notebook = nbformat.read(lesson, as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix="photonicslab-lesson-") as directory:
        NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": directory}},
        ).execute()
    print("Executed photonics.polarization.01 in an isolated kernel; all assertions passed.")


if __name__ == "__main__":
    main()
