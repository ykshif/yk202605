"""Run the 63-node Abaqus hinge benchmark in an isolated results folder."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DM_FEM_ROOT = Path(os.environ.get("RODM_DM_FEM_ROOT", r"E:\phd\Code\DM-FEM2D"))
SOURCE_INP = DM_FEM_ROOT / "Fem_inp" / "Job-1_largemesh_hinge_1.inp"
OUTPUT_ROOT = REPO_ROOT / "results" / "hinge_validation" / "abaqus_work"
JOB_NAME = "Job-1_largemesh_hinge_1"


def main() -> int:
    if not SOURCE_INP.exists():
        raise FileNotFoundError(SOURCE_INP)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    target_inp = OUTPUT_ROOT / SOURCE_INP.name
    shutil.copy2(SOURCE_INP, target_inp)

    command = [
        "abaqus",
        f"job={JOB_NAME}",
        f"input={target_inp.name}",
        "interactive",
    ]
    print(f"Running {' '.join(command)} in {OUTPUT_ROOT}")
    result = subprocess.run(command, cwd=OUTPUT_ROOT, text=True)
    if result.returncode != 0:
        return result.returncode

    dat_path = OUTPUT_ROOT / f"{JOB_NAME}.dat"
    if not dat_path.exists():
        raise FileNotFoundError(dat_path)
    print(f"Wrote {dat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
