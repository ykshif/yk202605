"""Plot solver variants for the 300 m x 60 m reference case.

This is a staged copy intended for
``RODM_20250310/scripts/plot_reference_case_300_solver_variants.py`` once the
repository path is writable.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.postprocess.metrics import rmse  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    default_paths,
    extract_centerline_heave,
    load_xy,
)


def _load_heave(response_path: Path) -> tuple[np.ndarray, np.ndarray]:
    response = np.load(response_path)
    return extract_centerline_heave(response)


def _interp_rmse(
    x_reference: np.ndarray,
    y_reference: np.ndarray,
    x_candidate: np.ndarray,
    y_candidate: np.ndarray,
) -> float:
    return rmse(y_reference, np.interp(x_reference, x_candidate, y_candidate))


def main() -> int:
    import matplotlib.pyplot as plt

    paths = default_paths(REPO_ROOT)
    default_solver_path = REPO_ROOT / "results" / "reference_case_300_rodm_generated.npy"
    reversed_solver_path = REPO_ROOT / "results" / "reference_case_300_rodm_hydro_reversed.npy"

    x_saved, heave_saved = _load_heave(paths.response_file)
    x_default, heave_default = _load_heave(default_solver_path)
    x_reversed, heave_reversed = _load_heave(reversed_solver_path)
    x_exp, heave_exp = load_xy(paths.experiment_file)
    x_fu, heave_fu = load_xy(paths.fu_sim_file)

    default_vs_saved = _interp_rmse(x_saved, heave_saved, x_default, heave_default)
    reversed_vs_saved = _interp_rmse(x_saved, heave_saved, x_reversed, heave_reversed)

    paths.figure_dir.mkdir(exist_ok=True)
    png_path = paths.figure_dir / "reference_case_300_solver_variants.png"
    pdf_path = paths.figure_dir / "reference_case_300_solver_variants.pdf"

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x_saved, heave_saved, color="#111111", linewidth=2.2, label="Saved reference response")
    ax.plot(
        x_default,
        heave_default,
        color="#1f77b4",
        linewidth=1.7,
        linestyle="--",
        label=f"Packaged solver / DM_Method (RMSE={default_vs_saved:.4f})",
    )
    ax.plot(
        x_reversed,
        heave_reversed,
        color="#2ca02c",
        linewidth=1.7,
        linestyle="-.",
        label=f"Hydro-node-reversed candidate (RMSE={reversed_vs_saved:.4f})",
    )
    ax.scatter(
        x_exp,
        heave_exp,
        color="#d62728",
        s=30,
        marker="o",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label="Experiment",
    )
    ax.plot(x_fu, heave_fu, color="#666666", linewidth=1.3, linestyle=":", label="Fu et al. simulation")
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_title("300 m x 60 m Floating Body, Solver Variant Comparison")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.4)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"default_vs_saved_heave_rmse: {default_vs_saved}")
    print(f"hydro_reversed_vs_saved_heave_rmse: {reversed_vs_saved}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
