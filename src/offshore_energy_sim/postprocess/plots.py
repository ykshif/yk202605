"""Reusable validation plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_heave_rao_comparison(
    x_present: np.ndarray,
    heave_present: np.ndarray,
    x_experiment: np.ndarray,
    heave_experiment: np.ndarray,
    png_path: str | Path,
    pdf_path: str | Path,
    *,
    x_reference: np.ndarray | None = None,
    heave_reference: np.ndarray | None = None,
    reference_label: str = "Reference simulation",
    title: str = "Heave RAO comparison",
) -> tuple[Path, Path]:
    """Plot present heave RAO against experiment and optional reference data."""

    import matplotlib.pyplot as plt

    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(exist_ok=True)
    pdf_path.parent.mkdir(exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 0.9,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(
        x_present,
        heave_present,
        color="#1f77b4",
        linewidth=2.0,
        label="Present RODM",
    )
    ax.scatter(
        x_experiment,
        heave_experiment,
        color="#d62728",
        s=34,
        marker="o",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Experiment",
    )

    if x_reference is not None and heave_reference is not None:
        ax.plot(
            x_reference,
            heave_reference,
            color="#555555",
            linewidth=1.4,
            linestyle="--",
            alpha=0.75,
            label=reference_label,
        )

    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.4)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path
