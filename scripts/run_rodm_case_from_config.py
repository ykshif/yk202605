"""Run a RODM frequency-domain case from a YAML configuration file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import build_rodm_frequency_case_from_config  # noqa: E402
from offshore_energy_sim.core import build_workflow_paths, write_metrics_json  # noqa: E402
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a RODM frequency-domain case from YAML configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "reference_case_300.yaml",
        help="Path to the case YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path. Defaults to standard workflow response.npy.",
    )
    parser.add_argument(
        "--case-output-dir",
        type=Path,
        default=None,
        help="Case output directory. Defaults to results/<case_id>.",
    )
    parser.add_argument(
        "--reverse-hydrodynamic-node-order",
        action="store_true",
        default=None,
        help="Reverse hydrodynamic node blocks before solving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = build_rodm_frequency_case_from_config(
        args.config,
        reverse_hydrodynamic_node_order=args.reverse_hydrodynamic_node_order,
    )

    variant_id = "hydro_reversed" if case.reverse_hydrodynamic_node_order else "default"
    case_output_dir = args.case_output_dir or (REPO_ROOT / "results" / case.case_id)
    paths = build_workflow_paths(case_output_dir, variant_id=variant_id)
    output = Path(args.output) if args.output is not None else paths.response_path
    output.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    result = solve_rodm_frequency_case(case)
    elapsed = time.perf_counter() - start
    np.save(output, result.global_displacement)
    write_metrics_json(
        paths.metrics_path,
        {
            "case_id": case.case_id,
            "variant_id": variant_id,
            "reverse_hydrodynamic_node_order": case.reverse_hydrodynamic_node_order,
            "response_shape": result.global_displacement.shape,
            "elapsed_seconds": elapsed,
            "response_path": output,
        },
    )

    print(f"case_id: {case.case_id}")
    print(f"variant_id: {variant_id}")
    print(f"reverse_hydrodynamic_node_order: {case.reverse_hydrodynamic_node_order}")
    print(f"response_shape: {result.global_displacement.shape}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"wrote: {output}")
    print(f"metrics: {paths.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
