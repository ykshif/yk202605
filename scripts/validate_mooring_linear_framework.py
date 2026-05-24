"""Validate the linear mooring framework with local analytic checks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.mooring import (  # noqa: E402
    LinearMooringMatrix,
    NodalMooringAttachment,
    assemble_nodal_mooring_terms,
    build_mooring_provider_from_config,
    build_nodal_mooring_reduced_terms,
)
from offshore_energy_sim.solver import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    fit_harmonic_amplitude,
    harmonic_amplitude_error,
    harmonic_force_time_series,
    solve_linear_time_domain,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mooring" / "linear_framework_validation"
DEFAULT_REPORT = REPO_ROOT / "docs" / "mooring_linear_framework_validation_2026_05_24.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def validate_force_formula(output_root: Path) -> dict[str, object]:
    """Check the WEC-Sim matrix convention F0 - K*q - C*qdot."""

    mooring = LinearMooringMatrix(
        stiffness=np.diag([2.0, 3.0]),
        damping=np.diag([0.5, 0.25]),
        pretension=np.array([10.0, -4.0]),
        dof_count=2,
    )
    actual = mooring.force(
        np.array([1.0, 2.0]),
        np.array([4.0, -8.0]),
    )
    expected = np.array([6.0, -8.0])
    error = max_abs_error(actual, expected)
    figure = plot_bar_comparison(
        output_root / "figures" / "force_formula_comparison.png",
        labels=("dof_0", "dof_1"),
        actual=actual,
        expected=expected,
        ylabel="force",
        title="Mooring force formula: actual vs expected",
    )
    return {
        "name": "force_formula",
        "max_abs_error": error,
        "passed": error < 1.0e-12,
        "actual": actual,
        "expected": expected,
        "figure": figure,
    }


def validate_nodal_assembly(output_root: Path) -> dict[str, object]:
    """Check 6-DOF nodal mooring assembly into retained 5-DOF order."""

    attachment = NodalMooringAttachment(
        node_one_based=2,
        matrix=LinearMooringMatrix(
            stiffness=np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 99.0]),
            damping=np.diag([0.1, 0.2, 0.3, 0.4, 0.5, 9.9]),
            pretension=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 990.0]),
        ),
        name="stern_line",
    )
    terms = assemble_nodal_mooring_terms(
        [attachment],
        total_nodes=2,
        retained_full_dofs_zero_based=(0, 1, 2, 3, 4),
    )
    expected_stiffness = np.zeros((10, 10))
    expected_stiffness[5:10, 5:10] = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_damping = np.zeros((10, 10))
    expected_damping[5:10, 5:10] = np.diag([0.1, 0.2, 0.3, 0.4, 0.5])
    expected_pretension = np.zeros(10)
    expected_pretension[5:10] = [10.0, 20.0, 30.0, 40.0, 50.0]
    stiffness_error = max_abs_error(terms.stiffness, expected_stiffness)
    damping_error = max_abs_error(terms.damping, expected_damping)
    pretension_error = max_abs_error(terms.pretension, expected_pretension)
    error = max(stiffness_error, damping_error, pretension_error)
    figure = plot_nodal_assembly_error(
        output_root / "figures" / "nodal_assembly_error.png",
        terms.stiffness - expected_stiffness,
        terms.damping - expected_damping,
        terms.pretension - expected_pretension,
    )
    return {
        "name": "nodal_assembly_6dof_to_5dof",
        "max_abs_error": error,
        "stiffness_error": stiffness_error,
        "damping_error": damping_error,
        "pretension_error": pretension_error,
        "passed": error < 1.0e-12,
        "metadata": terms.metadata,
        "figure": figure,
    }


def validate_reduced_projection(output_root: Path) -> dict[str, object]:
    """Check SEREP-style projection of K/C/F0 to one reduced coordinate."""

    attachment = NodalMooringAttachment(
        node_one_based=2,
        matrix=LinearMooringMatrix(
            stiffness=np.diag([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            damping=np.diag([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            pretension=np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    )
    transformation = np.array([[1.0], [0.5]])
    reduced = build_nodal_mooring_reduced_terms(
        [attachment],
        total_nodes=2,
        retained_full_dofs_zero_based=(0,),
        transformation=transformation,
        master_dofs=np.array([0]),
        slave_dofs=np.array([1]),
    )
    stiffness_error = max_abs_error(reduced.stiffness, np.array([[1.0]]))
    damping_error = max_abs_error(reduced.damping, np.array([[0.5]]))
    pretension_error = max_abs_error(reduced.pretension, np.array([4.0]))
    error = max(stiffness_error, damping_error, pretension_error)
    figure = plot_bar_comparison(
        output_root / "figures" / "reduced_projection_comparison.png",
        labels=("K_reduced", "C_reduced", "F0_reduced"),
        actual=np.array([
            reduced.stiffness[0, 0],
            reduced.damping[0, 0],
            reduced.pretension[0],
        ]),
        expected=np.array([1.0, 0.5, 4.0]),
        ylabel="reduced value",
        title="Reduced mooring terms: actual vs expected",
    )
    return {
        "name": "reduced_projection",
        "max_abs_error": error,
        "stiffness_error": stiffness_error,
        "damping_error": damping_error,
        "pretension_error": pretension_error,
        "passed": error < 1.0e-12,
        "reduced_stiffness_trace": float(np.trace(reduced.stiffness)),
        "reduced_damping_trace": float(np.trace(reduced.damping)),
        "reduced_pretension_norm": float(np.linalg.norm(reduced.pretension)),
        "figure": figure,
    }


def validate_config_provider(output_root: Path) -> dict[str, object]:
    """Check that YAML-like mooring config builds the same reduced K/C/F0."""

    config = {
        "mooring": {
            "enabled": True,
            "model": "linear_matrix",
            "retained_full_dofs_zero_based": [0],
            "attachments": [
                {
                    "name": "config_line",
                    "node_one_based": 2,
                    "stiffness": {"diagonal": [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                    "damping": {"diagonal": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                    "pretension": [8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ],
        }
    }
    case = SimpleNamespace(
        total_nodes=2,
        full_dofs_per_node=6,
        removed_full_dofs_zero_based=(1, 2, 3, 4, 5),
    )
    structural = SimpleNamespace(
        transformation=np.array([[1.0], [0.5]]),
        master_dofs=np.array([0]),
        slave_dofs=np.array([1]),
        reverse_master_order_for_reconstruction=False,
    )
    provider = build_mooring_provider_from_config(config)
    if provider is None:
        raise RuntimeError("enabled mooring config did not create a provider")
    reduced = provider(case, structural)
    expected = np.array([1.0, 0.5, 4.0])
    actual = np.array([
        reduced.stiffness[0, 0],
        reduced.damping[0, 0],
        reduced.pretension[0],
    ])
    error = max_abs_error(actual, expected)
    figure = plot_bar_comparison(
        output_root / "figures" / "config_provider_comparison.png",
        labels=("K_config", "C_config", "F0_config"),
        actual=actual,
        expected=expected,
        ylabel="reduced value",
        title="Config-driven mooring provider: actual vs expected",
    )
    return {
        "name": "config_provider",
        "max_abs_error": error,
        "passed": error < 1.0e-12,
        "actual": actual,
        "expected": expected,
        "figure": figure,
        "metadata": reduced.metadata,
    }


def validate_sdof_frequency_time_closure(output_root: Path) -> dict[str, object]:
    """Check a 1DOF oscillator with mooring K/C/F0 against analytic values."""

    output_root.mkdir(parents=True, exist_ok=True)
    mass = np.array([[2.5]])
    structural_damping = np.array([[0.3]])
    structural_stiffness = np.array([[8.0]])
    mooring = LinearMooringMatrix(
        stiffness=np.array([[4.0]]),
        damping=np.array([[0.5]]),
        pretension=np.array([1.2]),
        dof_count=1,
    )
    effective_damping = structural_damping + mooring.damping
    effective_stiffness = structural_stiffness + mooring.stiffness
    omega = 1.1
    force_hat = np.array([[2.0 + 0.75j]])
    period = 2.0 * np.pi / omega
    time_step = period / 240.0
    time = np.arange(0.0, 100.0 * period + 0.5 * time_step, time_step)
    force = harmonic_force_time_series(force_hat.reshape(-1), omega, time)
    force = force + mooring.pretension.reshape(1, -1)

    solved = solve_linear_time_domain(
        mass,
        effective_damping,
        effective_stiffness,
        force,
        time,
    )
    reference = solve_frequency_domain(
        mass,
        effective_damping,
        effective_stiffness,
        force_hat,
        omega,
    ).reshape(-1)
    fitted = fit_harmonic_amplitude(
        solved.displacement,
        solved.time,
        omega,
        start_time=70.0 * period,
    )
    amplitude_error = harmonic_amplitude_error(fitted, reference)
    static_reference = float(mooring.pretension[0] / effective_stiffness[0, 0])
    tail_start = int(np.searchsorted(time, 85.0 * period))
    static_estimate = float(np.mean(solved.displacement[tail_start:, 0]))
    static_error = abs(static_estimate - static_reference)
    np.savez(
        output_root / "sdof_time_history.npz",
        time=time,
        displacement=solved.displacement,
        force=force,
        reference_complex_amplitude=reference,
        fitted_complex_amplitude=fitted,
    )
    time_figure = plot_sdof_time_comparison(
        output_root / "figures" / "sdof_time_frequency_comparison.png",
        time,
        solved.displacement[:, 0],
        reference[0],
        fitted[0],
        omega,
        period,
        static_reference,
        static_estimate,
    )
    amplitude_figure = plot_complex_amplitude_comparison(
        output_root / "figures" / "sdof_complex_amplitude_comparison.png",
        reference[0],
        fitted[0],
    )
    return {
        "name": "sdof_frequency_time_closure",
        "omega_rad_s": omega,
        "time_step_s": time_step,
        "time_samples": int(time.size),
        "reference_complex_amplitude": complex_summary(reference[0]),
        "fitted_complex_amplitude": complex_summary(fitted[0]),
        "harmonic_amplitude_error": amplitude_error,
        "static_reference": static_reference,
        "static_estimate": static_estimate,
        "static_offset_abs_error": static_error,
        "passed": amplitude_error["l2_relative_error"] < 5.0e-4 and static_error < 5.0e-4,
        "saved_arrays": output_root / "sdof_time_history.npz",
        "figures": {
            "time_frequency_comparison": time_figure,
            "complex_amplitude_comparison": amplitude_figure,
        },
    }


def validate_2dof_coupled_frequency_time_closure(output_root: Path) -> dict[str, object]:
    """Check a 2DOF oscillator with coupled mooring K/C terms."""

    mass = np.diag([2.5, 1.8])
    structural_damping = np.array([[0.3, 0.0], [0.0, 0.25]])
    structural_stiffness = np.array([[8.0, 0.0], [0.0, 6.5]])
    mooring = LinearMooringMatrix(
        stiffness=np.array([[4.0, 0.7], [0.7, 2.8]]),
        damping=np.array([[0.5, 0.08], [0.08, 0.35]]),
        dof_count=2,
    )
    effective_damping = structural_damping + mooring.damping
    effective_stiffness = structural_stiffness + mooring.stiffness
    omega = 0.95
    force_hat = np.array([[2.0 + 0.4j, -0.8 + 0.6j]])
    period = 2.0 * np.pi / omega
    time_step = period / 260.0
    time = np.arange(0.0, 110.0 * period + 0.5 * time_step, time_step)
    force = harmonic_force_time_series(force_hat.reshape(-1), omega, time)
    solved = solve_linear_time_domain(
        mass,
        effective_damping,
        effective_stiffness,
        force,
        time,
    )
    reference = solve_frequency_domain(
        mass,
        effective_damping,
        effective_stiffness,
        force_hat,
        omega,
    ).reshape(-1)
    fitted = fit_harmonic_amplitude(
        solved.displacement,
        solved.time,
        omega,
        start_time=80.0 * period,
    )
    amplitude_error = harmonic_amplitude_error(fitted, reference)
    np.savez(
        output_root / "coupled_2dof_time_history.npz",
        time=time,
        displacement=solved.displacement,
        force=force,
        reference_complex_amplitude=reference,
        fitted_complex_amplitude=fitted,
    )
    figure = plot_mdof_complex_amplitude_comparison(
        output_root / "figures" / "coupled_2dof_complex_amplitude_comparison.png",
        reference,
        fitted,
    )
    return {
        "name": "coupled_2dof_frequency_time_closure",
        "omega_rad_s": omega,
        "time_step_s": time_step,
        "time_samples": int(time.size),
        "harmonic_amplitude_error": amplitude_error,
        "passed": amplitude_error["l2_relative_error"] < 8.0e-4,
        "saved_arrays": output_root / "coupled_2dof_time_history.npz",
        "figure": figure,
    }


def complex_summary(value: complex) -> dict[str, float]:
    number = complex(value)
    return {
        "real": float(number.real),
        "imag": float(number.imag),
        "abs": float(abs(number)),
        "phase_rad": float(np.angle(number)),
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_bar_comparison(
    path: Path,
    *,
    labels: tuple[str, ...],
    actual: np.ndarray,
    expected: np.ndarray,
    ylabel: str,
    title: str,
) -> Path:
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    expected = np.asarray(expected, dtype=float).reshape(-1)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.bar(x - width / 2.0, expected, width, label="expected", color="#4c78a8")
    ax.bar(x + width / 2.0, actual, width, label="actual", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_nodal_assembly_error(
    path: Path,
    stiffness_error: np.ndarray,
    damping_error: np.ndarray,
    pretension_error: np.ndarray,
) -> Path:
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    image0 = axes[0].imshow(stiffness_error, cmap="coolwarm")
    axes[0].set_title("K assembly error")
    axes[0].set_xlabel("DOF")
    axes[0].set_ylabel("DOF")
    fig.colorbar(image0, ax=axes[0], shrink=0.8)
    image1 = axes[1].imshow(damping_error, cmap="coolwarm")
    axes[1].set_title("C assembly error")
    axes[1].set_xlabel("DOF")
    axes[1].set_ylabel("DOF")
    fig.colorbar(image1, ax=axes[1], shrink=0.8)
    axes[2].bar(np.arange(pretension_error.size), pretension_error, color="#54a24b")
    axes[2].set_title("F0 assembly error")
    axes[2].set_xlabel("DOF")
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sdof_time_comparison(
    path: Path,
    time: np.ndarray,
    displacement: np.ndarray,
    reference: complex,
    fitted: complex,
    omega: float,
    period: float,
    static_reference: float,
    static_estimate: float,
) -> Path:
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time[-1] - 5.0 * period
    mask = time >= start
    local_time = (time[mask] - time[mask][0]) / period
    phase = np.exp(-1j * omega * time[mask])
    reference_series = static_reference + np.real(reference * phase)
    fitted_series = static_estimate + np.real(fitted * phase)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.plot(local_time, displacement[mask], label="time-domain displacement", color="#4c78a8")
    ax.plot(local_time, reference_series, "--", label="frequency reference + static", color="#f58518")
    ax.plot(local_time, fitted_series, ":", label="fitted harmonic + static", color="#54a24b")
    ax.set_xlabel("cycles in final 5 periods")
    ax.set_ylabel("displacement")
    ax.set_title("1DOF mooring validation: time vs frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_complex_amplitude_comparison(path: Path, reference: complex, fitted: complex) -> Path:
    return plot_bar_comparison(
        path,
        labels=("real", "imag", "abs"),
        actual=np.array([fitted.real, fitted.imag, abs(fitted)]),
        expected=np.array([reference.real, reference.imag, abs(reference)]),
        ylabel="complex amplitude component",
        title="1DOF complex amplitude: fitted vs frequency reference",
    )


def plot_mdof_complex_amplitude_comparison(
    path: Path,
    reference: np.ndarray,
    fitted: np.ndarray,
) -> Path:
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    fitted = np.asarray(fitted, dtype=np.complex128).reshape(-1)
    labels = [f"dof_{index}" for index in range(reference.size)]
    x = np.arange(reference.size)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].bar(x - width / 2.0, np.abs(reference), width, label="reference", color="#4c78a8")
    axes[0].bar(x + width / 2.0, np.abs(fitted), width, label="fitted", color="#f58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("amplitude abs")
    axes[0].set_title("2DOF amplitude")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(x - width / 2.0, np.angle(reference), width, label="reference", color="#4c78a8")
    axes[1].bar(x + width / 2.0, np.angle(fitted), width, label="fitted", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("phase rad")
    axes[1].set_title("2DOF phase")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_error_summary(path: Path, checks: list[dict[str, object]]) -> Path:
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels: list[str] = []
    values: list[float] = []
    for check in checks:
        labels.append(str(check["name"]))
        error = check.get("max_abs_error")
        if error is None:
            error = check.get("harmonic_amplitude_error", {}).get("l2_relative_error", 0.0)
        values.append(max(float(error), 1.0e-16))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.2), constrained_layout=True)
    ax.bar(x, values, color="#4c78a8")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("error, log scale")
    ax.set_title("Linear mooring validation error summary")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(path: Path, metrics: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = metrics["checks"]
    lines = [
        "# 线性系泊框架验证报告",
        "",
        f"日期：{metrics['date']}",
        "",
        "本报告验证当前 `offshore_energy_sim.mooring` 线性系泊框架。验证不依赖外部 DM-FEM2D 数据，所有算例均为小矩阵解析或半解析检查。",
        "",
        "## 验证结论",
        "",
        f"- 总体结果：`{'passed' if metrics['passed'] else 'failed'}`",
        f"- 检查数量：`{len(checks)}`",
        f"- metrics：`{metrics['metrics_path']}`",
        "",
        "## 图件",
        "",
        f"- 误差汇总图：`{metrics['figures']['error_summary']}`",
        "",
        "## 检查项",
        "",
        "| 检查 | 是否通过 | 关键误差 |",
        "| --- | --- | ---: |",
    ]
    for check in checks:
        error = check.get("max_abs_error")
        if error is None:
            error = check.get("harmonic_amplitude_error", {}).get("l2_relative_error", 0.0)
        lines.append(f"| `{check['name']}` | `{check['passed']}` | `{float(error):.6e}` |")
    sdof = next(check for check in checks if check["name"] == "sdof_frequency_time_closure")
    lines.extend(
        [
            "",
            "## 1DOF 时域/频域闭合",
            "",
            f"- 谐波复幅值相对误差：`{sdof['harmonic_amplitude_error']['l2_relative_error']:.6e}`",
            f"- 静态偏置解析值：`{sdof['static_reference']:.6e}`",
            f"- 静态偏置时域估计：`{sdof['static_estimate']:.6e}`",
            f"- 静态偏置绝对误差：`{sdof['static_offset_abs_error']:.6e}`",
            f"- 时序对比图：`{sdof['figures']['time_frequency_comparison']}`",
            f"- 复幅值对比图：`{sdof['figures']['complex_amplitude_comparison']}`",
            "",
            "## 其他检查图件",
            "",
            f"- 公式对比图：`{checks[0]['figure']}`",
            f"- 节点装配误差图：`{checks[1]['figure']}`",
            f"- 降阶投影对比图：`{checks[2]['figure']}`",
            f"- 配置 provider 对比图：`{checks[3]['figure']}`",
            f"- 2DOF 耦合复幅值对比图：`{checks[5]['figure']}`",
            "",
            "## 数值影响",
            "",
            "本验证只新增验证脚本、结果和报告，不改变 RODM 频域核心。线性系泊项只有在用户显式传入 `K_moor/C_moor/F0` 时才会改变响应。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    checks = [
        validate_force_formula(args.output_root),
        validate_nodal_assembly(args.output_root),
        validate_reduced_projection(args.output_root),
        validate_config_provider(args.output_root),
        validate_sdof_frequency_time_closure(args.output_root),
        validate_2dof_coupled_frequency_time_closure(args.output_root),
    ]
    metrics_path = args.output_root / "metrics.json"
    error_summary = plot_error_summary(
        args.output_root / "figures" / "linear_mooring_error_summary.png",
        checks,
    )
    metrics = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "metrics_path": metrics_path,
        "figures": {
            "error_summary": error_summary,
        },
    }
    write_metrics_json(metrics_path, metrics)
    report_path = write_report(args.report, metrics)
    print(f"passed: {metrics['passed']}")
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
