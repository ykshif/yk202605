"""Markdown report helpers for simulation workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def format_metric_table(metrics: dict[str, object]) -> list[str]:
    """Format metrics as a simple Markdown table."""

    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | `{value}` |")
    return lines


def write_workflow_report(
    report_path: str | Path,
    *,
    title: str,
    scope_lines: list[str],
    input_output_lines: list[str],
    metric_sections: dict[str, dict[str, object]],
) -> Path:
    """Write a workflow report with metric sections."""

    report_path = Path(report_path)
    lines = [
        f"# {title}",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
    ]
    lines.extend(scope_lines)
    lines.extend(["", "## Inputs And Outputs", ""])
    lines.extend(input_output_lines)
    lines.append("")

    for section_title, metrics in metric_sections.items():
        lines.extend([f"## {section_title}", ""])
        lines.extend(format_metric_table(metrics))
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
