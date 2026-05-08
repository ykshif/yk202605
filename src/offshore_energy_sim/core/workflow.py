"""Workflow path and artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorkflowPaths:
    """Standard artifact paths for one case workflow variant."""

    case_root: Path
    variant_id: str = "default"

    @property
    def variant_root(self) -> Path:
        if self.variant_id == "default":
            return self.case_root
        return self.case_root / "variants" / self.variant_id

    @property
    def response_path(self) -> Path:
        return self.variant_root / "response.npy"

    @property
    def metrics_path(self) -> Path:
        return self.variant_root / "metrics.json"

    @property
    def report_path(self) -> Path:
        return self.variant_root / "report.md"

    @property
    def figures_dir(self) -> Path:
        return self.case_root / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.variant_root / "logs"

    def ensure_directories(self) -> None:
        """Create standard output directories for this workflow variant."""

        self.variant_root.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def build_workflow_paths(case_root: str | Path, variant_id: str = "default") -> WorkflowPaths:
    """Build standard output paths for a case and variant."""

    paths = WorkflowPaths(case_root=Path(case_root), variant_id=variant_id)
    paths.ensure_directories()
    return paths


def to_jsonable(value: Any) -> Any:
    """Convert common NumPy/path values to JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def write_metrics_json(path: str | Path, metrics: dict[str, Any]) -> Path:
    """Write metrics in a machine-readable JSON format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
