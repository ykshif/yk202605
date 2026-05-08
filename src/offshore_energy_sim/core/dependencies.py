"""Optional runtime dependency checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class DependencyStatus:
    """Availability of one optional dependency."""

    name: str
    available: bool


def is_dependency_available(module_name: str) -> bool:
    """Return whether a module can be imported without importing it eagerly."""

    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def check_optional_dependencies(module_names: list[str] | tuple[str, ...]) -> list[DependencyStatus]:
    """Check optional dependencies used by heavy solver paths."""

    return [
        DependencyStatus(name=module_name, available=is_dependency_available(module_name))
        for module_name in module_names
    ]


def missing_dependencies(module_names: list[str] | tuple[str, ...]) -> list[str]:
    """Return missing module names from a dependency list."""

    return [
        status.name
        for status in check_optional_dependencies(module_names)
        if not status.available
    ]


def require_optional_dependencies(module_names: list[str] | tuple[str, ...]) -> None:
    """Raise a helpful ImportError if optional dependencies are unavailable."""

    missing = missing_dependencies(module_names)
    if missing:
        missing_text = ", ".join(missing)
        raise ImportError(f"Missing optional dependencies required for this solver path: {missing_text}")
