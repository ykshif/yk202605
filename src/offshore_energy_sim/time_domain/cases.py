"""Case helpers for time-domain RODM simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeDomainSimulationConfig:
    """Time-grid and forcing controls for a linear time-domain run.

    The first implementation is deliberately narrow: one fixed time step,
    one regular-wave frequency from the paired frequency-domain RODM case, and
    an optional cosine ramp to suppress startup transients.
    """

    time_step: float
    duration: float
    excitation_model: str = "regular_wave"
    wave_amplitude: float = 1.0
    phase_rad: float = 0.0
    ramp_time: float = 0.0
    spectrum_type: str = "jonswap"
    significant_wave_height: float = 1.0
    peak_period: float | None = None
    peak_enhancement_factor: float = 3.3
    spectrum_seed: int | None = 1
    external_force_time: np.ndarray | None = None
    external_force: np.ndarray | None = None
    radiation_model: str = "constant"
    memory_duration: float | None = None
    damping_convention: str = "physical"
    infinite_added_mass_method: str = "high_frequency"
    added_mass_tail_count: int = 3
    radiation_passivity_correction: str = "none"
    radiation_residual_model: str = "none"
    radiation_frequency_window: str = "none"
    radiation_window_start_omega: float | None = None
    radiation_window_stop_omega: float | None = None
    radiation_convolution_rule: str = "rectangular"

    def __post_init__(self) -> None:
        excitation_model = str(self.excitation_model).lower()
        spectrum_type = str(self.spectrum_type).lower()
        radiation_model = str(self.radiation_model).lower()
        damping_convention = str(self.damping_convention).lower()
        infinite_added_mass_method = str(self.infinite_added_mass_method).lower()
        radiation_passivity_correction = str(self.radiation_passivity_correction).lower()
        radiation_residual_model = str(self.radiation_residual_model).lower()
        radiation_frequency_window = str(self.radiation_frequency_window).lower()
        radiation_convolution_rule = str(self.radiation_convolution_rule).lower()
        object.__setattr__(self, "excitation_model", excitation_model)
        object.__setattr__(self, "spectrum_type", spectrum_type)
        object.__setattr__(self, "radiation_model", radiation_model)
        object.__setattr__(self, "damping_convention", damping_convention)
        object.__setattr__(
            self,
            "infinite_added_mass_method",
            infinite_added_mass_method,
        )
        object.__setattr__(
            self,
            "radiation_passivity_correction",
            radiation_passivity_correction,
        )
        object.__setattr__(self, "radiation_residual_model", radiation_residual_model)
        object.__setattr__(self, "radiation_frequency_window", radiation_frequency_window)
        object.__setattr__(self, "radiation_convolution_rule", radiation_convolution_rule)
        if self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.duration < self.time_step:
            raise ValueError("duration must be at least one time_step")
        if self.wave_amplitude < 0.0:
            raise ValueError("wave_amplitude must be non-negative")
        if self.ramp_time < 0.0:
            raise ValueError("ramp_time must be non-negative")
        if excitation_model not in {"regular_wave", "wave_spectrum", "external_force"}:
            raise ValueError(
                "excitation_model must be 'regular_wave', 'wave_spectrum', or "
                "'external_force'"
            )
        if spectrum_type not in {"jonswap", "pierson_moskowitz"}:
            raise ValueError("spectrum_type must be 'jonswap' or 'pierson_moskowitz'")
        if self.significant_wave_height < 0.0:
            raise ValueError("significant_wave_height must be non-negative")
        if self.peak_period is not None and self.peak_period <= 0.0:
            raise ValueError("peak_period must be positive")
        if self.peak_enhancement_factor <= 0.0:
            raise ValueError("peak_enhancement_factor must be positive")
        if self.spectrum_seed is not None and int(self.spectrum_seed) != self.spectrum_seed:
            raise ValueError("spectrum_seed must be an integer or None")
        if excitation_model == "wave_spectrum" and radiation_model == "constant":
            raise ValueError("wave_spectrum excitation requires direct_convolution radiation")
        if excitation_model == "external_force":
            if self.external_force_time is None or self.external_force is None:
                raise ValueError("external_force excitation requires external force arrays")
            force_time = np.asarray(self.external_force_time, dtype=float).reshape(-1)
            force_values = np.asarray(self.external_force, dtype=float)
            if force_time.size < 2:
                raise ValueError("external_force_time must contain at least two samples")
            if np.any(np.diff(force_time) <= 0.0):
                raise ValueError("external_force_time must be strictly increasing")
            if force_values.ndim != 2 or force_values.shape[0] != force_time.size:
                raise ValueError("external_force must have shape (n_time, ndof)")
        if radiation_model not in {"constant", "direct_convolution"}:
            raise ValueError("radiation_model must be 'constant' or 'direct_convolution'")
        if self.memory_duration is not None and self.memory_duration < 0.0:
            raise ValueError("memory_duration must be non-negative")
        if damping_convention not in {"physical", "wec_sim_bemio"}:
            raise ValueError("unsupported damping_convention")
        if infinite_added_mass_method not in {"high_frequency", "ogilvie"}:
            raise ValueError(
                "infinite_added_mass_method must be 'high_frequency' or 'ogilvie'"
            )
        if self.added_mass_tail_count < 1:
            raise ValueError("added_mass_tail_count must be positive")
        if radiation_passivity_correction not in {"none", "clip_negative_eigenvalues"}:
            raise ValueError(
                "radiation_passivity_correction must be 'none' or "
                "'clip_negative_eigenvalues'"
            )
        if radiation_residual_model not in {"none", "selected_frequency"}:
            raise ValueError(
                "radiation_residual_model must be 'none' or 'selected_frequency'"
            )
        if radiation_frequency_window not in {"none", "linear_tail", "cosine_tail"}:
            raise ValueError(
                "radiation_frequency_window must be 'none', 'linear_tail', or "
                "'cosine_tail'"
            )
        if self.radiation_window_start_omega is not None and self.radiation_window_start_omega <= 0.0:
            raise ValueError("radiation_window_start_omega must be positive")
        if self.radiation_window_stop_omega is not None and self.radiation_window_stop_omega <= 0.0:
            raise ValueError("radiation_window_stop_omega must be positive")
        if (
            self.radiation_window_start_omega is not None
            and self.radiation_window_stop_omega is not None
            and self.radiation_window_stop_omega <= self.radiation_window_start_omega
        ):
            raise ValueError(
                "radiation_window_stop_omega must be greater than "
                "radiation_window_start_omega"
            )
        if radiation_convolution_rule not in {"rectangular", "trapezoidal"}:
            raise ValueError(
                "radiation_convolution_rule must be 'rectangular' or 'trapezoidal'"
            )

    def time_values(self) -> np.ndarray:
        """Return an inclusive fixed-step time vector."""

        step_count = int(np.floor(self.duration / self.time_step))
        return np.arange(step_count + 1, dtype=float) * self.time_step
