"""Export the MATLAB FINAL UVW and FEMbrick series from one computed mesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .tecplot_ascii import write_tecplot_ascii
from .tecplot_v112 import write_tecplot_v112_tetra
from .vtk_legacy import FINAL_HEADERS


@dataclass(frozen=True, slots=True)
class MatlabTecplotSeriesResult:
    output_directory: Path
    uvw_files: tuple[Path, ...]
    plt_files: tuple[Path, ...]


def export_matlab_dat_plt_series(
    variables_by_phase: Sequence[NDArray[np.generic]],
    final_node_results: NDArray[np.generic],
    tetrahedra: NDArray[np.generic],
    case_directory: str | Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> MatlabTecplotSeriesResult:
    """Write dat/UVW_<1..T>.dat and dat/FEMbrick_<1..T>.plt."""
    final = np.asarray(final_node_results)
    elements = np.asarray(tetrahedra)
    phase_count = len(variables_by_phase)
    if final.ndim != 3 or final.shape[1] != len(FINAL_HEADERS) or final.shape[2] != phase_count or phase_count < 1:
        raise ValueError("FINAL data must have 27 columns and match the UVW phase count")
    if elements.ndim != 2 or elements.shape[1] != 4 or len(elements) < 1:
        raise ValueError("tetrahedra must be nonempty four-node elements")
    for phase_values in variables_by_phase:
        if np.asarray(phase_values).shape != (final.shape[0], 6):
            raise ValueError("each UVW phase must contain the same nodes and six columns")

    directory = Path(case_directory).expanduser().resolve() / "dat"
    directory.mkdir(parents=True, exist_ok=True)
    uvw_files: list[Path] = []
    plt_files: list[Path] = []
    for phase_index, phase_values in enumerate(variables_by_phase):
        phase_number = phase_index + 1
        uvw_files.append(write_tecplot_ascii(
            phase_values, ("X", "Y", "Z", "U", "V", "W"), elements,
            directory / f"UVW_{phase_number}.dat",
            time_frame=phase_number, connectivity_index_base=0,
        ))
        plt_files.append(write_tecplot_v112_tetra(
            final[:, :, phase_index], FINAL_HEADERS, elements,
            directory / f"FEMbrick_{phase_number}.plt", phase_number=phase_number,
        ))
        if progress is not None:
            progress(phase_number, phase_count)
    return MatlabTecplotSeriesResult(directory, tuple(uvw_files), tuple(plt_files))
