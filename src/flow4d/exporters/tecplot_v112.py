"""Write the single-zone Tecplot 112 tetrahedral subset used by main_final.m.

The layout follows calLiutexfromFLUENTdat/mat2tecplot.m for one FEVolume,
all 27 variables at nodes, float32 data blocks and zero-based connectivity.
"""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def _int32(handle, value: int) -> None:
    handle.write(struct.pack("<i", int(value)))


def _float32(handle, value: float) -> None:
    handle.write(struct.pack("<f", float(value)))


def _float64(handle, value: float) -> None:
    handle.write(struct.pack("<d", float(value)))


def _string(handle, value: str) -> None:
    for char in value:
        _int32(handle, ord(char))
    _int32(handle, 0)


def write_tecplot_v112_tetra(
    node_values: NDArray[np.generic],
    variable_names: Sequence[str],
    tetrahedra: NDArray[np.generic],
    filename: str | Path,
    *,
    phase_number: int,
    overwrite: bool = True,
) -> Path:
    """Write MATLAB's FEMbrick_<phase>.plt format without a Tecplot engine."""
    values = np.asarray(node_values, dtype=np.float64)
    names = tuple(str(name).strip().strip('"') for name in variable_names)
    raw_elements = np.asarray(tetrahedra)
    if values.ndim != 2 or not values.size or values.shape[1] != len(names):
        raise ValueError("node_values must be nonempty [nodes, variables]")
    if not names or any(not name or not name.isascii() for name in names):
        raise ValueError("variable_names must be nonempty ASCII names")
    if not np.all(np.isfinite(values)):
        raise ValueError("node_values must be finite")
    if raw_elements.ndim != 2 or raw_elements.shape[1] != 4 or not len(raw_elements):
        raise ValueError("connectivity must contain four nodes per tetrahedron")
    if not np.issubdtype(raw_elements.dtype, np.integer):
        if not np.all(np.isfinite(raw_elements)) or not np.all(raw_elements == np.floor(raw_elements)):
            raise ValueError("connectivity must contain integers")
    elements = np.asarray(raw_elements, dtype=np.int64)
    if elements.min() < 0 or elements.max() >= len(values) or elements.max() > np.iinfo(np.int32).max:
        raise ValueError("connectivity index is outside the node array")
    if not isinstance(phase_number, int) or phase_number < 1:
        raise ValueError("phase_number must be a positive integer")

    target = Path(filename).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            # File header: magic, little-endian marker, full data file.
            handle.write(b"#!TDV112")
            _int32(handle, 1)
            _int32(handle, 0)
            _string(handle, "tecplot data")
            _int32(handle, len(names))
            for name in names:
                _string(handle, name)

            # Single FEVolume1 zone, all variables at nodes.
            _float32(handle, 299.0)
            _string(handle, "FEVolume1")
            _int32(handle, -1)  # Parent zone.
            _int32(handle, 1)   # Strand ID.
            _float64(handle, phase_number)
            _int32(handle, -1)  # Automatic colour.
            _int32(handle, 4)   # FETetrahedron.
            _int32(handle, 1)   # MATLAB specifies variable locations.
            for _ in names:
                _int32(handle, 0)
            _int32(handle, 0)  # No raw face neighbours.
            _int32(handle, 0)  # No extra neighbour connections.
            for value in (len(values), len(elements), 0, 0, 0, 0):
                _int32(handle, value)
            _float32(handle, 357.0)  # End of header.

            # Zone data: 27 float32 blocks, min/max as float64, tetra indices.
            _float32(handle, 299.0)
            for _ in names:
                _int32(handle, 1)
            _int32(handle, 0)   # No passive variables.
            _int32(handle, 0)   # No variable sharing.
            _int32(handle, -1)  # No connectivity sharing.
            for column in range(len(names)):
                _float64(handle, np.min(values[:, column]))
                _float64(handle, np.max(values[:, column]))
            for column in range(len(names)):
                np.ascontiguousarray(values[:, column], dtype="<f4").tofile(handle)
            np.ascontiguousarray(elements, dtype="<i4").tofile(handle)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
