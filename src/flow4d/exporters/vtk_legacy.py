"""FINAL 四面体节点结果的传统 ASCII VTK 导出器"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..base_function import RuntimeTimer, TimingRecord


FINAL_HEADERS = (
    "X", "Y", "Z", "U", "V", "W",
    "dUdX", "dUdY", "dUdZ", "dVdX", "dVdY", "dVdZ",
    "dWdX", "dWdY", "dWdZ",
    "vorticity_x", "vorticity_y", "vorticity_z",
    "r1", "r2", "r3", "LiutexR", "Omega_Liutex",
    "omega_numer", "omega_denomin", "lambda_ci", "Pressure",
)


class VtkExportError(RuntimeError):
    """VTK 数据或输出路径无效时抛出"""


@dataclass(frozen=True, slots=True)
class VtkSeriesExportResult:
    output_directory: Path
    vtk_files: tuple[Path, ...]
    series_file: Path
    time_points_seconds: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _prepare_output_path(path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise VtkExportError(f"输出文件已存在，未获得覆盖许可：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _atomic_text_write(target: Path, content: str, *, overwrite: bool) -> None:
    _prepare_output_path(target, overwrite=overwrite)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        if target.exists() and not overwrite:
            raise VtkExportError(f"输出文件已存在，未获得覆盖许可：{target}")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _validated_connectivity(
    connectivity: NDArray[np.generic],
    *,
    number_of_nodes: int,
    connectivity_index_base: int,
) -> NDArray[np.int64]:
    raw = np.asarray(connectivity)
    if raw.ndim != 2 or raw.shape[1] != 4 or raw.shape[0] == 0:
        raise VtkExportError("connectivity 必须为非空 [四面体数,4]")
    if connectivity_index_base not in (0, 1):
        raise VtkExportError("connectivity_index_base 只能为 0 或 1")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise VtkExportError("connectivity 必须只包含有限整数")
    zero_based = np.asarray(raw, dtype=np.int64) - connectivity_index_base
    if np.min(zero_based) < 0 or np.max(zero_based) >= number_of_nodes:
        raise VtkExportError("connectivity 含有超出节点范围的索引")
    return zero_based


def write_final_tetra_vtk(
    node_values: NDArray[np.generic],
    connectivity: NDArray[np.generic],
    filename: str | Path,
    time_seconds: float,
    *,
    headers: Sequence[str] = FINAL_HEADERS,
    connectivity_index_base: int,
    overwrite: bool = True,
) -> Path:
    """对应 ``main_final.m`` 中的 ``saveFinalTetraVTK``"""

    node_raw = np.asarray(node_values)
    if np.iscomplexobj(node_raw):
        raise VtkExportError("VTK 节点数据不能包含复数")
    node = np.asarray(node_raw, dtype=np.float64)
    names = tuple(str(name).strip().strip('"') for name in headers)
    if node.ndim != 2 or node.shape[0] == 0 or node.shape[1] != 27:
        raise VtkExportError("node_values 必须为非空 [节点数,27] FINAL 数据")
    if len(names) != 27 or any(not name for name in names):
        raise VtkExportError("headers 必须包含 27 个非空字段名")
    if not np.all(np.isfinite(node)):
        raise VtkExportError("VTK 节点数据必须为有限实数")
    if not np.isfinite(time_seconds) or time_seconds < 0.0:
        raise VtkExportError("time_seconds 必须为非负有限数值")
    cells = _validated_connectivity(
        connectivity,
        number_of_nodes=node.shape[0],
        connectivity_index_base=connectivity_index_base,
    )

    lines = [
        "# vtk DataFile Version 3.0",
        f"FINAL Philips 4D Flow MRI, time = {time_seconds:.17g} s",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        "FIELD FieldData 1",
        "TIME 1 1 double",
        f"{time_seconds:.17g}",
        f"POINTS {node.shape[0]} double",
    ]
    lines.extend(" ".join(f"{value:.17g}" for value in row) for row in node[:, :3])
    lines.append(f"CELLS {cells.shape[0]} {5 * cells.shape[0]}")
    lines.extend(f"4 {a} {b} {c} {d}" for a, b, c, d in cells)
    lines.append(f"CELL_TYPES {cells.shape[0]}")
    lines.extend("10" for _ in range(cells.shape[0]))
    lines.append(f"POINT_DATA {node.shape[0]}")
    lines.append(f"FIELD NodeVariables {len(names) - 3 + 4}")
    for column in range(3, len(names)):
        lines.append(f"{names[column]} 1 {node.shape[0]} double")
        lines.extend(f"{value:.17g}" for value in node[:, column])

    for vector_name, columns in (
        ("Velocity", (3, 4, 5)),
        ("Vorticity", (15, 16, 17)),
        ("Rotex", (18, 19, 20)),
    ):
        lines.append(f"{vector_name} 3 {node.shape[0]} double")
        lines.extend(
            " ".join(f"{value:.17g}" for value in row)
            for row in node[:, columns]
        )
    lines.append(f"LiutexVector 3 {node.shape[0]} double")
    liutex_vector = node[:, 18:21] * node[:, 21, None]
    lines.extend(
        " ".join(f"{value:.17g}" for value in row)
        for row in liutex_vector
    )

    target = Path(filename).expanduser().resolve()
    _atomic_text_write(target, "\n".join(lines) + "\n", overwrite=overwrite)
    return target


def write_vtk_series(
    time_points_seconds: NDArray[np.generic],
    prefix: str,
    filename: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """对应 ``saveFinalVTKSeries`` / ``saveTime2vtkseries.m``"""

    times = np.asarray(time_points_seconds, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise VtkExportError("time_points_seconds 必须为非空一维有限数值")
    if np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
        raise VtkExportError("时间点必须非负且单调不减")
    clean_prefix = str(prefix)
    if not clean_prefix or any(character in clean_prefix for character in ('/', '\\', '"')):
        raise VtkExportError("prefix 不能为空，也不能包含路径分隔符或双引号")

    entries = [
        f'    {{ "name": "{clean_prefix}{phase}.vtk", "time": {time:.17g} }}'
        for phase, time in enumerate(times, start=1)
    ]
    content = "{\n  \"file-series-version\": \"1.0\",\n  \"files\": [\n"
    content += ",\n".join(entries)
    content += "\n  ]\n}\n"
    target = Path(filename).expanduser().resolve()
    _atomic_text_write(target, content, overwrite=overwrite)
    return target


def export_final_vtk_series(
    output_t: NDArray[np.generic],
    connectivity: NDArray[np.generic],
    case_directory: str | Path,
    time_spacing_ms: float,
    *,
    connectivity_index_base: int,
    prefix: str = "NodeLiutexPressure_",
    overwrite: bool = True,
) -> VtkSeriesExportResult:
    """对应完整 FINAL VTK 循环，并在最后写出时间序列索引"""

    values_raw = np.asarray(output_t)
    if np.iscomplexobj(values_raw):
        raise VtkExportError("output_t 不能包含复数")
    values = np.asarray(values_raw, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 27 or values.shape[2] == 0:
        raise VtkExportError("output_t 必须为 [节点数,27,时相数]")
    if not np.isfinite(time_spacing_ms) or time_spacing_ms <= 0.0:
        raise VtkExportError("time_spacing_ms 必须为正有限数值")
    output_directory = Path(case_directory).expanduser().resolve() / "vtk"
    output_directory.mkdir(parents=True, exist_ok=True)
    times = np.arange(values.shape[2], dtype=np.float64) * time_spacing_ms / 1000.0
    vtk_files = tuple(
        output_directory / f"{prefix}{phase}.vtk"
        for phase in range(1, values.shape[2] + 1)
    )
    series_file = output_directory / "timeseries.vtk.series"
    existing = tuple(path for path in (*vtk_files, series_file) if path.exists())
    if existing and not overwrite:
        raise VtkExportError(
            "以下结果已存在，未获得覆盖许可：" + ", ".join(str(path) for path in existing)
        )

    timer = RuntimeTimer()
    with timer.measure("逐时相导出 FINAL VTK"):
        for phase_index, target in enumerate(vtk_files):
            write_final_tetra_vtk(
                values[:, :, phase_index],
                connectivity,
                target,
                float(times[phase_index]),
                connectivity_index_base=connectivity_index_base,
                overwrite=overwrite,
            )
    with timer.measure("写入 VTK 时间序列索引"):
        write_vtk_series(times, prefix, series_file, overwrite=overwrite)
    return VtkSeriesExportResult(
        output_directory=output_directory,
        vtk_files=vtk_files,
        series_file=series_file,
        time_points_seconds=times,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
