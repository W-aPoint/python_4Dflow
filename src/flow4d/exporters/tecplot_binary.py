"""FINAL 当前 ``mat2tecplot`` 路径的可选 Tecplot 二进制导出"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .vtk_legacy import FINAL_HEADERS


class TecplotBinaryExportError(RuntimeError):
    """Tecplot 二进制导出无法完成时抛出"""


@dataclass(frozen=True, slots=True)
class TecplotBinarySeriesResult:
    """MATLAB ``FEMbrick_<phase>.plt`` 循环生成的文件"""

    output_directory: Path
    phase_files: tuple[Path, ...]


def _validated_phase_inputs(
    node_values: NDArray[np.generic],
    variable_names: Sequence[str],
    tetrahedra: NDArray[np.generic],
    connectivity_index_base: int,
) -> tuple[NDArray[np.float64], tuple[str, ...], NDArray[np.int64]]:
    values_raw = np.asarray(node_values)
    elements_raw = np.asarray(tetrahedra)
    names = tuple(str(name).strip().strip('"') for name in variable_names)

    if np.iscomplexobj(values_raw) or np.iscomplexobj(elements_raw):
        raise TecplotBinaryExportError("Tecplot 二进制输入不能包含 complex 数值")
    if values_raw.ndim != 2 or values_raw.shape[0] == 0:
        raise TecplotBinaryExportError("node_values 必须为非空 [节点数,变量数]")
    if len(names) != values_raw.shape[1] or any(not name for name in names):
        raise TecplotBinaryExportError("变量名必须非空且数量与节点变量列数一致")
    if elements_raw.ndim != 2 or elements_raw.shape[0] == 0 or elements_raw.shape[1] < 4:
        raise TecplotBinaryExportError("tetrahedra 必须至少包含四列节点连接")
    if connectivity_index_base not in (0, 1):
        raise TecplotBinaryExportError("connectivity_index_base 只能为 0 或 1")
    if not np.issubdtype(elements_raw.dtype, np.integer):
        if not np.all(np.isfinite(elements_raw)) or not np.all(
            elements_raw == np.floor(elements_raw)
        ):
            raise TecplotBinaryExportError("四面体连接必须只包含有限整数")

    values = np.asarray(values_raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise TecplotBinaryExportError("节点变量必须全部为有限实数")
    elements = np.asarray(elements_raw[:, :4], dtype=np.int64) - connectivity_index_base
    if int(elements.min()) < 0 or int(elements.max()) >= values.shape[0]:
        raise TecplotBinaryExportError("四面体连接包含超出节点范围的索引")
    return values, names, elements


def write_tecplot_binary_tetra(
    node_values: NDArray[np.generic],
    variable_names: Sequence[str],
    tetrahedra: NDArray[np.generic],
    filename: str | Path,
    *,
    solution_time: float,
    connectivity_index_base: int,
    strand_id: int = 1,
    overwrite: bool = True,
) -> Path:
    """写出 MATLAB FINAL 使用的单区域 FETetra 子集

    原始 ``mat2tecplot.m`` 写出 ``#!TDV112`` 数据，所有变量均以节点
    float 值保存PyTecplot 的 Tecplot 2009 目标是该 v112 格式的受支持
    等价方案依赖采用延迟导入，使普通计算和其他导出器无需 Tecplot
    """

    values, names, elements = _validated_phase_inputs(
        node_values,
        variable_names,
        tetrahedra,
        connectivity_index_base,
    )
    if not np.isfinite(solution_time):
        raise TecplotBinaryExportError("solution_time 必须为有限数值")
    if not isinstance(strand_id, (int, np.integer)) or int(strand_id) < 0:
        raise TecplotBinaryExportError("strand_id 必须为非负整数")

    target = Path(filename).expanduser().resolve()
    if target.exists() and not overwrite:
        raise TecplotBinaryExportError(f"输出文件已存在，未获得覆盖许可：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.plt")

    try:
        import tecplot as tp
        from tecplot.constant import (
            BinaryFileVersion,
            FieldDataType,
            PlotType,
            ZoneType,
        )
    except ImportError as exc:
        raise TecplotBinaryExportError(
            "二进制 .plt 导出需要官方 PyTecplot 和可用的 Tecplot Engine；"
            "未配置时可继续使用现有 ASCII DAT 或 VTK 导出"
        ) from exc

    try:
        with tp.session.suspend():
            original_frame = tp.active_frame()
            page = tp.active_page()
            export_frame = page.add_frame()
            try:
                export_frame.plot_type = PlotType.Sketch
                dataset = export_frame.create_dataset("tecplot data", list(names))
                zone = dataset.add_fe_zone(
                    ZoneType.FETetra,
                    name="FEVolume1",
                    num_points=values.shape[0],
                    num_elements=elements.shape[0],
                    dtypes=FieldDataType.Float,
                    solution_time=float(solution_time),
                    strand_id=int(strand_id),
                )
                for column, name in enumerate(names):
                    zone.values(name)[:] = values[:, column]
                zone.nodemap[:] = elements
                tp.data.save_tecplot_plt(
                    temporary,
                    dataset=dataset,
                    version=BinaryFileVersion.Tecplot2009,
                )
            finally:
                try:
                    page.delete_frame(export_frame)
                finally:
                    original_frame.activate()
        temporary.replace(target)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise TecplotBinaryExportError(f"Tecplot 二进制文件写出失败：{target}") from exc
    return target


def export_final_tecplot_binary_series(
    final_node_results: NDArray[np.generic],
    tetrahedra: NDArray[np.generic],
    output_directory: str | Path,
    *,
    connectivity_index_base: int,
    variable_names: Sequence[str] = FINAL_HEADERS,
    filename_prefix: str = "FEMbrick_",
    overwrite: bool = True,
) -> TecplotBinarySeriesResult:
    """对应 FINAL 每时相一个 PLT 的循环，不改变时相时间"""

    results_raw = np.asarray(final_node_results)
    if np.iscomplexobj(results_raw):
        raise TecplotBinaryExportError("final_node_results 不能包含 complex 数值")
    if results_raw.ndim != 3 or results_raw.shape[2] == 0:
        raise TecplotBinaryExportError(
            "final_node_results 必须为非空 [节点数,变量数,时相数]"
        )
    names = tuple(str(name).strip().strip('"') for name in variable_names)
    if results_raw.shape[1] != len(names):
        raise TecplotBinaryExportError("FINAL 变量列数与变量名数量不一致")
    if not filename_prefix:
        raise TecplotBinaryExportError("filename_prefix 不能为空")

    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    phase_files: list[Path] = []
    for phase_index in range(results_raw.shape[2]):
        matlab_phase = phase_index + 1
        phase_files.append(
            write_tecplot_binary_tetra(
                results_raw[:, :, phase_index],
                names,
                tetrahedra,
                directory / f"{filename_prefix}{matlab_phase}.plt",
                solution_time=float(matlab_phase),
                connectivity_index_base=connectivity_index_base,
                strand_id=1,
                overwrite=overwrite,
            )
        )
    return TecplotBinarySeriesResult(directory, tuple(phase_files))
