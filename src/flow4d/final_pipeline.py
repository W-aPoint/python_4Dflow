"""串联 FINAL PART 11，生成可供 VTK 使用的 27 列结果"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import perf_counter
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

from .base_function import RuntimeTimer, TimingRecord
from .exporters import (
    TecplotBinarySeriesResult,
    VtkSeriesExportResult,
    export_final_tecplot_binary_series,
    export_final_vtk_series,
)
from .mesh import NodeVelocityResult, sample_velocity_at_nodes
from .processing import (
    GradientOperators,
    LiutexTimeSeriesResult,
    LsqVelocityGradientResult,
    PressureResult,
    SpatialDerivativeResult,
    TemporalDerivativeResult,
    TetraGaussianKernel,
    assemble_final_node_results,
    assemble_tetra_state,
    build_lsq_gradient_operators,
    calculate_lsq_velocity_gradients,
    compute_liutex_timeseries,
    compute_periodic_temporal_derivatives,
    compute_pressure_tetra_sparse,
    compute_spatial_velocity_derivatives,
    prepare_tetra_gaussian_kernel,
    smooth_tetra_node_fields,
)


class FinalPipelineError(RuntimeError):
    """FINAL 后处理管线任一阶段失败时抛出"""


@dataclass(frozen=True, slots=True)
class FinalPostprocessingResult:
    """FINAL PART 11 到导出阶段的全部关键输出"""

    node_sampling: NodeVelocityResult
    lsq_reference: LsqVelocityGradientResult
    gradient_operators: GradientOperators
    smoothing_kernel: TetraGaussianKernel
    temporal_derivatives: TemporalDerivativeResult
    raw_spatial_derivatives: SpatialDerivativeResult
    spatial_derivatives: SpatialDerivativeResult
    tetra_state: NDArray[np.float64]
    pressure: PressureResult
    liutex: LiutexTimeSeriesResult
    final_node_results: NDArray[np.float64]
    tecplot_binary_export: TecplotBinarySeriesResult | None
    vtk_export: VtkSeriesExportResult | None
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


_ResultT = TypeVar("_ResultT")


def _run_stage(
    timer: RuntimeTimer,
    name: str,
    action: Callable[[], _ResultT],
    *,
    progress: Callable[[str, str, float | None], None] | None = None,
) -> _ResultT:
    started = perf_counter()
    if progress is not None:
        progress(name, "running", None)
    try:
        with timer.measure(name):
            result = action()
    except KeyboardInterrupt:
        if progress is not None:
            progress(name, "interrupted", perf_counter() - started)
        raise
    except Exception as exc:
        if progress is not None:
            progress(name, "failed", perf_counter() - started)
        raise FinalPipelineError(f"{name}失败：{exc}") from exc
    if progress is not None:
        progress(name, "completed", perf_counter() - started)
    return result


def _validate_pipeline_inputs(
    velocity_cm_per_second: NDArray[np.generic],
    node_coordinates_mm: NDArray[np.generic],
    tetrahedra: NDArray[np.generic],
    voxel_size_mm: tuple[float, float, float],
    time_spacing_ms: float,
    connectivity_index_base: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    tuple[float, float, float],
    float,
]:
    velocity_raw = np.asarray(velocity_cm_per_second)
    nodes_raw = np.asarray(node_coordinates_mm)
    elements_raw = np.asarray(tetrahedra)
    spacing_raw = np.asarray(voxel_size_mm)
    for name, values in (
        ("velocity_cm_per_second", velocity_raw),
        ("node_coordinates_mm", nodes_raw),
        ("tetrahedra", elements_raw),
        ("voxel_size_mm", spacing_raw),
    ):
        if np.iscomplexobj(values):
            raise FinalPipelineError(f"{name} 不能包含 complex 数值")
    if np.iscomplexobj(time_spacing_ms):
        raise FinalPipelineError("time_spacing_ms 不能为 complex 数值")

    try:
        velocity = np.asarray(velocity_raw, dtype=np.float64)
        nodes = np.asarray(nodes_raw, dtype=np.float64)
        spacing = np.asarray(spacing_raw, dtype=np.float64)
        time_step_ms = float(time_spacing_ms)
    except (TypeError, ValueError) as exc:
        raise FinalPipelineError("FINAL 主流程输入必须为实数数值") from exc

    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise FinalPipelineError(
            "velocity_cm_per_second 必须为 [row,column,slice,time,3]"
        )
    if velocity.shape[3] < 19:
        raise FinalPipelineError(
            "当前 MATLAB 主脚本固定使用第 19 时相建立 LSQ 算子，"
            "因此至少需要 19 个时相"
        )
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0:
        raise FinalPipelineError("node_coordinates_mm 必须为非空 [节点数,3]")
    if elements_raw.ndim != 2 or elements_raw.shape[1] != 4 or len(elements_raw) == 0:
        raise FinalPipelineError("tetrahedra 必须为非空 [四面体数,4]")
    if not np.issubdtype(elements_raw.dtype, np.integer):
        if not np.all(np.isfinite(elements_raw)) or not np.all(
            elements_raw == np.floor(elements_raw)
        ):
            raise FinalPipelineError("tetrahedra 必须只包含有限整数索引")
    if connectivity_index_base not in (0, 1):
        raise FinalPipelineError("connectivity_index_base 只能为 0 或 1")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise FinalPipelineError("voxel_size_mm 必须包含三个正有限数值")
    if not np.isfinite(time_step_ms) or time_step_ms <= 0.0:
        raise FinalPipelineError("time_spacing_ms 必须为正有限数值")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(nodes)):
        raise FinalPipelineError("速度和节点坐标必须全部为有限数值")

    elements = np.asarray(elements_raw, dtype=np.int64)
    zero_based = elements - connectivity_index_base
    if int(zero_based.min()) < 0 or int(zero_based.max()) >= len(nodes):
        raise FinalPipelineError("tetrahedra 包含超出节点范围的索引")
    return (
        velocity,
        nodes,
        elements,
        (float(spacing[0]), float(spacing[1]), float(spacing[2])),
        time_step_ms,
    )


def run_final_postprocessing(
    velocity_cm_per_second: NDArray[np.generic],
    node_coordinates_mm: NDArray[np.generic],
    tetrahedra: NDArray[np.generic],
    voxel_size_mm: tuple[float, float, float],
    time_spacing_ms: float,
    *,
    connectivity_index_base: int,
    vtk_case_directory: str | Path | None = None,
    vtk_overwrite: bool = True,
    tecplot_binary_directory: str | Path | None = None,
    tecplot_binary_overwrite: bool = True,
    progress: Callable[[str, str, float | None], None] | None = None,
) -> FinalPostprocessingResult:
    """按 MATLAB 源码顺序执行 FINAL 的网格后计算"""

    velocity, nodes, elements, spacing, time_step_ms = _validate_pipeline_inputs(
        velocity_cm_per_second,
        node_coordinates_mm,
        tetrahedra,
        voxel_size_mm,
        time_spacing_ms,
        connectivity_index_base,
    )
    timer = RuntimeTimer()
    run_stage = partial(_run_stage, progress=progress)

    node_sampling = run_stage(
        timer,
        "PART 11 网格节点速度采样",
        lambda: sample_velocity_at_nodes(
            velocity,
            nodes,
            spacing,
            time_step_ms,
        ),
    )
    nodal_velocity = node_sampling.velocity_mm_per_second
    temporal = run_stage(
        timer,
        "周期时间导数",
        lambda: compute_periodic_temporal_derivatives(
            nodal_velocity,
            time_step_ms / 1000.0,
            interpolation_factor=10,
        ),
    )
    lsq_reference = run_stage(
        timer,
        "第 19 时相 LSQ 几何系数",
        lambda: calculate_lsq_velocity_gradients(
            nodes,
            elements,
            nodal_velocity[:, :, 18],
            connectivity_index_base=connectivity_index_base,
        ),
    )
    operators = run_stage(
        timer,
        "构建 LSQ 稀疏梯度算子",
        lambda: build_lsq_gradient_operators(
            lsq_reference.point_neighbors,
            lsq_reference.weights,
            lsq_reference.gradient_matrices,
            number_of_nodes=len(nodes),
        ),
    )
    raw_spatial = run_stage(
        timer,
        "全部时相空间导数",
        lambda: compute_spatial_velocity_derivatives(nodal_velocity, operators),
    )
    smoothing_kernel = run_stage(
        timer,
        "准备四面体 Gaussian kernel",
        lambda: prepare_tetra_gaussian_kernel(
            nodes,
            elements,
            sigma=0.8,
            connectivity_index_base=connectivity_index_base,
        ),
    )
    smoothed_first = run_stage(
        timer,
        "平滑九列一阶空间导数",
        lambda: smooth_tetra_node_fields(
            raw_spatial.first_derivatives,
            smoothing_kernel,
        ),
    )
    spatial = SpatialDerivativeResult(
        first_derivatives=smoothed_first,
        second_derivatives=raw_spatial.second_derivatives,
        timings=raw_spatial.timings,
        total_elapsed_seconds=raw_spatial.total_elapsed_seconds,
    )
    tetra_state = run_stage(
        timer,
        "组装 24 列 TetraMesh 状态",
        lambda: assemble_tetra_state(
            nodal_velocity,
            temporal.derivatives,
            spatial,
        ),
    )
    pressure = run_stage(
        timer,
        "稀疏压力重建",
        lambda: compute_pressure_tetra_sparse(
            tetra_state,
            operators,
            dynamic_viscosity=3.36e-3,
            density=1050.0,
            tolerance=1.0e-4,
            max_iterations=1_000_000,
            preserve_matlab_dudy_column_bug=True,
        ),
    )
    liutex = run_stage(
        timer,
        "多时相 Liutex 与最终 LiutexR 平滑",
        lambda: compute_liutex_timeseries(
            spatial.first_derivatives,
            err=1.0e-8,
            smoothing_kernel=smoothing_kernel,
        ),
    )
    final_results = run_stage(
        timer,
        "组装 FINAL 27 列结果",
        lambda: assemble_final_node_results(
            node_sampling.node_coordinates_mm,
            nodal_velocity,
            spatial.first_derivatives,
            liutex,
            pressure.pressure_pa,
        ),
    )

    tecplot_binary_export: TecplotBinarySeriesResult | None = None
    if tecplot_binary_directory is not None:
        tecplot_binary_export = run_stage(
            timer,
            "导出 FINAL Tecplot 二进制时间序列",
            lambda: export_final_tecplot_binary_series(
                final_results,
                elements,
                tecplot_binary_directory,
                connectivity_index_base=connectivity_index_base,
                overwrite=tecplot_binary_overwrite,
            ),
        )

    vtk_export: VtkSeriesExportResult | None = None
    if vtk_case_directory is not None:
        vtk_export = run_stage(
            timer,
            "导出 FINAL VTK 时间序列",
            lambda: export_final_vtk_series(
                final_results,
                elements,
                vtk_case_directory,
                time_step_ms,
                connectivity_index_base=connectivity_index_base,
                overwrite=vtk_overwrite,
            ),
        )

    return FinalPostprocessingResult(
        node_sampling=node_sampling,
        lsq_reference=lsq_reference,
        gradient_operators=operators,
        smoothing_kernel=smoothing_kernel,
        temporal_derivatives=temporal,
        raw_spatial_derivatives=raw_spatial,
        spatial_derivatives=spatial,
        tetra_state=tetra_state,
        pressure=pressure,
        liutex=liutex,
        final_node_results=final_results,
        tecplot_binary_export=tecplot_binary_export,
        vtk_export=vtk_export,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
