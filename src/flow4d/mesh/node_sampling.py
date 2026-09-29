"""在四面体节点坐标处对裁剪后的速度体数据采样"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, make_interp_spline

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "mesh":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]


class NodeSamplingError(RuntimeError):
    """第 11 部分的节点采样输入不兼容时抛出"""


@dataclass(frozen=True, slots=True)
class NodeVelocityResult:
    """节点坐标及所有心动时相采样得到的 U/V/W"""

    node_coordinates_mm: NDArray[np.float64]
    velocity_mm_per_second: NDArray[np.float64]
    variables_by_phase: tuple[NDArray[np.float64], ...]
    times_seconds: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _matlab_round_nonnegative(
    values: NDArray[np.float64],
    decimals: int,
) -> NDArray[np.float64]:
    """对非负时相时间复现 MATLAB ``round(x, n)`` 的行为"""

    factor = 10.0**decimals
    return np.floor(values * factor + 0.5) / factor


def sample_velocity_at_nodes(
    velocity_cm_per_second: NDArray[np.generic],
    node_coordinates_mm: NDArray[np.generic],
    voxel_size_mm: tuple[float, float, float],
    time_spacing_ms: float,
) -> NodeVelocityResult:
    """移植 FINAL 第 11 部分的样条采样循环"""

    velocity = np.asarray(velocity_cm_per_second, dtype=np.float64)
    nodes = np.asarray(node_coordinates_mm, dtype=np.float64)
    spacing = np.asarray(voxel_size_mm, dtype=np.float64)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise NodeSamplingError("velocity 必须为 [row,column,slice,time,3]")
    if nodes.ndim != 2 or nodes.shape[1] < 3:
        raise NodeSamplingError("node_coordinates_mm 必须为 N×3")
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise NodeSamplingError("voxel_size_mm 必须包含三个正数")
    if not np.isfinite(time_spacing_ms) or time_spacing_ms <= 0:
        raise NodeSamplingError("time_spacing_ms 必须为正有限数值")

    timer = RuntimeTimer()
    with timer.measure("构建网格节点采样坐标"):
        coordinates = np.vstack(
            (
                nodes[:, 1] / spacing[0],
                nodes[:, 0] / spacing[1],
                nodes[:, 2] / spacing[2],
            )
        )
        if not np.all(np.isfinite(coordinates)):
            raise NodeSamplingError("网格节点采样坐标包含 NaN 或 Inf")
        # MATLAB griddedInterpolant 的 spline 方法默认也用于网格外的样条外推。

    phase_count = velocity.shape[3]
    sampled = np.empty((len(nodes), 3, phase_count), dtype=np.float64)
    spatial_grids = tuple(
        np.arange(length, dtype=np.float64) * spacing[axis]
        for axis, length in enumerate(velocity.shape[:3])
    )
    query_points = coordinates.T * spacing
    with timer.measure("构建三次样条节点权重"):
        # cubic_legacy 逐节点重复拟合 1D spline；预先生成同一 not-a-knot 基函数。
        basis = []
        for axis, grid in enumerate(spatial_grids):
            knots = make_interp_spline(
                grid, np.zeros(len(grid), dtype=np.float64), k=3
            ).t
            matrix = BSpline.design_matrix(
                query_points[:, axis], knots, 3, extrapolate=True
            )
            basis.append((
                matrix.indices.reshape(len(nodes), 4),
                matrix.data.reshape(len(nodes), 4),
            ))

    with timer.measure("逐时相三次样条采样速度"):
        for phase in range(phase_count):
            for component in range(3):
                coefficients = velocity[:, :, :, phase, component] * 10.0
                for axis, grid in enumerate(spatial_grids):
                    spline = make_interp_spline(
                        grid, coefficients, k=3, axis=axis
                    )
                    coefficients = np.moveaxis(spline.c, 0, axis)

                values = np.zeros(len(nodes), dtype=np.float64)
                for i in range(4):
                    xi, wi = basis[0][0][:, i], basis[0][1][:, i]
                    for j in range(4):
                        yj = basis[1][0][:, j]
                        wij = wi * basis[1][1][:, j]
                        for k in range(4):
                            values += (
                                wij * basis[2][1][:, k]
                                * coefficients[xi, yj, basis[2][0][:, k]]
                            )
                sampled[:, component, phase] = values
    variables = tuple(
        np.column_stack((nodes[:, :3], sampled[:, :, phase]))
        for phase in range(phase_count)
    )
    time_points_ms = (
        np.arange(phase_count, dtype=np.float64) * time_spacing_ms
    )
    times = _matlab_round_nonnegative(time_points_ms, 4) / 1000.0
    return NodeVelocityResult(
        node_coordinates_mm=nodes[:, :3].copy(),
        velocity_mm_per_second=sampled,
        variables_by_phase=variables,
        times_seconds=times,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
