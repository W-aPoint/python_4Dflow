"""时间/空间导数计算与稀疏压力重建"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cgs

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]

from .mesh_gradients import GradientOperators


class PressureReconstructionError(RuntimeError):
    """压力输入无效或 CGS 不收敛时抛出"""


@dataclass(frozen=True, slots=True)
class TemporalDerivativeResult:
    derivatives: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SpatialDerivativeResult:
    first_derivatives: NDArray[np.float64]
    second_derivatives: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PressureResult:
    pressure_native_units: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    pressure_gradient: NDArray[np.float64]
    solver_info: tuple[int, ...]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _as_velocity_timeseries(velocity: NDArray[np.generic]) -> NDArray[np.float64]:
    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 3:
        raise PressureReconstructionError("velocity 必须为 [节点数,3,时相数]")
    if values.shape[2] < 2:
        raise PressureReconstructionError("至少需要 2 个时相")
    if not np.all(np.isfinite(values)):
        raise PressureReconstructionError("velocity 必须为有限数值")
    return values


def compute_periodic_temporal_derivatives(
    velocity: NDArray[np.generic],
    dt_seconds: float,
    *,
    interpolation_factor: int = 10,
) -> TemporalDerivativeResult:
    """移植 FINAL 第 417–483 行：双周期样条插值与中心差分"""

    values = _as_velocity_timeseries(velocity)
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise PressureReconstructionError("dt_seconds 必须为正有限数值")
    if int(interpolation_factor) != interpolation_factor or interpolation_factor < 1:
        raise PressureReconstructionError("interpolation_factor 必须为正整数")
    interpolation_factor = int(interpolation_factor)

    timer = RuntimeTimer()
    number_of_phases = values.shape[2]
    with timer.measure("周期速度场样条时间插值"):
        repeated = np.concatenate((values, values), axis=2)
        original_time = np.arange(1, 2 * number_of_phases + 1, dtype=np.float64)
        interpolated_count = (2 * number_of_phases - 1) * interpolation_factor + 1
        new_time = 1.0 + np.arange(interpolated_count, dtype=np.float64) / interpolation_factor
        interpolated = CubicSpline(
            original_time,
            repeated,
            axis=2,
            bc_type="not-a-knot",
        )(new_time)

    with timer.measure("计算周期中心时间导数"):
        centered_difference = interpolated[:, :, 2:] - interpolated[:, :, :-2]
        derivatives = np.empty_like(values)
        derivatives[:, :, 1:] = centered_difference[
            :, :, interpolation_factor : number_of_phases * interpolation_factor : interpolation_factor
        ]
        derivatives[:, :, 0] = centered_difference[
            :, :, number_of_phases * interpolation_factor
        ]
        derivatives /= 2.0 * dt_seconds / interpolation_factor

    return TemporalDerivativeResult(
        derivatives=derivatives,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )


def compute_spatial_velocity_derivatives(
    velocity: NDArray[np.generic],
    operators: GradientOperators,
) -> SpatialDerivativeResult:
    """移植 FINAL 第 503–526 行，处理各速度分量和时相"""

    values = _as_velocity_timeseries(velocity)
    node_count, _, phase_count = values.shape
    if any(operator.shape != (node_count, node_count) for operator in (operators.x, operators.y, operators.z)):
        raise PressureReconstructionError("梯度算子尺寸与速度节点数不一致")

    first = np.empty((node_count, 9, phase_count), dtype=np.float64)
    second = np.empty((node_count, 9, phase_count), dtype=np.float64)
    axes = (operators.x, operators.y, operators.z)
    timer = RuntimeTimer()
    with timer.measure("计算各时相一阶与同轴二阶空间导数"):
        for phase_index in range(phase_count):
            for component in range(3):
                component_values = values[:, component, phase_index]
                column = component * 3
                for axis_index, operator in enumerate(axes):
                    first[:, column + axis_index, phase_index] = operator @ component_values
                    second[:, column + axis_index, phase_index] = operator @ (
                        operator @ component_values
                    )

    return SpatialDerivativeResult(
        first_derivatives=first,
        second_derivatives=second,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )


def assemble_tetra_state(
    velocity: NDArray[np.generic],
    temporal_derivatives: NDArray[np.generic],
    spatial_derivatives: SpatialDerivativeResult,
) -> NDArray[np.float64]:
    """组装 FINAL 的 24 列 ``TetraMesh`` 状态"""

    values = _as_velocity_timeseries(velocity)
    temporal = np.asarray(temporal_derivatives, dtype=np.float64)
    if temporal.shape != values.shape:
        raise PressureReconstructionError("temporal_derivatives 尺寸必须与 velocity 一致")
    expected_shape = (values.shape[0], 9, values.shape[2])
    if spatial_derivatives.first_derivatives.shape != expected_shape:
        raise PressureReconstructionError("first_derivatives 尺寸必须为 [节点数,9,时相数]")
    if spatial_derivatives.second_derivatives.shape != expected_shape:
        raise PressureReconstructionError("second_derivatives 尺寸必须为 [节点数,9,时相数]")
    return np.concatenate(
        (
            values,
            temporal,
            spatial_derivatives.first_derivatives,
            spatial_derivatives.second_derivatives,
        ),
        axis=1,
    )


def build_pressure_laplacian(operators: GradientOperators) -> csr_matrix:
    """按 FINAL 脚本构建 ``Ax.T@Ax + Ay.T@Ay + Az.T@Az``"""

    return (
        operators.x.T @ operators.x
        + operators.y.T @ operators.y
        + operators.z.T @ operators.z
    ).tocsr()


def compute_pressure_tetra_sparse(
    tetra_state: NDArray[np.generic],
    operators: GradientOperators,
    *,
    dynamic_viscosity: float = 3.36e-3,
    density: float = 1050.0,
    tolerance: float = 1.0e-4,
    max_iterations: int = 1_000_000,
    preserve_matlab_dudy_column_bug: bool = True,
) -> PressureResult:
    """使用 SciPy CGS 移植 ``calc_pressure_tetra_sparse.m``

    MATLAB 源代码同时从第 9 列读取 ``dudy`` 和 ``dudz``，默认保留该行为
    只有明确要求修正计算时，才将 ``preserve_matlab_dudy_column_bug``
    设置为 ``False``
    """

    state = np.asarray(tetra_state, dtype=np.float64)
    if state.ndim != 3 or state.shape[1] != 24:
        raise PressureReconstructionError("tetra_state 必须为 [节点数,24,时相数]")
    if not np.all(np.isfinite(state)):
        raise PressureReconstructionError("tetra_state 必须为有限数值")
    if not np.isfinite(dynamic_viscosity) or dynamic_viscosity < 0.0:
        raise PressureReconstructionError("dynamic_viscosity 必须为非负有限数值")
    if not np.isfinite(density) or density <= 0.0:
        raise PressureReconstructionError("density 必须为正有限数值")
    if not np.isfinite(tolerance) or tolerance <= 0.0 or max_iterations <= 0:
        raise PressureReconstructionError("CGS 容差和最大迭代次数必须为正值")
    node_count, _, phase_count = state.shape
    if any(operator.shape != (node_count, node_count) for operator in (operators.x, operators.y, operators.z)):
        raise PressureReconstructionError("梯度算子尺寸与 tetra_state 节点数不一致")

    u, v, w = (state[:, index, :] for index in range(3))
    dudt, dvdt, dwdt = (state[:, index, :] for index in range(3, 6))
    dudx = state[:, 6, :]
    dudy = state[:, 8 if preserve_matlab_dudy_column_bug else 7, :]
    dudz = state[:, 8, :]
    dvdx, dvdy, dvdz = (state[:, index, :] for index in range(9, 12))
    dwdx, dwdy, dwdz = (state[:, index, :] for index in range(12, 15))
    d2udx2, d2udy2, d2udz2 = (state[:, index, :] for index in range(15, 18))
    d2vdx2, d2vdy2, d2vdz2 = (state[:, index, :] for index in range(18, 21))
    d2wdx2, d2wdy2, d2wdz2 = (state[:, index, :] for index in range(21, 24))

    timer = RuntimeTimer()
    with timer.measure("计算 Navier-Stokes 压力梯度"):
        dpdx = dynamic_viscosity * (d2udx2 + d2udy2 + d2udz2) - density * (
            dudt + u * dudx + v * dudy + w * dudz
        )
        dpdy = dynamic_viscosity * (d2vdx2 + d2vdy2 + d2vdz2) - density * (
            dvdt + u * dvdx + v * dvdy + w * dvdz
        )
        dpdz = dynamic_viscosity * (d2wdx2 + d2wdy2 + d2wdz2) - density * (
            dwdt + u * dwdx + v * dwdy + w * dwdz
        )
        pressure_gradient = np.stack((dpdx, dpdy, dpdz), axis=1)

    laplacian = build_pressure_laplacian(operators)
    pressure = np.empty((node_count, 1, phase_count), dtype=np.float64)
    solver_info: list[int] = []
    with timer.measure("逐时相 CGS 压力重建"):
        for phase_index in range(phase_count):
            right_hand_side = (
                operators.x.T @ dpdx[:, phase_index]
                + operators.y.T @ dpdy[:, phase_index]
                + operators.z.T @ dpdz[:, phase_index]
            )
            solution, info = cgs(
                laplacian,
                right_hand_side,
                rtol=tolerance,
                atol=0.0,
                maxiter=max_iterations,
            )
            solver_info.append(int(info))
            if info != 0:
                raise PressureReconstructionError(
                    f"时相 {phase_index + 1} 的 CGS 未收敛，solver info={info}"
                )
            pressure[:, 0, phase_index] = solution

    return PressureResult(
        pressure_native_units=pressure,
        pressure_pa=pressure * 1.0e-6,
        pressure_gradient=pressure_gradient,
        solver_info=tuple(solver_info),
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
