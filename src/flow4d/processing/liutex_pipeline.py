"""面向 FINAL 节点数据的多时相涡量与 Liutex 组装"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]

from .liutex import compute_liutex_pagewise
from .tetra_smoothing import TetraGaussianKernel, smooth_tetra_node_fields


class LiutexPipelineError(RuntimeError):
    """多时相梯度数据无法组装时抛出"""


@dataclass(frozen=True, slots=True)
class LiutexTimeSeriesResult:
    vorticity: NDArray[np.float64]
    direction: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    omega_r: NDArray[np.float64]
    omega_numerator: NDArray[np.float64]
    omega_denominator: NDArray[np.float64]
    lambda_ci: NDArray[np.float64]
    epsilon_r_by_phase: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def compute_vorticity_from_velocity_gradients(
    first_derivatives: NDArray[np.generic],
) -> NDArray[np.float64]:
    """移植 FINAL 第 596–604 行，对应 ``[node,9,phase]`` 梯度"""

    gradients_raw = np.asarray(first_derivatives)
    if np.iscomplexobj(gradients_raw):
        raise LiutexPipelineError("first_derivatives 不能包含复数")
    gradients = np.asarray(gradients_raw, dtype=np.float64)
    if gradients.ndim != 3 or gradients.shape[1] != 9:
        raise LiutexPipelineError("first_derivatives 必须为 [节点数,9,时相数]")
    if not np.all(np.isfinite(gradients)):
        raise LiutexPipelineError("first_derivatives 必须为有限数值")
    return np.stack(
        (
            gradients[:, 7, :] - gradients[:, 5, :],
            gradients[:, 2, :] - gradients[:, 6, :],
            gradients[:, 3, :] - gradients[:, 1, :],
        ),
        axis=1,
    )


def compute_liutex_timeseries(
    first_derivatives: NDArray[np.generic],
    *,
    err: float = 1.0e-8,
    smoothing_kernel: TetraGaussianKernel | None = None,
) -> LiutexTimeSeriesResult:
    """移植 FINAL 的 ``getR_pagewise`` 循环及可选 LiutexR 平滑

    传入可复用的四面体核时，将复现 FINAL 第 648–660 行：
    只平滑最终 Liutex 幅值；方向和诊断量仍保留
    ``getR_pagewise`` 的直接计算结果
    """

    gradients_raw = np.asarray(first_derivatives)
    if np.iscomplexobj(gradients_raw):
        raise LiutexPipelineError("first_derivatives 不能包含复数")
    gradients = np.asarray(gradients_raw, dtype=np.float64)
    if gradients.ndim != 3 or gradients.shape[1] != 9:
        raise LiutexPipelineError("first_derivatives 必须为 [节点数,9,时相数]")
    node_count, _, phase_count = gradients.shape
    vorticity = compute_vorticity_from_velocity_gradients(gradients)
    direction = np.empty((node_count, 3, phase_count), dtype=np.float64)
    scalar_shape = (node_count, 1, phase_count)
    magnitude = np.empty(scalar_shape, dtype=np.float64)
    omega_r = np.empty(scalar_shape, dtype=np.float64)
    omega_numerator = np.empty(scalar_shape, dtype=np.float64)
    omega_denominator = np.empty(scalar_shape, dtype=np.float64)
    lambda_ci = np.empty(scalar_shape, dtype=np.float64)
    epsilon_r = np.empty(phase_count, dtype=np.float64)

    timer = RuntimeTimer()
    with timer.measure("逐时相计算 Liutex"):
        for phase_index in range(phase_count):
            component_by_derivative = gradients[:, :, phase_index].reshape(
                node_count, 3, 3
            )
            # MATLAB 的 reshape+permute 得到“求导轴 × 分量”的排列
            matlab_vgt = np.transpose(component_by_derivative, (0, 2, 1))
            result = compute_liutex_pagewise(
                matlab_vgt,
                vorticity[:, :, phase_index],
                err=err,
            )
            direction[:, :, phase_index] = result.direction
            magnitude[:, 0, phase_index] = result.magnitude
            omega_r[:, 0, phase_index] = result.omega_r
            omega_numerator[:, 0, phase_index] = result.omega_numerator
            omega_denominator[:, 0, phase_index] = result.omega_denominator
            lambda_ci[:, 0, phase_index] = result.lambda_ci
            epsilon_r[phase_index] = result.epsilon_r

    if smoothing_kernel is not None:
        with timer.measure("四面体网格 LiutexR 高斯平滑"):
            magnitude = smooth_tetra_node_fields(magnitude, smoothing_kernel)

    return LiutexTimeSeriesResult(
        vorticity=vorticity,
        direction=direction,
        magnitude=magnitude,
        omega_r=omega_r,
        omega_numerator=omega_numerator,
        omega_denominator=omega_denominator,
        lambda_ci=lambda_ci,
        epsilon_r_by_phase=epsilon_r,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
