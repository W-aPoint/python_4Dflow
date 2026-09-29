"""由 FINAL MATLAB 脚本移植的逐页 Liutex 指标计算"""

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


class LiutexError(RuntimeError):
    """速度梯度或涡量输入无效时抛出"""


@dataclass(frozen=True, slots=True)
class LiutexResult:
    """Liutex 方向、幅值、归一化强度及中间量"""

    direction: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    omega_r: NDArray[np.float64]
    omega_numerator: NDArray[np.float64]
    omega_denominator: NDArray[np.float64]
    lambda_ci: NDArray[np.float64]
    epsilon_r: float
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def compute_liutex_pagewise(
    gradient_tensors: NDArray[np.generic],
    vorticity: NDArray[np.generic],
    *,
    err: float = 1.0e-8,
) -> LiutexResult:
    """移植 ``main_for4DFlowMRI_FINAL_philips.m`` 中的 ``getR_pagewise``

    输入采用 Python 的节点优先布局：``[node,3,3]`` 和 ``[node,3]``
    公式、全实特征值分支、方向约定及全局 ``epsilon_r``
    均与 MATLAB 源代码保持一致
    """

    tensors_raw = np.asarray(gradient_tensors)
    omega_raw = np.asarray(vorticity)
    if np.iscomplexobj(tensors_raw) or np.iscomplexobj(omega_raw):
        raise LiutexError("速度梯度张量和涡量不能包含复数")
    tensors = np.asarray(tensors_raw, dtype=np.float64)
    omega = np.asarray(omega_raw, dtype=np.float64)
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
        raise LiutexError("gradient_tensors 必须为 [节点数,3,3]")
    if omega.shape != (tensors.shape[0], 3):
        raise LiutexError("vorticity 必须为 [节点数,3]")
    if tensors.shape[0] == 0:
        raise LiutexError("至少需要一个节点")
    if not np.all(np.isfinite(tensors)) or not np.all(np.isfinite(omega)):
        raise LiutexError("速度梯度张量和涡量必须为有限数值")
    if not np.isfinite(err) or err <= 0.0:
        raise LiutexError("err 必须为正有限数值")

    timer = RuntimeTimer()
    with timer.measure("逐节点速度梯度特征分解"):
        eigenvalues, eigenvectors = np.linalg.eig(tensors)

    with timer.measure("计算 Liutex 指标"):
        imaginary_exists = np.abs(np.imag(eigenvalues)) > err
        all_real = np.sum(imaginary_exists, axis=1) < err
        # MATLAB 的 sort(...,'ascend') 对逻辑值 0/1 使用稳定排序
        order = np.argsort(imaginary_exists, axis=1, kind="stable")
        node_indices = np.arange(tensors.shape[0])
        real_indices = order[:, 0]
        complex_indices = order[:, 2]

        lambda_r = eigenvalues[node_indices, real_indices]
        lambda_c = eigenvalues[node_indices, complex_indices]
        lambda_ci = np.abs(np.imag(lambda_c))
        lambda_cr = np.real(lambda_c)

        direction = np.real(
            eigenvectors[node_indices, :, real_indices]
        ).astype(np.float64, copy=False)
        norms = np.linalg.norm(direction, axis=1)
        if np.any((norms <= err) & ~all_real):
            raise LiutexError("复特征值节点的实特征向量无法归一化")
        nonzero = norms > 0.0
        direction[nonzero] /= norms[nonzero, None]
        direction[all_real] = 0.0

        projected_vorticity = np.sum(omega * direction, axis=1)
        flip = projected_vorticity < 0.0
        direction[flip] *= -1.0
        projected_vorticity = np.sum(omega * direction, axis=1)

        radical = projected_vorticity * projected_vorticity - 4.0 * lambda_ci**2
        local_scale = np.maximum(projected_vorticity**2, 4.0 * lambda_ci**2)
        tensor_scale = np.linalg.norm(tensors, ord="fro", axis=(1, 2))
        # 特征值求解器的绝对误差随 ||VGT|| 缩放，
        # 不能只依据根号内相消后更小的项估计误差
        eig_roundoff_scale = tensor_scale * (
            np.abs(projected_vorticity) + 2.0 * lambda_ci
        )
        roundoff_tolerance = (
            32.0
            * np.finfo(np.float64).eps
            * np.maximum(local_scale, eig_roundoff_scale)
        )
        if np.any(radical < -roundoff_tolerance):
            raise LiutexError("Liutex 根号项显著小于零，输入不满足当前实数公式")
        radical = np.maximum(radical, 0.0)
        magnitude = projected_vorticity - np.sqrt(radical)
        epsilon_r = float(np.max(1.0e-3 * lambda_ci**2))
        omega_numerator = projected_vorticity**2
        omega_denominator = 2.0 * (
            projected_vorticity**2
            - 2.0 * lambda_ci**2
            + 2.0 * lambda_cr**2
            + np.real(lambda_r) ** 2
        )
        omega_r = omega_numerator / (omega_denominator + epsilon_r)

        magnitude[all_real] = 0.0
        omega_r[all_real] = 0.0
        omega_numerator[all_real] = 0.0
        omega_denominator[all_real] = err

    return LiutexResult(
        direction=direction,
        magnitude=np.asarray(magnitude, dtype=np.float64),
        omega_r=np.asarray(omega_r, dtype=np.float64),
        omega_numerator=np.asarray(omega_numerator, dtype=np.float64),
        omega_denominator=np.asarray(omega_denominator, dtype=np.float64),
        lambda_ci=np.asarray(lambda_ci, dtype=np.float64),
        epsilon_r=epsilon_r,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
