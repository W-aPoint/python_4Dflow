"""FINAL 第 10 部分的速度高斯平滑准备"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]


class VelocitySmoothingError(RuntimeError):
    """第 10 部分的数组或参数无效时抛出"""


@dataclass(frozen=True, slots=True)
class GaussianVelocityResult:
    """未加掩膜及掩膜加权的高斯平滑速度场"""

    velocity_smoothed: NDArray[np.float64]
    masked_velocity_smoothed: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def smooth_velocity_gaussian(
    velocity_crop: NDArray[np.generic],
    mask_crop: NDArray[np.generic],
    *,
    sigma: float = 0.5,
) -> GaussianVelocityResult:
    """移植 FINAL 第 10 部分中的两段 ``imgaussfilt3`` 循环

    MATLAB 调用 ``imgaussfilt3`` 时没有传入局部变量 ``sigma=0.8``，
    因此此处保留其默认 sigma（0.5）
    """

    velocity = np.asarray(velocity_crop, dtype=np.float64)
    mask = np.asarray(mask_crop, dtype=np.float64)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise VelocitySmoothingError(
            "velocity_crop 必须为 [row,column,slice,time,3]"
        )
    if mask.ndim != 3 or velocity.shape[:3] != mask.shape:
        raise VelocitySmoothingError("mask_crop 与速度场的前三维尺寸不一致")
    if not np.isfinite(sigma) or sigma <= 0:
        raise VelocitySmoothingError("sigma 必须为正有限数值")

    timer = RuntimeTimer()
    spatial_sigma = (sigma, sigma, sigma, 0.0, 0.0)
    with timer.measure("裁剪速度场高斯平滑"):
        smoothed = gaussian_filter(
            velocity,
            spatial_sigma,
            mode="nearest",
            radius=1,
        )
    with timer.measure("Mask 内速度场高斯平滑"):
        masked_source = velocity * mask[:, :, :, None, None]
        masked_smoothed = gaussian_filter(
            masked_source,
            spatial_sigma,
            mode="nearest",
            radius=1,
        )
    return GaussianVelocityResult(
        velocity_smoothed=smoothed,
        masked_velocity_smoothed=masked_smoothed,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
