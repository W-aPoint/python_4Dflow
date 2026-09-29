"""FINAL 第 10 部分的多时相无散度小波处理流程"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]

from .dfw_backend import DfwNativeBackend


class DfwPipelineError(RuntimeError):
    """第 10 部分的 DFW 流程无法完成时抛出"""


@dataclass(frozen=True, slots=True)
class DfwTimeSeriesResult:
    """去噪后的速度时间序列及按顺序记录的运行耗时"""

    velocity_denoised: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _validate_pipeline_parameters(
    voxel_resolution: Sequence[float],
    minimum_size: Sequence[int],
    spins: int,
    random_shift: bool,
) -> None:
    raw_resolution = np.asarray(voxel_resolution)
    if np.iscomplexobj(raw_resolution):
        raise DfwPipelineError("voxel_resolution 不接受 complex 输入")
    try:
        resolution = np.asarray(voxel_resolution, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DfwPipelineError("voxel_resolution 必须是长度为 3 的数值序列") from exc
    if (
        resolution.shape != (3,)
        or not np.all(np.isfinite(resolution))
        or np.any(resolution <= 0.0)
    ):
        raise DfwPipelineError("voxel_resolution 必须包含 3 个正有限数值")

    try:
        minimum_values = tuple(minimum_size)
    except TypeError as exc:
        raise DfwPipelineError("minimum_size 必须是长度为 3 的正整数序列") from exc
    if len(minimum_values) != 3:
        raise DfwPipelineError("minimum_size 必须包含 3 个值")
    for value in minimum_values:
        if isinstance(value, (bool, np.bool_)):
            raise DfwPipelineError("minimum_size 必须只包含正整数")
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise DfwPipelineError("minimum_size 必须只包含正整数") from exc
        if integer_value <= 0 or integer_value > np.iinfo(np.int32).max:
            raise DfwPipelineError(
                "minimum_size 必须只包含 C int 范围内的正整数"
            )

    if isinstance(spins, (bool, np.bool_)):
        raise DfwPipelineError("spins 必须是正整数")
    try:
        spin_count = operator.index(spins)
    except TypeError as exc:
        raise DfwPipelineError("spins 必须是正整数") from exc
    if spin_count <= 0 or spin_count > np.iinfo(np.int32).max:
        raise DfwPipelineError("spins 必须是 C int 范围内的正整数")
    if not isinstance(random_shift, (bool, np.bool_)):
        raise DfwPipelineError("random_shift 必须为 bool")


def denoise_velocity_dfw_timeseries(
    masked_velocity_smoothed: NDArray[np.generic],
    mask_crop: NDArray[np.generic],
    voxel_resolution: Sequence[float],
    backend: DfwNativeBackend,
    *,
    minimum_size: Sequence[int] = (8, 8, 8),
    spins: int = 2,
    random_shift: bool = True,
) -> DfwTimeSeriesResult:
    """对每个心动时相执行 MATLAB 第 10 部分对应的 DFW 调用"""

    raw_velocity = np.asarray(masked_velocity_smoothed)
    raw_mask = np.asarray(mask_crop)
    if np.iscomplexobj(raw_velocity):
        raise DfwPipelineError("masked_velocity_smoothed 不接受 complex 输入")
    if np.iscomplexobj(raw_mask):
        raise DfwPipelineError("mask_crop 不接受 complex 输入")
    try:
        velocity = np.asarray(masked_velocity_smoothed, dtype=np.float64)
        mask = np.asarray(mask_crop, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DfwPipelineError("速度场和 Mask 必须为数值数组") from exc

    if velocity.ndim != 5 or velocity.shape[-1] != 3 or velocity.size == 0:
        raise DfwPipelineError(
            "masked_velocity_smoothed 必须为非空 [row,column,slice,time,3] 数组"
        )
    if velocity.shape[3] == 0:
        raise DfwPipelineError("masked_velocity_smoothed 必须至少包含一个时相")
    if mask.ndim != 3 or mask.shape != velocity.shape[:3]:
        raise DfwPipelineError("mask_crop 必须是与速度场前三维相同的三维数组")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(mask)):
        raise DfwPipelineError("速度场和 Mask 必须只包含有限数值")
    if backend is None or not callable(
        getattr(backend, "denoise_sure_mad_spin", None)
    ):
        raise DfwPipelineError("backend 必须提供 denoise_sure_mad_spin()")
    _validate_pipeline_parameters(
        voxel_resolution,
        minimum_size,
        spins,
        random_shift,
    )

    output = np.empty(velocity.shape, dtype=np.float64)
    timer = RuntimeTimer()
    for phase_index in range(velocity.shape[3]):
        try:
            with timer.measure(f"DFW 时相 {phase_index + 1}"):
                phase_vx, phase_vy, phase_vz = backend.denoise_sure_mad_spin(
                    velocity[:, :, :, phase_index, 0],
                    velocity[:, :, :, phase_index, 1],
                    velocity[:, :, :, phase_index, 2],
                    minimum_size=minimum_size,
                    voxel_resolution=voxel_resolution,
                    spins=spins,
                    random_shift=random_shift,
                )
        except Exception as exc:
            raise DfwPipelineError(
                f"DFW 第 {phase_index + 1} 时相处理失败：{exc}"
            ) from exc
        if (
            phase_vx.shape != velocity.shape[:3]
            or phase_vy.shape != velocity.shape[:3]
            or phase_vz.shape != velocity.shape[:3]
        ):
            raise DfwPipelineError(
                f"DFW 第 {phase_index + 1} 时相返回了错误的空间 shape"
            )
        output[:, :, :, phase_index, 0] = phase_vx
        output[:, :, :, phase_index, 1] = phase_vy
        output[:, :, :, phase_index, 2] = phase_vz

    with timer.measure("DFW 最终 Mask 约束"):
        output *= mask[:, :, :, None, None]
    return DfwTimeSeriesResult(
        velocity_denoised=output,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
