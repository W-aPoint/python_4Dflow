"""FINAL 第 08 部分的解混叠与速度调整流程"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

try:
    from ..base_function import RuntimeTimer, TimingRecord
    from ..dicom.case_loader import LoadedDicomCase
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]
    from dicom.case_loader import LoadedDicomCase  # type: ignore[no-redef]

from .phase_unwrap import unwrap_phase_4d


class UnwrapPipelineError(RuntimeError):
    """已加载病例无法完成 FINAL 第 08 部分时抛出"""


@dataclass(frozen=True, slots=True)
class UnwrapPipelineResult:
    """解混叠速度及可选的兼容性表示"""

    corrected_case: LoadedDicomCase | None
    original_velocity: NDArray[np.float64] | None
    unwrapped_velocity: NDArray[np.float64]
    velocity: NDArray[np.float64]
    wrap_counts_rl: NDArray[np.int8]
    wrap_counts_fh: NDArray[np.int8]
    wrap_counts_ap: NDArray[np.int8]
    flipped_components: tuple[bool, bool, bool]
    adj_velocity: int
    note: str
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float

    @property
    def inspection_message(self) -> str:
        """返回适合后续桌面界面显示的中文摘要"""

        component_names = ("RL", "FH", "AP")
        flipped = [
            name
            for name, selected in zip(component_names, self.flipped_components)
            if selected
        ]
        flipped_text = "、".join(flipped) if flipped else "无"
        timing_text = "\n".join(
            f"- {item.step_name}：{item.elapsed_seconds:.3f} 秒"
            for item in self.timings
        )
        return (
            "4D Laplacian 解混叠与速度方向处理完成\n"
            f"RL 非零 wrap 数：{int(np.count_nonzero(self.wrap_counts_rl))}\n"
            f"FH 非零 wrap 数：{int(np.count_nonzero(self.wrap_counts_fh))}\n"
            f"AP 非零 wrap 数：{int(np.count_nonzero(self.wrap_counts_ap))}\n"
            f"人工反转分量：{flipped_text}\n"
            f"校正后速度形状：{self.velocity.shape}\n"
            f"{timing_text}\n"
            f"- 总耗时：{self.total_elapsed_seconds:.3f} 秒"
        )


def _matlab_uint16(values: NDArray[np.generic]) -> NDArray[np.uint16]:
    array = np.asarray(values, dtype=np.float64)
    rounded = np.copysign(np.floor(np.abs(array) + 0.5), array)
    rounded = np.nan_to_num(
        rounded,
        nan=0.0,
        posinf=float(np.iinfo(np.uint16).max),
        neginf=0.0,
    )
    return np.clip(rounded, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def apply_velocity_component_flips(
    velocity: NDArray[np.generic],
    flip_rl: bool = False,
    flip_fh: bool = False,
    flip_ap: bool = False,
) -> NDArray[np.float64]:
    """将 MATLAB 对话框中的三项选择应用于 RL、FH 和 AP 分量"""

    array = np.asarray(velocity, dtype=np.float64)
    if array.ndim != 5 or array.shape[-1] != 3:
        raise UnwrapPipelineError("velocity 必须为 [row,column,slice,time,3]")
    selected = (bool(flip_rl), bool(flip_fh), bool(flip_ap))
    if not any(selected):
        return array
    adjusted = array.copy()
    for component, should_flip in enumerate(selected):
        if should_flip:
            adjusted[..., component] *= -1.0
    return adjusted


def run_unwrap_pipeline(
    case: LoadedDicomCase,
    *,
    temporal_scale: float = 2.0,
    flip_rl: bool = False,
    flip_fh: bool = False,
    flip_ap: bool = False,
    keep_original_velocity: bool = False,
    reencode_phase: bool = False,
) -> UnwrapPipelineResult:
    """执行 FINAL 第 08 部分，仅在请求时保留兼容性数据

    默认结果保留定量速度和圈数只有调用方需要对应的 MATLAB 风格兼容表示时，
    才设置 ``keep_original_velocity`` 或 ``reencode_phase``
    """

    arrays = (case.magnitude, case.ap, case.fh, case.rl)
    if len({array.shape for array in arrays}) != 1 or case.ap.ndim != 4:
        raise UnwrapPipelineError("M、AP、FH、RL 必须是形状一致的四维数组")
    if case.shape[3] != case.cardiac_phases:
        raise UnwrapPipelineError("CardiacPhases 与第四维长度不一致")
    if not np.isfinite(case.rescale_slope) or case.rescale_slope == 0:
        raise UnwrapPipelineError("RescaleSlope 必须是有限非零数值")
    if not np.isfinite(case.rescale_intercept):
        raise UnwrapPipelineError("RescaleIntercept 必须是有限数值")
    if not np.isfinite(case.venc) or case.venc == 0:
        raise UnwrapPipelineError("VENC 必须是有限非零数值")

    timer = RuntimeTimer()
    original_velocity = (
        np.empty(case.shape + (3,), dtype=np.float64)
        if keep_original_velocity
        else None
    )
    unwrapped_velocity = np.empty(case.shape + (3,), dtype=np.float64)
    wrap_counts: list[NDArray[np.int8]] = []
    labels = ("RL", "FH", "AP")
    source_components = (case.rl, case.fh, case.ap)
    for component, (label, source) in enumerate(zip(labels, source_components)):
        with timer.measure(f"从 {label} 重建原始速度与相位"):
            component_phase = source.astype(np.float64)
            component_phase *= case.rescale_slope
            component_phase += case.rescale_intercept
            if original_velocity is not None:
                original_velocity[..., component] = component_phase
            component_phase *= np.pi
            component_phase /= case.venc

        with timer.measure(f"{label} 四维 Laplacian 解混叠"):
            component_wrap_counts = unwrap_phase_4d(
                component_phase,
                temporal_scale=temporal_scale,
                real_output=True,
            )
            wrap_counts.append(component_wrap_counts)

        with timer.measure(f"恢复 {label} 解混叠速度"):
            component_phase += (
                2.0 * np.pi * component_wrap_counts.astype(np.float32)
            )
            component_phase *= case.venc
            component_phase /= np.pi
            unwrapped_velocity[..., component] = component_phase
            del component_phase

    flipped_components = (bool(flip_rl), bool(flip_fh), bool(flip_ap))
    with timer.measure("应用 RL/FH/AP 人工方向选择"):
        adjusted_velocity = apply_velocity_component_flips(
            unwrapped_velocity,
            flip_rl=flip_rl,
            flip_fh=flip_fh,
            flip_ap=flip_ap,
        )

    corrected_case: LoadedDicomCase | None = None
    if reencode_phase:
        with timer.measure("重新编码 AP/FH/RL 相位像素"):
            corrected_ap = _matlab_uint16(
                (adjusted_velocity[..., 2] - case.rescale_intercept)
                / case.rescale_slope
            )
            corrected_fh = _matlab_uint16(
                (adjusted_velocity[..., 1] - case.rescale_intercept)
                / case.rescale_slope
            )
            corrected_rl = _matlab_uint16(
                (adjusted_velocity[..., 0] - case.rescale_intercept)
                / case.rescale_slope
            )
            corrected_magnitude = (
                case.magnitude
                if case.magnitude.dtype == np.uint16
                else _matlab_uint16(case.magnitude)
            )
            corrected_case = replace(
                case,
                magnitude=corrected_magnitude,
                ap=corrected_ap,
                fh=corrected_fh,
                rl=corrected_rl,
                adj_velocity=1,
            )

    return UnwrapPipelineResult(
        corrected_case=corrected_case,
        original_velocity=original_velocity,
        unwrapped_velocity=unwrapped_velocity,
        velocity=adjusted_velocity,
        wrap_counts_rl=wrap_counts[0],
        wrap_counts_fh=wrap_counts[1],
        wrap_counts_ap=wrap_counts[2],
        flipped_components=flipped_components,
        adj_velocity=1,
        note="DONE:BG|velocity direction|UNWRAP",
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
