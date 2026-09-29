"""将 Philips 原始相位像素转换为定向的 RL/FH/AP 速度数组"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


class VelocityConversionError(RuntimeError):
    """已加载病例无法组成定向速度矩阵时抛出"""


@dataclass(frozen=True, slots=True)
class ClassicDicomFileGrid:
    """按 ``[slice][cardiac_phase]`` 排列的经典 DICOM 路径"""

    magnitude: tuple[tuple[Path, ...], ...]
    velocity_1_rl: tuple[tuple[Path, ...], ...]
    velocity_2_fh: tuple[tuple[Path, ...], ...]
    velocity_3_ap: tuple[tuple[Path, ...], ...]


@dataclass(frozen=True, slots=True)
class VelocityConversionResult:
    """物理速度、幅值强度及方向判断依据"""

    velocity: NDArray[np.float64]
    intensity: NDArray[np.float64]
    classic_file_grid: ClassicDicomFileGrid | None
    orientation_signs: tuple[int, int, int]
    adj_velocity_after_orientation: int
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float

    @property
    def inspection_message(self) -> str:
        x_sign, z_sign, y_sign = self.orientation_signs
        timing_text = "\n".join(
            f"- {item.step_name}：{item.elapsed_seconds:.3f} 秒"
            for item in self.timings
        )
        return (
            "速度矩阵生成完成\n"
            f"velocity 形状：{self.velocity.shape}\n"
            "分量顺序：[RL(U), FH(V), AP(W)]\n"
            f"方向符号：[x={x_sign}, z={z_sign}, y={y_sign}]\n"
            f"方向调整状态：{self.adj_velocity_after_orientation}\n"
            f"{timing_text}\n"
            f"- 总耗时：{self.total_elapsed_seconds:.3f} 秒"
        )


def _orientation_values(case: LoadedDicomCase) -> tuple[float, ...]:
    if case.is_enhanced_dicom:
        if case.enhanced_data is None:
            raise VelocityConversionError("Enhanced 病例缺少 enhanced_data")
        raw_orientation = (
            case.enhanced_data.classic_metadata.image_orientation_patient
        )
    else:
        if "ImageOrientationPatient" not in case.phase_info:
            raise VelocityConversionError(
                "Classic P 图 DICOM 缺少 ImageOrientationPatient"
            )
        raw_orientation = case.phase_info.ImageOrientationPatient

    try:
        orientation = tuple(float(value) for value in raw_orientation)
    except (TypeError, ValueError) as error:
        raise VelocityConversionError(
            "ImageOrientationPatient 不是有效数值向量"
        ) from error
    if len(orientation) != 6 or not np.all(np.isfinite(orientation)):
        raise VelocityConversionError(
            "ImageOrientationPatient 必须包含 6 个有限数值"
        )
    return orientation


def _path_grid(
    files: tuple[Path, ...],
    slices: int,
    periods: int,
) -> tuple[tuple[Path, ...], ...]:
    expected = slices * periods
    if len(files) != expected:
        raise VelocityConversionError(
            f"Classic 文件列表应有 {expected} 项，实际为 {len(files)} 项"
        )
    return tuple(
        tuple(files[slice_index * periods : (slice_index + 1) * periods])
        for slice_index in range(slices)
    )


def convert_case_to_velocity(
    case: LoadedDicomCase,
) -> VelocityConversionResult:
    """针对已加载的经典或增强型病例移植 ``run_getVelocity.m``"""

    expected_shape = case.magnitude.shape
    arrays = (case.magnitude, case.ap, case.fh, case.rl)
    if len({array.shape for array in arrays}) != 1 or len(expected_shape) != 4:
        raise VelocityConversionError(
            "M、AP、FH、RL 必须是形状一致的四维数组"
        )
    if expected_shape[3] != case.cardiac_phases:
        raise VelocityConversionError(
            "CardiacPhases 与相位数组第四维不一致"
        )
    if not np.isfinite(case.rescale_slope) or not np.isfinite(
        case.rescale_intercept
    ):
        raise VelocityConversionError("RescaleSlope 和 RescaleIntercept 必须有限")

    timer = RuntimeTimer()
    with timer.measure("相位像素转换为速度"):
        intensity = case.magnitude.astype(np.float64, copy=True)
        velocity = np.empty(expected_shape + (3,), dtype=np.float64)
        velocity[..., 0] = (
            case.rl.astype(np.float64) * case.rescale_slope
            + case.rescale_intercept
        )
        velocity[..., 1] = (
            case.fh.astype(np.float64) * case.rescale_slope
            + case.rescale_intercept
        )
        velocity[..., 2] = (
            case.ap.astype(np.float64) * case.rescale_slope
            + case.rescale_intercept
        )

    with timer.measure("整理 Classic 文件路径"):
        if case.is_enhanced_dicom:
            file_grid = None
        else:
            slices, periods = expected_shape[2], expected_shape[3]
            file_grid = ClassicDicomFileGrid(
                magnitude=_path_grid(
                    case.inventory.magnitude_files,
                    slices,
                    periods,
                ),
                velocity_1_rl=_path_grid(
                    case.inventory.rl_files,
                    slices,
                    periods,
                ),
                velocity_2_fh=_path_grid(
                    case.inventory.fh_files,
                    slices,
                    periods,
                ),
                velocity_3_ap=_path_grid(
                    case.inventory.ap_files,
                    slices,
                    periods,
                ),
            )

    with timer.measure("应用 DICOM 方向符号"):
        orientation = _orientation_values(case)
        x_orientation = int(np.sign(orientation[0]))
        z_orientation = int(np.sign(orientation[5]))
        y_orientation = int(
            np.cross(
                np.asarray([x_orientation, 0, 0]),
                np.asarray([0, 0, z_orientation]),
            )[1]
        )
        velocity[..., 0] *= x_orientation
        velocity[..., 1] *= z_orientation
        velocity[..., 2] *= y_orientation

    adj_velocity = (int(case.adj_velocity) + 1) % 2
    return VelocityConversionResult(
        velocity=velocity,
        intensity=intensity,
        classic_file_grid=file_grid,
        orientation_signs=(x_orientation, z_orientation, y_orientation),
        adj_velocity_after_orientation=adj_velocity,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
