"""生成用于制作血管掩膜的 PCMRA 与 R1 体数据"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from ..base_function import RuntimeTimer, TimingRecord
    from ..dicom.case_loader import LoadedDicomCase
    from ..dicom.volume_export import DicomVolumeExportResult, save_dicom_volume
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]
    from dicom.case_loader import LoadedDicomCase  # type: ignore[no-redef]
    from dicom.volume_export import (  # type: ignore[no-redef]
        DicomVolumeExportResult,
        save_dicom_volume,
    )

from .discontinuity import compute_discontinuity_function_third_order
from .velocity import VelocityConversionResult


class SegmentationPreparationError(RuntimeError):
    """第 06 部分的体数据无法生成时抛出"""


@dataclass(frozen=True, slots=True)
class SegmentationVolumes:
    """由 ``run_getDCMforSegmentation.m`` 生成的三个 uint16 体数据"""

    pcmra_gamma_02: NDArray[np.uint16]
    pcmra_gamma_05: NDArray[np.uint16]
    r1: NDArray[np.uint16]
    phase_numbers: tuple[int, ...]
    peak_phase_number: int
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SegmentationExportResult:
    """计算得到的体数据及导出的三个 DICOM 序列"""

    volumes: SegmentationVolumes
    pcmra_gamma_02_export: DicomVolumeExportResult
    pcmra_gamma_05_export: DicomVolumeExportResult
    r1_export: DicomVolumeExportResult
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _first_float(value: object, label: str) -> float:
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            value = next(iter(value))  # type: ignore[arg-type]
        except StopIteration as error:
            raise SegmentationPreparationError(f"{label} 为空") from error
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SegmentationPreparationError(f"{label} 不是有效数值") from error
    if not np.isfinite(result):
        raise SegmentationPreparationError(f"{label} 必须为有限数值")
    return result


def _window_parameters(case: LoadedDicomCase) -> tuple[float, float]:
    info = case.magnitude_info
    try:
        if case.is_enhanced_dicom:
            frame = info.PerFrameFunctionalGroupsSequence[0]
            voi = frame.FrameVOILUTSequence[0]
            width = _first_float(voi.WindowWidth, "WindowWidth")
            center = _first_float(voi.WindowCenter, "WindowCenter")
        else:
            width = _first_float(info.WindowWidth, "WindowWidth")
            center = _first_float(info.WindowCenter, "WindowCenter")
    except (AttributeError, IndexError, TypeError) as error:
        raise SegmentationPreparationError("M 图缺少窗宽或窗位信息") from error
    if width == 0:
        raise SegmentationPreparationError("WindowWidth 不能为 0")
    return width, center


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


def _normalize_pcmra(values: NDArray[np.float64]) -> NDArray[np.uint16]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (values - minimum) / (maximum - minimum) * 8191.0 + 1.0
    return _matlab_uint16(normalized)


def compute_segmentation_volumes(
    case: LoadedDicomCase,
    velocity_result: VelocityConversionResult,
    phase_numbers: Sequence[int] | None = None,
    peak_phase_number: int = 15,
    sigma: float = 0.5,
) -> SegmentationVolumes:
    """移植 ``run_getDCMforSegmentation.m`` 的数值计算部分

    ``phase_numbers`` 和 ``peak_phase_number`` 刻意保留从 1 开始的编号，
    以匹配 MATLAB 界面与脚本
    """

    if velocity_result.velocity.shape != case.shape + (3,):
        raise SegmentationPreparationError("velocity 形状与病例不一致")
    if velocity_result.intensity.shape != case.shape:
        raise SegmentationPreparationError("intensity 形状与病例不一致")
    if not np.isfinite(case.venc):
        raise SegmentationPreparationError("VENC 必须为有限数值")
    selected = tuple(
        range(1, case.cardiac_phases + 1)
        if phase_numbers is None
        else (int(value) for value in phase_numbers)
    )
    if not selected or any(value < 1 or value > case.cardiac_phases for value in selected):
        raise SegmentationPreparationError("phase_numbers 超出有效心动时相范围")
    peak = int(peak_phase_number)
    if peak < 1 or peak > case.cardiac_phases:
        raise SegmentationPreparationError("peak_phase_number 超出有效心动时相范围")

    width, center = _window_parameters(case)
    timer = RuntimeTimer()
    with timer.measure("计算 PCMRA 共同输入"):
        velocity = np.asarray(velocity_result.velocity, dtype=np.float64)
        intensity = np.asarray(velocity_result.intensity, dtype=np.float64)
        velocity_magnitude_squared = np.sum(velocity**2, axis=-1)
        lower_bound = center - 0.5 * width
        intensity_ratio = (intensity - lower_bound) / width
        intensity_2venc = _matlab_uint16(
            2.0 * case.venc * np.clip(intensity_ratio, 0.0, 1.0)
        ).astype(np.float64)
        phase_indices = np.asarray(selected, dtype=np.int64) - 1

    with timer.measure("计算 gamma 0.2 PCMRA"):
        pcmra_t = intensity_2venc * velocity_magnitude_squared**0.2
        pcmra_gamma_02 = _normalize_pcmra(np.mean(pcmra_t[..., phase_indices], axis=3))

    with timer.measure("计算 gamma 0.5 PCMRA"):
        pcmra_t2 = intensity_2venc * velocity_magnitude_squared**0.5
        pcmra_gamma_05 = _normalize_pcmra(np.mean(pcmra_t2[..., phase_indices], axis=3))

    with timer.measure("计算 R1 速度不连续函数"):
        discontinuity = compute_discontinuity_function_third_order(
            velocity[..., peak - 1, :],
            sigma,
        )
        r1 = _matlab_uint16(128.0 * np.exp(discontinuity))

    return SegmentationVolumes(
        pcmra_gamma_02=pcmra_gamma_02,
        pcmra_gamma_05=pcmra_gamma_05,
        r1=r1,
        phase_numbers=selected,
        peak_phase_number=peak,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )


def generate_segmentation_dicoms(
    case: LoadedDicomCase,
    velocity_result: VelocityConversionResult,
    case_directory: str | Path | None = None,
    phase_numbers: Sequence[int] | None = None,
    peak_phase_number: int = 15,
    sigma: float = 0.5,
) -> SegmentationExportResult:
    """执行 FINAL 第 06 部分并写入 ``results_PCA``、``PCA2`` 和 ``R1``"""

    output_root = case.case_directory if case_directory is None else Path(case_directory)
    timer = RuntimeTimer()
    with timer.measure("生成分割用三维图像"):
        volumes = compute_segmentation_volumes(
            case,
            velocity_result,
            phase_numbers=phase_numbers,
            peak_phase_number=peak_phase_number,
            sigma=sigma,
        )
    with timer.measure("导出 gamma 0.2 PCMRA"):
        export_02 = save_dicom_volume(
            volumes.pcmra_gamma_02,
            output_root / "results_PCA",
            case,
            "PCA0.2" if case.is_enhanced_dicom else "pcmra0.2",
        )
    with timer.measure("导出 gamma 0.5 PCMRA"):
        export_05 = save_dicom_volume(
            volumes.pcmra_gamma_05,
            output_root / "results_PCA2",
            case,
            "PCA0.5" if case.is_enhanced_dicom else "pcmra0.5",
        )
    with timer.measure("导出 R1"):
        export_r1 = save_dicom_volume(
            volumes.r1,
            output_root / "results_R1",
            case,
            "R1",
        )
    return SegmentationExportResult(
        volumes=volumes,
        pcmra_gamma_02_export=export_02,
        pcmra_gamma_05_export=export_05,
        r1_export=export_r1,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
