"""用于 4D Flow MRI 的 MSAC 多项式背景相位校正"""

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

from .polynomial import build_polynomial_system
from .velocity import VelocityConversionResult


class MsacCorrectionError(RuntimeError):
    """MSAC 准备或多项式校正无法继续时抛出"""


@dataclass(frozen=True, slots=True)
class MsacResult:
    """MATLAB ``msac_function4D`` 结果的直接 Python 对应结构"""

    magnitude_mask: NDArray[np.bool_]
    msac_inlier_mask: NDArray[np.bool_]
    background_phase: NDArray[np.float64]
    corrected_average_phase: NDArray[np.float64]
    corrected_time_resolved_phase: NDArray[np.float64]
    inlier_indices: NDArray[np.bool_]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float

    @property
    def inspection_message(self) -> str:
        timing_text = "\n".join(
            f"- {item.step_name}：{item.elapsed_seconds:.3f} 秒"
            for item in self.timings
        )
        return (
            "MSAC 背景相位校正完成\n"
            f"Magnitude mask 点数：{int(self.magnitude_mask.sum())}\n"
            f"背景相位形状：{self.background_phase.shape}\n"
            f"校正后时序相位形状：{self.corrected_time_resolved_phase.shape}\n"
            f"{timing_text}\n"
            f"- 总耗时：{self.total_elapsed_seconds:.3f} 秒"
        )


@dataclass(frozen=True, slots=True)
class MsacCorrectionResult:
    """校正后的速度及按 MATLAB 方式重新编码的相位像素"""

    corrected_case: LoadedDicomCase
    velocity: NDArray[np.float64]
    msac: MsacResult
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float

    @property
    def inspection_message(self) -> str:
        timing_text = "\n".join(
            f"- {item.step_name}：{item.elapsed_seconds:.3f} 秒"
            for item in self.timings
        )
        return (
            "run_MSAC 迁移流程完成\n"
            f"校正后速度形状：{self.velocity.shape}\n"
            f"重新编码相位类型：{self.corrected_case.ap.dtype}\n"
            f"{timing_text}\n"
            f"- 总耗时：{self.total_elapsed_seconds:.3f} 秒"
        )


def _validate_msac_inputs(
    phase_images: NDArray[np.generic],
    magnitude_images: NDArray[np.generic],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    phase = np.asarray(phase_images, dtype=np.float64)
    magnitude = np.asarray(magnitude_images, dtype=np.float64)
    if phase.ndim != 5 or phase.shape[-1] != 3:
        raise MsacCorrectionError(
            "phase_images 必须为 [row,column,slice,time,3]"
        )
    if magnitude.ndim != 4 or magnitude.shape != phase.shape[:4]:
        raise MsacCorrectionError(
            "magnitude_images 必须为 [row,column,slice,time]，"
            "且前四维与 phase_images 一致"
        )
    if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(magnitude)):
        raise MsacCorrectionError("MSAC 输入包含 NaN 或 Inf")
    return phase, magnitude


def msac_correct_4d(
    phase_images: NDArray[np.generic],
    magnitude_images: NDArray[np.generic],
    msac_fit_order: int = 1,
    correction_fit_order: int = 3,
) -> MsacResult:
    """以确定性的 NumPy 采样方式移植 ``msac_function4D.m``"""

    phase_images_f64, magnitude_images_f64 = _validate_msac_inputs(
        phase_images,
        magnitude_images,
    )
    timer = RuntimeTimer()
    threshold = 0.01
    sample_count = 10
    trial_count = 100
    magnitude_threshold = 0.12
    spatial_shape = magnitude_images_f64.shape[:3]

    with timer.measure("准备 MSAC 输入数据"):
        magnitude_average = np.mean(magnitude_images_f64, axis=3)
        phase_average = np.mean(phase_images_f64, axis=3)
        magnitude_mask = magnitude_average > magnitude_threshold
        coordinates = (
            np.indices(spatial_shape, dtype=np.float64)
            .reshape(3, -1)
            .T
            + 1.0
        )
        all_values = phase_average.reshape(-1, 3)
        all_points = np.column_stack((all_values, coordinates))
        mask_flat = magnitude_mask.reshape(-1)
        mask_points = all_points[mask_flat]
        point_count = mask_points.shape[0]
        if point_count < sample_count:
            raise MsacCorrectionError(
                "Magnitude mask 中至少需要 10 个点，"
                f"实际为 {point_count} 个"
            )
        mask_system_msac = build_polynomial_system(msac_fit_order, mask_points)

    with timer.measure("执行 100 次 MSAC 试验"):
        best_cost = np.full(3, threshold * point_count, dtype=np.float64)
        best_inliers = np.zeros((point_count, 3), dtype=np.bool_)
        for matlab_trial_index in range(1, trial_count + 1):
            random_state = np.random.RandomState(matlab_trial_index + 8179)
            sample_indices = random_state.permutation(point_count)[:sample_count]
            sample_system = build_polynomial_system(
                msac_fit_order,
                mask_points[sample_indices],
            )
            coefficients = np.linalg.lstsq(
                sample_system.design_matrix,
                sample_system.values,
                rcond=None,
            )[0]
            estimated = mask_system_msac.design_matrix @ coefficients
            residuals = np.abs((mask_system_msac.values - estimated) / 2.0)
            capped_residuals = np.minimum(residuals, threshold)
            inliers = residuals < threshold
            cost = np.sum(capped_residuals, axis=0)
            improved = best_cost > cost
            best_cost[improved] = cost[improved]
            best_inliers[:, improved] = inliers[:, improved]

    with timer.measure("拟合最终背景相位"):
        correction_system = build_polynomial_system(
            correction_fit_order,
            mask_points,
        )
        correction_coefficients = np.zeros(
            (correction_system.term_count, 3),
            dtype=np.float64,
        )
        for dimension in range(3):
            inliers = best_inliers[:, dimension]
            if not np.any(inliers):
                raise MsacCorrectionError(
                    f"第 {dimension + 1} 个速度分量没有 MSAC 内点"
                )
            if correction_fit_order == 0:
                correction_coefficients[0, dimension] = np.mean(
                    correction_system.values[inliers, dimension]
                )
            else:
                correction_coefficients[:, dimension] = np.linalg.lstsq(
                    correction_system.design_matrix[inliers],
                    correction_system.values[inliers, dimension],
                    rcond=None,
                )[0]
        all_system = build_polynomial_system(correction_fit_order, all_points)
        background_phase = (
            all_system.design_matrix @ correction_coefficients
        ).reshape(spatial_shape + (3,))

        inlier_mask_flat = np.zeros((mask_flat.size, 3), dtype=np.bool_)
        inlier_mask_flat[mask_flat] = best_inliers
        msac_inlier_mask = inlier_mask_flat.reshape(spatial_shape + (3,))

    with timer.measure("校正时间分辨相位"):
        corrected_average = phase_average - background_phase
        corrected_time_resolved = (
            phase_images_f64 - background_phase[:, :, :, None, :]
        )

    return MsacResult(
        magnitude_mask=magnitude_mask,
        msac_inlier_mask=msac_inlier_mask,
        background_phase=background_phase,
        corrected_average_phase=corrected_average,
        corrected_time_resolved_phase=corrected_time_resolved,
        inlier_indices=best_inliers,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )


def _matlab_uint16(values: NDArray[np.generic]) -> NDArray[np.uint16]:
    array = np.asarray(values, dtype=np.float64)
    rounded = np.copysign(np.floor(np.abs(array) + 0.5), array)
    return np.clip(rounded, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def apply_msac_correction(
    case: LoadedDicomCase,
    velocity_result: VelocityConversionResult,
) -> MsacCorrectionResult:
    """移植 ``run_MSAC.m``，返回校正后的病例数据和速度"""

    if not np.isfinite(case.venc) or case.venc == 0:
        raise MsacCorrectionError("VENC 必须是有限非零数值")
    if not np.isfinite(case.rescale_slope) or case.rescale_slope == 0:
        raise MsacCorrectionError("RescaleSlope 必须是有限非零数值")
    if velocity_result.velocity.shape != case.shape + (3,):
        raise MsacCorrectionError("velocity 形状与病例四维图像不一致")
    if velocity_result.intensity.shape != case.shape:
        raise MsacCorrectionError("intensity 形状与病例四维图像不一致")

    timer = RuntimeTimer()
    with timer.measure("归一化 MSAC 输入"):
        phase_images = velocity_result.velocity / case.venc
        center_column = (velocity_result.intensity.shape[1] - 1) // 2
        center_slice = velocity_result.intensity[:, center_column, :, :]
        maximum = float(np.max(center_slice))
        if not np.isfinite(maximum) or maximum == 0:
            raise MsacCorrectionError(
                "Magnitude 中心切片最大值必须是有限非零数值"
            )
        normalized_magnitude = velocity_result.intensity / maximum

    with timer.measure("执行 MSAC 背景相位校正"):
        msac = msac_correct_4d(
            phase_images,
            normalized_magnitude,
            msac_fit_order=1,
            correction_fit_order=3,
        )

    with timer.measure("恢复速度并重新编码相位像素"):
        corrected_velocity = msac.corrected_time_resolved_phase * case.venc
        corrected_ap = _matlab_uint16(
            (corrected_velocity[..., 2] - case.rescale_intercept)
            / case.rescale_slope
        )
        corrected_fh = _matlab_uint16(
            (corrected_velocity[..., 1] - case.rescale_intercept)
            / case.rescale_slope
        )
        corrected_rl = _matlab_uint16(
            (corrected_velocity[..., 0] - case.rescale_intercept)
            / case.rescale_slope
        )
        corrected_case = replace(
            case,
            ap=corrected_ap,
            fh=corrected_fh,
            rl=corrected_rl,
        )

    return MsacCorrectionResult(
        corrected_case=corrected_case,
        velocity=corrected_velocity,
        msac=msac,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
