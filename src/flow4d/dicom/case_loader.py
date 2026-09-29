"""从 Classic 或 Enhanced DICOM 文件夹读取一个 4D Flow MRI 病例

本模块对应 ``runFunction/run_loadDCMfromFiles.m``目录和格式选择由
后续桌面界面负责；核心函数显式接收这两个值，并组合已经迁移的 DICOM 工具
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

try:
    from ..base_function import (
        DicomSeriesInventory,
        RuntimeTimer,
        TimingRecord,
        inspect_dicom_series,
    )
except ImportError:
    if __package__ != "dicom":
        raise
    from base_function import (  # type: ignore[no-redef]
        DicomSeriesInventory,
        RuntimeTimer,
        TimingRecord,
        inspect_dicom_series,
    )

from .enhanced_loader import Enhanced4DData, convert_enhanced_dicom_data
from .series_metadata import DicomSeriesMetadata, read_dicom_series_metadata
from .volume_loader import (
    Classic4DVolumes,
    ProgressCallback,
    load_classic_4d_volumes,
)


class CaseLoadError(RuntimeError):
    """所选病例无法进入任一 DICOM 读取分支时抛出"""


@dataclass(frozen=True, slots=True)
class LoadedDicomCase:
    """Classic 与 Enhanced 读取分支返回的统一结果"""

    case_directory: Path
    inventory: DicomSeriesInventory
    is_enhanced_dicom: bool
    adj_velocity: int
    magnitude_info: Dataset
    phase_info: Dataset
    reference_image: NDArray[np.generic]
    manufacturer: str
    series: object
    rescale_intercept: float
    rescale_slope: float
    venc: float
    cardiac_phases: int
    voxel_size_mm: tuple[float, float, float]
    heart_rate_bpm: float
    time_spacing_ms: float
    magnitude: NDArray[np.generic]
    ap: NDArray[np.generic]
    fh: NDArray[np.generic]
    rl: NDArray[np.generic]
    classic_series_metadata: DicomSeriesMetadata | None
    classic_volumes: Classic4DVolumes | None
    enhanced_data: Enhanced4DData | None
    timings: tuple[TimingRecord, ...] = ()
    total_elapsed_seconds: float = 0.0

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """返回共享的行、列、切片、心动时相形状"""

        return self.magnitude.shape

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """返回公共输出数据类型"""

        return self.magnitude.dtype

    @property
    def mode_name(self) -> str:
        """返回所选 DICOM 分支的用户可读名称"""

        return "Enhanced DICOM" if self.is_enhanced_dicom else "Classic DICOM"

    @property
    def inspection_message(self) -> str:
        """返回适合后续桌面界面显示的中文摘要"""

        rows, columns, slices, phases = self.shape
        if self.classic_volumes is None:
            direction_text = "方向来源：Enhanced DICOM 已按 AP、FH、RL 输入目录装载"
        else:
            direction_text = (
                "方向重排："
                f"AP←{self.classic_volumes.ap_source}，"
                f"FH←{self.classic_volumes.fh_source}，"
                f"RL←{self.classic_volumes.rl_source}"
            )

        timing_lines = "\n".join(
            f"- {record.step_name}：{record.elapsed_seconds:.3f} 秒"
            for record in self.timings
        )
        if not timing_lines:
            timing_lines = "- 暂无分步骤计时记录"

        return (
            "4D Flow MRI 病例读取完成\n"
            f"病例目录：{self.case_directory}\n"
            f"输入格式：{self.mode_name}\n"
            f"矩阵形状：[row={rows}, column={columns}, "
            f"slice={slices}, cardiac_phase={phases}]\n"
            f"数据类型：{self.dtype}\n"
            f"VENC：{self.venc:g}\n"
            "体素间距："
            f"{self.voxel_size_mm[0]:g} × {self.voxel_size_mm[1]:g} × "
            f"{self.voxel_size_mm[2]:g} mm\n"
            f"时间间隔：{self.time_spacing_ms:g} ms\n"
            f"{direction_text}\n"
            "速度方向人工校正次数：0\n"
            "运行耗时：\n"
            f"{timing_lines}\n"
            f"- 总耗时：{self.total_elapsed_seconds:.3f} 秒"
        )


def _require_ready_inventory(
    inventory: DicomSeriesInventory,
    *,
    is_enhanced: bool,
) -> None:
    if is_enhanced:
        if inventory.counts != (1, 1, 1, 1):
            counts = "，".join(
                f"{name}={len(files)}" for name, files in inventory.series
            )
            raise CaseLoadError(
                "Enhanced DICOM 模式要求 M、AP、FH、RL 四个目录各有且仅有 "
                f"1 个多帧 DICOM 文件；当前为：{counts}"
            )
        return

    if not inventory.ready_for_metadata_reading:
        raise CaseLoadError(inventory.inspection_message)


def _read_dicom_metadata(path: Path, series_name: str) -> Dataset:
    try:
        return dcmread(path, stop_before_pixels=True)
    except InvalidDicomError as error:
        raise CaseLoadError(
            f"{series_name} 的首个文件不是可识别的 DICOM：{path}"
        ) from error
    except OSError as error:
        raise CaseLoadError(
            f"无法读取 {series_name} 的首个 DICOM：{path}"
            "请检查文件权限或文件状态"
        ) from error


def _read_enhanced_dicom(
    path: Path,
    series_name: str,
) -> tuple[Dataset, NDArray[np.generic]]:
    try:
        dataset = dcmread(path)
    except InvalidDicomError as error:
        raise CaseLoadError(
            f"{series_name} 文件不是可识别的 Enhanced DICOM：{path}"
        ) from error
    except OSError as error:
        raise CaseLoadError(
            f"无法读取 {series_name} Enhanced DICOM：{path}"
            "请检查文件权限或文件状态"
        ) from error

    try:
        pixels = np.asarray(dataset.pixel_array)
    except (
        AttributeError,
        ImportError,
        NotImplementedError,
        RuntimeError,
        ValueError,
    ) as error:
        raise CaseLoadError(
            f"无法解码 {series_name} Enhanced DICOM 的像素数据：{path}"
            "请检查 PixelData 和传输语法；压缩 DICOM 可能需要额外解码组件"
        ) from error
    return dataset, pixels


def _integer_cardiac_phases(metadata: DicomSeriesMetadata) -> int:
    cardiac_phases = float(metadata.cardiac_phases)
    if not cardiac_phases.is_integer():
        raise CaseLoadError(
            "Classic DICOM 的 CardiacPhases 必须是整数，"
            f"实际读取为 {cardiac_phases}"
        )
    return int(cardiac_phases)


def _load_classic_case(
    inventory: DicomSeriesInventory,
    expected_timeframes: int,
    progress_callback: ProgressCallback | None,
    timer: RuntimeTimer,
) -> LoadedDicomCase:
    with timer.measure("读取 Classic 元数据"):
        metadata = read_dicom_series_metadata(
            inventory.ap_files,
            expected_timeframes,
            check_file_count=True,
        )
    cardiac_phases = _integer_cardiac_phases(metadata)
    with timer.measure("读取 Classic 四维像素"):
        volumes = load_classic_4d_volumes(
            inventory.magnitude_files,
            inventory.ap_files,
            inventory.fh_files,
            inventory.rl_files,
            num_periods=cardiac_phases,
            progress_callback=progress_callback,
        )
    with timer.measure("读取 M 图元数据"):
        magnitude_info = _read_dicom_metadata(
            inventory.magnitude_files[0],
            "M 图",
        )

    return LoadedDicomCase(
        case_directory=inventory.case_directory,
        inventory=inventory,
        is_enhanced_dicom=False,
        adj_velocity=0,
        magnitude_info=magnitude_info,
        phase_info=metadata.dicom_info,
        reference_image=metadata.image_data,
        manufacturer=metadata.manufacturer,
        series=metadata.series,
        rescale_intercept=metadata.rescale_intercept,
        rescale_slope=metadata.rescale_slope,
        venc=metadata.venc,
        cardiac_phases=cardiac_phases,
        voxel_size_mm=metadata.voxel_size_mm,
        heart_rate_bpm=metadata.heart_rate_bpm,
        time_spacing_ms=metadata.time_spacing_ms,
        magnitude=volumes.magnitude,
        ap=volumes.ap,
        fh=volumes.fh,
        rl=volumes.rl,
        classic_series_metadata=metadata,
        classic_volumes=volumes,
        enhanced_data=None,
    )


def _load_enhanced_case(
    inventory: DicomSeriesInventory,
    timer: RuntimeTimer,
) -> LoadedDicomCase:
    with timer.measure("读取四个 Enhanced DICOM"):
        magnitude_info, magnitude_pixels = _read_enhanced_dicom(
            inventory.magnitude_files[0],
            "M 图",
        )
        _, ap_pixels = _read_enhanced_dicom(inventory.ap_files[0], "AP 图")
        phase_info, fh_pixels = _read_enhanced_dicom(
            inventory.fh_files[0],
            "FH 图",
        )
        _, rl_pixels = _read_enhanced_dicom(inventory.rl_files[0], "RL 图")

    with timer.measure("转换 Enhanced 四维数据"):
        enhanced = convert_enhanced_dicom_data(
            magnitude_info,
            phase_info,
            magnitude_pixels,
            ap_pixels,
            fh_pixels,
            rl_pixels,
        )
    return LoadedDicomCase(
        case_directory=inventory.case_directory,
        inventory=inventory,
        is_enhanced_dicom=True,
        adj_velocity=0,
        magnitude_info=enhanced.magnitude_dicom_info,
        phase_info=enhanced.phase_dicom_info,
        reference_image=enhanced.reference_image,
        manufacturer=enhanced.manufacturer,
        series=enhanced.series,
        rescale_intercept=enhanced.rescale_intercept,
        rescale_slope=enhanced.rescale_slope,
        venc=enhanced.venc,
        cardiac_phases=enhanced.cardiac_phases,
        voxel_size_mm=enhanced.voxel_size_mm,
        heart_rate_bpm=enhanced.heart_rate_bpm,
        time_spacing_ms=enhanced.time_spacing_ms,
        magnitude=enhanced.magnitude,
        ap=enhanced.ap,
        fh=enhanced.fh,
        rl=enhanced.rl,
        classic_series_metadata=None,
        classic_volumes=None,
        enhanced_data=enhanced,
    )


def load_dicom_case(
    case_directory: str | Path,
    *,
    is_enhanced: bool,
    expected_classic_timeframes: int = 25,
    progress_callback: ProgressCallback | None = None,
) -> LoadedDicomCase:
    """读取一个已经准备好的 ``DICOM_4D_Qflow`` 目录

    参数：
        case_directory: 包含固定 M、AP、FH、RL 子目录的病例目录
        is_enhanced: 调用方或界面提供的显式格式选择
        expected_classic_timeframes: MATLAB 默认用于比较的 25 时相；实际
            重排仍由 Philips 心动时相标签决定
        progress_callback: Classic 序列可选的进度接收器；Enhanced DICOM
            当前整文件读取四个序列，不提供分段回调

    返回：
        包含路径、元数据和四个四维数组的统一病例记录

    异常：
        TypeError: ``is_enhanced`` 不是布尔值
        ValueError: Classic 期望时相数无效
        CaseLoadError: 所选目录与请求的 DICOM 分支不匹配，或必要文件
            无法读取
    """

    if not isinstance(is_enhanced, bool):
        raise TypeError("is_enhanced 必须明确传入 True 或 False")

    timer = RuntimeTimer()
    with timer.measure("扫描 DICOM 目录"):
        inventory = inspect_dicom_series(case_directory)
    _require_ready_inventory(inventory, is_enhanced=is_enhanced)

    if is_enhanced:
        loaded_case = _load_enhanced_case(inventory, timer)
        return replace(
            loaded_case,
            timings=timer.records,
            total_elapsed_seconds=timer.total_elapsed_seconds,
        )

    if (
        isinstance(expected_classic_timeframes, bool)
        or not isinstance(expected_classic_timeframes, int)
        or expected_classic_timeframes <= 0
    ):
        raise ValueError("Classic DICOM 的预期时相数必须是正整数")

    loaded_case = _load_classic_case(
        inventory,
        expected_classic_timeframes,
        progress_callback,
        timer,
    )
    return replace(
        loaded_case,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
