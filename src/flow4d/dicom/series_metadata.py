"""读取经典 4D Flow DICOM 序列的核心元数据和第一幅图像

本模块对应 ``baseFunction/GetDicominfo4D.m``，只读取有序序列的第一份
文件，并保留 MATLAB 的参数来源和公式，不尝试进行厂商无关的推断
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag


MetadataSource: TypeAlias = Literal["private_tag", "rescale_intercept", "input"]

_PHILIPS_SERIES_TAG = Tag(0x2001, 0x1020)
_PHILIPS_CARDIAC_PHASES_TAG = Tag(0x2001, 0x1017)
_PHILIPS_VENC_TAG = Tag(0x2005, 0x100B)


class SeriesMetadataError(RuntimeError):
    """无法读取必要元数据或第一幅 DICOM 图像时抛出"""


@dataclass(frozen=True, slots=True)
class DicomSeriesMetadata:
    """从一个速度序列的第一份 DICOM 提取的核心参数"""

    first_file: Path
    image_data: NDArray[np.generic]
    dicom_info: Dataset
    manufacturer: str
    series: object
    rescale_intercept: float
    rescale_slope: float
    venc: float
    venc_source: MetadataSource
    venc_consistency_difference: float | None
    cardiac_phases: float
    cardiac_phases_source: MetadataSource
    voxel_size_mm: tuple[float, float, float]
    heart_rate_bpm: float
    time_spacing_ms: float
    file_count: int
    requested_timeframes: int
    file_count_is_timeframe_multiple: bool | None

    @property
    def inspection_message(self) -> str:
        """返回适合后续桌面界面显示的中文摘要"""

        if self.file_count_is_timeframe_multiple is None:
            count_status = "未启用文件数量与时相数的整倍数检查"
        elif self.file_count_is_timeframe_multiple:
            count_status = (
                f"文件数量检查通过：{self.file_count} 个文件是 "
                f"{self.requested_timeframes} 个时相的整倍数"
            )
        else:
            count_status = (
                f"文件数量检查未通过：{self.file_count} 个文件不是 "
                f"{self.requested_timeframes} 个时相的整倍数"
            )

        venc_source_text = (
            "Philips 私有标签 (2005,100B)"
            if self.venc_source == "private_tag"
            else "RescaleIntercept 的相反数"
        )
        cardiac_source_text = (
            "Philips 私有标签 (2001,1017)"
            if self.cardiac_phases_source == "private_tag"
            else "用户输入的预期时相数"
        )

        return (
            "DICOM 参数读取完成\n"
            f"首个文件：{self.first_file}\n"
            f"设备厂商：{self.manufacturer}\n"
            f"VENC：{self.venc:g}（来源：{venc_source_text}）\n"
            f"心动周期时相数：{self.cardiac_phases:g}"
            f"（来源：{cardiac_source_text}）\n"
            "体素间距："
            f"{self.voxel_size_mm[0]:g} × {self.voxel_size_mm[1]:g} × "
            f"{self.voxel_size_mm[2]:g} mm\n"
            f"心率：{self.heart_rate_bpm:g} 次/分钟\n"
            f"估算帧间隔：{self.time_spacing_ms:g} ms\n"
            f"{count_status}"
        )


def _required_keyword(dataset: Dataset, keyword: str) -> object:
    if keyword not in dataset:
        raise SeriesMetadataError(f"首个 DICOM 缺少必要字段 {keyword}")
    return getattr(dataset, keyword)


def _required_private_value(dataset: Dataset, tag: Tag, matlab_name: str) -> object:
    if tag not in dataset:
        raise SeriesMetadataError(
            f"首个 DICOM 缺少必要的 Philips 私有标签 {matlab_name}"
        )
    return dataset[tag].value


def _as_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise SeriesMetadataError(
            f"首个 DICOM 的 {field_name} 不是可识别的数值"
        ) from error


def read_dicom_series_metadata(
    dicom_files: Sequence[str | Path],
    num_timeframes: int,
    check_file_count: bool = True,
) -> DicomSeriesMetadata:
    """从第一份 DICOM 读取 MATLAB ``GetDicominfo4D`` 所需参数

    参数：
        dicom_files: 一个 DICOM 序列有序且非空的路径
        num_timeframes: 用于文件数检查的期望心动时相数；Philips 标签
            ``(2001,1017)`` 缺失时也用作回退值
        check_file_count: 是否报告文件数能否被 ``num_timeframes`` 整除

    返回：
        包含第一幅图像和提取参数的元数据记录

    异常：
        TypeError: 传入单一路径而不是路径序列
        ValueError: 路径序列为空，或 ``num_timeframes`` 不是正整数
        FileNotFoundError: 第一条 DICOM 路径不存在
        IsADirectoryError: 第一条 DICOM 路径不是文件
        SeriesMetadataError: 无法读取 DICOM、像素数据或必要元数据
    """

    if isinstance(dicom_files, (str, bytes, Path)):
        raise TypeError(
            "参数读取需要一个有序的 DICOM 文件列表，不能只传入单个路径值"
        )
    if not dicom_files:
        raise ValueError("DICOM 文件列表为空，无法读取参数和首张图像")
    if isinstance(num_timeframes, bool) or not isinstance(num_timeframes, int):
        raise ValueError("预期时相数必须是正整数")
    if num_timeframes <= 0:
        raise ValueError("预期时相数必须大于 0")

    first_path = Path(dicom_files[0]).expanduser()
    if not first_path.exists():
        raise FileNotFoundError(f"首个 DICOM 文件不存在：{first_path}")
    if not first_path.is_file():
        raise IsADirectoryError(f"首个 DICOM 路径不是文件：{first_path}")

    first_path = first_path.resolve()
    try:
        dataset = dcmread(first_path)
    except InvalidDicomError as error:
        raise SeriesMetadataError(
            f"首个文件不是可识别的 DICOM 文件：{first_path}"
        ) from error
    except OSError as error:
        raise SeriesMetadataError(
            f"无法读取首个 DICOM 文件：{first_path}请检查文件权限或文件状态"
        ) from error

    try:
        image_data = dataset.pixel_array
    except (AttributeError, ImportError, NotImplementedError, RuntimeError, ValueError) as error:
        raise SeriesMetadataError(
            "无法解码首个 DICOM 的像素数据请检查 PixelData 和传输语法，"
            "压缩 DICOM 还可能需要额外解码组件"
        ) from error

    manufacturer = str(_required_keyword(dataset, "Manufacturer"))
    series = _required_private_value(dataset, _PHILIPS_SERIES_TAG, "Private_2001_1020")
    rescale_intercept = _as_float(
        _required_keyword(dataset, "RescaleIntercept"), "RescaleIntercept"
    )
    rescale_slope = _as_float(
        _required_keyword(dataset, "RescaleSlope"), "RescaleSlope"
    )

    if _PHILIPS_VENC_TAG in dataset:
        venc = _as_float(dataset[_PHILIPS_VENC_TAG].value, "Private_2005_100b")
        venc_source: MetadataSource = "private_tag"
        venc_consistency_difference = venc + rescale_intercept
    else:
        venc = -rescale_intercept
        venc_source = "rescale_intercept"
        venc_consistency_difference = None

    if _PHILIPS_CARDIAC_PHASES_TAG in dataset:
        cardiac_phases = _as_float(
            dataset[_PHILIPS_CARDIAC_PHASES_TAG].value,
            "Private_2001_1017",
        )
        cardiac_phases_source: MetadataSource = "private_tag"
    else:
        cardiac_phases = float(num_timeframes)
        cardiac_phases_source = "input"

    if cardiac_phases <= 0:
        raise SeriesMetadataError("心动周期时相数必须大于 0，无法计算帧间隔")

    pixel_spacing = _required_keyword(dataset, "PixelSpacing")
    try:
        pixel_spacing_x = float(pixel_spacing[0])  # type: ignore[index]
        pixel_spacing_y = float(pixel_spacing[1])  # type: ignore[index]
    except (IndexError, TypeError, ValueError) as error:
        raise SeriesMetadataError(
            "首个 DICOM 的 PixelSpacing 不能提供两个有效数值"
        ) from error
    pixel_spacing_z = _as_float(
        _required_keyword(dataset, "SpacingBetweenSlices"),
        "SpacingBetweenSlices",
    )

    heart_rate_bpm = _as_float(
        _required_keyword(dataset, "HeartRate"),
        "HeartRate",
    )
    if heart_rate_bpm <= 0:
        raise SeriesMetadataError("HeartRate 必须大于 0，无法计算帧间隔")

    time_spacing_ms = (60.0 / heart_rate_bpm) * 1000.0 / cardiac_phases
    file_count = len(dicom_files)
    file_count_is_timeframe_multiple = (
        file_count % num_timeframes == 0 if check_file_count else None
    )

    return DicomSeriesMetadata(
        first_file=first_path,
        image_data=image_data,
        dicom_info=dataset,
        manufacturer=manufacturer,
        series=series,
        rescale_intercept=rescale_intercept,
        rescale_slope=rescale_slope,
        venc=venc,
        venc_source=venc_source,
        venc_consistency_difference=venc_consistency_difference,
        cardiac_phases=cardiac_phases,
        cardiac_phases_source=cardiac_phases_source,
        voxel_size_mm=(pixel_spacing_x, pixel_spacing_y, pixel_spacing_z),
        heart_rate_bpm=heart_rate_bpm,
        time_spacing_ms=time_spacing_ms,
        file_count=file_count,
        requested_timeframes=num_timeframes,
        file_count_is_timeframe_multiple=file_count_is_timeframe_multiple,
    )
