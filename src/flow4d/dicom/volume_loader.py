"""把四个经典 DICOM 序列读取为与 MATLAB 兼容的四维数组

本模块对应 ``baseFunction/Get4D_Dicom_Matrix_fromList.m``，严格保留文件
顺序：每连续 ``num_periods`` 个文件构成一个切片，最终数组顺序为
行、列、切片、时相
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag


VelocitySeriesSource: TypeAlias = Literal["v1", "v2", "v3"]
ProgressCallback: TypeAlias = Callable[[str, int, int], None]

_PHILIPS_VELOCITY_ENCODING_SEQUENCE_TAG = Tag(0x2005, 0x140F)


class VolumeLoadError(RuntimeError):
    """Classic DICOM 文件无法组成四个有效四维体数据时抛出"""


@dataclass(frozen=True, slots=True)
class Classic4DVolumes:
    """幅度体数据和按方向排列的相位体数据"""

    magnitude: NDArray[np.generic]
    ap: NDArray[np.generic]
    fh: NDArray[np.generic]
    rl: NDArray[np.generic]
    file_count_per_series: int
    num_slices: int
    num_periods: int
    ap_source: VelocitySeriesSource
    fh_source: VelocitySeriesSource
    rl_source: VelocitySeriesSource

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """返回共享的行、列、切片、时相形状"""

        return self.magnitude.shape

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """返回从第一份 v1 DICOM 推断出的公共输出数据类型"""

        return self.magnitude.dtype

    @property
    def inspection_message(self) -> str:
        """返回适合后续桌面界面的中文完成摘要"""

        rows, columns, slices, periods = self.shape
        return (
            "Classic DICOM 四维矩阵读取完成\n"
            f"每组文件数：{self.file_count_per_series}\n"
            f"矩阵形状：[row={rows}, column={columns}, "
            f"slice={slices}, cardiac_phase={periods}]\n"
            f"数据类型：{self.dtype}\n"
            "方向重排："
            f"AP←{self.ap_source}，FH←{self.fh_source}，RL←{self.rl_source}"
        )


def _normalise_paths(
    files: Sequence[str | Path],
    series_name: str,
) -> tuple[Path, ...]:
    if isinstance(files, (str, bytes, Path)):
        raise TypeError(
            f"{series_name} 需要一个有序的 DICOM 文件列表，不能只传入单个路径值"
        )
    if not files:
        raise ValueError(f"{series_name} 的 DICOM 文件列表为空")
    return tuple(Path(path).expanduser() for path in files)


def _read_image(
    path: Path,
    series_name: str,
    one_based_index: int,
) -> tuple[Dataset, NDArray[np.generic]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{series_name} 的第 {one_based_index} 个 DICOM 文件不存在：{path}"
        )
    if not path.is_file():
        raise IsADirectoryError(
            f"{series_name} 的第 {one_based_index} 个 DICOM 路径不是文件：{path}"
        )

    resolved_path = path.resolve()
    try:
        dataset = dcmread(resolved_path)
    except InvalidDicomError as error:
        raise VolumeLoadError(
            f"{series_name} 的第 {one_based_index} 个文件不是可识别的 DICOM："
            f"{resolved_path}"
        ) from error
    except OSError as error:
        raise VolumeLoadError(
            f"无法读取 {series_name} 的第 {one_based_index} 个 DICOM："
            f"{resolved_path}请检查权限或文件状态"
        ) from error

    try:
        image = dataset.pixel_array
    except (AttributeError, ImportError, NotImplementedError, RuntimeError, ValueError) as error:
        raise VolumeLoadError(
            f"无法解码 {series_name} 的第 {one_based_index} 个 DICOM 像素数据："
            f"{resolved_path}压缩 DICOM 可能需要额外解码组件"
        ) from error

    if image.ndim != 2:
        raise VolumeLoadError(
            f"{series_name} 的第 {one_based_index} 张图像不是二维矩阵，"
            f"实际形状为 {image.shape}"
        )
    return dataset, image


def _load_series(
    paths: tuple[Path, ...],
    series_name: str,
    reference_shape: tuple[int, int],
    output_dtype: np.dtype[np.generic],
    num_slices: int,
    num_periods: int,
    progress_callback: ProgressCallback | None,
    first_item: tuple[Dataset, NDArray[np.generic]] | None = None,
) -> tuple[NDArray[np.generic], Dataset]:
    volume = np.empty(
        (*reference_shape, num_slices, num_periods),
        dtype=output_dtype,
    )
    first_dataset: Dataset | None = None

    for zero_based_index, path in enumerate(paths):
        if zero_based_index == 0 and first_item is not None:
            dataset, image = first_item
        else:
            dataset, image = _read_image(
                path,
                series_name,
                zero_based_index + 1,
            )

        if image.shape != reference_shape:
            raise VolumeLoadError(
                f"{series_name} 的第 {zero_based_index + 1} 张图像尺寸不一致："
                f"预期 {reference_shape}，实际 {image.shape}"
            )

        slice_index = zero_based_index // num_periods
        period_index = zero_based_index % num_periods
        volume[:, :, slice_index, period_index] = image

        if first_dataset is None:
            first_dataset = dataset
        if progress_callback is not None:
            progress_callback(series_name, zero_based_index + 1, len(paths))

    if first_dataset is None:
        raise VolumeLoadError(f"{series_name} 没有可读取的 DICOM 文件")
    return volume, first_dataset


def _velocity_axis(dataset: Dataset, series_name: str) -> Literal["AP", "FH", "RL"]:
    if _PHILIPS_VELOCITY_ENCODING_SEQUENCE_TAG not in dataset:
        raise VolumeLoadError(
            f"{series_name} 首个 DICOM 缺少 Philips 私有序列 (2005,140F)，"
            "无法重排 AP、FH、RL"
        )

    sequence = dataset[_PHILIPS_VELOCITY_ENCODING_SEQUENCE_TAG].value
    if not sequence:
        raise VolumeLoadError(
            f"{series_name} 首个 DICOM 的 Philips 私有序列 (2005,140F) 为空"
        )

    first_item = sequence[0]
    if "VelocityEncodingDirection" not in first_item:
        raise VolumeLoadError(
            f"{series_name} 首个 DICOM 缺少 VelocityEncodingDirection"
        )

    raw_vector = first_item.VelocityEncodingDirection
    if isinstance(raw_vector, Iterable) and not isinstance(raw_vector, (str, bytes)):
        components = raw_vector
    else:
        components = (raw_vector,)

    try:
        vector = tuple(float(component) for component in components)
    except (TypeError, ValueError) as error:
        raise VolumeLoadError(
            f"{series_name} 的 VelocityEncodingDirection 不是有效数值向量"
        ) from error

    if len(vector) != 3:
        raise VolumeLoadError(
            f"{series_name} 的 VelocityEncodingDirection 应有 3 个分量，"
            f"实际为 {len(vector)} 个"
        )

    nonzero_axes = tuple(index for index, value in enumerate(vector) if value != 0.0)
    if len(nonzero_axes) != 1:
        raise VolumeLoadError(
            f"{series_name} 的方向向量 {vector} 不能唯一对应 AP、FH 或 RL"
        )

    return ("RL", "AP", "FH")[nonzero_axes[0]]


def load_classic_4d_volumes(
    magnitude_files: Sequence[str | Path],
    velocity_1_files: Sequence[str | Path],
    velocity_2_files: Sequence[str | Path],
    velocity_3_files: Sequence[str | Path],
    *,
    num_periods: int,
    progress_callback: ProgressCallback | None = None,
) -> Classic4DVolumes:
    """读取四个 Classic DICOM 序列并按方向排列

    四个列表必须已经按 MATLAB 兼容顺序组织：先是切片 1 的全部时相，
    再是切片 2 的全部时相，依此类推本函数不按 DICOM 标签或文件名排序
    """

    if isinstance(num_periods, bool) or not isinstance(num_periods, int):
        raise ValueError("心动周期时相数必须是正整数")
    if num_periods <= 0:
        raise ValueError("心动周期时相数必须大于 0")

    paths_by_series = {
        "M": _normalise_paths(magnitude_files, "M"),
        "v1": _normalise_paths(velocity_1_files, "v1"),
        "v2": _normalise_paths(velocity_2_files, "v2"),
        "v3": _normalise_paths(velocity_3_files, "v3"),
    }
    counts = {name: len(paths) for name, paths in paths_by_series.items()}
    if len(set(counts.values())) != 1:
        count_text = "，".join(f"{name}={count}" for name, count in counts.items())
        raise VolumeLoadError(f"四组 DICOM 文件数量不一致：{count_text}")

    file_count = counts["M"]
    if file_count % num_periods != 0:
        raise VolumeLoadError(
            f"每组有 {file_count} 个文件，不能被 {num_periods} 个时相整除"
        )
    num_slices = file_count // num_periods

    v1_first_item = _read_image(paths_by_series["v1"][0], "v1", 1)
    reference_shape = v1_first_item[1].shape
    output_dtype = v1_first_item[1].dtype

    magnitude, _ = _load_series(
        paths_by_series["M"],
        "M",
        reference_shape,
        output_dtype,
        num_slices,
        num_periods,
        progress_callback,
    )
    v1, v1_first_dataset = _load_series(
        paths_by_series["v1"],
        "v1",
        reference_shape,
        output_dtype,
        num_slices,
        num_periods,
        progress_callback,
        first_item=v1_first_item,
    )
    v2, v2_first_dataset = _load_series(
        paths_by_series["v2"],
        "v2",
        reference_shape,
        output_dtype,
        num_slices,
        num_periods,
        progress_callback,
    )
    v3, v3_first_dataset = _load_series(
        paths_by_series["v3"],
        "v3",
        reference_shape,
        output_dtype,
        num_slices,
        num_periods,
        progress_callback,
    )

    source_volumes: dict[VelocitySeriesSource, NDArray[np.generic]] = {
        "v1": v1,
        "v2": v2,
        "v3": v3,
    }
    datasets: dict[VelocitySeriesSource, Dataset] = {
        "v1": v1_first_dataset,
        "v2": v2_first_dataset,
        "v3": v3_first_dataset,
    }
    direction_sources: dict[str, VelocitySeriesSource] = {}
    for source, dataset in datasets.items():
        direction = _velocity_axis(dataset, source)
        if direction in direction_sources:
            raise VolumeLoadError(
                f"速度方向重复：{direction} 同时出现在 "
                f"{direction_sources[direction]} 和 {source}"
            )
        direction_sources[direction] = source

    missing_directions = {"AP", "FH", "RL"} - direction_sources.keys()
    if missing_directions:
        missing_text = "、".join(sorted(missing_directions))
        raise VolumeLoadError(f"速度方向不完整，缺少：{missing_text}")

    ap_source = direction_sources["AP"]
    fh_source = direction_sources["FH"]
    rl_source = direction_sources["RL"]
    return Classic4DVolumes(
        magnitude=magnitude,
        ap=source_volumes[ap_source],
        fh=source_volumes[fh_source],
        rl=source_volumes[rl_source],
        file_count_per_series=file_count,
        num_slices=num_slices,
        num_periods=num_periods,
        ap_source=ap_source,
        fh_source=fh_source,
        rl_source=rl_source,
    )
