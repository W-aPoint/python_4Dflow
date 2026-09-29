"""读取外部分割的 mask/root DICOM 堆栈并生成二值 Mask"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydicom import dcmread

try:
    from ..base_function import MaskDirectoryInventory, RuntimeTimer, TimingRecord
    from .case_loader import LoadedDicomCase
except ImportError:
    if __package__ != "dicom":
        raise
    from base_function import (  # type: ignore[no-redef]
        MaskDirectoryInventory,
        RuntimeTimer,
        TimingRecord,
    )
    from dicom.case_loader import LoadedDicomCase  # type: ignore[no-redef]


class MaskLoadError(RuntimeError):
    """mask/root DICOM 堆栈与已读取病例不匹配时抛出"""


@dataclass(frozen=True, slots=True)
class MaskVolumeResult:
    """原始 mask/root 体数据及与 MATLAB 兼容的二值差"""

    mask_source: NDArray[np.float64]
    root_source: NDArray[np.float64]
    mask: NDArray[np.uint16]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _load_stack(
    files: tuple,
    expected_shape: tuple[int, int, int],
    label: str,
) -> NDArray[np.float64]:
    rows, columns, slices = expected_shape
    if len(files) != slices:
        raise MaskLoadError(
            f"{label} 应包含 {slices} 个 DICOM，实际为 {len(files)} 个"
        )
    output = np.empty(expected_shape, dtype=np.float64)
    for index, path in enumerate(files):
        try:
            pixels = np.asarray(dcmread(path).pixel_array)
        except Exception as error:
            raise MaskLoadError(f"无法读取 {label} DICOM：{path}") from error
        if pixels.shape != (rows, columns):
            raise MaskLoadError(
                f"{label} 第 {index + 1} 层尺寸应为 {(rows, columns)}，"
                f"实际为 {pixels.shape}"
            )
        output[:, :, index] = pixels
    return output


def load_mask_volumes(
    case: LoadedDicomCase,
    inventory: MaskDirectoryInventory,
) -> MaskVolumeResult:
    """对应 ``addMask2data.m``，返回 ``uint16(mask-root != 0)``"""

    expected_shape = case.shape[:3]
    timer = RuntimeTimer()
    with timer.measure("读取 mask DICOM"):
        mask_source = _load_stack(inventory.mask_files, expected_shape, "mask")
    with timer.measure("读取 root DICOM"):
        root_source = _load_stack(inventory.root_files, expected_shape, "root")
    with timer.measure("生成二值血管蒙版"):
        mask = np.asarray(mask_source - root_source != 0, dtype=np.uint16)
    return MaskVolumeResult(
        mask_source=mask_source,
        root_source=root_source,
        mask=mask,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
