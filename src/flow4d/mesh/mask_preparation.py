"""第 09 部分的掩膜裁剪、Makima 加密与二值平滑"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .makima3d import (
    Makima3DError,
    _interp3_makima_half_grid,
    interp3_makima,
)

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "mesh":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]


class MaskPreparationError(RuntimeError):
    """掩膜无法进入第 09 部分网格流程时抛出"""


@dataclass(frozen=True, slots=True)
class MaskCropBounds:
    """与 MATLAB 首末索引对应、包含端点且从 0 开始的边界"""

    row_start: int
    row_stop: int
    column_start: int
    column_stop: int
    slice_start: int
    slice_stop: int

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.row_start, self.row_stop + 1),
            slice(self.column_start, self.column_stop + 1),
            slice(self.slice_start, self.slice_stop + 1),
        )


@dataclass(frozen=True, slots=True)
class MaskPreparationResult:
    """裁剪后的输入及用于网格生成的加密二值掩膜"""

    bounds: MaskCropBounds
    mask_crop: NDArray[np.uint8]
    velocity_crop: NDArray[np.float64]
    refined_mask: NDArray[np.uint8]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def crop_mask_and_velocity(
    mask: NDArray[np.generic],
    velocity: NDArray[np.generic],
) -> tuple[MaskCropBounds, NDArray[np.uint8], NDArray[np.float64]]:
    """裁剪三维掩膜及与之匹配的五维速度数组"""

    mask_array = np.asarray(mask)
    velocity_array = np.asarray(velocity)
    if mask_array.ndim != 3:
        raise MaskPreparationError("Mask 必须为三维数组")
    if velocity_array.ndim != 5 or velocity_array.shape[-1] != 3:
        raise MaskPreparationError(
            "velocity 必须为 [row,column,slice,time,3]"
        )
    if velocity_array.shape[:3] != mask_array.shape:
        raise MaskPreparationError("Mask 与 velocity 的前三维尺寸不一致")
    occupied = np.argwhere(mask_array != 0)
    if occupied.size == 0:
        raise MaskPreparationError("Mask 为空，无法确定裁剪范围")
    lower = occupied.min(axis=0)
    upper = occupied.max(axis=0)
    bounds = MaskCropBounds(
        row_start=int(lower[0]),
        row_stop=int(upper[0]),
        column_start=int(lower[1]),
        column_stop=int(upper[1]),
        slice_start=int(lower[2]),
        slice_stop=int(upper[2]),
    )
    spatial_slices = bounds.slices
    mask_crop = np.asarray(mask_array[spatial_slices] != 0, dtype=np.uint8)
    velocity_crop = np.asarray(
        velocity_array[spatial_slices + (slice(None), slice(None))],
        dtype=np.float64,
    ).copy()
    return bounds, mask_crop, velocity_crop


def upsample_mask_makima(mask: NDArray[np.generic]) -> NDArray[np.float64]:
    """使用原生 MATLAB R2021a 三维 Makima 加密 ``mask[y,x,z]``

    非单节点轴至少需要 7 个节点，以符合已验证的生产内核约束
    单节点轴保持为单节点，并在内部以常值延拓表示；长度为 2–6 的轴
    直接拒绝，不转交其他插值后端
    """

    mask_array = np.asarray(mask)
    if np.iscomplexobj(mask_array):
        raise MaskPreparationError("待加密 Mask 不能包含 complex 数值")
    output = np.asarray(mask_array, dtype=np.float64)
    if output.ndim != 3 or 0 in output.shape:
        raise MaskPreparationError("待加密 Mask 必须是非空三维数组")
    if not np.all(np.isfinite(output)):
        raise MaskPreparationError("待加密 Mask 不能包含 NaN 或 Inf")

    for axis, length in enumerate(output.shape):
        if length != 1 and length < 7:
            raise MaskPreparationError(
                "Makima 加密要求每个非单节点轴至少有 7 个节点；"
                f"axis={axis}, length={length}"
            )

    if all(length > 1 for length in output.shape):
        y_grid = np.arange(output.shape[0], dtype=np.float64)
        x_grid = np.arange(output.shape[1], dtype=np.float64)
        z_grid = np.arange(output.shape[2], dtype=np.float64)
        try:
            return _interp3_makima_half_grid(
                x_grid, y_grid, z_grid, output
            )
        except Makima3DError as exc:
            raise MaskPreparationError(
                f"Mask Makima 加密失败：{exc}"
            ) from exc

    expanded = output
    grids: list[NDArray[np.float64]] = []
    queries: list[NDArray[np.float64]] = []
    for axis, length in enumerate(output.shape):
        if length == 1:
            expanded = np.repeat(expanded, 7, axis=axis)
            grids.append(np.arange(7, dtype=np.float64))
            queries.append(np.array([3.0], dtype=np.float64))
            continue
        grids.append(np.arange(length, dtype=np.float64))
        queries.append(
            np.linspace(0.0, float(length - 1), 2 * length - 1)
        )

    y_grid, x_grid, z_grid = grids
    y_query, x_query, z_query = queries
    try:
        return np.asarray(
            interp3_makima(
                x_grid,
                y_grid,
                z_grid,
                expanded,
                x_query[None, :, None],
                y_query[:, None, None],
                z_query[None, None, :],
            ),
            dtype=np.float64,
        )
    except Makima3DError as exc:
        raise MaskPreparationError(f"Mask Makima 加密失败：{exc}") from exc


def smooth_binary_volume(
    volume: NDArray[np.generic],
    iterations: int = 5,
) -> NDArray[np.float64]:
    """忠实复现 iso2mesh ``smoothbinvol.m`` 的线性索引行为"""

    output = np.array(volume, dtype=np.float64, order="F", copy=True)
    if output.ndim != 3:
        raise MaskPreparationError("smooth_binary_volume 需要三维数组")
    if iterations < 0:
        raise MaskPreparationError("iterations 不能为负数")

    dimensions = output.shape
    plane_size = dimensions[0] * dimensions[1]
    full_length = int(np.prod(dimensions))
    weight = 1.0 / 6.0
    offsets = (
        1,
        -1,
        dimensions[0],
        -dimensions[0],
        plane_size,
        -plane_size,
    )
    flat = output.reshape(-1, order="F")

    for _ in range(iterations):
        index_zero_based = np.flatnonzero(flat != 0)
        index_one_based = index_zero_based + 1
        values = flat[index_zero_based].copy()

        for offset in offsets:
            next_index_one_based = index_one_based + offset
            valid = (next_index_one_based > 0) & (
                next_index_one_based < full_length
            )
            target_zero_based = next_index_one_based[valid] - 1
            flat[target_zero_based] = (
                flat[target_zero_based] + weight * values[valid]
            )

        flat[index_zero_based] = values

    return flat.reshape(dimensions, order="F")


def prepare_mask_for_meshing(
    mask: NDArray[np.generic],
    velocity: NDArray[np.generic],
    *,
    gaussian_sigma: float = 0.8,
    binary_smoothing_iterations: int = 5,
) -> MaskPreparationResult:
    """执行 ``run_iso2mesh.m`` 中裁剪和掩膜加密部分"""

    if not np.isfinite(gaussian_sigma) or gaussian_sigma <= 0:
        raise MaskPreparationError("gaussian_sigma 必须为正有限数值")
    timer = RuntimeTimer()
    with timer.measure("根据 Mask 裁剪速度场"):
        bounds, mask_crop, velocity_crop = crop_mask_and_velocity(mask, velocity)
    with timer.measure("Mask Makima 二倍加密"):
        first_smoothed = gaussian_filter(
            mask_crop.astype(np.float64),
            gaussian_sigma,
            mode="nearest",
            radius=2,
        )
        refined = upsample_mask_makima(first_smoothed)
    with timer.measure("Mask 二值平滑"):
        refined = np.asarray(refined >= 0.5, dtype=np.float64)
        refined = gaussian_filter(
            refined,
            gaussian_sigma,
            mode="nearest",
            radius=2,
        )
        refined = smooth_binary_volume(refined, binary_smoothing_iterations)
        refined_mask = np.asarray(refined >= 0.5, dtype=np.uint8)
    return MaskPreparationResult(
        bounds=bounds,
        mask_crop=mask_crop,
        velocity_crop=velocity_crop,
        refined_mask=refined_mask,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
