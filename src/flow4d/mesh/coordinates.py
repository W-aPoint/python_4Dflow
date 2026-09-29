"""FINAL 四面体网格生成后的坐标转换"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def scale_tetrahedral_nodes(
    nodes: NDArray[np.generic],
    voxel_size_mm: tuple[float, float, float],
    *,
    refinement_scale: float = 0.5,
) -> NDArray[np.float64]:
    """应用 ``run_iso2mesh.m`` 中的坐标轴交换和 0.5 缩放"""

    source = np.asarray(nodes, dtype=np.float64)
    spacing = np.asarray(voxel_size_mm, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError("nodes 必须为 N×3 或包含更多列的二维数组")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)):
        raise ValueError("voxel_size_mm 必须包含三个有限数值")
    if np.any(spacing <= 0) or not np.isfinite(refinement_scale) or refinement_scale <= 0:
        raise ValueError("体素间距和 refinement_scale 必须为正数")

    scaled = source[:, :3].copy()
    scaled[:, 0] = source[:, 1] * spacing[0] * refinement_scale
    scaled[:, 1] = source[:, 0] * spacing[1] * refinement_scale
    scaled[:, 2] = source[:, 2] * spacing[2] * refinement_scale
    return scaled
