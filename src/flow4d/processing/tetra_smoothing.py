"""四面体网格节点的一环邻域高斯平滑"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]


class TetraSmoothingError(RuntimeError):
    """四面体平滑输入无效时抛出"""


@dataclass(frozen=True, slots=True)
class TetraGaussianKernel:
    """可复用的填充一环索引、有效性掩膜和高斯权重"""

    neighbor_indices: NDArray[np.int64]
    valid_mask: NDArray[np.bool_]
    weights: NDArray[np.float64]
    normalization: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def build_tetra_one_ring_neighbors(
    number_of_nodes: int,
    connectivity: NDArray[np.generic],
    *,
    connectivity_index_base: int,
) -> tuple[NDArray[np.int64], ...]:
    """移植 ``find_neighbors_1ring_tetra.m``，输出使用从 0 开始的索引"""

    elements_raw = np.asarray(connectivity)
    if number_of_nodes <= 0:
        raise TetraSmoothingError("number_of_nodes 必须为正整数")
    if elements_raw.ndim != 2 or elements_raw.shape[1] != 4:
        raise TetraSmoothingError("connectivity 必须为 [四面体数,4]")
    if connectivity_index_base not in (0, 1):
        raise TetraSmoothingError("connectivity_index_base 只能为 0 或 1")
    if not np.issubdtype(elements_raw.dtype, np.integer):
        if not np.all(np.equal(elements_raw, np.floor(elements_raw))):
            raise TetraSmoothingError("connectivity 必须只包含整数索引")

    elements = np.asarray(elements_raw, dtype=np.int64) - connectivity_index_base
    if elements.size and (np.min(elements) < 0 or np.max(elements) >= number_of_nodes):
        raise TetraSmoothingError("connectivity 含有超出节点范围的索引")

    neighbors: list[set[int]] = [set() for _ in range(number_of_nodes)]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for element in elements:
        for left_local, right_local in edge_pairs:
            left = int(element[left_local])
            right = int(element[right_local])
            neighbors[left].add(right)
            neighbors[right].add(left)
    return tuple(
        np.asarray(sorted(node_neighbors), dtype=np.int64)
        for node_neighbors in neighbors
    )


def prepare_tetra_gaussian_kernel(
    vertices: NDArray[np.generic],
    connectivity: NDArray[np.generic],
    *,
    sigma: float = 0.8,
    connectivity_index_base: int,
) -> TetraGaussianKernel:
    """移植 ``gaussian_smoothing_tetra.m`` 中可复用的几何计算部分"""

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise TetraSmoothingError("vertices 必须为 [节点数,3]")
    if not np.all(np.isfinite(points)):
        raise TetraSmoothingError("vertices 必须为有限数值")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise TetraSmoothingError("sigma 必须为正有限数值")

    timer = RuntimeTimer()
    with timer.measure("构建四面体节点 1-ring 邻接"):
        neighbors = build_tetra_one_ring_neighbors(
            points.shape[0],
            connectivity,
            connectivity_index_base=connectivity_index_base,
        )
    max_neighbors = max((item.size for item in neighbors), default=0)
    neighbor_indices = np.zeros((points.shape[0], max_neighbors), dtype=np.int64)
    valid_mask = np.zeros((points.shape[0], max_neighbors), dtype=np.bool_)
    for node_index, node_neighbors in enumerate(neighbors):
        count = node_neighbors.size
        if count:
            neighbor_indices[node_index, :count] = node_neighbors
            valid_mask[node_index, :count] = True

    with timer.measure("计算四面体节点高斯权重"):
        if max_neighbors:
            neighbor_positions = points[neighbor_indices]
            differences = points[:, None, :] - neighbor_positions
            distances_squared = np.sum(differences * differences, axis=2)
            weights = np.exp(-distances_squared / (2.0 * sigma * sigma))
            weights[~valid_mask] = 0.0
        else:
            weights = np.empty((points.shape[0], 0), dtype=np.float64)
        normalization = np.sum(weights, axis=1)

    return TetraGaussianKernel(
        neighbor_indices=neighbor_indices,
        valid_mask=valid_mask,
        weights=weights,
        normalization=normalization,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )


def smooth_tetra_node_fields(
    field_values: NDArray[np.generic],
    kernel: TetraGaussianKernel,
    *,
    isolated_threshold: float = 1.0e-6,
) -> NDArray[np.float64]:
    """将一个可复用核应用于 ``[node,...]`` 标量场"""

    values = np.asarray(field_values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != kernel.neighbor_indices.shape[0]:
        raise TetraSmoothingError("field_values 第一维必须与节点数一致")
    if not np.all(np.isfinite(values)):
        raise TetraSmoothingError("field_values 必须为有限数值")
    if not np.isfinite(isolated_threshold) or isolated_threshold < 0.0:
        raise TetraSmoothingError("isolated_threshold 必须为非负有限数值")

    if kernel.neighbor_indices.shape[1] == 0:
        return values.copy()
    gathered = values[kernel.neighbor_indices]
    expanded_weights = kernel.weights.reshape(
        kernel.weights.shape + (1,) * (values.ndim - 1)
    )
    weighted_sum = np.sum(gathered * expanded_weights, axis=1)
    normalization = kernel.normalization.reshape(
        (kernel.normalization.size,) + (1,) * (values.ndim - 1)
    )
    result = np.divide(
        weighted_sum,
        normalization,
        out=np.zeros_like(weighted_sum),
        where=normalization != 0.0,
    )
    isolated = kernel.normalization < isolated_threshold
    result[isolated] = values[isolated]
    return result
