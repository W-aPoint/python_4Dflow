"""Liutex 与压力计算前使用的四面体节点最小二乘梯度"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]


class MeshGradientError(RuntimeError):
    """四面体输入无法生成 MATLAB 风格的最小二乘梯度时抛出"""


@dataclass(frozen=True, slots=True)
class LsqVelocityGradientResult:
    """各节点的邻域、权重、最小二乘矩阵及速度梯度"""

    point_neighbors: tuple[NDArray[np.int64], ...]
    weights: tuple[NDArray[np.float64], ...]
    gradient_matrices: tuple[NDArray[np.float64], ...]
    gradients: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class GradientOperators:
    """将节点标量场映射为 x、y、z 方向导数的稀疏算子"""

    x: csr_matrix
    y: csr_matrix
    z: csr_matrix


def _validate_mesh_inputs(
    nodes_coords: NDArray[np.generic],
    connectivity: NDArray[np.generic],
    velocity: NDArray[np.generic],
    *,
    connectivity_index_base: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    nodes = np.asarray(nodes_coords, dtype=np.float64)
    elements_raw = np.asarray(connectivity)
    values = np.asarray(velocity, dtype=np.float64)

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise MeshGradientError("nodes_coords 必须为 [节点数,3]")
    if elements_raw.ndim != 2 or elements_raw.shape[1] != 4:
        raise MeshGradientError("connectivity 必须为 [四面体数,4]")
    if values.shape != nodes.shape:
        raise MeshGradientError("velocity 必须为 [节点数,3]，并与节点坐标对应")
    if connectivity_index_base not in (0, 1):
        raise MeshGradientError("connectivity_index_base 只能为 0 或 1")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(values)):
        raise MeshGradientError("节点坐标和速度必须为有限数值")
    if not np.issubdtype(elements_raw.dtype, np.integer):
        if not np.all(np.equal(elements_raw, np.floor(elements_raw))):
            raise MeshGradientError("connectivity 必须只包含整数索引")

    elements = np.asarray(elements_raw, dtype=np.int64) - connectivity_index_base
    if elements.size == 0:
        raise MeshGradientError("connectivity 不能为空")
    if np.min(elements) < 0 or np.max(elements) >= nodes.shape[0]:
        raise MeshGradientError("connectivity 含有超出节点范围的索引")
    if np.any(np.apply_along_axis(lambda row: np.unique(row).size, 1, elements) != 4):
        raise MeshGradientError("每个四面体必须包含 4 个不同节点")
    return nodes, elements, values


def _build_point_neighbors(
    number_of_nodes: int,
    connectivity: NDArray[np.int64],
) -> tuple[NDArray[np.int64], ...]:
    attached: list[set[int]] = [set() for _ in range(number_of_nodes)]
    for element in connectivity:
        element_nodes = tuple(int(index) for index in element)
        for node in element_nodes:
            attached[node].update(element_nodes)
            attached[node].discard(node)
    return tuple(np.asarray(sorted(items), dtype=np.int64) for items in attached)


def calculate_lsq_velocity_gradients(
    nodes_coords: NDArray[np.generic],
    connectivity: NDArray[np.generic],
    velocity: NDArray[np.generic],
    *,
    connectivity_index_base: int,
) -> LsqVelocityGradientResult:
    """为三分量场移植 ``calc_gradVector3D_LSQ.m``

    ``gradients[node, component, derivative_axis]`` 保存速度分量
    ``component`` 对 x、y、z 的导数
    """

    nodes, elements, values = _validate_mesh_inputs(
        nodes_coords,
        connectivity,
        velocity,
        connectivity_index_base=connectivity_index_base,
    )
    timer = RuntimeTimer()
    with timer.measure("构建四面体节点邻接关系"):
        point_neighbors = _build_point_neighbors(nodes.shape[0], elements)

    weights: list[NDArray[np.float64]] = []
    gradient_matrices: list[NDArray[np.float64]] = []
    gradients = np.empty((nodes.shape[0], 3, 3), dtype=np.float64)

    with timer.measure("计算节点最小二乘速度梯度"):
        for node_index, neighbors in enumerate(point_neighbors):
            if neighbors.size < 3:
                raise MeshGradientError(
                    f"节点 {node_index + connectivity_index_base} 的邻点少于 3 个，"
                    "无法计算三维梯度"
                )
            offsets = nodes[neighbors] - nodes[node_index]
            radius_squared = np.sum(offsets * offsets, axis=1)
            mean_radius_squared = float(np.mean(radius_squared))
            if mean_radius_squared <= 0.0:
                raise MeshGradientError(
                    f"节点 {node_index + connectivity_index_base} 存在重合邻点"
                )
            weight = mean_radius_squared / (
                radius_squared + 0.1 * mean_radius_squared
            )
            distance_matrix = offsets * weight[:, None]
            normal_matrix = distance_matrix.T @ distance_matrix
            try:
                gradient_matrix = np.linalg.solve(
                    normal_matrix,
                    distance_matrix.T,
                )
            except np.linalg.LinAlgError as exc:
                raise MeshGradientError(
                    f"节点 {node_index + connectivity_index_base} 的邻域几何退化，"
                    "最小二乘矩阵不可逆"
                ) from exc

            weighted_differences = (
                values[neighbors] - values[node_index]
            ) * weight[:, None]
            # MATLAB 的 GradVar 以 U/V/W 为列；Python 对外采用分量优先
            gradients[node_index] = (gradient_matrix @ weighted_differences).T
            weights.append(weight)
            gradient_matrices.append(gradient_matrix)

    return LsqVelocityGradientResult(
        point_neighbors=point_neighbors,
        weights=tuple(weights),
        gradient_matrices=tuple(gradient_matrices),
        gradients=gradients,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )


def build_lsq_gradient_operators(
    point_neighbors: Sequence[NDArray[np.generic]],
    weights: Sequence[NDArray[np.generic]],
    gradient_matrices: Sequence[NDArray[np.generic]],
    *,
    number_of_nodes: int,
) -> GradientOperators:
    """将 ``Get_gradVector3D_LSQ_SparseMatrix.m`` 移植为 SciPy CSR"""

    if number_of_nodes <= 0:
        raise MeshGradientError("number_of_nodes 必须为正整数")
    if not (
        len(point_neighbors)
        == len(weights)
        == len(gradient_matrices)
        == number_of_nodes
    ):
        raise MeshGradientError("邻接、权重和梯度矩阵的节点数量不一致")

    rows: list[int] = []
    columns: list[int] = []
    data = ([], [], [])

    for node_index in range(number_of_nodes):
        neighbors = np.asarray(point_neighbors[node_index], dtype=np.int64)
        weight = np.asarray(weights[node_index], dtype=np.float64)
        matrix = np.asarray(gradient_matrices[node_index], dtype=np.float64)
        if weight.shape != (neighbors.size,) or matrix.shape != (3, neighbors.size):
            raise MeshGradientError(
                f"节点 {node_index} 的邻接、权重或梯度矩阵尺寸不匹配"
            )
        if neighbors.size and (np.min(neighbors) < 0 or np.max(neighbors) >= number_of_nodes):
            raise MeshGradientError(f"节点 {node_index} 含有越界邻点索引")

        coefficients = matrix * weight[None, :]
        rows.append(node_index)
        columns.append(node_index)
        for axis in range(3):
            data[axis].append(float(-np.sum(coefficients[axis])))
        for local_index, neighbor in enumerate(neighbors):
            rows.append(node_index)
            columns.append(int(neighbor))
            for axis in range(3):
                data[axis].append(float(coefficients[axis, local_index]))

    shape = (number_of_nodes, number_of_nodes)
    return GradientOperators(
        x=csr_matrix((data[0], (rows, columns)), shape=shape),
        y=csr_matrix((data[1], (rows, columns)), shape=shape),
        z=csr_matrix((data[2], (rows, columns)), shape=shape),
    )
