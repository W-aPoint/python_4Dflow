"""将 iso2mesh ``meshconn.m`` 移植为从 0 开始的 NumPy 连接关系"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_node_connectivity(
    elements: NDArray[np.integer],
    node_count: int,
) -> tuple[NDArray[np.int64], ...]:
    """返回每个网格节点排序并去重后的邻接节点"""

    table = np.asarray(elements)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("elements 必须是至少包含两个节点列的二维数组")
    if node_count <= 0:
        raise ValueError("node_count 必须为正整数")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("elements 必须使用整数节点索引")
    if table.size and (int(table.min()) < 0 or int(table.max()) >= node_count):
        raise ValueError("elements 含有超出节点范围的索引")

    neighbours: list[set[int]] = [set() for _ in range(node_count)]
    for element in table:
        indices = [int(value) for value in element]
        for node in indices:
            neighbours[node].update(indices)
            neighbours[node].discard(node)
    return tuple(
        np.asarray(sorted(values), dtype=np.int64) for values in neighbours
    )
