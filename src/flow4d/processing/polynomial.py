"""构建 MATLAB MSAC 代码使用的精确多项式设计矩阵"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class PolynomialSystemError(ValueError):
    """多项式输入或阶数无效时抛出"""


@dataclass(frozen=True, slots=True)
class PolynomialSystem:
    """向量取值及其空间多项式设计矩阵"""

    values: NDArray[np.float64]
    design_matrix: NDArray[np.float64]
    term_count: int


def build_polynomial_system(
    order: int,
    points: NDArray[np.generic],
) -> PolynomialSystem:
    """移植 ``getPolynomial.m``，支持 0 至 3 阶"""

    if isinstance(order, bool) or not isinstance(order, int) or order not in range(4):
        raise PolynomialSystemError("多项式阶数必须是 0、1、2 或 3")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6:
        raise PolynomialSystemError(
            "points 必须是 [point,6] 数组，列顺序为 [u,v,w,i,j,k]"
        )
    if not np.all(np.isfinite(array)):
        raise PolynomialSystemError("points 包含 NaN 或 Inf")

    values = array[:, :3].copy()
    p1, p2, p3 = array[:, 3], array[:, 4], array[:, 5]
    columns: list[NDArray[np.float64]] = [np.ones(array.shape[0])]
    if order >= 1:
        columns.extend((p1, p2, p3))
    if order >= 2:
        columns.extend((p1**2, p1 * p2, p1 * p3, p2**2, p2 * p3, p3**2))
    if order >= 3:
        columns.extend(
            (
                p1**3,
                p1**2 * p2,
                p1**2 * p3,
                p2**3,
                p2**2 * p1,
                p2**2 * p3,
                p3**3,
                p3**2 * p1,
                p3**2 * p2,
                p1 * p2 * p3,
            )
        )
    design_matrix = np.column_stack(columns).astype(np.float64, copy=False)
    return PolynomialSystem(
        values=values,
        design_matrix=design_matrix,
        term_count=design_matrix.shape[1],
    )
