"""移植 iso2mesh 的 ``sms.m`` 和 ``smoothsurf.m``"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .connectivity import build_node_connectivity


class SurfaceSmoothingError(RuntimeError):
    """三角表面平滑输入无效时抛出"""


def _neighbour_means(
    values: NDArray[np.float64],
    connectivity: tuple[NDArray[np.int64], ...],
) -> NDArray[np.float64]:
    output = values.copy()
    for index, neighbours in enumerate(connectivity):
        if neighbours.size:
            output[index] = values[neighbours].mean(axis=0)
    return output


def smooth_surface(
    nodes: NDArray[np.generic],
    faces: NDArray[np.integer],
    *,
    iterations: int = 10,
    alpha: float = 0.3,
    method: str = "laplacianhc",
    beta: float | None = None,
) -> NDArray[np.float64]:
    """使用与 MATLAB 兼容的具名方法平滑三角表面"""

    original = np.asarray(nodes, dtype=np.float64)
    face_array = np.asarray(faces)
    if original.ndim != 2 or original.shape[1] < 3:
        raise SurfaceSmoothingError("nodes 必须是 N×3 数组")
    if face_array.ndim != 2 or face_array.shape[1] < 3:
        raise SurfaceSmoothingError("faces 必须是 M×3 三角形连接数组")
    if iterations < 0 or not 0 <= alpha <= 1:
        raise SurfaceSmoothingError("iterations 和 alpha 参数无效")
    selected_method = method.lower()
    if selected_method not in {"laplacian", "laplacianhc", "lowpass"}:
        raise SurfaceSmoothingError(f"不支持的表面平滑方法：{method}")
    selected_beta = alpha if beta is None else float(beta)
    connectivity = build_node_connectivity(face_array[:, :3], len(original))
    movable = np.asarray([item.size > 0 for item in connectivity])
    p = original[:, :3].copy()

    if selected_method == "laplacian":
        for _ in range(iterations):
            means = _neighbour_means(p, connectivity)
            p[movable] = (1.0 - alpha) * p[movable] + alpha * means[movable]
        return p

    if selected_method == "laplacianhc":
        for _ in range(iterations):
            q = p.copy()
            means = _neighbour_means(q, connectivity)
            p[movable] = means[movable]
            b = p - (alpha * original[:, :3] + (1.0 - alpha) * q)
            b_means = _neighbour_means(b, connectivity)
            p[movable] -= (
                selected_beta * b[movable]
                + (1.0 - selected_beta) * b_means[movable]
            )
        return p

    lowpass_beta = -1.02 * alpha
    for _ in range(iterations):
        means = _neighbour_means(p, connectivity)
        p[movable] = (1.0 - alpha) * p[movable] + alpha * means[movable]
        means = _neighbour_means(p, connectivity)
        p[movable] = (1.0 - lowpass_beta) * p[movable] + lowpass_beta * means[movable]
    return p
