"""与 MATLAB R2021a 兼容的实数域内三维 Makima 插值

该内部生产内核实现了经验证的实数闭域子集，
对应 ``interp3(..., 'makima')`` 的当前生产范围
数据采用 MATLAB 的 ``F[y_index, x_index, z_index]`` 布局，
查询数组遵循 NumPy 广播规则；网格必须为有限值、
严格递增，且每个网格至少包含 7 个节点
当前明确不支持外推和复数输入

实现取自已冻结的实验验证器，
并保留经验证的深层内部直接差商路径，
以避免不必要地改变浮点运算顺序
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Makima3DError(ValueError):
    """输入超出已验证的生产约束时抛出"""


@dataclass(frozen=True, slots=True)
class _NodeDofs3D:
    f: float
    fx: float
    fy: float
    fz: float
    fxy: float
    fxz: float
    fyz: float
    fxyz: float


@dataclass(frozen=True, slots=True)
class _NodeDofFields:
    """每个网格节点连续存储的 Hermite 自由度"""

    f: NDArray[np.float64]
    fx: NDArray[np.float64]
    fy: NDArray[np.float64]
    fz: NDArray[np.float64]
    fxy: NDArray[np.float64]
    fxz: NDArray[np.float64]
    fyz: NDArray[np.float64]
    fxyz: NDArray[np.float64]


def _as_real_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise Makima3DError(f"{name} must be real-valued")
    return np.asarray(array, dtype=np.float64)


def _as_strict_grid(values: ArrayLike, name: str) -> NDArray[np.float64]:
    grid = _as_real_array(values, name)
    if grid.ndim != 1 or grid.size < 7:
        raise Makima3DError(
            f"{name} must be a 1-D grid with at least 7 nodes"
        )
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise Makima3DError(f"{name} must be finite and strictly increasing")
    return grid


def _cell_dxy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    field: NDArray[np.float64],
    row: int,
    column: int,
) -> float:
    hx = x[column + 1] - x[column]
    hy = y[row + 1] - y[row]
    return float(
        (
            field[row + 1, column + 1]
            - field[row + 1, column]
            - field[row, column + 1]
            + field[row, column]
        )
        / (hx * hy)
    )


def _makima_node_blend(
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
    index: int,
) -> tuple[float, float]:
    slopes = np.diff(values) / np.diff(coordinates)
    dm2 = slopes[index - 2]
    dm1 = slopes[index - 1]
    d0 = slopes[index]
    dp1 = slopes[index + 1]
    w1 = abs(dp1 - d0) + 0.5 * abs(dp1 + d0)
    w2 = abs(dm1 - dm2) + 0.5 * abs(dm1 + dm2)
    denominator = w1 + w2
    if denominator == 0.0:
        return 0.0, 0.0
    return float(w1 / denominator), float(w2 / denominator)


def _extended_secants(slopes: NDArray[np.float64]) -> NDArray[np.float64]:
    left_m1 = 2.0 * slopes[0] - slopes[1]
    left_m2 = 2.0 * left_m1 - slopes[0]
    right_n = 2.0 * slopes[-1] - slopes[-2]
    right_n1 = 2.0 * right_n - slopes[-1]
    return np.concatenate(
        ([left_m2, left_m1], slopes, [right_n, right_n1])
    )


def _makima_node_blend_with_zero_fallback(
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
    index: int,
) -> tuple[float, float]:
    slopes = np.diff(values) / np.diff(coordinates)
    if index < 2 or index > coordinates.size - 3:
        extended = _extended_secants(slopes)
        dm2, dm1, d0, dp1 = extended[index : index + 4]
    else:
        dm2 = slopes[index - 2]
        dm1 = slopes[index - 1]
        d0 = slopes[index]
        dp1 = slopes[index + 1]
    w1 = abs(dp1 - d0) + 0.5 * abs(dp1 + d0)
    w2 = abs(dm1 - dm2) + 0.5 * abs(dm1 + dm2)
    if w1 + w2 == 0.0:
        return 0.0, 0.0
    if index < 2 or index > coordinates.size - 3:
        denominator = w1 + w2
        return float(w1 / denominator), float(w2 / denominator)
    return _makima_node_blend(coordinates, values, index)


def _makima_node_slope(
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
    index: int,
) -> float:
    if index < 0 or index >= coordinates.size:
        raise Makima3DError(f"node index {index} is outside the grid")
    left, right = _makima_node_blend_with_zero_fallback(
        coordinates, values, index
    )
    slopes = np.diff(values) / np.diff(coordinates)
    if index < 2 or index > coordinates.size - 3:
        slopes = _extended_secants(slopes)
        return float(left * slopes[index + 1] + right * slopes[index + 2])
    return float(left * slopes[index - 1] + right * slopes[index])


def _extend_cell_axis(
    values: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    first = np.take(values, 0, axis=axis)
    second = np.take(values, 1, axis=axis)
    penultimate = np.take(values, -2, axis=axis)
    last = np.take(values, -1, axis=axis)
    return np.concatenate(
        (
            np.expand_dims(3.0 * first - 2.0 * second, axis),
            np.expand_dims(2.0 * first - second, axis),
            values,
            np.expand_dims(2.0 * last - penultimate, axis),
            np.expand_dims(3.0 * last - 2.0 * penultimate, axis),
        ),
        axis=axis,
    )


def _plane_mixed_derivative(
    first_grid: NDArray[np.float64],
    second_grid: NDArray[np.float64],
    plane: NDArray[np.float64],
    second_index: int,
    first_index: int,
) -> float:
    first_left, first_right = _makima_node_blend_with_zero_fallback(
        first_grid, plane[second_index, :], first_index
    )
    second_left, second_right = _makima_node_blend_with_zero_fallback(
        second_grid, plane[:, first_index], second_index
    )

    first_interior = 2 <= first_index <= first_grid.size - 3
    second_interior = 2 <= second_index <= second_grid.size - 3
    if first_interior and second_interior:
        lower_left = _cell_dxy(
            first_grid, second_grid, plane, second_index - 1, first_index - 1
        )
        lower_right = _cell_dxy(
            first_grid, second_grid, plane, second_index - 1, first_index
        )
        upper_left = _cell_dxy(
            first_grid, second_grid, plane, second_index, first_index - 1
        )
        upper_right = _cell_dxy(
            first_grid, second_grid, plane, second_index, first_index
        )
    else:
        cell_dxy = np.diff(np.diff(plane, axis=0), axis=1) / (
            np.diff(second_grid)[:, None] * np.diff(first_grid)[None, :]
        )
        cell_dxy = _extend_cell_axis(cell_dxy, axis=1)
        cell_dxy = _extend_cell_axis(cell_dxy, axis=0)
        lower_left = float(cell_dxy[second_index + 1, first_index + 1])
        lower_right = float(cell_dxy[second_index + 1, first_index + 2])
        upper_left = float(cell_dxy[second_index + 2, first_index + 1])
        upper_right = float(cell_dxy[second_index + 2, first_index + 2])
    lower = first_left * lower_left + first_right * lower_right
    upper = first_left * upper_left + first_right * upper_right
    return float(second_left * lower + second_right * upper)


def _cell_dxyz(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    field: NDArray[np.float64],
    ix: int,
    iy: int,
    iz: int,
) -> float:
    block = field[iy : iy + 2, ix : ix + 2, iz : iz + 2]
    numerator = np.diff(np.diff(np.diff(block, axis=0), axis=1), axis=2).item()
    denominator = (x[ix + 1] - x[ix]) * (y[iy + 1] - y[iy]) * (
        z[iz + 1] - z[iz]
    )
    return float(numerator / denominator)


def _fxyz_tensor_dxyz(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    field: NDArray[np.float64],
    ix: int,
    iy: int,
    iz: int,
) -> float:
    wx = _makima_node_blend_with_zero_fallback(x, field[iy, :, iz], ix)
    wy = _makima_node_blend_with_zero_fallback(y, field[:, ix, iz], iy)
    wz = _makima_node_blend_with_zero_fallback(z, field[iy, ix, :], iz)

    interior = (
        2 <= ix <= x.size - 3
        and 2 <= iy <= y.size - 3
        and 2 <= iz <= z.size - 3
    )
    result = 0.0
    if interior:
        for side_z in range(2):
            for side_y in range(2):
                for side_x in range(2):
                    result += (
                        wz[side_z]
                        * wy[side_y]
                        * wx[side_x]
                        * _cell_dxyz(
                            x,
                            y,
                            z,
                            field,
                            ix - 1 + side_x,
                            iy - 1 + side_y,
                            iz - 1 + side_z,
                        )
                    )
        return float(result)

    cell_dxyz = np.diff(
        np.diff(np.diff(field, axis=0), axis=1), axis=2
    ) / (
        np.diff(y)[:, None, None]
        * np.diff(x)[None, :, None]
        * np.diff(z)[None, None, :]
    )
    cell_dxyz = _extend_cell_axis(cell_dxyz, axis=0)
    cell_dxyz = _extend_cell_axis(cell_dxyz, axis=1)
    cell_dxyz = _extend_cell_axis(cell_dxyz, axis=2)
    for side_z in range(2):
        for side_y in range(2):
            for side_x in range(2):
                result += (
                    wz[side_z]
                    * wy[side_y]
                    * wx[side_x]
                    * cell_dxyz[
                        iy + 1 + side_y,
                        ix + 1 + side_x,
                        iz + 1 + side_z,
                    ]
                )
    return float(result)


def _node_dofs(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    field: NDArray[np.float64],
    ix: int,
    iy: int,
    iz: int,
) -> _NodeDofs3D:
    return _NodeDofs3D(
        f=float(field[iy, ix, iz]),
        fx=_makima_node_slope(x, field[iy, :, iz], ix),
        fy=_makima_node_slope(y, field[:, ix, iz], iy),
        fz=_makima_node_slope(z, field[iy, ix, :], iz),
        fxy=_plane_mixed_derivative(x, y, field[:, :, iz], iy, ix),
        fxz=_plane_mixed_derivative(x, z, field[iy, :, :].T, iz, ix),
        fyz=_plane_mixed_derivative(y, z, field[:, ix, :].T, iz, iy),
        fxyz=_fxyz_tensor_dxyz(x, y, z, field, ix, iy, iz),
    )


def _precompute_node_dof_fields(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    field: NDArray[np.float64],
) -> _NodeDofFields:
    """按冻结的标量语义计算连续节点自由度数组"""

    fxy = _mixed_derivative_planes(
        x, y, np.transpose(field, (2, 0, 1))
    ).transpose(1, 2, 0)
    fxz = _mixed_derivative_planes(
        x, z, np.transpose(field, (0, 2, 1))
    ).transpose(0, 2, 1)
    fyz = _mixed_derivative_planes(
        y, z, np.transpose(field, (1, 2, 0))
    ).transpose(2, 0, 1)
    fxyz = _triple_mixed_derivative_field(x, y, z, field)
    return _NodeDofFields(
        f=np.array(field, dtype=np.float64, copy=True),
        fx=_makima_node_slopes_axis(x, field, axis=1),
        fy=_makima_node_slopes_axis(y, field, axis=0),
        fz=_makima_node_slopes_axis(z, field, axis=2),
        fxy=fxy,
        fxz=fxz,
        fyz=fyz,
        fxyz=fxyz,
    )


def _makima_blend_weights_axis(
    coordinates: NDArray[np.float64],
    field: NDArray[np.float64],
    axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """返回冻结的 Makima 左右权重及沿指定轴扩展的割线"""

    moved = np.moveaxis(field, axis, -1)
    slopes = np.diff(moved, axis=-1) / np.diff(coordinates)
    left_m1 = 2.0 * slopes[..., 0] - slopes[..., 1]
    left_m2 = 2.0 * left_m1 - slopes[..., 0]
    right_n = 2.0 * slopes[..., -1] - slopes[..., -2]
    right_n1 = 2.0 * right_n - slopes[..., -1]
    extended = np.concatenate(
        (
            left_m2[..., None],
            left_m1[..., None],
            slopes,
            right_n[..., None],
            right_n1[..., None],
        ),
        axis=-1,
    )
    size = coordinates.size
    dm2 = extended[..., 0:size]
    dm1 = extended[..., 1 : size + 1]
    d0 = extended[..., 2 : size + 2]
    dp1 = extended[..., 3 : size + 3]
    w1 = np.abs(dp1 - d0) + 0.5 * np.abs(dp1 + d0)
    w2 = np.abs(dm1 - dm2) + 0.5 * np.abs(dm1 + dm2)
    denominator = w1 + w2
    left = np.zeros_like(denominator)
    right = np.zeros_like(denominator)
    nonzero = denominator != 0.0
    left[nonzero] = w1[nonzero] / denominator[nonzero]
    right[nonzero] = w2[nonzero] / denominator[nonzero]
    return (
        np.moveaxis(left, -1, axis),
        np.moveaxis(right, -1, axis),
        extended,
    )


def _makima_node_slopes_axis(
    coordinates: NDArray[np.float64],
    field: NDArray[np.float64],
    axis: int,
) -> NDArray[np.float64]:
    """沿一个场轴批量计算冻结的节点斜率"""

    left, right, extended = _makima_blend_weights_axis(
        coordinates, field, axis
    )
    moved_left = np.moveaxis(left, axis, -1)
    moved_right = np.moveaxis(right, axis, -1)
    size = coordinates.size
    result = (
        moved_left * extended[..., 1 : size + 1]
        + moved_right * extended[..., 2 : size + 2]
    )
    return np.moveaxis(result, -1, axis)


def _mixed_derivative_planes(
    first_grid: NDArray[np.float64],
    second_grid: NDArray[np.float64],
    planes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """批量计算 ``[plane,second,first]`` 的冻结平面混合导数"""

    first_left, first_right, _ = _makima_blend_weights_axis(
        first_grid, planes, axis=2
    )
    second_left, second_right, _ = _makima_blend_weights_axis(
        second_grid, planes, axis=1
    )
    denominator = (
        np.diff(second_grid)[None, :, None]
        * np.diff(first_grid)[None, None, :]
    )

    direct_cells = (
        planes[:, 1:, 1:]
        - planes[:, 1:, :-1]
        - planes[:, :-1, 1:]
        + planes[:, :-1, :-1]
    ) / denominator
    boundary_cells = np.diff(
        np.diff(planes, axis=1), axis=2
    ) / denominator
    extended = _extend_cell_axis(boundary_cells, axis=2)
    extended = _extend_cell_axis(extended, axis=1)

    second_size = second_grid.size
    first_size = first_grid.size
    lower_left = extended[
        :, 1 : second_size + 1, 1 : first_size + 1
    ]
    lower_right = extended[
        :, 1 : second_size + 1, 2 : first_size + 2
    ]
    upper_left = extended[
        :, 2 : second_size + 2, 1 : first_size + 1
    ]
    upper_right = extended[
        :, 2 : second_size + 2, 2 : first_size + 2
    ]
    lower = first_left * lower_left + first_right * lower_right
    upper = first_left * upper_left + first_right * upper_right
    result = second_left * lower + second_right * upper

    second_nodes = slice(2, second_size - 2)
    first_nodes = slice(2, first_size - 2)
    lower_second_cells = slice(1, second_size - 3)
    upper_second_cells = slice(2, second_size - 2)
    left_first_cells = slice(1, first_size - 3)
    right_first_cells = slice(2, first_size - 2)
    interior_first_left = first_left[:, second_nodes, first_nodes]
    interior_first_right = first_right[:, second_nodes, first_nodes]
    interior_second_left = second_left[:, second_nodes, first_nodes]
    interior_second_right = second_right[:, second_nodes, first_nodes]
    interior_lower = (
        interior_first_left
        * direct_cells[:, lower_second_cells, left_first_cells]
        + interior_first_right
        * direct_cells[:, lower_second_cells, right_first_cells]
    )
    interior_upper = (
        interior_first_left
        * direct_cells[:, upper_second_cells, left_first_cells]
        + interior_first_right
        * direct_cells[:, upper_second_cells, right_first_cells]
    )
    result[:, second_nodes, first_nodes] = (
        interior_second_left * interior_lower
        + interior_second_right * interior_upper
    )
    return result


def _triple_mixed_derivative_field(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    field: NDArray[np.float64],
) -> NDArray[np.float64]:
    """在各网格节点批量计算冻结的张量加权单元 ``Dxyz``"""

    wx = _makima_blend_weights_axis(x, field, axis=1)[:2]
    wy = _makima_blend_weights_axis(y, field, axis=0)[:2]
    wz = _makima_blend_weights_axis(z, field, axis=2)[:2]
    cell_dxyz = np.diff(
        np.diff(np.diff(field, axis=0), axis=1), axis=2
    ) / (
        np.diff(y)[:, None, None]
        * np.diff(x)[None, :, None]
        * np.diff(z)[None, None, :]
    )
    extended = _extend_cell_axis(cell_dxyz, axis=0)
    extended = _extend_cell_axis(extended, axis=1)
    extended = _extend_cell_axis(extended, axis=2)
    ny, nx, nz = field.shape
    result = np.zeros(field.shape, dtype=np.float64)
    for side_z in range(2):
        for side_y in range(2):
            for side_x in range(2):
                result += (
                    wz[side_z]
                    * wy[side_y]
                    * wx[side_x]
                    * extended[
                        1 + side_y : 1 + side_y + ny,
                        1 + side_x : 1 + side_x + nx,
                        1 + side_z : 1 + side_z + nz,
                    ]
                )

    node_y = slice(2, ny - 2)
    node_x = slice(2, nx - 2)
    node_z = slice(2, nz - 2)
    interior = np.zeros((ny - 4, nx - 4, nz - 4), dtype=np.float64)
    for side_z in range(2):
        for side_y in range(2):
            for side_x in range(2):
                interior += (
                    wz[side_z][node_y, node_x, node_z]
                    * wy[side_y][node_y, node_x, node_z]
                    * wx[side_x][node_y, node_x, node_z]
                    * cell_dxyz[
                        1 + side_y : ny - 3 + side_y,
                        1 + side_x : nx - 3 + side_x,
                        1 + side_z : nz - 3 + side_z,
                    ]
                )
    result[node_y, node_x, node_z] = interior
    return result


def _half_grid_axis_data(
    grid: NDArray[np.float64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    tuple[NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    """为半网格查询构建冻结的单元索引、宽度和 Hermite 基函数"""

    query = np.empty(2 * grid.size - 1, dtype=np.float64)
    query[::2] = grid
    query[1::2] = 0.5 * (grid[:-1] + grid[1:])
    cells = np.searchsorted(grid, query, side="right") - 1
    cells[query == grid[-1]] = grid.size - 2
    widths = grid[cells + 1] - grid[cells]
    t = (query - grid[cells]) / widths
    t2 = t * t
    t3 = t2 * t
    values = (
        2.0 * t3 - 3.0 * t2 + 1.0,
        -2.0 * t3 + 3.0 * t2,
    )
    derivatives = (t3 - 2.0 * t2 + t, t3 - t2)
    return cells.astype(np.int64, copy=False), widths, values, derivatives


def _interp3_makima_half_grid(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    field: ArrayLike,
    *,
    block_depth: int = 1,
) -> NDArray[np.float64]:
    """按 z 分块计算内部规则半网格掩膜模式"""

    x_grid = _as_strict_grid(x, "x")
    y_grid = _as_strict_grid(y, "y")
    z_grid = _as_strict_grid(z, "z")
    values = _as_real_array(field, "field")
    expected_shape = (y_grid.size, x_grid.size, z_grid.size)
    if values.shape != expected_shape:
        raise Makima3DError(
            "field must have MATLAB layout (len(y),len(x),len(z))="
            f"{expected_shape}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise Makima3DError("field must contain only finite values")
    if isinstance(block_depth, bool) or not isinstance(block_depth, int):
        raise Makima3DError("block_depth must be a positive integer")
    if block_depth <= 0:
        raise Makima3DError("block_depth must be a positive integer")

    dofs = _precompute_node_dof_fields(
        x_grid, y_grid, z_grid, values
    )
    cell_x, width_x, hu, dhu = _half_grid_axis_data(x_grid)
    cell_y, width_y, hv, dhv = _half_grid_axis_data(y_grid)
    cell_z, width_z, hw, dhw = _half_grid_axis_data(z_grid)
    output = np.empty(
        (cell_y.size, cell_x.size, cell_z.size), dtype=np.float64
    )

    index_y = cell_y[:, None, None]
    index_x = cell_x[None, :, None]
    vx = tuple(item[None, :, None] for item in hu)
    dx = tuple(
        (dhu[side] * width_x)[None, :, None] for side in range(2)
    )
    vy = tuple(item[:, None, None] for item in hv)
    dy = tuple(
        (dhv[side] * width_y)[:, None, None] for side in range(2)
    )

    for start in range(0, cell_z.size, block_depth):
        stop = min(start + block_depth, cell_z.size)
        index_z = cell_z[None, None, start:stop]
        vz = tuple(item[None, None, start:stop] for item in hw)
        dz = tuple(
            (dhw[side][start:stop] * width_z[start:stop])[None, None, :]
            for side in range(2)
        )
        block = np.zeros(
            (cell_y.size, cell_x.size, stop - start), dtype=np.float64
        )
        for side_z in range(2):
            for side_y in range(2):
                for side_x in range(2):
                    node_index = (
                        index_y + side_y,
                        index_x + side_x,
                        index_z + side_z,
                    )
                    block += (
                        vx[side_x]
                        * vy[side_y]
                        * vz[side_z]
                        * dofs.f[node_index]
                    )
                    block += (
                        dx[side_x]
                        * vy[side_y]
                        * vz[side_z]
                        * dofs.fx[node_index]
                    )
                    block += (
                        vx[side_x]
                        * dy[side_y]
                        * vz[side_z]
                        * dofs.fy[node_index]
                    )
                    block += (
                        vx[side_x]
                        * vy[side_y]
                        * dz[side_z]
                        * dofs.fz[node_index]
                    )
                    block += (
                        dx[side_x]
                        * dy[side_y]
                        * vz[side_z]
                        * dofs.fxy[node_index]
                    )
                    block += (
                        dx[side_x]
                        * vy[side_y]
                        * dz[side_z]
                        * dofs.fxz[node_index]
                    )
                    block += (
                        vx[side_x]
                        * dy[side_y]
                        * dz[side_z]
                        * dofs.fyz[node_index]
                    )
                    block += (
                        dx[side_x]
                        * dy[side_y]
                        * dz[side_z]
                        * dofs.fxyz[node_index]
                    )
        output[:, :, start:stop] = block

    output[::2, ::2, ::2] = values
    return output


def _hermite_basis(
    t: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    t2 = t * t
    t3 = t2 * t
    values = (2.0 * t3 - 3.0 * t2 + 1.0, -2.0 * t3 + 3.0 * t2)
    derivatives = (t3 - 2.0 * t2 + t, t3 - t2)
    return values, derivatives


def _evaluate_cell(
    corners: tuple[
        tuple[
            tuple[_NodeDofs3D, _NodeDofs3D],
            tuple[_NodeDofs3D, _NodeDofs3D],
        ],
        tuple[
            tuple[_NodeDofs3D, _NodeDofs3D],
            tuple[_NodeDofs3D, _NodeDofs3D],
        ],
    ],
    u: float,
    v: float,
    w: float,
    hx: float,
    hy: float,
    hz: float,
) -> float:
    hu, dhu = _hermite_basis(u)
    hv, dhv = _hermite_basis(v)
    hw, dhw = _hermite_basis(w)
    result = 0.0
    for side_z in range(2):
        for side_y in range(2):
            for side_x in range(2):
                dof = corners[side_z][side_y][side_x]
                vx, dx = hu[side_x], dhu[side_x] * hx
                vy, dy = hv[side_y], dhv[side_y] * hy
                vz, dz = hw[side_z], dhw[side_z] * hz
                result += vx * vy * vz * dof.f
                result += dx * vy * vz * dof.fx
                result += vx * dy * vz * dof.fy
                result += vx * vy * dz * dof.fz
                result += dx * dy * vz * dof.fxy
                result += dx * vy * dz * dof.fxz
                result += vx * dy * dz * dof.fyz
                result += dx * dy * dz * dof.fxyz
    return float(result)


def interp3_makima(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    field: ArrayLike,
    xq: ArrayLike,
    yq: ArrayLike,
    zq: ArrayLike,
) -> float | NDArray[np.float64]:
    """计算与 MATLAB R2021a 兼容的实数域内三维 Makima 插值

    ``field`` 必须采用 ``(len(y), len(x), len(z))`` 布局
    查询数组按 NumPy 规则共同广播，
    并且必须位于三个闭合网格区间内
    网格与数据必须为有限实数
    复数输入和外推会被明确拒绝
    """

    x_grid = _as_strict_grid(x, "x")
    y_grid = _as_strict_grid(y, "y")
    z_grid = _as_strict_grid(z, "z")
    values = _as_real_array(field, "field")
    expected_shape = (y_grid.size, x_grid.size, z_grid.size)
    if values.shape != expected_shape:
        raise Makima3DError(
            "field must have MATLAB layout (len(y),len(x),len(z))="
            f"{expected_shape}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise Makima3DError("field must contain only finite values")

    query_x, query_y, query_z = np.broadcast_arrays(
        _as_real_array(xq, "xq"),
        _as_real_array(yq, "yq"),
        _as_real_array(zq, "zq"),
    )
    if not (
        np.all(np.isfinite(query_x))
        and np.all(np.isfinite(query_y))
        and np.all(np.isfinite(query_z))
    ):
        raise Makima3DError("query coordinates must be finite")
    flat_x = query_x.ravel()
    flat_y = query_y.ravel()
    flat_z = query_z.ravel()
    cell_x = np.searchsorted(x_grid, flat_x, side="right") - 1
    cell_y = np.searchsorted(y_grid, flat_y, side="right") - 1
    cell_z = np.searchsorted(z_grid, flat_z, side="right") - 1
    cell_x[flat_x == x_grid[-1]] = x_grid.size - 2
    cell_y[flat_y == y_grid[-1]] = y_grid.size - 2
    cell_z[flat_z == z_grid[-1]] = z_grid.size - 2
    in_domain = (
        (flat_x >= x_grid[0])
        & (flat_x <= x_grid[-1])
        & (flat_y >= y_grid[0])
        & (flat_y <= y_grid[-1])
        & (flat_z >= z_grid[0])
        & (flat_z <= z_grid[-1])
    )
    actual_cells = (
        (cell_x >= 0)
        & (cell_x < x_grid.size - 1)
        & (cell_y >= 0)
        & (cell_y < y_grid.size - 1)
        & (cell_z >= 0)
        & (cell_z < z_grid.size - 1)
    )
    valid = in_domain & actual_cells
    if not np.all(valid):
        first = int(np.flatnonzero(~valid)[0])
        raise Makima3DError(
            "query requires extrapolation outside the validated closed-domain "
            "contract: "
            f"xq={flat_x[first]!r}, yq={flat_y[first]!r}, "
            f"zq={flat_z[first]!r}"
        )

    output = np.empty(flat_x.size, dtype=np.float64)
    cell_cache: dict[tuple[int, int, int], object] = {}
    for query_index, (qx_value, qy_value, qz_value, ix, iy, iz) in enumerate(
        zip(flat_x, flat_y, flat_z, cell_x, cell_y, cell_z, strict=True)
    ):
        key = (int(ix), int(iy), int(iz))
        corners = cell_cache.get(key)
        if corners is None:
            corners = tuple(
                tuple(
                    tuple(
                        _node_dofs(
                            x_grid,
                            y_grid,
                            z_grid,
                            values,
                            key[0] + side_x,
                            key[1] + side_y,
                            key[2] + side_z,
                        )
                        for side_x in range(2)
                    )
                    for side_y in range(2)
                )
                for side_z in range(2)
            )
            cell_cache[key] = corners
        hx = float(x_grid[key[0] + 1] - x_grid[key[0]])
        hy = float(y_grid[key[1] + 1] - y_grid[key[1]])
        hz = float(z_grid[key[2] + 1] - z_grid[key[2]])
        u = float((qx_value - x_grid[key[0]]) / hx)
        v = float((qy_value - y_grid[key[1]]) / hy)
        w = float((qz_value - z_grid[key[2]]) / hz)
        output[query_index] = _evaluate_cell(
            corners, u, v, w, hx, hy, hz  # type: ignore[arg-type]
        )

    reshaped = output.reshape(query_x.shape)
    if reshaped.ndim == 0:
        return float(reshaped)
    return reshaped


__all__ = ["Makima3DError", "interp3_makima"]
