"""组装 FINAL 的 27 列多时相节点结果张量"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .liutex_pipeline import LiutexTimeSeriesResult


class FinalResultAssemblyError(RuntimeError):
    """FINAL 结果分量的形状不兼容时抛出"""


def assemble_final_node_results(
    coordinates: NDArray[np.generic],
    velocity: NDArray[np.generic],
    first_derivatives: NDArray[np.generic],
    liutex: LiutexTimeSeriesResult,
    pressure: NDArray[np.generic],
) -> NDArray[np.float64]:
    """按 FINAL 中 ``Nodedata``、Liutex 和压力的顺序拼接结果"""

    velocity_values = np.asarray(velocity, dtype=np.float64)
    gradients = np.asarray(first_derivatives, dtype=np.float64)
    pressure_values = np.asarray(pressure, dtype=np.float64)
    if velocity_values.ndim != 3 or velocity_values.shape[1] != 3:
        raise FinalResultAssemblyError("velocity 必须为 [节点数,3,时相数]")
    node_count, _, phase_count = velocity_values.shape
    if gradients.shape != (node_count, 9, phase_count):
        raise FinalResultAssemblyError("first_derivatives 必须为 [节点数,9,时相数]")
    if pressure_values.shape == (node_count, phase_count):
        pressure_values = pressure_values[:, None, :]
    if pressure_values.shape != (node_count, 1, phase_count):
        raise FinalResultAssemblyError("pressure 必须为 [节点数,1,时相数]")

    coordinate_values = np.asarray(coordinates, dtype=np.float64)
    if coordinate_values.shape == (node_count, 3):
        coordinate_values = np.repeat(
            coordinate_values[:, :, None],
            phase_count,
            axis=2,
        )
    if coordinate_values.shape != (node_count, 3, phase_count):
        raise FinalResultAssemblyError(
            "coordinates 必须为 [节点数,3] 或 [节点数,3,时相数]"
        )

    expected_vector = (node_count, 3, phase_count)
    expected_scalar = (node_count, 1, phase_count)
    if liutex.vorticity.shape != expected_vector:
        raise FinalResultAssemblyError("liutex.vorticity 尺寸不一致")
    if liutex.direction.shape != expected_vector:
        raise FinalResultAssemblyError("liutex.direction 尺寸不一致")
    for name, values in (
        ("magnitude", liutex.magnitude),
        ("omega_r", liutex.omega_r),
        ("omega_numerator", liutex.omega_numerator),
        ("omega_denominator", liutex.omega_denominator),
        ("lambda_ci", liutex.lambda_ci),
    ):
        if values.shape != expected_scalar:
            raise FinalResultAssemblyError(f"liutex.{name} 尺寸不一致")

    components = (
        coordinate_values,
        velocity_values,
        gradients,
        liutex.vorticity,
        liutex.direction,
        liutex.magnitude,
        liutex.omega_r,
        liutex.omega_numerator,
        liutex.omega_denominator,
        liutex.lambda_ci,
        pressure_values,
    )
    if any(not np.all(np.isfinite(values)) for values in components):
        raise FinalResultAssemblyError("FINAL 结果组件必须全部为有限实数")
    result = np.concatenate(components, axis=1)
    if result.shape != (node_count, 27, phase_count):
        raise FinalResultAssemblyError("FINAL 结果组装后不是预期的 27 列")
    return result
