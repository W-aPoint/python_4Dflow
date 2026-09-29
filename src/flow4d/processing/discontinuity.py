"""分割流程使用的速度不连续性计算"""

from __future__ import annotations

from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


class DiscontinuityError(RuntimeError):
    """R 函数输入无法处理时抛出"""


def eig3matrix_batch(
    a: NDArray[np.generic],
    b: NDArray[np.generic],
    c: NDArray[np.generic],
    d: NDArray[np.generic],
) -> NDArray[np.generic]:
    """移植 ``eig3matrix_batch.m``，逐元素求解三次方程"""

    coefficients = tuple(np.asarray(item, dtype=np.float64) for item in (a, b, c, d))
    if len({item.shape for item in coefficients}) != 1:
        raise ValueError("三次方程的 a、b、c、d 必须具有相同形状")
    a_f64, b_f64, c_f64, d_f64 = coefficients
    if np.any(a_f64 == 0) or not all(np.all(np.isfinite(x)) for x in coefficients):
        raise ValueError("三次方程系数必须有限，且 a 不能为 0")

    p = (3.0 * a_f64 * c_f64 - b_f64**2) / (3.0 * a_f64**2)
    q = (
        27.0 * a_f64**2 * d_f64
        - 9.0 * a_f64 * b_f64 * c_f64
        + 2.0 * b_f64**3
    ) / (27.0 * a_f64**3)
    determinant = 0.25 * q**2 + (1.0 / 27.0) * p**3
    determinant[np.abs(determinant) < 1e-14] = 0.0

    roots = np.zeros(a_f64.shape + (3,), dtype=np.complex128)
    nonnegative = determinant >= 0
    if np.any(nonnegative):
        sqrt_determinant = np.sqrt(determinant[nonnegative])
        y_plus = -0.5 * q[nonnegative] + sqrt_determinant
        y_minus = -0.5 * q[nonnegative] - sqrt_determinant
        y1 = np.cbrt(y_plus)
        y2 = np.cbrt(y_minus)
        omega = complex(-0.5, np.sqrt(3.0) / 2.0)
        omega_squared = complex(-0.5, -np.sqrt(3.0) / 2.0)
        shift = b_f64[nonnegative] / (3.0 * a_f64[nonnegative])
        roots[nonnegative, 0] = y1 + y2 - shift
        roots[nonnegative, 1] = omega * y1 + omega_squared * y2 - shift
        roots[nonnegative, 2] = omega_squared * y1 + omega * y2 - shift

    negative = determinant < 0
    if np.any(negative):
        radius = np.sqrt(-(p[negative] / 3.0) ** 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine_argument = np.clip(-q[negative] / (2.0 * radius), -1.0, 1.0)
        theta = np.arccos(cosine_argument) / 3.0
        amplitude = 2.0 * np.cbrt(radius)
        shift = b_f64[negative] / (3.0 * a_f64[negative])
        roots[negative, 0] = amplitude * np.cos(theta) - shift
        roots[negative, 1] = amplitude * np.cos(theta + 2.0 * np.pi / 3.0) - shift
        roots[negative, 2] = amplitude * np.cos(theta + 4.0 * np.pi / 3.0) - shift

    roots[np.abs(roots) < 1e-14] = 0.0
    order = np.argsort(roots.real, axis=-1)[..., ::-1]
    sorted_roots = np.take_along_axis(roots, order, axis=-1)
    return np.real_if_close(sorted_roots, tol=1000)


def compute_discontinuity_function_third_order(
    velocity: NDArray[np.generic],
    sigma: float,
) -> NDArray[np.float64]:
    """移植 ``compute_discontinuity_function3order.m``"""

    velocity_f64 = np.asarray(velocity, dtype=np.float64)
    if velocity_f64.ndim != 4 or velocity_f64.shape[-1] != 3:
        raise DiscontinuityError("velocity 必须为 [row,column,slice,3]")
    if not np.all(np.isfinite(velocity_f64)):
        raise DiscontinuityError("velocity 包含 NaN 或 Inf")
    if not np.isfinite(float(sigma)) or float(sigma) <= 0:
        raise DiscontinuityError("sigma 必须是有限正数")

    radius = int(ceil(2.0 * float(sigma)))

    def smooth(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return gaussian_filter(
            values,
            sigma=float(sigma),
            mode="nearest",
            radius=radius,
        )

    vx, vy, vz = (velocity_f64[..., index] for index in range(3))
    mv11 = smooth(vx**2)
    mv12 = smooth(vx * vy)
    mv13 = smooth(vx * vz)
    mv22 = smooth(vy**2)
    mv23 = smooth(vy * vz)
    mv33 = smooth(vz**2)

    a = -np.ones_like(mv11)
    b = mv11 + mv22 + mv33
    c = (
        mv12**2
        - mv11 * mv22
        + mv13**2
        + mv23**2
        - mv11 * mv33
        - mv22 * mv33
    )
    d = (
        -mv13 * mv22 * mv13
        + mv12 * mv23 * mv13
        + mv13 * mv12 * mv23
        - mv11 * mv23 * mv23
        - mv12 * mv12 * mv33
        + mv11 * mv22 * mv33
    )
    eigenvalues = np.asarray(eig3matrix_batch(a, b, c, d))
    if np.iscomplexobj(eigenvalues):
        imaginary_magnitude = np.abs(eigenvalues.imag)
        local_real_scale = np.maximum(
            np.max(np.abs(eigenvalues.real), axis=-1, keepdims=True),
            1.0,
        )
        imaginary_tolerance = 1e-8 * local_real_scale + 1e-12
        if np.any(imaginary_magnitude > imaginary_tolerance):
            raise DiscontinuityError(
                "结构张量特征值出现超过数值误差范围的复数"
            )
        eigenvalues = eigenvalues.real
    lambda1 = eigenvalues[..., 0]
    lambda2 = eigenvalues[..., 1]
    denominator = lambda1 + lambda2
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (4.0 * lambda1 * lambda2) / denominator**2
    result[np.abs(denominator) < 1e-4] = 0.0
    return np.asarray(result, dtype=np.float64)
