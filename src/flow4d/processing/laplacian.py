"""由 ``lap3.m`` 和 ``lap4.m`` 移植的 FFT Laplacian 基础运算"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fftn, ifftn


class LaplacianError(RuntimeError):
    """FFT Laplacian 的输入或核无效时抛出"""


@dataclass(frozen=True, slots=True)
class _PreparedLaplacianKernel:
    """Validated kernel in the native FFT frequency order, scoped to one run."""

    native: NDArray[np.generic]


def _prepare_laplacian_kernel(kernel: NDArray[np.generic]) -> _PreparedLaplacianKernel:
    values = np.asarray(kernel)
    if not np.all(np.isfinite(values)):
        raise LaplacianError("Laplacian 频域核包含 NaN/Inf")
    native = np.fft.ifftshift(values)
    native.setflags(write=False)
    return _PreparedLaplacianKernel(native)


def _laplacian_fft_prepared(
    values: NDArray[np.generic],
    direction: int,
    kernel: _PreparedLaplacianKernel,
    real_output: bool,
) -> NDArray[np.generic]:
    # The unwrap pipeline always uses double precision, independently of the
    # FFT library's version-dependent single-precision behavior.
    array = np.asarray(
        values, dtype=np.complex128 if np.iscomplexobj(values) else np.float64
    )
    if array.shape != kernel.native.shape:
        raise LaplacianError("输入形状与 Laplacian 核不一致")
    if not np.all(np.isfinite(array)):
        raise LaplacianError("Laplacian 输入包含 NaN/Inf")
    if direction not in (1, -1):
        raise ValueError("direction 只能为 1（正向）或 -1（逆向）")

    # A Fourier multiplier commutes with circular spatial shifts. Moving only
    # the kernel to native order cancels the original input/output shifts,
    # including odd dimensions; retain the exact legacy kernel coefficients.
    spectrum = fftn(array, workers=1)
    if direction == 1:
        np.multiply(spectrum, kernel.native, out=spectrum)
    else:
        inverse_kernel = kernel.native.astype(np.float64, copy=True)
        inverse_kernel[(0,) * array.ndim] = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(spectrum, inverse_kernel, out=spectrum)
        del inverse_kernel
    output = ifftn(spectrum, workers=1)
    if real_output:
        return np.array(output.real, copy=True)
    return np.asarray(output)


def _validated_shape(shape: Sequence[int], dimensions: int) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in shape)
    if len(normalized) != dimensions or any(value <= 0 for value in normalized):
        raise ValueError(f"形状必须包含 {dimensions} 个正整数")
    return normalized


def build_laplacian_kernel_3d(shape: Sequence[int]) -> NDArray[np.float32]:
    """构建 MATLAB ``unwrap_3D`` 使用的频域核"""

    sx, sy, sz = _validated_shape(shape, 3)
    axes = tuple(
        np.arange(-size / 2.0, size / 2.0, dtype=np.float32)
        for size in (sx, sy, sz)
    )
    x = axes[0][:, None, None]
    y = axes[1][None, :, None]
    z = axes[2][None, None, :]
    kernel = (
        2.0 * np.cos(np.pi * x / sx)
        + 2.0 * np.cos(np.pi * y / sy)
        + 2.0 * np.cos(np.pi * z / sz)
        - 6.0
    )
    return np.asarray(kernel, dtype=np.float32)


def build_laplacian_kernel_4d(
    shape: Sequence[int],
    temporal_scale: float = 2.0,
) -> NDArray[np.float32]:
    """构建 MATLAB ``unwrap_4D`` 使用的四维频域核"""

    sx, sy, sz, st = _validated_shape(shape, 4)
    scale = float(temporal_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("temporal_scale 必须是有限正数")
    axes = tuple(
        np.arange(-size / 2.0, size / 2.0, dtype=np.float32)
        for size in (sx, sy, sz, st)
    )
    x = axes[0][:, None, None, None]
    y = axes[1][None, :, None, None]
    z = axes[2][None, None, :, None]
    t = axes[3][None, None, None, :]
    kernel = (
        2.0 * np.cos(np.pi * x / sx)
        + 2.0 * np.cos(np.pi * y / sy)
        + 2.0 * np.cos(np.pi * z / sz)
        + scale * np.cos(np.pi * t / st)
        - 6.0
        - scale
    )
    return np.asarray(kernel, dtype=np.float32)


def _laplacian_fft(
    values: NDArray[np.generic],
    direction: int,
    kernel: NDArray[np.generic],
    real_output: bool,
) -> NDArray[np.generic]:
    array = np.asarray(values)
    kernel_array = np.asarray(kernel)
    if array.shape != kernel_array.shape:
        raise LaplacianError(
            f"输入形状 {array.shape} 与 Laplacian 核 {kernel_array.shape} 不一致"
        )
    if not np.all(np.isfinite(array)) or not np.all(np.isfinite(kernel_array)):
        raise LaplacianError("Laplacian 输入或频域核包含 NaN/Inf")
    if direction not in (1, -1):
        raise ValueError("direction 只能为 1（正向）或 -1（逆向）")

    spectrum = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(array)))
    if direction == 1:
        np.multiply(spectrum, kernel_array, out=spectrum)
    else:
        inverse_kernel = kernel_array.astype(np.float64, copy=True)
        center = tuple(size // 2 for size in array.shape)
        inverse_kernel[center] = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(spectrum, inverse_kernel, out=spectrum)

    output = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(spectrum)))
    if real_output:
        return np.array(output.real, copy=True)
    return np.asarray(output)


def laplacian_fft_3d(
    values: NDArray[np.generic],
    direction: int,
    kernel: NDArray[np.generic],
    real_output: bool = False,
) -> NDArray[np.generic]:
    """执行由 ``lap3.m`` 移植的变换"""

    if np.asarray(values).ndim != 3:
        raise LaplacianError("laplacian_fft_3d 需要三维输入")
    return _laplacian_fft(values, direction, kernel, real_output)


def laplacian_fft_4d(
    values: NDArray[np.generic],
    direction: int,
    kernel: NDArray[np.generic],
    real_output: bool = False,
) -> NDArray[np.generic]:
    """执行由 ``lap4.m`` 移植的变换"""

    if np.asarray(values).ndim != 4:
        raise LaplacianError("laplacian_fft_4d 需要四维输入")
    # Preserve the existing public NumPy behavior for other input dtypes.
    # The production unwrap path converts phase to float64 before arriving here.
    if np.asarray(values).dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        return _laplacian_fft(values, direction, kernel, real_output)
    return _laplacian_fft_prepared(
        values, direction, _prepare_laplacian_kernel(kernel), real_output
    )
