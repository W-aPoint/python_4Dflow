"""基于 Laplacian 的三维和四维相位解混叠圈数估计"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .laplacian import (
    _PreparedLaplacianKernel,
    _laplacian_fft_prepared,
    _prepare_laplacian_kernel,
    build_laplacian_kernel_3d,
    build_laplacian_kernel_4d,
    laplacian_fft_3d,
)


class PhaseUnwrapError(RuntimeError):
    """包裹相位无法处理时抛出"""


def _matlab_int8(values: NDArray[np.generic]) -> NDArray[np.int8]:
    array = np.asarray(values, dtype=np.float64)
    rounded = np.copysign(np.floor(np.abs(array) + 0.5), array)
    return np.clip(rounded, -128, 127).astype(np.int8)


def unwrap_phase_3d(
    wrapped_phase: NDArray[np.generic],
    real_output: bool = True,
) -> NDArray[np.int8]:
    """移植 ``unwrap_3D.m``，返回每个体素的整数圈数"""

    phase = np.asarray(wrapped_phase, dtype=np.float64)
    if phase.ndim != 3 or not np.all(np.isfinite(phase)):
        raise PhaseUnwrapError("wrapped_phase 必须是有限的三维数组")
    kernel = build_laplacian_kernel_3d(phase.shape)
    laplacian_wrapped = laplacian_fft_3d(phase, 1, kernel, real_output)
    laplacian_phase = (
        np.cos(phase)
        * laplacian_fft_3d(np.sin(phase), 1, kernel, real_output)
        - np.sin(phase)
        * laplacian_fft_3d(np.cos(phase), 1, kernel, real_output)
    )
    inverse_difference = laplacian_fft_3d(
        laplacian_phase - laplacian_wrapped,
        -1,
        kernel,
        real_output,
    )
    if np.iscomplexobj(inverse_difference):
        inverse_difference = np.real_if_close(inverse_difference, tol=1000)
        if np.iscomplexobj(inverse_difference):
            raise PhaseUnwrapError("三维解混叠结果包含非数值误差级复数")
    return _matlab_int8(np.asarray(inverse_difference) / (2.0 * np.pi))


def unwrap_phase_4d(
    wrapped_phase: NDArray[np.generic],
    temporal_scale: float = 2.0,
    real_output: bool = True,
) -> NDArray[np.int8]:
    """移植 ``unwrap_4D.m``，返回每个体素的整数圈数"""

    phase = np.asarray(wrapped_phase, dtype=np.float64)
    if phase.ndim != 4 or not np.all(np.isfinite(phase)):
        raise PhaseUnwrapError("wrapped_phase 必须是有限的四维数组")
    kernel = _prepare_laplacian_kernel(
        build_laplacian_kernel_4d(phase.shape, temporal_scale)
    )
    return _unwrap_phase_4d_prepared(phase, kernel, real_output)


def _unwrap_phase_4d_prepared(
    wrapped_phase: NDArray[np.generic],
    kernel: _PreparedLaplacianKernel,
    real_output: bool = True,
) -> NDArray[np.int8]:
    """Internal entry point for the pipeline's shared, prepared kernel."""
    phase = np.asarray(wrapped_phase, dtype=np.float64)
    if phase.ndim != 4 or not np.all(np.isfinite(phase)):
        raise PhaseUnwrapError("wrapped_phase 必须是有限的四维数组")
    laplacian_wrapped = _laplacian_fft_prepared(phase, 1, kernel, real_output)
    laplacian_phase = _laplacian_fft_prepared(np.sin(phase), 1, kernel, real_output)
    np.multiply(laplacian_phase, np.cos(phase), out=laplacian_phase)
    cosine_term = _laplacian_fft_prepared(np.cos(phase), 1, kernel, real_output)
    np.multiply(cosine_term, np.sin(phase), out=cosine_term)
    np.subtract(laplacian_phase, cosine_term, out=laplacian_phase)
    del cosine_term
    np.subtract(laplacian_phase, laplacian_wrapped, out=laplacian_phase)
    del laplacian_wrapped
    inverse_difference = _laplacian_fft_prepared(
        laplacian_phase,
        -1,
        kernel,
        real_output,
    )
    if np.iscomplexobj(inverse_difference):
        inverse_difference = np.real_if_close(inverse_difference, tol=1000)
        if np.iscomplexobj(inverse_difference):
            raise PhaseUnwrapError("四维解混叠结果包含非数值误差级复数")
    return _matlab_int8(np.asarray(inverse_difference) / (2.0 * np.pi))
