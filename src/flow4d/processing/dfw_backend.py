"""FINAL 第 10 部分使用的原生无散度小波后端"""

from __future__ import annotations

import ctypes
import operator
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class DfwBackendError(RuntimeError):
    """原生 DFW 后端无法加载或执行时抛出"""


_Float64Volume = np.ctypeslib.ndpointer(
    dtype=np.float64,
    ndim=3,
    flags=("F_CONTIGUOUS", "ALIGNED"),
)
_ERROR_BUFFER_SIZE = 1024


def _normalise_minimum_size(values: Sequence[int]) -> tuple[int, int, int]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise DfwBackendError("minimum_size 必须是长度为 3 的正整数序列") from exc
    if len(raw_values) != 3:
        raise DfwBackendError("minimum_size 必须包含 3 个值")

    result: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)):
            raise DfwBackendError("minimum_size 必须只包含正整数")
        try:
            integer_value = operator.index(value)
        except TypeError as exc:
            raise DfwBackendError("minimum_size 必须只包含正整数") from exc
        if integer_value <= 0 or integer_value > np.iinfo(np.int32).max:
            raise DfwBackendError("minimum_size 必须只包含 C int 范围内的正整数")
        result.append(integer_value)
    return result[0], result[1], result[2]


def _normalise_resolution(values: Sequence[float]) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise DfwBackendError("voxel_resolution 不接受 complex 输入")
    try:
        resolution = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DfwBackendError("voxel_resolution 必须是长度为 3 的数值序列") from exc
    if resolution.shape != (3,):
        raise DfwBackendError("voxel_resolution 必须包含 3 个值")
    if not np.all(np.isfinite(resolution)) or np.any(resolution <= 0.0):
        raise DfwBackendError("voxel_resolution 必须只包含正有限数值")
    return np.ascontiguousarray(resolution, dtype=np.float64)


def _normalise_volume(
    value: NDArray[np.generic],
    name: str,
) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise DfwBackendError(f"{name} 不接受 complex 输入")
    try:
        volume = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DfwBackendError(f"{name} 必须是数值三维数组") from exc
    if volume.ndim != 3 or volume.size == 0:
        raise DfwBackendError(f"{name} 必须是非空三维数组")
    if not np.all(np.isfinite(volume)):
        raise DfwBackendError(f"{name} 必须只包含有限数值")
    if any(size > np.iinfo(np.int32).max for size in volume.shape):
        raise DfwBackendError(f"{name} 的维度超过 C int 范围")
    return np.array(volume, dtype=np.float64, order="F", copy=True)


class DfwNativeBackend:
    """通过稳定的 C ABI 加载并调用项目内置的 DFW 核心"""

    def __init__(self, library_path: str | Path | None = None) -> None:
        if library_path is None:
            path = Path(__file__).resolve().parents[1] / "native" / "flow4d_dfwavelet.dll"
        else:
            path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise DfwBackendError(f"未找到 DFW native DLL；已检查：{path}")

        try:
            library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise DfwBackendError(f"无法加载 DFW native DLL：{path}；{exc}") from exc
        try:
            function = library.flow4d_dfwavelet_sure_mad_spin_3d
        except AttributeError as exc:
            raise DfwBackendError(
                "DFW native DLL 缺少导出符号 "
                "flow4d_dfwavelet_sure_mad_spin_3d"
            ) from exc

        function.argtypes = [
            _Float64Volume,
            _Float64Volume,
            _Float64Volume,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            _Float64Volume,
            _Float64Volume,
            _Float64Volume,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int

        self.library_path = path
        self._library = library
        self._function = function

    def denoise_sure_mad_spin(
        self,
        vx: NDArray[np.generic],
        vy: NDArray[np.generic],
        vz: NDArray[np.generic],
        *,
        minimum_size: Sequence[int] = (8, 8, 8),
        voxel_resolution: Sequence[float],
        spins: int = 2,
        random_shift: bool = True,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """对单个时相的三分量速度体数据进行去噪"""

        input_vx = _normalise_volume(vx, "vx")
        input_vy = _normalise_volume(vy, "vy")
        input_vz = _normalise_volume(vz, "vz")
        if input_vx.shape != input_vy.shape or input_vx.shape != input_vz.shape:
            raise DfwBackendError("vx、vy、vz 的 shape 必须完全相同")

        minimum = _normalise_minimum_size(minimum_size)
        resolution = _normalise_resolution(voxel_resolution)
        if isinstance(spins, (bool, np.bool_)):
            raise DfwBackendError("spins 必须是正整数")
        try:
            spin_count = operator.index(spins)
        except TypeError as exc:
            raise DfwBackendError("spins 必须是正整数") from exc
        if spin_count <= 0 or spin_count > np.iinfo(np.int32).max:
            raise DfwBackendError("spins 必须是 C int 范围内的正整数")
        if not isinstance(random_shift, (bool, np.bool_)):
            raise DfwBackendError("random_shift 必须为 bool")

        minimum_native = (ctypes.c_int * 3)(*minimum)
        resolution_native = (ctypes.c_double * 3)(*resolution)
        output_vx = np.empty(input_vx.shape, dtype=np.float64, order="F")
        output_vy = np.empty(input_vx.shape, dtype=np.float64, order="F")
        output_vz = np.empty(input_vx.shape, dtype=np.float64, order="F")
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)

        status = self._function(
            input_vx,
            input_vy,
            input_vz,
            input_vx.shape[0],
            input_vx.shape[1],
            input_vx.shape[2],
            minimum_native,
            resolution_native,
            spin_count,
            int(random_shift),
            output_vx,
            output_vy,
            output_vz,
            error_buffer,
            len(error_buffer),
        )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace").strip()
            if not detail:
                detail = "native backend 未返回错误文字"
            raise DfwBackendError(
                "flow4d_dfwavelet_sure_mad_spin_3d "
                f"失败（status={status}）：{detail}"
            )
        return output_vx, output_vy, output_vz
