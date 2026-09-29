"""当前 iso2mesh ``v2s`` 路径使用的外部 CGALMesh 后端"""

from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray

from .backend import SurfaceMesh
from .cgalmesh_formats import (
    CgalMeshFormatError,
    read_medit_mesh,
    write_inr_uint8,
)


class CgalMeshBackendError(RuntimeError):
    """外部 CGALMesh 表面阶段无法完成时抛出"""


def _as_finite_real_scalar(
    value: object,
    *,
    name: str,
    positive: bool,
) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        requirement = "正有限实数" if positive else "有限实数"
        raise CgalMeshBackendError(f"{name} 必须为{requirement}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        requirement = "正有限实数" if positive else "有限实数"
        raise CgalMeshBackendError(f"{name} 必须为{requirement}") from exc
    if not np.isfinite(number) or (positive and number <= 0.0):
        requirement = "正有限实数" if positive else "有限实数"
        raise CgalMeshBackendError(f"{name} 必须为{requirement}")
    return number


def _as_matlab_uint8_volume(mask: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(mask)
    if np.iscomplexobj(array):
        raise CgalMeshBackendError("CGALMesh 输入体数据不能包含 complex 数值")
    if array.ndim != 3 or any(length == 0 for length in array.shape):
        raise CgalMeshBackendError("CGALMesh 输入必须是非空三维数组")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise CgalMeshBackendError("CGALMesh 输入体数据必须是数值或布尔数组")
    if not np.all(np.isfinite(array)):
        raise CgalMeshBackendError("CGALMesh 输入体数据包含 NaN 或 Inf")

    if array.dtype == np.bool_:
        converted = array.astype(np.uint8, copy=True)
    else:
        converted = np.clip(array, 0, 255).astype(np.uint8)
    volume = np.ascontiguousarray(converted, dtype=np.uint8)
    if not np.any(volume):
        raise CgalMeshBackendError("CGALMesh 输入体数据中没有非零标签区域")
    return volume


def _sort_surface_like_iso2mesh(
    nodes: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """复现 ``sortmesh.m`` 的节点与面片处理部分"""

    node_array = np.asarray(nodes, dtype=np.float64)
    face_array = np.asarray(triangles, dtype=np.int64)
    if node_array.ndim != 2 or node_array.shape[1] != 3 or len(node_array) == 0:
        raise CgalMeshBackendError("CGALMesh 节点必须是非空 N×3 数组")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or len(face_array) == 0:
        raise CgalMeshBackendError("CGALMesh 三角面必须是非空 M×3 数组")
    if not np.all(np.isfinite(node_array)):
        raise CgalMeshBackendError("CGALMesh 节点包含 NaN 或 Inf")
    if int(face_array.min()) < 0 or int(face_array.max()) >= len(node_array):
        raise CgalMeshBackendError("CGALMesh 三角面包含越界节点索引")

    offset = node_array - node_array[0]
    theta = np.arctan2(offset[:, 1], offset[:, 0])
    phi = np.arctan2(
        offset[:, 2],
        np.hypot(offset[:, 0], offset[:, 1]),
    )
    radius = np.linalg.norm(offset, axis=1)
    original_index = np.arange(len(node_array), dtype=np.int64)
    node_order = np.lexsort((original_index, theta, phi, radius))

    inverse = np.empty(len(node_array), dtype=np.int64)
    inverse[node_order] = np.arange(len(node_array), dtype=np.int64)
    remapped_faces = np.sort(inverse[face_array], axis=1)
    face_order = np.lexsort(
        (
            remapped_faces[:, 2],
            remapped_faces[:, 1],
            remapped_faces[:, 0],
        )
    )
    sorted_nodes = np.ascontiguousarray(
        node_array[node_order] + 0.5, dtype=np.float64
    )
    sorted_faces = np.ascontiguousarray(
        remapped_faces[face_order], dtype=np.int64
    )
    return sorted_nodes, sorted_faces


def _deduplicate_and_compact_surface(
    nodes: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> SurfaceMesh:
    """复现 ``unique(...,'rows')`` 和 ``removeisolatednode``"""

    unique_triangles = np.unique(np.asarray(triangles, dtype=np.int64), axis=0)
    if len(unique_triangles) == 0:
        raise CgalMeshBackendError("CGALMesh 后处理后没有可用三角面")
    used_nodes = np.unique(unique_triangles.reshape(-1))
    if len(used_nodes) == 0:
        raise CgalMeshBackendError("CGALMesh 后处理后没有可用节点")

    inverse = np.full(len(nodes), -1, dtype=np.int64)
    inverse[used_nodes] = np.arange(len(used_nodes), dtype=np.int64)
    compact_faces = inverse[unique_triangles]
    if np.any(compact_faces < 0):
        raise CgalMeshBackendError("CGALMesh 孤立节点清理后的连接映射无效")
    compact_nodes = np.asarray(nodes, dtype=np.float64)[used_nodes]
    return SurfaceMesh(
        nodes=np.ascontiguousarray(compact_nodes, dtype=np.float64),
        faces=np.ascontiguousarray(compact_faces, dtype=np.int64),
        face_markers=None,
    )


class CgalMeshSurfaceBackend:
    """运行 iso2mesh ``cgalv2m.m`` 使用的外部 CGAL 网格器"""

    def __init__(
        self,
        executable: str | Path,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        candidate = Path(executable).expanduser()
        if not candidate.is_file():
            raise CgalMeshBackendError(
                f"找不到 cgalmesh 可执行文件：{candidate}"
            )
        if timeout_seconds is None:
            timeout = None
        else:
            timeout = _as_finite_real_scalar(
                timeout_seconds,
                name="timeout_seconds",
                positive=True,
            )
        self.executable_path = candidate.resolve()
        self.timeout_seconds = timeout

    def volume_to_surface(
        self,
        mask: NDArray[np.float64],
        *,
        iso_value: float,
        radius_bound: float,
    ) -> SurfaceMesh:
        """创建 ``v2s(..., 'cgalmesh')`` 返回的表面网格"""

        _as_finite_real_scalar(iso_value, name="iso_value", positive=False)
        radius = _as_finite_real_scalar(
            radius_bound, name="radius_bound", positive=True
        )
        volume = _as_matlab_uint8_volume(mask)

        with TemporaryDirectory(prefix="flow4d-cgalmesh-") as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "pre_cgalmesh.inr"
            output_path = temporary_path / "post_cgalmesh.mesh"
            try:
                write_inr_uint8(volume, input_path)
            except CgalMeshFormatError as exc:
                raise CgalMeshBackendError(
                    f"CGALMesh 输入写出失败：{exc}"
                ) from exc

            command = [
                str(self.executable_path),
                str(input_path),
                str(output_path),
                f"{30.0:f}",
                f"{radius:f}",
                f"{0.5:f}",
                f"{3.0:f}",
                f"{1000.0:f}",
                str(0x623F9A9E),
            ]
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                completed = subprocess.run(
                    command,
                    cwd=temporary_path,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                    creationflags=creation_flags,
                )
            except subprocess.TimeoutExpired as exc:
                raise CgalMeshBackendError(
                    "CGALMesh 执行超过限定时间"
                    f"（timeout_seconds={self.timeout_seconds}）"
                ) from exc
            except OSError as exc:
                raise CgalMeshBackendError(
                    f"无法启动 CGALMesh：{self.executable_path}；{exc}"
                ) from exc

            stdout = completed.stdout[-4000:]
            stderr = completed.stderr[-4000:]
            if completed.returncode != 0:
                raise CgalMeshBackendError(
                    "CGALMesh 执行失败"
                    f"（executable={self.executable_path}，"
                    f"returncode={completed.returncode}）\n"
                    f"stdout：{stdout}\n"
                    f"stderr：{stderr}"
                )
            if not output_path.is_file():
                raise CgalMeshBackendError(
                    "CGALMesh 未生成输出文件"
                    f"（executable={self.executable_path}，"
                    f"returncode={completed.returncode}）\n"
                    f"stdout：{stdout}\n"
                    f"stderr：{stderr}"
                )
            try:
                mesh = read_medit_mesh(output_path)
            except CgalMeshFormatError as exc:
                raise CgalMeshBackendError(
                    f"CGALMesh 输出解析失败：{exc}"
                ) from exc

        sorted_nodes, sorted_faces = _sort_surface_like_iso2mesh(
            mesh.nodes, mesh.triangles
        )
        return _deduplicate_and_compact_surface(sorted_nodes, sorted_faces)
