"""MATLAB iso2mesh/CGAL 操作的显式后端边界"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SurfaceMesh:
    """体数据转表面后端返回的三角表面网格"""

    nodes: NDArray[np.float64]
    faces: NDArray[np.int64]
    face_markers: NDArray[np.int64] | None = None


@dataclass(frozen=True, slots=True)
class TetrahedralMesh:
    """采用 Python 从 0 开始连接索引的四节点四面体网格"""

    nodes: NDArray[np.float64]
    elements: NDArray[np.int64]
    element_attributes: NDArray[np.int64] | None = None
    boundary_faces: NDArray[np.int64] | None = None
    boundary_markers: NDArray[np.int64] | None = None


class SurfaceExtractionBackend(Protocol):
    """替代 ``v2s(...,'cgalmesh')`` 阶段的接口"""

    def volume_to_surface(
        self,
        mask: NDArray[np.float64],
        *,
        iso_value: float,
        radius_bound: float,
    ) -> SurfaceMesh: ...


class TetrahedralizationBackend(Protocol):
    """替代 ``s2m`` 表面转体网格阶段的接口"""

    def surface_to_tetrahedra(
        self,
        surface: SurfaceMesh,
        *,
        keep_ratio: float,
        max_volume: float,
        regions: NDArray[np.float64] | None = None,
        holes: NDArray[np.float64] | None = None,
    ) -> TetrahedralMesh: ...


class MeshBackend(
    SurfaceExtractionBackend,
    TetrahedralizationBackend,
    Protocol,
):
    """替代 ``v2s(...,'cgalmesh')`` 和 ``s2m`` 的组合后端

    在该协议接入真实病例前，必须选择具体后端并完成数值验证
    保持边界显式，可以避免把不同的 Python 网格器未经验证地视为
    已复现 iso2mesh/CGAL
    """
