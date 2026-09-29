"""将第 09 部分的掩膜准备与显式网格后端串联"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

try:
    from ..base_function import RuntimeTimer, TimingRecord
except ImportError:
    if __package__ != "mesh":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]

from .backend import (
    SurfaceExtractionBackend,
    SurfaceMesh,
    TetrahedralizationBackend,
    TetrahedralMesh,
)
from .coordinates import scale_tetrahedral_nodes
from .iso2mesh_formats import merge_surface_meshes
from .mask_preparation import MaskPreparationResult, prepare_mask_for_meshing
from .surface_smoothing import smooth_surface


class MeshPipelineError(RuntimeError):
    """所选网格后端返回不兼容数据时抛出"""


@dataclass(frozen=True, slots=True)
class MeshPipelineResult:
    """速度平滑和节点采样前的第 09 部分输出"""

    preparation: MaskPreparationResult
    surface: SurfaceMesh
    tetrahedral: TetrahedralMesh
    node_coordinates_mm: NDArray[np.float64]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _validate_surface(surface: SurfaceMesh) -> None:
    nodes = np.asarray(surface.nodes)
    faces = np.asarray(surface.faces)
    if np.iscomplexobj(nodes) or np.iscomplexobj(faces):
        raise MeshPipelineError("网格后端返回的表面网格不能包含 complex 数值")
    if nodes.ndim != 2 or nodes.shape[1] < 3 or len(nodes) == 0:
        raise MeshPipelineError("网格后端返回的表面节点不是 N×3")
    if not np.all(np.isfinite(nodes[:, :3])):
        raise MeshPipelineError("网格后端返回的表面节点含有 NaN 或 Inf")
    if faces.ndim != 2 or faces.shape[1] < 3 or len(faces) == 0:
        raise MeshPipelineError("网格后端返回的表面连接不是 M×3")
    if not np.issubdtype(faces.dtype, np.integer):
        raise MeshPipelineError("网格后端返回的表面连接必须为整数")
    face_nodes = faces[:, :3]
    if int(face_nodes.min()) < 0 or int(face_nodes.max()) >= len(nodes):
        raise MeshPipelineError("表面连接含有超出节点范围的索引")
    if surface.face_markers is not None:
        markers = np.asarray(surface.face_markers)
        if (
            markers.ndim != 1
            or len(markers) != len(faces)
            or not np.issubdtype(markers.dtype, np.integer)
        ):
            raise MeshPipelineError("表面 face_markers 与三角面不兼容")


def _validate_tetrahedra(mesh: TetrahedralMesh) -> None:
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    if np.iscomplexobj(nodes) or np.iscomplexobj(elements):
        raise MeshPipelineError("网格后端返回的四面体网格不能包含 complex 数值")
    if nodes.ndim != 2 or nodes.shape[1] < 3 or len(nodes) == 0:
        raise MeshPipelineError("网格后端返回的四面体节点不是 N×3")
    if not np.all(np.isfinite(nodes[:, :3])):
        raise MeshPipelineError("四面体节点含有 NaN 或 Inf")
    if elements.ndim != 2 or elements.shape[1] < 4 or len(elements) == 0:
        raise MeshPipelineError("网格后端返回的四面体连接不是 M×4")
    if not np.issubdtype(elements.dtype, np.integer):
        raise MeshPipelineError("四面体连接必须为整数")
    tetrahedra = elements[:, :4]
    if int(tetrahedra.min()) < 0 or int(tetrahedra.max()) >= len(nodes):
        raise MeshPipelineError("四面体连接含有超出节点范围的索引")
    a, b, c, d = (tetrahedra[:, index] for index in range(4))
    six_volume = np.einsum(
        "ij,ij->i",
        np.cross(nodes[b, :3] - nodes[a, :3], nodes[c, :3] - nodes[a, :3]),
        nodes[d, :3] - nodes[a, :3],
    )
    if np.any(six_volume == 0.0):
        raise MeshPipelineError("四面体网格含有零体积单元")

    if mesh.element_attributes is not None:
        attributes = np.asarray(mesh.element_attributes)
        if (
            attributes.ndim != 2
            or len(attributes) != len(elements)
            or not np.issubdtype(attributes.dtype, np.integer)
        ):
            raise MeshPipelineError("element_attributes 与四面体数量不一致")
    if mesh.boundary_faces is not None:
        boundary_faces = np.asarray(mesh.boundary_faces)
        if (
            boundary_faces.ndim != 2
            or boundary_faces.shape[1] != 3
            or not np.issubdtype(boundary_faces.dtype, np.integer)
        ):
            raise MeshPipelineError("boundary_faces 必须为整数 F×3 数组")
        if boundary_faces.size and (
            int(boundary_faces.min()) < 0
            or int(boundary_faces.max()) >= len(nodes)
        ):
            raise MeshPipelineError("boundary_faces 含有超出节点范围的索引")
        if mesh.boundary_markers is not None:
            boundary_markers = np.asarray(mesh.boundary_markers)
            if (
                boundary_markers.ndim != 1
                or len(boundary_markers) != len(boundary_faces)
                or not np.issubdtype(boundary_markers.dtype, np.integer)
            ):
                raise MeshPipelineError(
                    "boundary_markers 与 boundary_faces 数量不一致"
                )
    elif mesh.boundary_markers is not None:
        raise MeshPipelineError("存在 boundary_markers 但缺少 boundary_faces")


def run_mesh_pipeline(
    mask: NDArray[np.generic],
    velocity: NDArray[np.generic],
    voxel_size_mm: tuple[float, float, float],
    backend: SurfaceExtractionBackend,
    *,
    tetrahedralization_backend: TetrahedralizationBackend | None = None,
    radius_bound: float = 1.2,
    tetra_keep_ratio: float = 1.0,
    tetra_max_volume: float = 1.0,
) -> MeshPipelineResult:
    """移植 ``run_iso2mesh.m`` 的编排流程，同时保留 CGAL 边界可见"""

    if not np.isfinite(radius_bound) or radius_bound <= 0:
        raise MeshPipelineError("radius_bound 必须为正有限数值")
    timer = RuntimeTimer()
    with timer.measure("裁剪并加密 Mask"):
        preparation = prepare_mask_for_meshing(mask, velocity)
    with timer.measure("体数据生成表面网格"):
        surface = backend.volume_to_surface(
            preparation.refined_mask.astype(np.float64),
            iso_value=0.5,
            radius_bound=2.0 * radius_bound,
        )
        _validate_surface(surface)
    with timer.measure("表面网格 Laplacian-HC 平滑"):
        smoothed_nodes = smooth_surface(
            surface.nodes[:, :3],
            surface.faces[:, :3],
            iterations=10,
            alpha=0.3,
            method="laplacianhc",
            beta=0.3,
        )
        smoothed_surface = merge_surface_meshes(
            SurfaceMesh(smoothed_nodes, surface.faces[:, :3])
        )
    tetra_backend = tetrahedralization_backend
    if tetra_backend is None:
        surface_to_tetrahedra = getattr(backend, "surface_to_tetrahedra", None)
        if not callable(surface_to_tetrahedra):
            raise MeshPipelineError(
                "未提供 tetrahedralization_backend，且表面 backend 不支持 "
                "surface_to_tetrahedra()"
            )
        tetra_backend = cast(TetrahedralizationBackend, backend)
    with timer.measure("表面网格生成四面体网格"):
        try:
            tetrahedral = tetra_backend.surface_to_tetrahedra(
                smoothed_surface,
                keep_ratio=tetra_keep_ratio,
                max_volume=tetra_max_volume,
                regions=None,
                holes=None,
            )
            _validate_tetrahedra(tetrahedral)
        except Exception as exc:
            raise MeshPipelineError(
                f"表面网格生成四面体网格失败：{exc}"
            ) from exc
    with timer.measure("网格节点坐标缩放"):
        coordinates = scale_tetrahedral_nodes(
            tetrahedral.nodes, voxel_size_mm, refinement_scale=0.5
        )
    return MeshPipelineResult(
        preparation=preparation,
        surface=smoothed_surface,
        tetrahedral=tetrahedral,
        node_coordinates_mm=coordinates,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
