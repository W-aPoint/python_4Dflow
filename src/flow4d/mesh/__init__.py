"""FINAL 第 09–11 部分中从掩膜到四面体网格的准备工具"""

from .backend import (
    MeshBackend,
    SurfaceExtractionBackend,
    SurfaceMesh,
    TetrahedralizationBackend,
    TetrahedralMesh,
)
from .cgalmesh_backend import CgalMeshBackendError, CgalMeshSurfaceBackend
from .cgalmesh_formats import CgalMeshFormatError
from .connectivity import build_node_connectivity
from .coordinates import scale_tetrahedral_nodes
from .iso2mesh_formats import (
    Iso2MeshFormatError,
    merge_surface_meshes,
    read_tetgen_output,
    write_tetgen_poly,
)
from .mask_preparation import (
    MaskCropBounds,
    MaskPreparationError,
    MaskPreparationResult,
    crop_mask_and_velocity,
    prepare_mask_for_meshing,
    smooth_binary_volume,
    upsample_mask_makima,
)
from .node_sampling import NodeSamplingError, NodeVelocityResult, sample_velocity_at_nodes
from .pipeline import MeshPipelineError, MeshPipelineResult, run_mesh_pipeline
from .surface_smoothing import SurfaceSmoothingError, smooth_surface
from .tetgen_backend import TetGenBackendError, TetGenTetrahedralizationBackend

__all__ = [
    "MeshBackend",
    "SurfaceExtractionBackend",
    "SurfaceMesh",
    "TetrahedralizationBackend",
    "TetrahedralMesh",
    "CgalMeshBackendError",
    "CgalMeshSurfaceBackend",
    "CgalMeshFormatError",
    "build_node_connectivity",
    "scale_tetrahedral_nodes",
    "Iso2MeshFormatError",
    "merge_surface_meshes",
    "read_tetgen_output",
    "write_tetgen_poly",
    "MaskCropBounds",
    "MaskPreparationError",
    "MaskPreparationResult",
    "crop_mask_and_velocity",
    "prepare_mask_for_meshing",
    "smooth_binary_volume",
    "upsample_mask_makima",
    "NodeSamplingError",
    "NodeVelocityResult",
    "sample_velocity_at_nodes",
    "MeshPipelineError",
    "MeshPipelineResult",
    "run_mesh_pipeline",
    "SurfaceSmoothingError",
    "smooth_surface",
    "TetGenBackendError",
    "TetGenTetrahedralizationBackend",
]
