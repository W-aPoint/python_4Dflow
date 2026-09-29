"""相位预处理、速度转换与背景场校正"""

from .msac import (
    MsacCorrectionError,
    MsacCorrectionResult,
    MsacResult,
    apply_msac_correction,
    msac_correct_4d,
)
from .laplacian import (
    LaplacianError,
    build_laplacian_kernel_3d,
    build_laplacian_kernel_4d,
    laplacian_fft_3d,
    laplacian_fft_4d,
)
from .phase_unwrap import PhaseUnwrapError, unwrap_phase_3d, unwrap_phase_4d
from .discontinuity import (
    DiscontinuityError,
    compute_discontinuity_function_third_order,
    eig3matrix_batch,
)
from .phase_boundary import (
    PhaseBoundaryCorrectionResult,
    PhaseBoundaryError,
    apply_phase_boundary_correction,
    get_edge_ring_mask,
)
from .polynomial import (
    PolynomialSystem,
    PolynomialSystemError,
    build_polynomial_system,
)
from .velocity import (
    ClassicDicomFileGrid,
    VelocityConversionError,
    VelocityConversionResult,
    convert_case_to_velocity,
)
from .segmentation import (
    SegmentationExportResult,
    SegmentationPreparationError,
    SegmentationVolumes,
    compute_segmentation_volumes,
    generate_segmentation_dicoms,
)
from .unwrap_pipeline import (
    UnwrapPipelineError,
    UnwrapPipelineResult,
    apply_velocity_component_flips,
    run_unwrap_pipeline,
)
from .velocity_smoothing import (
    GaussianVelocityResult,
    VelocitySmoothingError,
    smooth_velocity_gaussian,
)
from .dfw_backend import DfwBackendError, DfwNativeBackend
from .dfw_pipeline import (
    DfwPipelineError,
    DfwTimeSeriesResult,
    denoise_velocity_dfw_timeseries,
)
from .mesh_gradients import (
    GradientOperators,
    LsqVelocityGradientResult,
    MeshGradientError,
    build_lsq_gradient_operators,
    calculate_lsq_velocity_gradients,
)
from .liutex import LiutexError, LiutexResult, compute_liutex_pagewise
from .tetra_smoothing import (
    TetraGaussianKernel,
    TetraSmoothingError,
    build_tetra_one_ring_neighbors,
    prepare_tetra_gaussian_kernel,
    smooth_tetra_node_fields,
)
from .pressure import (
    PressureReconstructionError,
    PressureResult,
    SpatialDerivativeResult,
    TemporalDerivativeResult,
    assemble_tetra_state,
    build_pressure_laplacian,
    compute_periodic_temporal_derivatives,
    compute_pressure_tetra_sparse,
    compute_spatial_velocity_derivatives,
)
from .liutex_pipeline import (
    LiutexPipelineError,
    LiutexTimeSeriesResult,
    compute_liutex_timeseries,
    compute_vorticity_from_velocity_gradients,
)
from .final_result import FinalResultAssemblyError, assemble_final_node_results

__all__ = [
    "MsacCorrectionError",
    "MsacCorrectionResult",
    "MsacResult",
    "apply_msac_correction",
    "msac_correct_4d",
    "LaplacianError",
    "build_laplacian_kernel_3d",
    "build_laplacian_kernel_4d",
    "laplacian_fft_3d",
    "laplacian_fft_4d",
    "PhaseUnwrapError",
    "unwrap_phase_3d",
    "unwrap_phase_4d",
    "DiscontinuityError",
    "compute_discontinuity_function_third_order",
    "eig3matrix_batch",
    "PhaseBoundaryCorrectionResult",
    "PhaseBoundaryError",
    "apply_phase_boundary_correction",
    "get_edge_ring_mask",
    "PolynomialSystem",
    "PolynomialSystemError",
    "build_polynomial_system",
    "ClassicDicomFileGrid",
    "VelocityConversionError",
    "VelocityConversionResult",
    "convert_case_to_velocity",
    "SegmentationExportResult",
    "SegmentationPreparationError",
    "SegmentationVolumes",
    "compute_segmentation_volumes",
    "generate_segmentation_dicoms",
    "UnwrapPipelineError",
    "UnwrapPipelineResult",
    "apply_velocity_component_flips",
    "run_unwrap_pipeline",
    "GaussianVelocityResult",
    "VelocitySmoothingError",
    "smooth_velocity_gaussian",
    "DfwBackendError",
    "DfwNativeBackend",
    "DfwPipelineError",
    "DfwTimeSeriesResult",
    "denoise_velocity_dfw_timeseries",
    "GradientOperators",
    "LsqVelocityGradientResult",
    "MeshGradientError",
    "build_lsq_gradient_operators",
    "calculate_lsq_velocity_gradients",
    "LiutexError",
    "LiutexResult",
    "compute_liutex_pagewise",
    "TetraGaussianKernel",
    "TetraSmoothingError",
    "build_tetra_one_ring_neighbors",
    "prepare_tetra_gaussian_kernel",
    "smooth_tetra_node_fields",
    "PressureReconstructionError",
    "PressureResult",
    "SpatialDerivativeResult",
    "TemporalDerivativeResult",
    "assemble_tetra_state",
    "build_pressure_laplacian",
    "compute_periodic_temporal_derivatives",
    "compute_pressure_tetra_sparse",
    "compute_spatial_velocity_derivatives",
    "LiutexPipelineError",
    "LiutexTimeSeriesResult",
    "compute_liutex_timeseries",
    "compute_vorticity_from_velocity_gradients",
    "FinalResultAssemblyError",
    "assemble_final_node_results",
]
