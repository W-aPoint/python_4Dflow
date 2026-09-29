"""检测并替换与图像角点连通的 Philips 相位图填充值"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

try:
    from ..base_function import RuntimeTimer, TimingRecord
    from ..dicom.case_loader import LoadedDicomCase
except ImportError:
    if __package__ != "processing":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]
    from dicom.case_loader import LoadedDicomCase  # type: ignore[no-redef]


class PhaseBoundaryError(RuntimeError):
    """相位边界校正输入无效时抛出"""


@dataclass(frozen=True, slots=True)
class PhaseBoundaryCorrectionResult:
    """校正后的病例数据及被替换边界体素的记录"""

    corrected_case: LoadedDicomCase
    edge_ring_mask: NDArray[np.bool_]
    replacement_pixel_value: int
    corrected_voxels_per_component: int
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float

    @property
    def inspection_message(self) -> str:
        timing_text = "\n".join(
            f"- {item.step_name}：{item.elapsed_seconds:.3f} 秒"
            for item in self.timings
        )
        return (
            "相位图边界归零完成\n"
            f"二维边缘像素数：{int(self.edge_ring_mask.sum())}\n"
            f"每个方向修正体素数：{self.corrected_voxels_per_component}\n"
            f"替换后的原始像素值：{self.replacement_pixel_value}\n"
            f"{timing_text}\n"
            f"- 总耗时：{self.total_elapsed_seconds:.3f} 秒"
        )


def _matlab_round(values: NDArray[np.float64] | float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(array) + 0.5), array)


def get_edge_ring_mask(
    reference_phase: NDArray[np.generic],
    venc: float,
) -> NDArray[np.bool_]:
    """使用半径为 1 的圆盘邻域移植 ``GetEdgeRingmask.m``"""

    image = np.asarray(reference_phase)
    if image.ndim != 2:
        raise ValueError(
            "边界识别需要二维参考相位图，"
            f"实际形状为 {image.shape}"
        )
    if image.size == 0:
        raise ValueError("边界识别不能使用空图像")
    if not np.isfinite(float(venc)) or float(venc) < 0:
        raise ValueError("VENC 必须是有限且非负的数值")

    candidate = np.abs(_matlab_round(image.astype(np.float64, copy=False))) == float(
        venc
    )
    rows, columns = candidate.shape
    corners = ((0, 0), (0, columns - 1), (rows - 1, 0), (rows - 1, columns - 1))
    edge_ring = np.zeros_like(candidate, dtype=np.bool_)
    pending: deque[tuple[int, int]] = deque()

    for row, column in corners:
        if candidate[row, column] and not edge_ring[row, column]:
            edge_ring[row, column] = True
            pending.append((row, column))

    neighbours = (
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
    )
    while pending:
        row, column = pending.popleft()
        for row_delta, column_delta in neighbours:
            next_row = row + row_delta
            next_column = column + column_delta
            if not (0 <= next_row < rows and 0 <= next_column < columns):
                continue
            if candidate[next_row, next_column] and not edge_ring[
                next_row, next_column
            ]:
                edge_ring[next_row, next_column] = True
                pending.append((next_row, next_column))

    return edge_ring


def apply_phase_boundary_correction(
    case: LoadedDicomCase,
) -> PhaseBoundaryCorrectionResult:
    """在保留输入病例的同时，对 AP、FH 和 RL 执行 FINAL 第 03 部分"""

    shapes = {case.ap.shape, case.fh.shape, case.rl.shape}
    if len(shapes) != 1 or case.ap.ndim != 4:
        raise PhaseBoundaryError(
            "AP、FH、RL 必须是形状一致的四维数组，才能执行边界归零"
        )
    if not np.isfinite(case.rescale_slope) or case.rescale_slope == 0:
        raise PhaseBoundaryError("RescaleSlope 必须是有限非零数值")

    timer = RuntimeTimer()
    with timer.measure("识别相位图边缘"):
        reference_phase = (
            case.ap[:, :, 0, 0].astype(np.float64) * case.rescale_slope
            + case.rescale_intercept
        )
        edge_ring = get_edge_ring_mask(reference_phase, case.venc)

    replacement = int(
        np.clip(
            _matlab_round(-case.rescale_intercept / case.rescale_slope).item(),
            0,
            np.iinfo(np.uint16).max,
        )
    )
    with timer.measure("替换 AP、FH、RL 边缘像素"):
        expanded_mask = np.broadcast_to(edge_ring[:, :, None, None], case.ap.shape)
        corrected_arrays: list[NDArray[np.generic]] = []
        for source in (case.ap, case.fh, case.rl):
            corrected = source.copy()
            corrected[expanded_mask] = replacement
            corrected_arrays.append(corrected)

    corrected_case = replace(
        case,
        ap=corrected_arrays[0],
        fh=corrected_arrays[1],
        rl=corrected_arrays[2],
    )
    return PhaseBoundaryCorrectionResult(
        corrected_case=corrected_case,
        edge_ring_mask=edge_ring,
        replacement_pixel_value=replacement,
        corrected_voxels_per_component=int(expanded_mask.sum()),
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
