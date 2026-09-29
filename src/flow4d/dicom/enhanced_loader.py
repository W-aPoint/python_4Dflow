"""把 Philips Enhanced MR 帧转换为项目使用的经典四维布局

本模块对应 ``baseFunction/GetDCMfromEnhancedDCM.m``输入为已经由
pydicom 解码、按 ``[frame, row, column]`` 排列的多帧数组，输出为
``[row, column, slice, cardiac_phase]`` 数组
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


SliceAxis: TypeAlias = Literal[0, 1, 2]

_PHILIPS_CARDIAC_PHASES_TAG = Tag(0x2001, 0x1017)
_PHILIPS_FRAME_DETAILS_TAG = Tag(0x2005, 0x140F)


class EnhancedDicomError(RuntimeError):
    """Enhanced DICOM 输入无法安全转换时抛出"""


@dataclass(frozen=True, slots=True)
class ClassicMetadataTemplate:
    """由 Enhanced DICOM 帧生成的向量化经典样式元数据"""

    sop_class_uid: str
    media_storage_sop_class_uid: str
    transfer_syntax_uid: str
    series_instance_uid: str
    study_instance_uid: str
    media_storage_sop_instance_uid: str
    series_description: str
    series_number: object
    pixel_spacing: tuple[float, float]
    image_orientation_patient: tuple[float, float, float, float, float, float]
    slice_thickness: float
    spacing_between_slices: float
    slice_axis: SliceAxis
    image_positions_patient: NDArray[np.float64]
    slice_locations: NDArray[np.float64]
    instance_numbers: NDArray[np.int64]
    rows: int
    columns: int
    bits_allocated: int
    bits_stored: int
    high_bit: int
    pixel_representation: int
    samples_per_pixel: int
    rescale_intercept: float
    rescale_slope: float
    patient_name: str | None
    patient_id: str | None
    study_date: str | None
    modality: str | None


@dataclass(frozen=True, slots=True)
class Enhanced4DData:
    """Enhanced DICOM 参数及四个与项目兼容的四维数组"""

    reference_image: NDArray[np.generic]
    manufacturer: str
    classic_metadata: ClassicMetadataTemplate
    phase_dicom_info: Dataset
    magnitude_dicom_info: Dataset
    series: str
    rescale_intercept: float
    rescale_slope: float
    venc: float
    cardiac_phases: int
    voxel_size_mm: tuple[float, float, float]
    heart_rate_bpm: float
    time_spacing_ms: float
    magnitude: NDArray[np.generic]
    ap: NDArray[np.generic]
    fh: NDArray[np.generic]
    rl: NDArray[np.generic]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """返回共享的行、列、切片、时相形状"""

        return self.fh.shape

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """返回从 FH 数组继承的公共输出数据类型"""

        return self.fh.dtype

    @property
    def inspection_message(self) -> str:
        """返回适合后续桌面界面显示的中文摘要"""

        rows, columns, slices, periods = self.shape
        return (
            "Enhanced DICOM 转换完成\n"
            f"矩阵形状：[row={rows}, column={columns}, "
            f"slice={slices}, cardiac_phase={periods}]\n"
            f"数据类型：{self.dtype}\n"
            f"VENC：{self.venc:g}\n"
            f"心动周期时相数：{self.cardiac_phases}\n"
            "体素间距："
            f"{self.voxel_size_mm[0]:g} × {self.voxel_size_mm[1]:g} × "
            f"{self.voxel_size_mm[2]:g} mm\n"
            f"TriggerTime 平均间隔：{self.time_spacing_ms:g} ms"
        )


def _required_keyword(dataset: Dataset, keyword: str, location: str) -> object:
    if keyword not in dataset:
        raise EnhancedDicomError(f"{location} 缺少必要字段 {keyword}")
    return getattr(dataset, keyword)


def _first_sequence_item(dataset: Dataset, keyword: str, location: str) -> Dataset:
    sequence = _required_keyword(dataset, keyword, location)
    if not sequence:
        raise EnhancedDicomError(f"{location} 的 {keyword} 为空")
    return sequence[0]  # type: ignore[index,no-any-return]


def _private_value(dataset: Dataset, tag: Tag, matlab_name: str, location: str) -> object:
    if tag not in dataset:
        raise EnhancedDicomError(
            f"{location} 缺少 Philips 私有标签 {matlab_name}"
        )
    return dataset[tag].value


def _as_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise EnhancedDicomError(f"{field_name} 不是可识别的数值") from error


def _as_int(value: object, field_name: str) -> int:
    numeric_value = _as_float(value, field_name)
    if not numeric_value.is_integer():
        raise EnhancedDicomError(f"{field_name} 必须是整数，实际为 {numeric_value}")
    return int(numeric_value)


def _numeric_tuple(value: object, length: int, field_name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(component) for component in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise EnhancedDicomError(f"{field_name} 不是有效数值向量") from error
    if len(result) != length:
        raise EnhancedDicomError(
            f"{field_name} 应有 {length} 个分量，实际为 {len(result)} 个"
        )
    return result


def _normalise_multiframe_pixels(
    pixels: NDArray[np.generic],
    dataset: Dataset,
    series_name: str,
) -> NDArray[np.generic]:
    array = np.asarray(pixels)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise EnhancedDicomError(
            f"{series_name} 多帧像素数组应为 [frame,row,column]，"
            f"实际形状为 {array.shape}"
        )

    rows = _as_int(_required_keyword(dataset, "Rows", series_name), f"{series_name}.Rows")
    columns = _as_int(
        _required_keyword(dataset, "Columns", series_name),
        f"{series_name}.Columns",
    )
    if array.shape[1:] != (rows, columns):
        raise EnhancedDicomError(
            f"{series_name} 像素尺寸与 DICOM 元数据不一致："
            f"元数据为 {(rows, columns)}，像素数组为 {array.shape[1:]}"
        )

    if "NumberOfFrames" in dataset:
        number_of_frames = _as_int(dataset.NumberOfFrames, f"{series_name}.NumberOfFrames")
        if number_of_frames != array.shape[0]:
            raise EnhancedDicomError(
                f"{series_name} 的 NumberOfFrames={number_of_frames}，"
                f"但像素数组包含 {array.shape[0]} 帧"
            )
    return array


def _reshape_to_project_4d(
    pixels: NDArray[np.generic],
    num_slices: int,
    num_periods: int,
    output_dtype: np.dtype[np.generic],
) -> NDArray[np.generic]:
    frames, rows, columns = pixels.shape
    return (
        pixels.astype(output_dtype, copy=False)
        .reshape(num_slices, num_periods, rows, columns)
        .transpose(2, 3, 0, 1)
        .copy()
    )


def convert_enhanced_dicom_data(
    magnitude_info: Dataset,
    phase_info: Dataset,
    magnitude_pixels: NDArray[np.generic],
    ap_pixels: NDArray[np.generic],
    fh_pixels: NDArray[np.generic],
    rl_pixels: NDArray[np.generic],
) -> Enhanced4DData:
    """把 MATLAB ``GetDCMfromEnhancedDCM`` 的输入转换为 Python 数据

    像素数组遵循 pydicom 的灰度多帧顺序
    ``[frame, row, column]``本函数不改变帧顺序，也不从元数据重新
    推断顺序；连续的心动时相被视为属于同一切片
    """

    magnitude = _normalise_multiframe_pixels(
        magnitude_pixels,
        magnitude_info,
        "M",
    )
    ap = _normalise_multiframe_pixels(ap_pixels, phase_info, "AP")
    fh = _normalise_multiframe_pixels(fh_pixels, phase_info, "FH")
    rl = _normalise_multiframe_pixels(rl_pixels, phase_info, "RL")

    shapes = {
        "M": magnitude.shape,
        "AP": ap.shape,
        "FH": fh.shape,
        "RL": rl.shape,
    }
    if len(set(shapes.values())) != 1:
        shape_text = "，".join(f"{name}={shape}" for name, shape in shapes.items())
        raise EnhancedDicomError(f"四组 Enhanced DICOM 像素形状不一致：{shape_text}")

    cardiac_phases = _as_int(
        _private_value(
            phase_info,
            _PHILIPS_CARDIAC_PHASES_TAG,
            "Private_2001_1017",
            "P 图 Enhanced DICOM",
        ),
        "Private_2001_1017",
    )
    if cardiac_phases < 2:
        raise EnhancedDicomError(
            "CardiacPhases 至少需要 2 个时相，才能计算 TriggerTime 平均间隔"
        )

    frame_count, rows, columns = fh.shape
    if frame_count % cardiac_phases != 0:
        raise EnhancedDicomError(
            f"P 图共有 {frame_count} 帧，不能被 {cardiac_phases} 个时相整除"
        )
    num_slices = frame_count // cardiac_phases

    shared_item = _first_sequence_item(
        phase_info,
        "SharedFunctionalGroupsSequence",
        "P 图 Enhanced DICOM",
    )
    mapping_item = _first_sequence_item(
        shared_item,
        "RealWorldValueMappingSequence",
        "SharedFunctionalGroupsSequence.Item_1",
    )
    rescale_intercept = _as_float(
        _required_keyword(
            mapping_item,
            "RealWorldValueIntercept",
            "RealWorldValueMappingSequence.Item_1",
        ),
        "RealWorldValueIntercept",
    )
    rescale_slope = _as_float(
        _required_keyword(
            mapping_item,
            "RealWorldValueSlope",
            "RealWorldValueMappingSequence.Item_1",
        ),
        "RealWorldValueSlope",
    )
    venc = abs(rescale_intercept)
    heart_rate_bpm = _as_float(
        _required_keyword(phase_info, "HeartRate", "P 图 Enhanced DICOM"),
        "HeartRate",
    )

    per_frame = _required_keyword(
        phase_info,
        "PerFrameFunctionalGroupsSequence",
        "P 图 Enhanced DICOM",
    )
    if len(per_frame) != frame_count:  # type: ignore[arg-type]
        raise EnhancedDicomError(
            "PerFrameFunctionalGroupsSequence 项目数与像素帧数不一致："
            f"Functional Groups={len(per_frame)}，像素帧={frame_count}"  # type: ignore[arg-type]
        )

    first_frame = per_frame[0]  # type: ignore[index]
    pixel_measures = _first_sequence_item(
        first_frame,
        "PixelMeasuresSequence",
        "PerFrameFunctionalGroupsSequence.Item_1",
    )
    pixel_spacing_values = _numeric_tuple(
        _required_keyword(
            pixel_measures,
            "PixelSpacing",
            "PixelMeasuresSequence.Item_1",
        ),
        2,
        "PixelSpacing",
    )
    pixel_spacing = (pixel_spacing_values[0], pixel_spacing_values[1])
    spacing_between_slices = _as_float(
        _required_keyword(
            pixel_measures,
            "SpacingBetweenSlices",
            "PixelMeasuresSequence.Item_1",
        ),
        "SpacingBetweenSlices",
    )
    slice_thickness = _as_float(
        _required_keyword(
            pixel_measures,
            "SliceThickness",
            "PixelMeasuresSequence.Item_1",
        ),
        "SliceThickness",
    )

    orientation_item = _first_sequence_item(
        first_frame,
        "PlaneOrientationSequence",
        "PerFrameFunctionalGroupsSequence.Item_1",
    )
    orientation_values = _numeric_tuple(
        _required_keyword(
            orientation_item,
            "ImageOrientationPatient",
            "PlaneOrientationSequence.Item_1",
        ),
        6,
        "ImageOrientationPatient",
    )
    orientation = (
        orientation_values[0],
        orientation_values[1],
        orientation_values[2],
        orientation_values[3],
        orientation_values[4],
        orientation_values[5],
    )
    normal = np.cross(
        np.asarray(orientation[:3], dtype=float),
        np.asarray(orientation[3:], dtype=float),
    )
    if not np.any(normal):
        raise EnhancedDicomError(
            "ImageOrientationPatient 的行列方向向量叉积为零，无法确定切片方向"
        )
    slice_axis: SliceAxis = int(np.argmax(np.abs(normal)))  # type: ignore[assignment]

    trigger_times: list[float] = []
    for frame_index in range(cardiac_phases):
        frame_item = per_frame[frame_index]  # type: ignore[index]
        private_item = _private_value(
            frame_item,
            _PHILIPS_FRAME_DETAILS_TAG,
            "Private_2005_140f",
            f"PerFrameFunctionalGroupsSequence.Item_{frame_index + 1}",
        )
        if not private_item:
            raise EnhancedDicomError(
                f"第 {frame_index + 1} 帧的 Private_2005_140f 为空"
            )
        trigger_times.append(
            _as_float(
                _required_keyword(
                    private_item[0],
                    "TriggerTime",
                    f"Private_2005_140f.Item_1（第 {frame_index + 1} 帧）",
                ),
                f"第 {frame_index + 1} 帧 TriggerTime",
            )
        )
    time_spacing_ms = float(np.mean(np.diff(np.asarray(trigger_times, dtype=float))))

    image_positions = np.empty((3, frame_count), dtype=np.float64)
    slice_locations = np.empty(frame_count, dtype=np.float64)
    for frame_index, frame_item in enumerate(per_frame):  # type: ignore[union-attr]
        position_item = _first_sequence_item(
            frame_item,
            "PlanePositionSequence",
            f"PerFrameFunctionalGroupsSequence.Item_{frame_index + 1}",
        )
        position = _numeric_tuple(
            _required_keyword(
                position_item,
                "ImagePositionPatient",
                f"PlanePositionSequence.Item_1（第 {frame_index + 1} 帧）",
            ),
            3,
            f"第 {frame_index + 1} 帧 ImagePositionPatient",
        )
        image_positions[:, frame_index] = position
        slice_locations[frame_index] = position[slice_axis]

    series_description = str(
        _required_keyword(phase_info, "SeriesDescription", "P 图 Enhanced DICOM")
    )
    series_number = _required_keyword(
        phase_info,
        "SeriesNumber",
        "P 图 Enhanced DICOM",
    )
    bits_allocated = _as_int(
        _required_keyword(phase_info, "BitsAllocated", "P 图 Enhanced DICOM"),
        "BitsAllocated",
    )
    bits_stored = _as_int(
        _required_keyword(phase_info, "BitsStored", "P 图 Enhanced DICOM"),
        "BitsStored",
    )
    high_bit = _as_int(
        _required_keyword(phase_info, "HighBit", "P 图 Enhanced DICOM"),
        "HighBit",
    )
    pixel_representation = _as_int(
        _required_keyword(phase_info, "PixelRepresentation", "P 图 Enhanced DICOM"),
        "PixelRepresentation",
    )

    patient_name: str | None = None
    if "PatientName" in phase_info:
        raw_patient_name = phase_info.PatientName
        patient_name = str(getattr(raw_patient_name, "family_name", raw_patient_name))

    classic_metadata = ClassicMetadataTemplate(
        sop_class_uid=str(MRImageStorage),
        media_storage_sop_class_uid=str(MRImageStorage),
        transfer_syntax_uid=str(ExplicitVRLittleEndian),
        series_instance_uid=generate_uid(),
        study_instance_uid=generate_uid(),
        media_storage_sop_instance_uid=generate_uid(),
        series_description=series_description,
        series_number=series_number,
        pixel_spacing=pixel_spacing,
        image_orientation_patient=orientation,
        slice_thickness=slice_thickness,
        spacing_between_slices=spacing_between_slices,
        slice_axis=slice_axis,
        image_positions_patient=image_positions,
        slice_locations=slice_locations,
        instance_numbers=np.arange(1, frame_count + 1, dtype=np.int64),
        rows=rows,
        columns=columns,
        bits_allocated=bits_allocated,
        bits_stored=bits_stored,
        high_bit=high_bit,
        pixel_representation=pixel_representation,
        samples_per_pixel=1,
        rescale_intercept=rescale_intercept,
        rescale_slope=rescale_slope,
        patient_name=patient_name,
        patient_id=str(phase_info.PatientID) if "PatientID" in phase_info else None,
        study_date=str(phase_info.StudyDate) if "StudyDate" in phase_info else None,
        modality=str(phase_info.Modality) if "Modality" in phase_info else None,
    )

    output_dtype = fh.dtype
    magnitude_4d = _reshape_to_project_4d(
        magnitude,
        num_slices,
        cardiac_phases,
        output_dtype,
    )
    ap_4d = _reshape_to_project_4d(ap, num_slices, cardiac_phases, output_dtype)
    fh_4d = _reshape_to_project_4d(fh, num_slices, cardiac_phases, output_dtype)
    rl_4d = _reshape_to_project_4d(rl, num_slices, cardiac_phases, output_dtype)

    return Enhanced4DData(
        reference_image=fh_4d[:, :, 0, 0].copy(),
        manufacturer="Philips",
        classic_metadata=classic_metadata,
        phase_dicom_info=phase_info,
        magnitude_dicom_info=magnitude_info,
        series="TFEPI",
        rescale_intercept=rescale_intercept,
        rescale_slope=rescale_slope,
        venc=venc,
        cardiac_phases=cardiac_phases,
        voxel_size_mm=(
            pixel_spacing[0],
            pixel_spacing[1],
            spacing_between_slices,
        ),
        heart_rate_bpm=heart_rate_bpm,
        time_spacing_ms=time_spacing_ms,
        magnitude=magnitude_4d,
        ap=ap_4d,
        fh=fh_4d,
        rl=rl_4d,
    )
