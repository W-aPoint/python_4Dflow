"""把派生的三维体数据写成 Classic MR DICOM 切片"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray
from pydicom import dcmread, dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

try:
    from ..base_function import RuntimeTimer, TimingRecord
    from .case_loader import LoadedDicomCase
except ImportError:
    if __package__ != "dicom":
        raise
    from base_function import RuntimeTimer, TimingRecord  # type: ignore[no-redef]
    from dicom.case_loader import LoadedDicomCase  # type: ignore[no-redef]


class DicomVolumeExportError(RuntimeError):
    """派生体数据无法写成 Classic DICOM 时抛出"""


@dataclass(frozen=True, slots=True)
class DicomVolumeExportResult:
    """一个已导出三维序列的文件清单和耗时证据"""

    output_directory: Path
    files: tuple[Path, ...]
    series_instance_uid: str
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _remove_enhanced_attributes(dataset: Dataset) -> None:
    for keyword in (
        "SharedFunctionalGroupsSequence",
        "PerFrameFunctionalGroupsSequence",
        "NumberOfFrames",
        "DimensionOrganizationSequence",
        "DimensionIndexSequence",
        "FrameIncrementPointer",
    ):
        if keyword in dataset:
            del dataset[keyword]


def _set_file_meta(dataset: Dataset, sop_instance_uid: str) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset.file_meta = file_meta
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False


def _set_derived_image_metadata(dataset: Dataset) -> None:
    """将输出标记为派生图像，并应用恒等的存储值映射"""

    dataset.ImageType = ["DERIVED", "SECONDARY"]
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    if "RealWorldValueMappingSequence" in dataset:
        del dataset.RealWorldValueMappingSequence


def _set_pixels(dataset: Dataset, pixels: NDArray[np.uint16]) -> None:
    rows, columns = pixels.shape
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.ascontiguousarray(pixels.astype("<u2", copy=False)).tobytes()


def _classic_slice_dataset(
    case: LoadedDicomCase,
    slice_index: int,
) -> Dataset:
    source_index = slice_index * case.cardiac_phases
    try:
        source_path = case.inventory.magnitude_files[source_index]
    except IndexError as error:
        raise DicomVolumeExportError(
            "Classic M 图文件数量不足，无法为每个空间层复制元数据"
        ) from error
    try:
        return deepcopy(dcmread(source_path))
    except Exception as error:
        raise DicomVolumeExportError(f"无法读取导出模板 DICOM：{source_path}") from error


def _enhanced_slice_dataset(
    case: LoadedDicomCase,
    slice_index: int,
) -> Dataset:
    if case.enhanced_data is None:
        raise DicomVolumeExportError("Enhanced 病例缺少 enhanced_data")
    dataset = deepcopy(case.phase_info)
    _remove_enhanced_attributes(dataset)
    template = case.enhanced_data.classic_metadata
    frame_index = slice_index * case.cardiac_phases
    if frame_index >= len(template.image_positions_patient):
        raise DicomVolumeExportError("Enhanced 几何模板的帧数不足")
    dataset.PixelSpacing = list(template.pixel_spacing)
    dataset.ImageOrientationPatient = list(template.image_orientation_patient)
    dataset.ImagePositionPatient = list(template.image_positions_patient[frame_index])
    dataset.SliceThickness = template.slice_thickness
    dataset.SpacingBetweenSlices = template.spacing_between_slices
    dataset.SliceLocation = float(template.slice_locations[frame_index])
    if template.patient_name is not None:
        dataset.PatientName = template.patient_name
    if template.patient_id is not None:
        dataset.PatientID = template.patient_id
    if template.study_date is not None:
        dataset.StudyDate = template.study_date
    if template.modality is not None:
        dataset.Modality = template.modality
    return dataset


def save_dicom_volume(
    volume: NDArray[np.generic],
    output_directory: str | Path,
    case: LoadedDicomCase,
    series_description: str,
) -> DicomVolumeExportResult:
    """对应两个 ``SaveDicom3D`` 变体，用于导出派生三维体数据"""

    array = np.asarray(volume)
    if array.ndim != 3 or array.shape != case.shape[:3]:
        raise DicomVolumeExportError(
            f"导出体数据应为 {case.shape[:3]}，实际为 {array.shape}"
        )
    if array.dtype != np.uint16:
        raise DicomVolumeExportError("导出体数据必须先转换为 uint16")
    description = str(series_description).strip()
    if not description:
        raise DicomVolumeExportError("SeriesDescription 不能为空")

    output_path = Path(output_directory).expanduser()
    timer = RuntimeTimer()
    with timer.measure("准备 DICOM 输出目录"):
        output_path.mkdir(parents=True, exist_ok=True)
        if not output_path.is_dir():
            raise NotADirectoryError(f"DICOM 输出路径不是文件夹：{output_path}")
        output_path = output_path.resolve()

    series_uid = generate_uid()
    files: list[Path] = []
    with timer.measure(f"写出 {description} DICOM"):
        for slice_index in range(array.shape[2]):
            if case.is_enhanced_dicom:
                dataset = _enhanced_slice_dataset(case, slice_index)
                filename = f"slice_{slice_index + 1:05d}.dcm"
            else:
                dataset = _classic_slice_dataset(case, slice_index)
                filename = f"DCM000_Sec_{slice_index + 1:03d}.dcm"

            dataset.SeriesInstanceUID = series_uid
            dataset.SeriesDescription = description
            dataset.ProtocolName = description
            dataset.InstanceNumber = slice_index + 1
            if "BodyPartExamined" in dataset:
                dataset.BodyPartExamined = re.sub(
                    r"[^A-Z0-9 ]",
                    "",
                    str(dataset.BodyPartExamined).upper(),
                )
            _set_derived_image_metadata(dataset)
            sop_instance_uid = generate_uid()
            _set_file_meta(dataset, sop_instance_uid)
            _set_pixels(dataset, array[:, :, slice_index])
            destination = output_path / filename
            try:
                dcmwrite(destination, dataset, enforce_file_format=True)
            except Exception as error:
                raise DicomVolumeExportError(
                    f"无法写出第 {slice_index + 1} 层 DICOM：{destination}"
                ) from error
            files.append(destination)

    return DicomVolumeExportResult(
        output_directory=output_path,
        files=tuple(files),
        series_instance_uid=series_uid,
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
