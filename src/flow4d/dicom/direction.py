"""从第一份 Philips DICOM 文件读取速度编码方向

本模块对应 ``baseFunction/GetDirection.m``，只读取元数据，并严格保留
MATLAB 的向量匹配规则：``[0, 1, 0]`` 为 AP，``[1, 0, 0]`` 为 RL，
``[0, 0, 1]`` 为 FH，其余向量均报告为 ``Unknown``
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag


DicomDirection: TypeAlias = Literal["AP", "RL", "FH", "Unknown"]

_PHILIPS_VELOCITY_ENCODING_SEQUENCE_TAG = Tag(0x2005, 0x140F)
_KNOWN_DIRECTIONS: dict[tuple[float, ...], DicomDirection] = {
    (0.0, 1.0, 0.0): "AP",
    (1.0, 0.0, 0.0): "RL",
    (0.0, 0.0, 1.0): "FH",
}


class DirectionMetadataError(RuntimeError):
    """无法从第一份 DICOM 读取方向元数据时抛出"""


def _normalise_vector(raw_value: object) -> tuple[float, ...]:
    """把 pydicom 值转为数值元组，不改变原始数值"""

    if isinstance(raw_value, Iterable) and not isinstance(raw_value, (str, bytes)):
        components = raw_value
    else:
        components = (raw_value,)

    try:
        return tuple(float(component) for component in components)
    except (TypeError, ValueError) as error:
        raise DirectionMetadataError(
            "VelocityEncodingDirection 不是可识别的数值向量，无法判断方向"
        ) from error


def get_velocity_encoding_direction(
    dicom_files: Sequence[str | Path],
) -> DicomDirection:
    """返回第一份 DICOM 的 Philips 速度编码方向

    参数：
        dicom_files: 有序且非空的 DICOM 路径序列与 MATLAB 一致，
            只读取第一份文件

    返回：
        ``"AP"``、``"RL"``、``"FH"`` 或 ``"Unknown"``

    异常：
        TypeError: 传入单一路径而不是路径序列
        ValueError: 提供的序列为空
        FileNotFoundError: 第一条路径不存在
        IsADirectoryError: 第一条路径不是文件
        DirectionMetadataError: 第一份文件不是可读 DICOM，或必要的 Philips
            方向元数据缺失、格式错误
    """

    if isinstance(dicom_files, (str, bytes, Path)):
        raise TypeError(
            "方向识别需要一个有序的 DICOM 文件列表，不能只传入单个路径值"
        )
    if not dicom_files:
        raise ValueError("DICOM 文件列表为空，无法判断速度编码方向")

    first_path = Path(dicom_files[0]).expanduser()
    if not first_path.exists():
        raise FileNotFoundError(f"首个 DICOM 文件不存在：{first_path}")
    if not first_path.is_file():
        raise IsADirectoryError(f"首个 DICOM 路径不是文件：{first_path}")

    first_path = first_path.resolve()
    try:
        dataset = dcmread(first_path, stop_before_pixels=True)
    except InvalidDicomError as error:
        raise DirectionMetadataError(
            f"首个文件不是可识别的 DICOM 文件：{first_path}"
        ) from error
    except OSError as error:
        raise DirectionMetadataError(
            f"无法读取首个 DICOM 文件：{first_path}请检查文件权限或文件状态"
        ) from error

    if _PHILIPS_VELOCITY_ENCODING_SEQUENCE_TAG not in dataset:
        raise DirectionMetadataError(
            "首个 DICOM 缺少 Philips 私有序列 (2005,140F)，"
            "无法读取速度编码方向"
        )

    sequence = dataset[_PHILIPS_VELOCITY_ENCODING_SEQUENCE_TAG].value
    if not sequence:
        raise DirectionMetadataError(
            "首个 DICOM 的 Philips 私有序列 (2005,140F) 为空，"
            "无法读取速度编码方向"
        )

    first_item = sequence[0]
    if "VelocityEncodingDirection" not in first_item:
        raise DirectionMetadataError(
            "首个 DICOM 的 Philips 私有序列缺少 VelocityEncodingDirection，"
            "无法判断 AP、RL 或 FH 方向"
        )

    vector = _normalise_vector(first_item.VelocityEncodingDirection)
    return _KNOWN_DIRECTIONS.get(vector, "Unknown")
