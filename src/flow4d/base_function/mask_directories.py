"""准备并盘点 FINAL PART 06–07 使用的 mask/root 文件夹"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from .runtime_timing import RuntimeTimer, TimingRecord
except ImportError:
    from runtime_timing import RuntimeTimer, TimingRecord  # type: ignore[no-redef]


@dataclass(frozen=True, slots=True)
class MaskDirectoryInventory:
    """mask 和 root 文件夹的路径及当前 DICOM 内容"""

    case_directory: Path
    mask_directory: Path
    root_directory: Path
    mask_files: tuple[Path, ...]
    root_files: tuple[Path, ...]
    created_directories: tuple[Path, ...]
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _dicom_files(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (item.resolve() for item in directory.iterdir() if item.is_file() and item.suffix.casefold() == ".dcm"),
            key=lambda item: (item.name.casefold(), item.name),
        )
    )


def prepare_mask_directories(case_directory: str | Path) -> MaskDirectoryInventory:
    """对应 ``mkdirFORmask.m``，同时消除 MATLAB 占位路径的歧义"""

    case_path = Path(case_directory).expanduser()
    if not case_path.exists():
        raise FileNotFoundError(f"病例目录不存在：{case_path}")
    if not case_path.is_dir():
        raise NotADirectoryError(f"病例路径不是文件夹：{case_path}")
    case_path = case_path.resolve()
    timer = RuntimeTimer()
    created: list[Path] = []
    with timer.measure("创建或复用 mask/root 目录"):
        mask_directory = case_path / "mask"
        root_directory = case_path / "root"
        for directory in (mask_directory, root_directory):
            if not directory.exists():
                directory.mkdir()
                created.append(directory)
            elif not directory.is_dir():
                raise NotADirectoryError(f"目标路径不是文件夹：{directory}")
    with timer.measure("扫描 mask/root DICOM"):
        mask_files = _dicom_files(mask_directory)
        root_files = _dicom_files(root_directory)
    return MaskDirectoryInventory(
        case_directory=case_path,
        mask_directory=mask_directory,
        root_directory=root_directory,
        mask_files=mask_files,
        root_files=root_files,
        created_directories=tuple(created),
        timings=timer.records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )
