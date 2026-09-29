"""发现 MATLAB 工作流使用的四个 DICOM 序列文件夹

本模块对应 ``baseFunction/GetDicomlist.m``，只盘点文件路径，不打开
DICOM、不解析元数据，也不推断速度编码方向
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .case_directory import DICOM_SERIES_FOLDER_NAMES


@dataclass(frozen=True, slots=True)
class DicomSeriesInventory:
    """幅度序列和三个速度序列的路径清单"""

    case_directory: Path
    magnitude_files: tuple[Path, ...]
    ap_files: tuple[Path, ...]
    fh_files: tuple[Path, ...]
    rl_files: tuple[Path, ...]

    @property
    def series(self) -> tuple[tuple[str, tuple[Path, ...]], ...]:
        """按 MATLAB 工作流的相同顺序返回四个序列"""

        return (
            ("4D_Qflow_M", self.magnitude_files),
            ("4D_Qflow_AP", self.ap_files),
            ("4D_Qflow_FH", self.fh_files),
            ("4D_Qflow_RL", self.rl_files),
        )

    @property
    def counts(self) -> tuple[int, int, int, int]:
        """按 M、AP、FH、RL 顺序返回文件数"""

        return tuple(len(files) for _, files in self.series)

    @property
    def ready_for_metadata_reading(self) -> bool:
        """判断四个非空序列的当前文件数是否一致"""

        return min(self.counts) > 0 and len(set(self.counts)) == 1

    @property
    def inspection_message(self) -> str:
        """返回适合桌面界面显示的中文状态提示"""

        count_lines = "\n".join(
            f"- {name}：{len(files)} 个 DICOM 文件" for name, files in self.series
        )

        if self.ready_for_metadata_reading:
            status = (
                "输入检查结果：准备就绪四组 DICOM 文件数量一致，可以进入元数据读取"
            )
            next_step = "下一步：读取首个 DICOM 的参数，并核对序列方向和时相信息"
        elif max(self.counts) == 0:
            status = "输入检查结果：等待输入四个目录中都没有发现 .dcm 文件"
            next_step = "下一步：将 M、AP、FH、RL 四组 DICOM 分别放入对应目录"
        elif min(self.counts) == 0:
            empty_names = [name for name, files in self.series if not files]
            status = "输入检查结果：未通过以下目录没有 DICOM 文件：" + "、".join(
                empty_names
            )
            next_step = "下一步：补齐空目录后重新检查；当前不要继续读取数据"
        else:
            status = "输入检查结果：未通过四组 DICOM 文件数量不一致"
            next_step = (
                "下一步：检查是否缺少文件、放错方向或存在重复文件；确认后重新检查"
            )

        return (
            "DICOM 文件清单检查完成\n"
            f"病例目录：{self.case_directory}\n"
            f"{count_lines}\n"
            f"{status}\n"
            f"{next_step}"
        )


def _list_dicom_files(directory: Path) -> tuple[Path, ...]:
    """列出直接子文件中扩展名不区分大小写的 ``.dcm`` 文件"""

    return tuple(
        sorted(
            (
                item.resolve()
                for item in directory.iterdir()
                if item.is_file() and item.suffix.casefold() == ".dcm"
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


def inspect_dicom_series(case_directory: str | Path) -> DicomSeriesInventory:
    """盘点所选 ``DICOM_4D_Qflow`` 目录下的 DICOM 路径

    参数：
        case_directory: 包含四个固定序列文件夹的病例目录

    返回：
        按 M、AP、FH、RL 顺序保存文件路径的不可变清单

    异常：
        FileNotFoundError: 病例目录或必要的序列目录不存在
        NotADirectoryError: 所选路径或必要序列路径不是目录
        OSError: 无法枚举目录内容
    """

    case_path = Path(case_directory).expanduser()
    if not case_path.exists():
        raise FileNotFoundError(f"所选病例目录不存在：{case_path}")
    if not case_path.is_dir():
        raise NotADirectoryError(f"所选病例路径不是文件夹：{case_path}")

    case_path = case_path.resolve()
    series_paths = tuple(case_path / name for name in DICOM_SERIES_FOLDER_NAMES)

    missing = tuple(path for path in series_paths if not path.exists())
    if missing:
        missing_names = "、".join(path.name for path in missing)
        raise FileNotFoundError(
            f"病例目录结构不完整，缺少以下文件夹：{missing_names}"
            "请先准备病例目录，再重新检查"
        )

    non_directories = tuple(path for path in series_paths if not path.is_dir())
    if non_directories:
        invalid_names = "、".join(path.name for path in non_directories)
        raise NotADirectoryError(
            f"以下路径应为文件夹，但当前不是文件夹：{invalid_names}"
        )

    magnitude_files, ap_files, fh_files, rl_files = (
        _list_dicom_files(path) for path in series_paths
    )

    return DicomSeriesInventory(
        case_directory=case_path,
        magnitude_files=magnitude_files,
        ap_files=ap_files,
        fh_files=fh_files,
        rl_files=rl_files,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查 4D Flow MRI 病例中 M、AP、FH、RL 四组 DICOM 文件清单"
    )
    parser.add_argument(
        "case_directory",
        help="包含 4D_Qflow_M、AP、FH、RL 四个子目录的 DICOM_4D_Qflow 路径",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """从命令行运行 DICOM 序列盘点"""

    args = _build_argument_parser().parse_args(argv)
    try:
        inventory = inspect_dicom_series(args.case_directory)
    except OSError as error:
        print(f"DICOM 文件清单检查失败：{error}", file=sys.stderr)
        print(
            "请确认选择的是 DICOM_4D_Qflow 文件夹，并检查四个方向子目录",
            file=sys.stderr,
        )
        return 1

    print(inventory.inspection_message)
    return 0 if inventory.ready_for_metadata_reading else 2


if __name__ == "__main__":
    raise SystemExit(main())
