"""准备 MATLAB 工作流使用的病例目录结构

本模块对应 ``baseFunction/CreateFolder.m``MATLAB 函数同时负责目录
选择和目录创建；在 Python 应用中，后续由 PySide6 界面负责选择目录，
再把用户选定的路径传入本模块
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence


DEFAULT_CASE_FOLDER_NAME: Final = "DICOM_4D_Qflow"

DICOM_SERIES_FOLDER_NAMES: Final[tuple[str, ...]] = (
    "4D_Qflow_M",
    "4D_Qflow_AP",
    "4D_Qflow_FH",
    "4D_Qflow_RL",
)


@dataclass(frozen=True, slots=True)
class CaseDirectoryResult:
    """一个 4D Flow 病例目录的准备结果"""

    case_directory: Path
    series_directories: tuple[Path, ...]
    created_directories: tuple[Path, ...]
    existing_directories: tuple[Path, ...]

    @property
    def completion_message(self) -> str:
        """返回适合桌面界面显示的具体中文提示"""

        series_lines = "\n".join(
            f"- {directory.name}：{directory}" for directory in self.series_directories
        )
        return (
            "病例目录准备完成\n"
            f"病例目录：{self.case_directory}\n"
            f"本次新建：{len(self.created_directories)} 个目录\n"
            f"原已存在：{len(self.existing_directories)} 个目录\n"
            "四组 DICOM 目录：\n"
            f"{series_lines}\n"
            "下一步：将幅度图放入 4D_Qflow_M，并将三个速度方向序列分别放入 "
            "4D_Qflow_AP、4D_Qflow_FH 和 4D_Qflow_RL放置完成后再进行数据检查"
        )


def prepare_case_directory(
    parent_directory: str | Path,
    folder_name: str = DEFAULT_CASE_FOLDER_NAME,
) -> CaseDirectoryResult:
    """创建或复用 ``main_final.m`` 所需的目录结构

    已有目录会被保留；本函数不会删除文件、清空目录，也不会用目录静默
    替换同名文件

    参数：
        parent_directory: 用户选择的、已经存在的病例父目录
        folder_name: 在父目录下创建的工作流目录名称

    返回：
        病例目录、四个 DICOM 序列目录及其创建状态

    异常：
        FileNotFoundError: 选择的父目录不存在
        NotADirectoryError: 选择的路径不是目录
        ValueError: ``folder_name`` 为空或包含路径分隔部分
        FileExistsError: 预期的目录位置已被同名文件占用
        OSError: 操作系统无法创建所需目录
    """

    parent = Path(parent_directory).expanduser()
    if not parent.exists():
        raise FileNotFoundError(f"所选父目录不存在：{parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"所选路径不是文件夹：{parent}")

    normalized_folder_name = folder_name.strip()
    if (
        not normalized_folder_name
        or normalized_folder_name in {".", ".."}
        or Path(normalized_folder_name).name != normalized_folder_name
    ):
        raise ValueError("病例文件夹名称必须是单个非空文件夹名称，不能包含路径")

    parent = parent.resolve()
    case_directory = parent / normalized_folder_name
    series_directories = tuple(
        case_directory / name for name in DICOM_SERIES_FOLDER_NAMES
    )
    required_directories = (case_directory, *series_directories)

    created: list[Path] = []
    existing: list[Path] = []

    for directory in required_directories:
        if directory.exists():
            if not directory.is_dir():
                raise FileExistsError(
                    f"无法创建所需文件夹，因为同名文件已经存在：{directory}"
                )
            existing.append(directory)
            continue

        directory.mkdir()
        created.append(directory)

    return CaseDirectoryResult(
        case_directory=case_directory,
        series_directories=series_directories,
        created_directories=tuple(created),
        existing_directories=tuple(existing),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="创建或检查 4D Flow MRI 病例所需的四组 DICOM 文件夹"
    )
    parser.add_argument(
        "parent_directory",
        help="病例父目录；程序会在其下创建 DICOM_4D_Qflow",
    )
    parser.add_argument(
        "--folder-name",
        default=DEFAULT_CASE_FOLDER_NAME,
        help=f"病例工作文件夹名称，默认：{DEFAULT_CASE_FOLDER_NAME}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """从命令行运行目录准备工具"""

    args = _build_argument_parser().parse_args(argv)
    try:
        result = prepare_case_directory(args.parent_directory, args.folder_name)
    except (OSError, ValueError) as error:
        print(f"病例目录准备失败：{error}", file=sys.stderr)
        print(
            "请检查所选路径是否存在、是否有写入权限，以及是否存在同名文件",
            file=sys.stderr,
        )
        return 1

    print(result.completion_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
