"""Fill MATLAB-named Tecplot exports from an already completed FINAL VTK series.

Run: python -m flow4d.backfill_tecplot --case PATH_TO_DICOM_4D_Qflow
The normal overall_flow path writes from in-memory arrays; this tool avoids
repeating the numerical pipeline for a case whose VTK files already exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Callable

import numpy as np

from .exporters.tecplot_ascii import write_tecplot_ascii
from .exporters.tecplot_v112 import write_tecplot_v112_tetra
from .exporters.vtk_legacy import FINAL_HEADERS


_PATTERN = re.compile(r"NodeLiutexPressure_(\d+)\.vtk$")


def backfill_tecplot_from_vtk(
    case_directory: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Convert all consecutive FINAL VTK phases to UVW DAT and FEMbrick PLT."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise RuntimeError("从已有 VTK 补导出需要 Python 的 vtk 包") from exc

    case = Path(case_directory).expanduser().resolve()
    vtk_directory = case / "vtk"
    indexed = {
        int(match.group(1)): path
        for path in vtk_directory.glob("NodeLiutexPressure_*.vtk")
        if (match := _PATTERN.fullmatch(path.name)) is not None
    }
    if not indexed or sorted(indexed) != list(range(1, len(indexed) + 1)):
        raise ValueError("VTK 时相必须从 1 开始连续编号")
    output = case / "dat"
    if not overwrite:
        existing = [
            output / f"{prefix}{phase}.{suffix}"
            for phase in indexed
            for prefix, suffix in (("UVW_", "dat"), ("FEMbrick_", "plt"))
            if (output / f"{prefix}{phase}.{suffix}").exists()
        ]
        if existing:
            raise FileExistsError(f"已有 DAT/PLT 文件，未覆盖：{existing[0]}")
    output.mkdir(parents=True, exist_ok=True)

    for phase, path in sorted(indexed.items()):
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(str(path))
        reader.ReadAllFieldsOn()
        reader.Update()
        grid = reader.GetOutput()
        point_count = grid.GetNumberOfPoints()
        cell_count = grid.GetNumberOfCells()
        if point_count == 0 or cell_count == 0:
            raise ValueError(f"VTK 网格为空：{path}")
        cell_types = vtk_to_numpy(grid.GetCellTypes())
        if not np.all(cell_types == vtk.VTK_TETRA):
            raise ValueError(f"VTK 含有非四节点四面体：{path}")
        nodes = np.asarray(vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64)
        elements = np.asarray(
            vtk_to_numpy(grid.GetCells().GetConnectivityArray()), dtype=np.int64
        ).reshape(cell_count, 4)
        values = np.empty((point_count, len(FINAL_HEADERS)), dtype=np.float64)
        values[:, :3] = nodes
        point_data = grid.GetPointData()
        for column, name in enumerate(FINAL_HEADERS[3:], start=3):
            field = point_data.GetArray(name)
            if field is None or field.GetNumberOfTuples() != point_count:
                raise ValueError(f"VTK 缺少节点字段 {name}：{path}")
            values[:, column] = vtk_to_numpy(field).reshape(point_count)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"VTK 含有 NaN/Inf：{path}")
        write_tecplot_ascii(
            values[:, :6], FINAL_HEADERS[:6], elements,
            output / f"UVW_{phase}.dat", time_frame=phase,
            connectivity_index_base=0, overwrite=overwrite,
        )
        write_tecplot_v112_tetra(
            values, FINAL_HEADERS, elements,
            output / f"FEMbrick_{phase}.plt", phase_number=phase,
            overwrite=overwrite,
        )
        if progress is not None:
            progress(phase, len(indexed))
    return len(indexed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="已有 vtk 文件夹的病例目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同名 DAT/PLT 文件")
    args = parser.parse_args()
    count = backfill_tecplot_from_vtk(
        args.case, overwrite=args.overwrite,
        progress=lambda phase, total: print(f"DAT/PLT {phase}/{total} 已写入", flush=True),
    )
    print(f"完成：{count} 组 DAT/PLT，位于 {args.case.resolve() / 'dat'}")


if __name__ == "__main__":
    main()
