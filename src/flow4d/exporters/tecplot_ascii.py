"""由 MATLAB 保存函数迁移而来的 Tecplot ASCII 节点/单元导出器"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class TecplotExportError(RuntimeError):
    """Tecplot ASCII 输入或路径无效时抛出"""


def write_tecplot_ascii(
    node_values: NDArray[np.generic],
    variable_names: Sequence[str],
    connectivity: NDArray[np.generic],
    filename: str | Path,
    *,
    time_frame: int = 0,
    zone_type: str = "FETetrahedron",
    connectivity_index_base: int,
    include_velocity_auxdata: bool = False,
    include_solution_time: bool = False,
    overwrite: bool = True,
) -> Path:
    """对应 ``FASTsavedata2dat.m`` 和 ``FASTsavedata2datFluent.m``"""

    node_raw = np.asarray(node_values)
    if np.iscomplexobj(node_raw):
        raise TecplotExportError("节点数据不能包含复数")
    node = np.asarray(node_raw, dtype=np.float64)
    elements_raw = np.asarray(connectivity)
    names = tuple(str(name).strip().strip('"') for name in variable_names)
    if node.ndim != 2 or node.shape[0] == 0 or node.shape[1] != len(names):
        raise TecplotExportError("节点数据列数必须与变量名数量一致，且节点不能为空")
    if not names or any(not name for name in names):
        raise TecplotExportError("变量名不能为空")
    if not np.all(np.isfinite(node)):
        raise TecplotExportError("节点数据必须为有限实数")
    if zone_type not in ("FETetrahedron", "FEBrick"):
        raise TecplotExportError("zone_type 只能为 FETetrahedron 或 FEBrick")
    expected_width = 4 if zone_type == "FETetrahedron" else 8
    if elements_raw.ndim != 2 or elements_raw.shape[0] == 0 or elements_raw.shape[1] < expected_width:
        raise TecplotExportError(f"{zone_type} 连接矩阵至少需要 {expected_width} 列")
    if connectivity_index_base not in (0, 1):
        raise TecplotExportError("connectivity_index_base 只能为 0 或 1")
    selected_elements = (
        elements_raw[:, :expected_width]
        if zone_type == "FETetrahedron"
        else elements_raw
    )
    if not np.issubdtype(selected_elements.dtype, np.integer):
        if not np.all(np.isfinite(selected_elements)) or not np.all(
            selected_elements == np.floor(selected_elements)
        ):
            raise TecplotExportError("连接矩阵必须只包含有限整数")
    elements = np.asarray(selected_elements, dtype=np.int64)
    if connectivity_index_base == 0:
        elements = elements + 1
    if np.min(elements) < 1 or np.max(elements) > node.shape[0]:
        raise TecplotExportError("连接矩阵含有超出节点范围的索引")

    target = Path(filename).expanduser().resolve()
    if target.exists() and not overwrite:
        raise TecplotExportError(f"输出文件已存在，未获得覆盖许可：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ['TITLE\t= "byAB"', f'VARIABLES = "{names[0]}"']
    lines.extend(f'"{name}"' for name in names[1:])
    if include_velocity_auxdata:
        lines.extend(
            (
                'DATASETAUXDATA Common.VectorVarsAreVelocity="TRUE"',
                'DATASETAUXDATA Common.UVar="4"',
                'DATASETAUXDATA Common.VVar="5"',
                'DATASETAUXDATA Common.WVar="6"',
            )
        )
    lines.append(f'ZONE T= "time_{int(time_frame)}"')
    if include_solution_time:
        lines.append(f" STRANDID=1, SOLUTIONTIME={int(time_frame)}")
    lines.extend(
        (
            f" Nodes={node.shape[0]}, Elements={elements.shape[0]}, ZONETYPE={zone_type}",
            " DATAPACKING=POINT",
            " DT=(" + "DOUBLE " * len(names) + ")",
        )
    )
    lines.extend(" ".join(f"{value:.6f}" for value in row) + " " for row in node)
    lines.extend(" ".join(str(int(value)) for value in row) + " " for row in elements)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return target
