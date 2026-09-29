"""三角表面或四面体体网格的 ASCII STL 导出"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class StlExportError(RuntimeError):
    """STL 网格数据或输出路径无效时抛出"""


def _boundary_faces(
    tetrahedra: NDArray[np.int64],
    vertices: NDArray[np.float64],
) -> NDArray[np.int64]:
    oriented: dict[tuple[int, int, int], list[NDArray[np.int64]]] = {}
    for tetrahedron in tetrahedra:
        a, b, c, d = (int(value) for value in tetrahedron)
        signed_volume = np.linalg.det(
            np.column_stack(
                (
                    vertices[b] - vertices[a],
                    vertices[c] - vertices[a],
                    vertices[d] - vertices[a],
                )
            )
        )
        if signed_volume < 0.0:
            a, b = b, a
        for face in (
            np.array((b, c, d), dtype=np.int64),
            np.array((a, d, c), dtype=np.int64),
            np.array((a, b, d), dtype=np.int64),
            np.array((a, c, b), dtype=np.int64),
        ):
            key = tuple(sorted(int(value) for value in face))
            oriented.setdefault(key, []).append(face)
    return np.asarray(
        [items[0] for items in oriented.values() if len(items) == 1],
        dtype=np.int64,
    )


def write_ascii_stl(
    vertices: NDArray[np.generic],
    elements: NDArray[np.generic],
    filename: str | Path,
    *,
    solid_name: str = "",
    connectivity_index_base: int,
    overwrite: bool = True,
) -> Path:
    """对应 iso2mesh ``savestl.m``，支持三角面或四面体输入"""

    points_raw = np.asarray(vertices)
    if np.iscomplexobj(points_raw):
        raise StlExportError("vertices 不能包含复数")
    points = np.asarray(points_raw, dtype=np.float64)
    cells_raw = np.asarray(elements)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        raise StlExportError("vertices 必须为非空 [节点数,至少3列]")
    points = points[:, :3]
    if not np.all(np.isfinite(points)):
        raise StlExportError("vertices 必须为有限实数")
    if cells_raw.ndim != 2 or cells_raw.shape[1] not in (3, 4) and cells_raw.shape[1] < 5:
        raise StlExportError("elements 必须为三角形或四面体连接矩阵")
    if cells_raw.shape[1] >= 5:
        cells_raw = cells_raw[:, :4]
    if connectivity_index_base not in (0, 1):
        raise StlExportError("connectivity_index_base 只能为 0 或 1")
    if not np.issubdtype(cells_raw.dtype, np.integer):
        if not np.all(np.isfinite(cells_raw)) or not np.all(cells_raw == np.floor(cells_raw)):
            raise StlExportError("elements 必须只包含有限整数")
    cells = np.asarray(cells_raw, dtype=np.int64) - connectivity_index_base
    if cells.size and (np.min(cells) < 0 or np.max(cells) >= points.shape[0]):
        raise StlExportError("elements 含有超出节点范围的索引")
    triangles = cells if cells.shape[1] == 3 else _boundary_faces(cells, points)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)

    if triangles.size:
        first = points[triangles[:, 0]]
        second = points[triangles[:, 1]]
        third = points[triangles[:, 2]]
        normals = np.cross(second - first, third - first)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0.0):
            raise StlExportError("表面包含退化三角形，无法计算单位法向量")
        normals /= lengths[:, None]
    else:
        normals = np.empty((0, 3), dtype=np.float64)

    name = str(solid_name).replace("\n", " ").replace("\r", " ")
    lines = [f"solid {name}"]
    for normal, triangle in zip(normals, triangles, strict=True):
        lines.append("facet normal " + " ".join(f"{value:e}" for value in normal))
        lines.append(" outer loop")
        for vertex_index in triangle:
            lines.append(
                "  vertex "
                + " ".join(f"{value:e}" for value in points[vertex_index])
            )
        lines.extend((" endloop", "endfacet"))
    lines.append(f"endsolid {name}")

    target = Path(filename).expanduser().resolve()
    if target.exists() and not overwrite:
        raise StlExportError(f"输出文件已存在，未获得覆盖许可：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return target
