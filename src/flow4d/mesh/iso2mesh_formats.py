"""当前 iso2mesh/TetGen 路径的忠实文本格式适配器"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .backend import SurfaceMesh, TetrahedralMesh


class Iso2MeshFormatError(RuntimeError):
    """iso2mesh/TetGen 网格或文本文件无效时抛出"""


def _normalise_nodes(
    values: NDArray[np.generic],
    *,
    name: str,
) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise Iso2MeshFormatError(f"{name} 不接受 complex 坐标")
    try:
        nodes = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Iso2MeshFormatError(f"{name} 必须为数值 N×3 数组") from exc
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0:
        raise Iso2MeshFormatError(f"{name} 必须为非空 N×3 数组")
    if not np.all(np.isfinite(nodes)):
        raise Iso2MeshFormatError(f"{name} 必须只包含有限坐标")
    return np.array(nodes, dtype=np.float64, copy=True)


def _normalise_integer_matrix(
    values: NDArray[np.generic],
    *,
    columns: int,
    name: str,
) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.integer):
        raise Iso2MeshFormatError(f"{name} 必须为整数数组")
    if raw.ndim != 2 or raw.shape[1] != columns or len(raw) == 0:
        raise Iso2MeshFormatError(f"{name} 必须为非空 N×{columns} 数组")
    return np.array(raw, dtype=np.int64, copy=True)


def _normalise_surface(
    surface: SurfaceMesh,
    surface_index: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64] | None,
]:
    nodes = _normalise_nodes(surface.nodes, name=f"第 {surface_index} 个表面节点")
    faces = _normalise_integer_matrix(
        surface.faces,
        columns=3,
        name=f"第 {surface_index} 个表面三角面",
    )
    if int(faces.min()) < 0 or int(faces.max()) >= len(nodes):
        raise Iso2MeshFormatError(
            f"第 {surface_index} 个表面三角面含有超出节点范围的索引"
        )

    if surface.face_markers is None:
        markers = None
    else:
        raw_markers = np.asarray(surface.face_markers)
        if np.iscomplexobj(raw_markers) or not np.issubdtype(
            raw_markers.dtype, np.integer
        ):
            raise Iso2MeshFormatError(
                f"第 {surface_index} 个表面的 face_markers 必须为整数数组"
            )
        if raw_markers.ndim != 1 or len(raw_markers) != len(faces):
            raise Iso2MeshFormatError(
                f"第 {surface_index} 个表面的 face_markers 数量与三角面不一致"
            )
        markers = np.array(raw_markers, dtype=np.int64, copy=True)
    return nodes, faces, markers


def _normalise_points(
    values: NDArray[np.generic] | Sequence[Sequence[float]] | None,
    *,
    name: str,
    allowed_columns: tuple[int, ...],
) -> NDArray[np.float64]:
    if values is None:
        return np.empty((0, allowed_columns[0]), dtype=np.float64)
    raw = np.asarray(values)
    if raw.size == 0:
        return np.empty((0, allowed_columns[0]), dtype=np.float64)
    if np.iscomplexobj(raw):
        raise Iso2MeshFormatError(f"{name} 不接受 complex 输入")
    try:
        points = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Iso2MeshFormatError(f"{name} 必须为数值二维数组") from exc
    if points.ndim != 2 or points.shape[1] not in allowed_columns:
        choices = " 或 ".join(str(value) for value in allowed_columns)
        raise Iso2MeshFormatError(f"{name} 必须具有 {choices} 列")
    if not np.all(np.isfinite(points)):
        raise Iso2MeshFormatError(f"{name} 必须只包含有限数值")
    if points.shape[1] == 4 and np.any(points[:, 3] <= 0.0):
        raise Iso2MeshFormatError(f"{name} 的第四列最大体积必须为正数")
    return np.array(points, dtype=np.float64, copy=True)


def merge_surface_meshes(*surfaces: SurfaceMesh) -> SurfaceMesh:
    """复现当前三角表面 ``mergemesh`` 的拼接行为"""

    if not surfaces:
        raise Iso2MeshFormatError("至少需要一个表面网格")
    node_blocks: list[NDArray[np.float64]] = []
    face_blocks: list[NDArray[np.int64]] = []
    marker_blocks: list[NDArray[np.int64]] = []
    node_offset = 0
    for surface_index, surface in enumerate(surfaces, start=1):
        nodes, faces, markers = _normalise_surface(surface, surface_index)
        node_blocks.append(nodes)
        face_blocks.append(faces + node_offset)
        marker_blocks.append(
            markers
            if markers is not None
            else np.full(len(faces), surface_index, dtype=np.int64)
        )
        node_offset += len(nodes)
    return SurfaceMesh(
        nodes=np.concatenate(node_blocks, axis=0),
        faces=np.concatenate(face_blocks, axis=0),
        face_markers=np.concatenate(marker_blocks, axis=0),
    )


def write_tetgen_poly(
    path: str | Path,
    surface: SurfaceMesh,
    *,
    holes: NDArray[np.generic] | Sequence[Sequence[float]] | None = None,
    regions: NDArray[np.generic] | Sequence[Sequence[float]] | None = None,
) -> Path:
    """写出 ``savesurfpoly.m`` 中闭合三角表面的子集格式"""

    target = Path(path)
    nodes, faces, markers = _normalise_surface(surface, 1)
    if markers is None:
        markers = np.ones(len(faces), dtype=np.int64)
    hole_points = _normalise_points(
        holes,
        name="holes",
        allowed_columns=(3,),
    )
    region_points = _normalise_points(
        regions,
        name="regions",
        allowed_columns=(3, 4),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("#node list\n")
            stream.write(f"{len(nodes)} 3 0 0\n")
            for node_id, (x, y, z) in enumerate(nodes):
                stream.write(f"{node_id} {x:.17g} {y:.17g} {z:.17g}\n")

            stream.write("#facet list\n")
            stream.write(f"{len(faces)} 1\n")
            for face, marker in zip(faces, markers, strict=True):
                stream.write(f"1 0 {int(marker)}\n")
                stream.write(
                    f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n"
                )

            stream.write("#hole list\n")
            stream.write(f"{len(hole_points)}\n")
            for hole_id, (x, y, z) in enumerate(hole_points, start=1):
                stream.write(f"{hole_id} {x:.17g} {y:.17g} {z:.17g}\n")

            stream.write("#region list\n")
            stream.write(f"{len(region_points)}\n")
            for region_id, region in enumerate(region_points, start=1):
                base = (
                    f"{region_id} {region[0]:.17g} {region[1]:.17g} "
                    f"{region[2]:.17g} {region_id}"
                )
                if region_points.shape[1] == 4:
                    base += f" {region[3]:.17g}"
                stream.write(base + "\n")
    except OSError as exc:
        raise Iso2MeshFormatError(f"无法写入 TetGen POLY 文件：{target}") from exc
    return target


def _read_data_lines(path: Path) -> list[tuple[int, list[str]]]:
    records: list[tuple[int, list[str]]] = []
    try:
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                data = line.split(b"#", 1)[0].strip()
                if not data:
                    continue
                try:
                    records.append((line_number, data.decode("ascii").split()))
                except UnicodeDecodeError as exc:
                    raise Iso2MeshFormatError(
                        f"{path}:{line_number} 的 TetGen 数值记录含非 ASCII 字节"
                    ) from exc
    except OSError as exc:
        raise Iso2MeshFormatError(f"无法读取 TetGen 文件：{path}") from exc
    if not records:
        raise Iso2MeshFormatError(f"TetGen 文件为空：{path}")
    return records


def _parse_int(token: str, path: Path, line_number: int, field: str) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise Iso2MeshFormatError(
            f"{path}:{line_number} 的 {field} 不是整数：{token!r}"
        ) from exc


def _parse_float(token: str, path: Path, line_number: int, field: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise Iso2MeshFormatError(
            f"{path}:{line_number} 的 {field} 不是数值：{token!r}"
        ) from exc
    if not np.isfinite(value):
        raise Iso2MeshFormatError(
            f"{path}:{line_number} 的 {field} 不是有限数值"
        )
    return value


def _require_field_count(
    tokens: list[str],
    expected: int,
    path: Path,
    line_number: int,
    record_type: str,
) -> None:
    if len(tokens) != expected:
        raise Iso2MeshFormatError(
            f"{path}:{line_number} 的 {record_type} 应有 {expected} 个字段，"
            f"实际为 {len(tokens)}"
        )


def _read_nodes(
    path: Path,
) -> tuple[NDArray[np.float64], dict[int, int]]:
    records = _read_data_lines(path)
    header_line, header = records[0]
    _require_field_count(header, 4, path, header_line, "NODE 头部")
    node_count = _parse_int(header[0], path, header_line, "节点数")
    dimension = _parse_int(header[1], path, header_line, "空间维数")
    attribute_count = _parse_int(header[2], path, header_line, "节点属性数")
    marker_flag = _parse_int(header[3], path, header_line, "节点 marker flag")
    if node_count <= 0:
        raise Iso2MeshFormatError(f"{path}:{header_line} 的节点数必须为正数")
    if dimension != 3 or attribute_count != 0 or marker_flag != 0:
        raise Iso2MeshFormatError(
            f"{path}:{header_line} 仅支持三维、零属性、零 marker 的 NODE 文件"
        )
    if len(records) != node_count + 1:
        raise Iso2MeshFormatError(
            f"{path} 声明 {node_count} 个节点，实际读取 {len(records) - 1} 个"
        )

    nodes = np.empty((node_count, 3), dtype=np.float64)
    node_id_to_row: dict[int, int] = {}
    for row, (line_number, tokens) in enumerate(records[1:]):
        _require_field_count(tokens, 4, path, line_number, "NODE 记录")
        node_id = _parse_int(tokens[0], path, line_number, "节点 ID")
        if node_id in node_id_to_row:
            raise Iso2MeshFormatError(
                f"{path}:{line_number} 出现重复节点 ID {node_id}"
            )
        node_id_to_row[node_id] = row
        nodes[row] = [
            _parse_float(tokens[1], path, line_number, "x"),
            _parse_float(tokens[2], path, line_number, "y"),
            _parse_float(tokens[3], path, line_number, "z"),
        ]
    return nodes, node_id_to_row


def _map_node_id(
    token: str,
    node_id_to_row: dict[int, int],
    path: Path,
    line_number: int,
) -> int:
    node_id = _parse_int(token, path, line_number, "连接节点 ID")
    try:
        return node_id_to_row[node_id]
    except KeyError as exc:
        raise Iso2MeshFormatError(
            f"{path}:{line_number} 引用了未知节点 ID {node_id}"
        ) from exc


def _read_elements(
    path: Path,
    node_id_to_row: dict[int, int],
) -> tuple[NDArray[np.int64], NDArray[np.int64] | None]:
    records = _read_data_lines(path)
    header_line, header = records[0]
    _require_field_count(header, 3, path, header_line, "ELE 头部")
    element_count = _parse_int(header[0], path, header_line, "四面体数")
    nodes_per_element = _parse_int(header[1], path, header_line, "每单元节点数")
    attribute_count = _parse_int(header[2], path, header_line, "单元属性数")
    if element_count <= 0 or nodes_per_element != 4 or attribute_count < 0:
        raise Iso2MeshFormatError(
            f"{path}:{header_line} 的 ELE 头部不符合四节点四面体格式"
        )
    if len(records) != element_count + 1:
        raise Iso2MeshFormatError(
            f"{path} 声明 {element_count} 个四面体，"
            f"实际读取 {len(records) - 1} 个"
        )

    elements = np.empty((element_count, 4), dtype=np.int64)
    attributes = (
        np.empty((element_count, attribute_count), dtype=np.int64)
        if attribute_count
        else None
    )
    seen_ids: set[int] = set()
    expected_fields = 1 + nodes_per_element + attribute_count
    for row, (line_number, tokens) in enumerate(records[1:]):
        _require_field_count(tokens, expected_fields, path, line_number, "ELE 记录")
        element_id = _parse_int(tokens[0], path, line_number, "四面体 ID")
        if element_id in seen_ids:
            raise Iso2MeshFormatError(
                f"{path}:{line_number} 出现重复四面体 ID {element_id}"
            )
        seen_ids.add(element_id)
        elements[row] = [
            _map_node_id(token, node_id_to_row, path, line_number)
            for token in tokens[1:5]
        ]
        if attributes is not None:
            attributes[row] = [
                _parse_int(token, path, line_number, "四面体属性")
                for token in tokens[5:]
            ]
    return elements, attributes


def _read_faces(
    path: Path,
    node_id_to_row: dict[int, int],
) -> tuple[NDArray[np.int64], NDArray[np.int64] | None]:
    records = _read_data_lines(path)
    header_line, header = records[0]
    _require_field_count(header, 2, path, header_line, "FACE 头部")
    face_count = _parse_int(header[0], path, header_line, "边界面数")
    marker_flag = _parse_int(header[1], path, header_line, "边界 marker flag")
    if face_count < 0 or marker_flag not in (0, 1):
        raise Iso2MeshFormatError(f"{path}:{header_line} 的 FACE 头部无效")
    if len(records) != face_count + 1:
        raise Iso2MeshFormatError(
            f"{path} 声明 {face_count} 个边界面，"
            f"实际读取 {len(records) - 1} 个"
        )

    faces = np.empty((face_count, 3), dtype=np.int64)
    markers = np.empty(face_count, dtype=np.int64) if marker_flag else None
    seen_ids: set[int] = set()
    expected_fields = 4 + marker_flag
    for row, (line_number, tokens) in enumerate(records[1:]):
        _require_field_count(tokens, expected_fields, path, line_number, "FACE 记录")
        face_id = _parse_int(tokens[0], path, line_number, "边界面 ID")
        if face_id in seen_ids:
            raise Iso2MeshFormatError(
                f"{path}:{line_number} 出现重复边界面 ID {face_id}"
            )
        seen_ids.add(face_id)
        faces[row] = [
            _map_node_id(token, node_id_to_row, path, line_number)
            for token in tokens[1:4]
        ]
        if markers is not None:
            markers[row] = _parse_int(tokens[4], path, line_number, "边界 marker")
    return faces, markers


def _reorient_tetrahedra(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    source_path: Path,
) -> NDArray[np.int64]:
    oriented = np.array(elements, dtype=np.int64, copy=True)
    a, b, c, d = (oriented[:, index] for index in range(4))
    signed_six_volume = -np.einsum(
        "ij,ij->i",
        np.cross(nodes[b] - nodes[a], nodes[c] - nodes[a]),
        nodes[d] - nodes[a],
    )
    if np.any(signed_six_volume == 0.0):
        count = int(np.count_nonzero(signed_six_volume == 0.0))
        raise Iso2MeshFormatError(
            f"{source_path} 含有 {count} 个零体积四面体"
        )
    negative = signed_six_volume < 0.0
    if np.any(negative):
        temporary = oriented[negative, 2].copy()
        oriented[negative, 2] = oriented[negative, 3]
        oriented[negative, 3] = temporary
    return oriented


def read_tetgen_output(stub: str | Path) -> TetrahedralMesh:
    """读取并规范化一组 TetGen ``.node/.ele/.face`` 输出"""

    base = Path(stub)
    node_path = Path(f"{base}.node")
    element_path = Path(f"{base}.ele")
    face_path = Path(f"{base}.face")
    nodes, node_id_to_row = _read_nodes(node_path)
    elements, attributes = _read_elements(element_path, node_id_to_row)
    boundary_faces, boundary_markers = _read_faces(face_path, node_id_to_row)
    elements = _reorient_tetrahedra(nodes, elements, element_path)
    return TetrahedralMesh(
        nodes=nodes,
        elements=elements,
        element_attributes=attributes,
        boundary_faces=boundary_faces,
        boundary_markers=boundary_markers,
    )
