"""iso2mesh CGAL 后端使用的 INR 与 MEDIT 交换格式"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class CgalMeshFormatError(ValueError):
    """CGALMesh 交换数据格式错误时抛出"""


@dataclass(frozen=True, slots=True)
class MeditMesh:
    """采用 Python 从 0 开始连接索引的 CGALMesh MEDIT 输出"""

    nodes: NDArray[np.float64]
    tetrahedra: NDArray[np.int64]
    tetrahedron_attributes: NDArray[np.int64]
    triangles: NDArray[np.int64]
    triangle_markers: NDArray[np.int64]


def write_inr_uint8(
    volume: NDArray[np.uint8],
    path: str | Path,
) -> Path:
    """写出 ``saveinr.m`` 使用的 uint8 INRIMAGE-4 表示"""

    array = np.asarray(volume)
    if array.ndim != 3 or any(length == 0 for length in array.shape):
        raise CgalMeshFormatError("INR 体数据必须是非空三维数组")
    if array.dtype != np.uint8:
        raise CgalMeshFormatError("INR 活动路径要求 uint8 体数据")

    xdim, ydim, zdim = array.shape
    prefix = (
        "#INRIMAGE-4#{\n"
        f"XDIM={xdim}\nYDIM={ydim}\nZDIM={zdim}\n"
        "VDIM=1\nTYPE=unsigned fixed\nPIXSIZE=8 bits\n"
        "CPU=decm\nVX=1\nVY=1\nVZ=1\n"
    ).encode("ascii")
    if len(prefix) > 252:
        raise CgalMeshFormatError("INR 头部超过 256 字节")
    header = prefix + b"\n" * (252 - len(prefix)) + b"##}\n"

    destination = Path(path)
    try:
        with destination.open("wb") as stream:
            stream.write(header)
            stream.write(np.asfortranarray(array).tobytes(order="F"))
    except OSError as exc:
        raise CgalMeshFormatError(f"无法写出 INR 文件：{destination}") from exc
    return destination


def _meaningful_lines(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise CgalMeshFormatError(f"无法读取 MEDIT 文件：{path}") from exc

    lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append((line_number, stripped))
    return lines


def _parse_integer(token: str, *, path: Path, line_number: int) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise CgalMeshFormatError(
            f"{path}:{line_number}: 预期整数，实际为 {token!r}"
        ) from exc


def _section_count(
    tokens: list[str],
    lines: list[tuple[int, str]],
    index: int,
    *,
    path: Path,
) -> tuple[int, int]:
    line_number = lines[index][0]
    if len(tokens) == 2:
        count = _parse_integer(tokens[1], path=path, line_number=line_number)
        next_index = index + 1
    elif len(tokens) == 1:
        if index + 1 >= len(lines):
            raise CgalMeshFormatError(
                f"{path}:{line_number}: 区段缺少记录数量"
            )
        count_line, count_text = lines[index + 1]
        count_tokens = count_text.split()
        if len(count_tokens) != 1:
            raise CgalMeshFormatError(
                f"{path}:{count_line}: 区段记录数量必须单独占一项"
            )
        count = _parse_integer(
            count_tokens[0], path=path, line_number=count_line
        )
        next_index = index + 2
    else:
        raise CgalMeshFormatError(
            f"{path}:{line_number}: 无效的 MEDIT 区段定义"
        )
    if count < 0:
        raise CgalMeshFormatError(
            f"{path}:{line_number}: 区段记录数量不能为负数"
        )
    return count, next_index


def _read_records(
    lines: list[tuple[int, str]],
    index: int,
    count: int,
    width: int,
    *,
    path: Path,
    section: str,
    integer: bool,
) -> tuple[list[list[float]] | list[list[int]], int]:
    if index + count > len(lines):
        raise CgalMeshFormatError(
            f"{path}: {section} 声明 {count} 条记录，但文件提前结束"
        )

    float_rows: list[list[float]] = []
    integer_rows: list[list[int]] = []
    for record_index in range(count):
        line_number, text = lines[index + record_index]
        tokens = text.split()
        if len(tokens) != width:
            raise CgalMeshFormatError(
                f"{path}:{line_number}: {section} 记录应有 {width} 列，"
                f"实际为 {len(tokens)} 列"
            )
        if integer:
            integer_rows.append(
                [
                    _parse_integer(token, path=path, line_number=line_number)
                    for token in tokens
                ]
            )
        else:
            try:
                float_rows.append([float(token) for token in tokens])
            except ValueError as exc:
                raise CgalMeshFormatError(
                    f"{path}:{line_number}: {section} 包含无效浮点数"
                ) from exc
    if integer:
        return integer_rows, index + count
    return float_rows, index + count


def _consume_header_scalar(
    tokens: list[str],
    lines: list[tuple[int, str]],
    index: int,
    *,
    path: Path,
) -> tuple[str, int]:
    line_number = lines[index][0]
    if len(tokens) == 2:
        return tokens[1], index + 1
    if len(tokens) != 1 or index + 1 >= len(lines):
        raise CgalMeshFormatError(
            f"{path}:{line_number}: 无效的 MEDIT 头部定义"
        )
    value_line, value_text = lines[index + 1]
    value_tokens = value_text.split()
    if len(value_tokens) != 1:
        raise CgalMeshFormatError(
            f"{path}:{value_line}: MEDIT 头部值必须单独占一项"
        )
    return value_tokens[0], index + 2


def read_medit_mesh(path: str | Path) -> MeditMesh:
    """读取 ``cgalmesh`` 输出的 ASCII MEDIT 各区段"""

    source = Path(path)
    lines = _meaningful_lines(source)
    if not lines:
        raise CgalMeshFormatError(f"MEDIT 文件为空：{source}")

    vertex_rows: list[list[float]] | None = None
    triangle_rows: list[list[int]] | None = None
    tetrahedron_rows: list[list[int]] | None = None
    index = 0
    saw_end = False
    while index < len(lines):
        line_number, text = lines[index]
        tokens = text.split()
        keyword = tokens[0]

        if keyword in {"MeshVersionFormatted", "Dimension"}:
            value, index = _consume_header_scalar(
                tokens, lines, index, path=source
            )
            if keyword == "Dimension" and value != "3":
                raise CgalMeshFormatError(
                    f"{source}:{line_number}: 仅支持三维 MEDIT 网格"
                )
            continue
        if keyword == "End":
            if len(tokens) != 1:
                raise CgalMeshFormatError(
                    f"{source}:{line_number}: End 后存在多余内容"
                )
            saw_end = True
            index += 1
            break
        if keyword not in {"Vertices", "Triangles", "Tetrahedra"}:
            raise CgalMeshFormatError(
                f"{source}:{line_number}: 不支持的 MEDIT 关键字 {keyword!r}"
            )

        count, records_index = _section_count(
            tokens, lines, index, path=source
        )
        if keyword == "Vertices":
            if vertex_rows is not None:
                raise CgalMeshFormatError(f"{source}: Vertices 区段重复")
            rows, index = _read_records(
                lines,
                records_index,
                count,
                4,
                path=source,
                section=keyword,
                integer=False,
            )
            vertex_rows = rows  # type: ignore[assignment]
        elif keyword == "Triangles":
            if triangle_rows is not None:
                raise CgalMeshFormatError(f"{source}: Triangles 区段重复")
            rows, index = _read_records(
                lines,
                records_index,
                count,
                4,
                path=source,
                section=keyword,
                integer=True,
            )
            triangle_rows = rows  # type: ignore[assignment]
        else:
            if tetrahedron_rows is not None:
                raise CgalMeshFormatError(f"{source}: Tetrahedra 区段重复")
            rows, index = _read_records(
                lines,
                records_index,
                count,
                5,
                path=source,
                section=keyword,
                integer=True,
            )
            tetrahedron_rows = rows  # type: ignore[assignment]

    if not saw_end:
        raise CgalMeshFormatError(f"{source}: 缺少 MEDIT End 标记")
    if index != len(lines):
        extra_line, _ = lines[index]
        raise CgalMeshFormatError(
            f"{source}:{extra_line}: End 后存在未解析内容"
        )
    if not vertex_rows:
        raise CgalMeshFormatError(f"{source}: 缺少非空 Vertices 区段")
    if not triangle_rows:
        raise CgalMeshFormatError(f"{source}: 缺少非空 Triangles 区段")
    if not tetrahedron_rows:
        raise CgalMeshFormatError(f"{source}: 缺少非空 Tetrahedra 区段")

    vertex_array = np.asarray(vertex_rows, dtype=np.float64)
    triangle_array = np.asarray(triangle_rows, dtype=np.int64)
    tetrahedron_array = np.asarray(tetrahedron_rows, dtype=np.int64)
    nodes = np.ascontiguousarray(vertex_array[:, :3], dtype=np.float64)
    triangles = np.ascontiguousarray(
        triangle_array[:, :3] - 1, dtype=np.int64
    )
    triangle_markers = np.ascontiguousarray(
        triangle_array[:, 3], dtype=np.int64
    )
    tetrahedra = np.ascontiguousarray(
        tetrahedron_array[:, :4] - 1, dtype=np.int64
    )
    tetrahedron_attributes = np.ascontiguousarray(
        tetrahedron_array[:, 4], dtype=np.int64
    )

    if not np.all(np.isfinite(nodes)):
        raise CgalMeshFormatError(f"{source}: Vertices 包含 NaN 或 Inf")
    for section, connectivity in (
        ("Triangles", triangles),
        ("Tetrahedra", tetrahedra),
    ):
        if int(connectivity.min()) < 0 or int(connectivity.max()) >= len(nodes):
            raise CgalMeshFormatError(
                f"{source}: {section} 包含超出 Vertices 范围的索引"
            )

    return MeditMesh(
        nodes=nodes,
        tetrahedra=tetrahedra,
        tetrahedron_attributes=tetrahedron_attributes,
        triangles=triangles,
        triangle_markers=triangle_markers,
    )
