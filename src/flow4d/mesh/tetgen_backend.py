"""当前 iso2mesh ``s2m`` 路径使用的 TetGen 可执行程序后端"""

from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray

from .backend import SurfaceMesh, TetrahedralMesh
from .iso2mesh_formats import (
    Iso2MeshFormatError,
    read_tetgen_output,
    write_tetgen_poly,
)


class TetGenBackendError(RuntimeError):
    """外部 TetGen 阶段无法完成时抛出"""


class TetGenTetrahedralizationBackend:
    """使用 ``s2m(..., 1, 1)`` 的参数运行 TetGen"""

    def __init__(
        self,
        executable_path: str | Path | None = None,
        *,
        timeout_seconds: float = 600.0,
    ) -> None:
        if executable_path is None:
            path = (
                Path(__file__).resolve().parents[1]
                / "native"
                / "tetgen"
                / "tetgen.exe"
            )
        else:
            path = Path(executable_path).expanduser().resolve()
        if not path.is_file():
            raise TetGenBackendError(
                f"未找到 TetGen 可执行文件；已检查：{path}"
            )
        if isinstance(timeout_seconds, (bool, np.bool_)):
            raise TetGenBackendError("timeout_seconds 必须为正有限数值")
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise TetGenBackendError("timeout_seconds 必须为正有限数值") from exc
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise TetGenBackendError("timeout_seconds 必须为正有限数值")

        self.executable_path = path
        self.timeout_seconds = timeout

    @staticmethod
    def _normalise_active_parameters(
        keep_ratio: float,
        max_volume: float,
    ) -> tuple[float, float]:
        if isinstance(keep_ratio, (bool, np.bool_)) or np.iscomplexobj(keep_ratio):
            raise TetGenBackendError("keep_ratio 必须为有限数值")
        if isinstance(max_volume, (bool, np.bool_)) or np.iscomplexobj(max_volume):
            raise TetGenBackendError("max_volume 必须为正有限数值")
        try:
            keep = float(keep_ratio)
            volume = float(max_volume)
        except (TypeError, ValueError) as exc:
            raise TetGenBackendError(
                "keep_ratio 和 max_volume 必须为数值"
            ) from exc
        if not np.isfinite(keep):
            raise TetGenBackendError("keep_ratio 必须为有限数值")
        if not np.isclose(keep, 1.0, rtol=0.0, atol=1e-12):
            raise TetGenBackendError(
                "当前仅支持 keep_ratio=1；meshresample/cgalsimp2 尚未迁移"
            )
        if not np.isfinite(volume) or volume <= 0.0:
            raise TetGenBackendError("max_volume 必须为正有限数值")
        return keep, volume

    def surface_to_tetrahedra(
        self,
        surface: SurfaceMesh,
        *,
        keep_ratio: float = 1.0,
        max_volume: float = 1.0,
        regions: NDArray[np.float64] | None = None,
        holes: NDArray[np.float64] | None = None,
    ) -> TetrahedralMesh:
        """在不改变 iso2mesh 质量参数的情况下生成四面体网格"""

        _, volume = self._normalise_active_parameters(keep_ratio, max_volume)
        volume_text = format(volume, ".17g")

        with TemporaryDirectory(prefix="flow4d-tetgen-") as temporary_directory:
            temporary_path = Path(temporary_directory)
            poly_path = temporary_path / "post_vmesh.poly"
            try:
                write_tetgen_poly(
                    poly_path,
                    surface,
                    holes=holes,
                    regions=regions,
                )
            except Iso2MeshFormatError as exc:
                raise TetGenBackendError(
                    f"TetGen 输入写出失败：{exc}"
                ) from exc

            command = [
                str(self.executable_path),
                "-A",
                f"-q1.414a{volume_text}",
                str(poly_path),
            ]
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                completed = subprocess.run(
                    command,
                    cwd=temporary_path,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                    creationflags=creation_flags,
                )
            except subprocess.TimeoutExpired as exc:
                raise TetGenBackendError(
                    f"TetGen 执行超过 {self.timeout_seconds:.17g} 秒"
                ) from exc
            except OSError as exc:
                raise TetGenBackendError(
                    f"无法启动 TetGen：{self.executable_path}；{exc}"
                ) from exc

            if completed.returncode != 0:
                stdout = completed.stdout[-4000:]
                stderr = completed.stderr[-4000:]
                raise TetGenBackendError(
                    "TetGen 执行失败"
                    f"（returncode={completed.returncode}）\n"
                    f"stdout：{stdout}\n"
                    f"stderr：{stderr}"
                )

            output_stub = temporary_path / "post_vmesh.1"
            expected_paths = [
                Path(f"{output_stub}.node"),
                Path(f"{output_stub}.ele"),
                Path(f"{output_stub}.face"),
            ]
            missing = [path.name for path in expected_paths if not path.is_file()]
            if missing:
                raise TetGenBackendError(
                    "TetGen 未生成所需输出文件：" + ", ".join(missing)
                )
            try:
                return read_tetgen_output(output_stub)
            except Iso2MeshFormatError as exc:
                raise TetGenBackendError(
                    f"TetGen 输出解析失败：{exc}"
                ) from exc
