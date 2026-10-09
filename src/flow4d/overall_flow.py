"""把已迁移模块串联为可逐步检查的命令行与 PySide 流程"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import traceback
from time import perf_counter, sleep, time
from typing import Any, Iterator

import numpy as np
from pydicom import dcmread

if not __package__:
    # 允许用 `python path/to/overall_flow.py` 启动同一份入口。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "flow4d"

from .base_function import RuntimeTimer, TimingRecord
from .base_function.dicom_series import inspect_dicom_series
from .base_function.mask_directories import MaskDirectoryInventory, prepare_mask_directories
from .dicom.case_loader import LoadedDicomCase, load_dicom_case
from .dicom.mask_loader import MaskVolumeResult, _load_stack, load_mask_volumes
from .final_pipeline import run_final_postprocessing
from .exporters.matlab_tecplot import export_matlab_dat_plt_series
from .mesh.cgalmesh_backend import CgalMeshSurfaceBackend
from .mesh.coordinates import scale_tetrahedral_nodes
from .mesh.pipeline import run_mesh_pipeline
from .mesh.tetgen_backend import TetGenTetrahedralizationBackend
from .processing.dfw_backend import DfwNativeBackend
from .processing.dfw_pipeline import denoise_velocity_dfw_timeseries
from .processing.msac import _matlab_uint16, msac_correct_4d
from .processing.phase_boundary import apply_phase_boundary_correction
from .processing.segmentation import generate_segmentation_dicoms
from .processing.unwrap_pipeline import run_unwrap_pipeline
from .processing.velocity import VelocityConversionResult, convert_case_to_velocity
from .processing.velocity_smoothing import smooth_velocity_gaussian


STEPS = (
    ("00", "检查输入与已有结果"),
    ("01", "读取 DICOM"),
    ("02", "相位边界归零"),
    ("03", "生成速度矩阵"),
    ("04", "MSAC 背景校正"),
    ("05", "导出分割辅助图像"),
    ("06", "导入 mask/root"),
    ("07", "自动解混叠和速度方向"),
    ("08", "裁剪 Mask 并生成网格"),
    ("09", "Gaussian 与 DFW"),
    ("10", "节点结果、压力与 Liutex"),
    ("11", "核对最终 VTK 导出"),
    ("12", "导出 UVW DAT 与 FEMbrick PLT"),
)

STEP_GUIDANCE = {
    "00": ("核对病例目录与已有结果", "四方向 DICOM；可选已有 PCA、mask/root", "文件数量与复用决定"),
    "01": ("读取原始图像和采集参数", "四方向同一病例 DICOM", "图像形状、VENC、体素和时间信息"),
    "02": ("修正相位图像边界", "已读取的相位图像", "边界校正后的病例"),
    "03": ("按 RL/FH/AP 生成速度", "校正后的相位图像", "速度矩阵及方向符号"),
    "04": ("校正背景相位；已有 PCA 且选原始数据时可跳过", "速度与 Magnitude 图像", "MSAC 校正速度及相位"),
    "05": ("生成分割参考图；完整 results_PCA 可复用", "速度及图像；或已有逐层 PCA DICOM", "results_PCA、results_PCA2、results_R1"),
    "06": ("导入外部分割", "病例目录下 mask 有完整逐层 DICOM；root 可为空或有同样层数", "二值血管 Mask；空 root 按 0 处理"),
    "07": ("自动解混叠并应用选定的方向", "原始或 MSAC 相位、方向勾选", "校正后的三方向速度及 wrap 数"),
    "08": ("根据 Mask 裁剪并建立四面体网格", "Mask、速度、CGALMesh 和 TetGen", "表面 STL、网格节点与四面体"),
    "09": ("对速度做 Gaussian 和 DFW 处理", "裁剪速度、Mask 和 DFW DLL", "平滑后的速度场"),
    "10": ("采样节点并计算压力与 Liutex", "网格、速度、时间和体素间距", "节点结果；内部子步骤逐项计时"),
    "11": ("检查导出的时间序列", "节点结果和四面体", "各时相 VTK 与 series 索引"),
    "12": ("保存 Tecplot 结果", "同一次运行的节点速度、FINAL 27 列和四面体", "dat 文件夹下每时相一个 UVW_*.dat 与 FEMbrick_*.plt"),
}


def _array_info(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _timings(result: Any) -> list[dict[str, Any]]:
    return [
        {"name": record.step_name, "seconds": round(record.elapsed_seconds, 3)}
        for record in getattr(result, "timings", ())
    ]


def _sampling_boundary_info(
    nodes_mm: np.ndarray,
    volume_shape: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
) -> dict[str, Any]:
    """报告进入节点采样前的网格边界；有限越界点按 MATLAB 样条外推。"""
    spacing = np.asarray(voxel_size_mm, dtype=np.float64)
    indices = np.column_stack((
        nodes_mm[:, 1] / spacing[0],
        nodes_mm[:, 0] / spacing[1],
        nodes_mm[:, 2] / spacing[2],
    ))
    upper = np.asarray(volume_shape, dtype=np.float64) - 1.0
    outside = np.any((indices < 0.0) | (indices > upper), axis=1)
    overhang = np.maximum.reduce((np.zeros_like(indices), -indices, indices - upper))
    return {
        "extrapolated_nodes": int(np.count_nonzero(outside)),
        "maximum_overhang_voxels": round(float(np.max(overhang)), 6),
    }

@dataclass(frozen=True, slots=True)
class _MsacStageResult:
    corrected_case: LoadedDicomCase
    velocity: np.ndarray
    timings: tuple[TimingRecord, ...]
    total_elapsed_seconds: float


def _run_msac_low_memory(
    case: LoadedDicomCase,
    velocity_result: VelocityConversionResult,
    *,
    reencode_phase: bool,
) -> _MsacStageResult:
    """复用速度数组，并逐时相执行与 MSAC 包装函数相同的重新编码。"""

    if case.venc == 0 or case.rescale_slope == 0:
        raise ValueError("MSAC 要求非零 VENC 和 RescaleSlope")
    timer = RuntimeTimer()
    with timer.measure("归一化 MSAC 输入"):
        phase_images = velocity_result.velocity
        np.divide(phase_images, case.venc, out=phase_images)
        center_column = (velocity_result.intensity.shape[1] - 1) // 2
        maximum = float(np.max(velocity_result.intensity[:, center_column, :, :]))
        if not np.isfinite(maximum) or maximum == 0:
            raise ValueError("Magnitude 中心切片最大值必须是有限非零数值")
        normalized_magnitude = velocity_result.intensity / maximum

    with timer.measure("执行 MSAC 背景相位校正"):
        core = msac_correct_4d(
            phase_images,
            normalized_magnitude,
            msac_fit_order=1,
            correction_fit_order=3,
        )
    del normalized_magnitude

    with timer.measure("恢复校正速度与逐时相相位像素"):
        corrected_velocity = core.corrected_time_resolved_phase
        np.multiply(corrected_velocity, case.venc, out=corrected_velocity)
        if reencode_phase:
            corrected_components: dict[str, np.ndarray] = {}
            for label, component_index in (("ap", 2), ("fh", 1), ("rl", 0)):
                encoded = np.empty(case.shape, dtype=np.uint16)
                for phase_index in range(case.cardiac_phases):
                    component = corrected_velocity[:, :, :, phase_index, component_index]
                    scaled = (component - case.rescale_intercept) / case.rescale_slope
                    encoded[:, :, :, phase_index] = _matlab_uint16(scaled)
                corrected_components[label] = encoded
            corrected_case = replace(case, **corrected_components)
        else:
            corrected_case = case

    records = (*timer.records, *core.timings)
    return _MsacStageResult(
        corrected_case=corrected_case,
        velocity=corrected_velocity,
        timings=records,
        total_elapsed_seconds=timer.total_elapsed_seconds,
    )


class ProgressReport:
    """逐步落盘状态；失败时也保留已完成步骤"""

    def __init__(self, output: Path, selections: dict[str, Any]) -> None:
        self.output = output
        self.path = output / "overall_flow_status.json"
        self.data: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selections": selections,
            "steps": [
                {"number": number, "name": name, "status": "not_started"}
                for number, name in STEPS
            ],
        }
        self._write()

    def _write(self) -> None:
        payload = json.dumps(self.data, ensure_ascii=False, indent=2)
        temporary: Path | None = None
        try:
            # A distinct, closed file avoids collisions and preserves atomic reads.
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.output,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            for attempt in range(6):
                try:
                    temporary.replace(self.path)
                    break
                except OSError as error:
                    # Windows readers can briefly deny deletion/rename sharing.
                    if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 5:
                        raise
                    sleep(0.05 * 2 ** attempt)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    # Do not mask a write failure with a cleanup failure.
                    pass

    @contextmanager
    def step(self, number: str) -> Iterator[dict[str, Any]]:
        item = next(step for step in self.data["steps"] if step["number"] == number)
        item["status"] = "running"
        item["started_utc"] = datetime.now(timezone.utc).isoformat()
        self._write()
        print(f"[{number}] {item['name']}：运行中", flush=True)
        purpose, required, produced = STEP_GUIDANCE[number]
        print(f"    目的：{purpose}；需要：{required}；完成后：{produced}", flush=True)
        started = perf_counter()
        try:
            yield item
        except (KeyboardInterrupt, EOFError) as error:
            item["status"] = "interrupted"
            item["error"] = type(error).__name__
            raise
        except Exception as error:
            item["status"] = "failed"
            item["error"] = f"{type(error).__name__}: {error}"
            item["traceback"] = traceback.format_exc()
            raise
        else:
            item["status"] = "completed"
        finally:
            item["elapsed_seconds"] = round(perf_counter() - started, 3)
            item["finished_utc"] = datetime.now(timezone.utc).isoformat()
            self._write()
            print(
                f"[{number}] {item['name']}：{item['status']}，"
                f"{item['elapsed_seconds']:.3f} 秒",
                flush=True,
            )
            if item["status"] == "completed" and item.get("output"):
                print(f"    输出：{json.dumps(item['output'], ensure_ascii=False)}", flush=True)

    def skip(self, number: str, reason: str, output: dict[str, Any] | None = None) -> None:
        item = next(step for step in self.data["steps"] if step["number"] == number)
        item.update(status="skipped", reason=reason, elapsed_seconds=0.0,
                    finished_utc=datetime.now(timezone.utc).isoformat())
        if output is not None:
            item["output"] = output
        self._write()
        print(f"[{number}] {item['name']}：已跳过。{reason}", flush=True)

    def wait_for_input(self, number: str, message: str) -> None:
        item = next(step for step in self.data["steps"] if step["number"] == number)
        item["wait_attempts"] = item.get("wait_attempts", 0) + 1
        item.update(status="waiting_input", reason=message, elapsed_seconds=0.0,
                    finished_utc=datetime.now(timezone.utc).isoformat())
        self._write()
        print(f"[{number}] {item['name']}：等待用户操作。{message}", flush=True)

    def input_ready(self, number: str) -> None:
        item = next(step for step in self.data["steps"] if step["number"] == number)
        item["status"] = "not_started"
        for key in ("reason", "elapsed_seconds", "finished_utc"):
            item.pop(key, None)
        self._write()

    def save_array(self, number: str, name: str, array: np.ndarray) -> str:
        checkpoint_directory = self.output / "checkpoints"
        checkpoint_directory.mkdir(exist_ok=True)
        target = checkpoint_directory / f"{number}_{name}.npy"
        np.save(target, array, allow_pickle=False)
        return str(target)


def _mark_unfinished_run(output: Path, exit_code: int) -> bool:
    """A vanished worker must not leave the last stage displayed as running."""
    path = output / "overall_flow_status.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc)
        changed = False
        for step in data.get("steps", ()):
            if step.get("status") != "running":
                continue
            step["status"] = "failed"
            step["error"] = f"计算子进程已退出（退出码 {exit_code}），此步骤未完成；请查看日志和系统错误记录"
            step["finished_utc"] = now.isoformat()
            if step.get("started_utc"):
                try:
                    elapsed = (now - datetime.fromisoformat(step["started_utc"])).total_seconds()
                    step["elapsed_seconds"] = round(max(0.0, elapsed), 3)
                except ValueError:
                    pass
            changed = True
        if changed:
            temporary = path.with_name(path.name + ".exit.tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        return changed
    except (OSError, ValueError, TypeError):
        return False


def _prompt_path(value: str | None, label: str) -> Path:
    selected = value if value is not None else input(f"{label}：").strip().strip('"')
    if not selected:
        raise ValueError(f"{label}不能为空")
    return Path(selected).expanduser().resolve()


def _output_path(value: str | None, case_directory: Path) -> Path:
    if value is None:
        selected = input(
            f"输出病例目录（直接回车使用输入目录 {case_directory}）："
        ).strip().strip('"')
        if not selected:
            return case_directory
    else:
        selected = value
    return Path(selected).expanduser().resolve()


def _choice(value: str | None, label: str, choices: tuple[str, ...]) -> str:
    while True:
        selected = value if value is not None else input(
            f"{label}（{'/'.join(choices)}）："
        )
        normalized = selected.strip().lower()
        if normalized in choices:
            return normalized
        message = f"{label}须选择 {'/'.join(choices)}"
        if value is not None:
            raise ValueError(message)
        print(message, file=sys.stderr)


def _flips(value: str | None) -> tuple[str, ...]:
    while True:
        selected = value if value is not None else input(
            "确认解剖方向后，输入需反转的速度分量（如 RL,FH；都不反转直接回车）："
        )
        normalized = selected.strip()
        if not normalized or normalized.lower() == "none":
            return ()
        components = tuple(part.strip().upper() for part in normalized.split(","))
        if len(set(components)) == len(components) and all(
            part in {"RL", "FH", "AP"} for part in components
        ):
            return components
        message = "反转分量只能是 none 或不重复的 RL,FH,AP 组合"
        if value is not None:
            raise ValueError(message)
        print(message, file=sys.stderr)

def _inspect(case_directory: Path, format_name: str | None) -> dict[str, Any]:
    if not case_directory.is_dir() or case_directory.name != "DICOM_4D_Qflow":
        raise ValueError("请选择名为 DICOM_4D_Qflow 的外层病例目录")
    inventory = inspect_dicom_series(case_directory)
    counts = {name: len(files) for name, files in inventory.series}
    if format_name == "classic" and not inventory.ready_for_metadata_reading:
        raise ValueError(inventory.inspection_message)
    if format_name == "enhanced" and any(count != 1 for count in counts.values()):
        raise ValueError("Enhanced 模式要求四个方向目录各有 1 个多帧 DICOM")
    mask_counts = {}
    for name in ("mask", "root"):
        directory = case_directory / name
        mask_counts[name] = (
            sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".dcm")
            if directory.is_dir()
            else 0
        )
    pca_directory = case_directory / "results_PCA"
    pca_count = _dicom_count(pca_directory)
    expected_slices = None
    image_shape = None
    if format_name in {"classic", "enhanced"} and inventory.ap_files:
        header = dcmread(inventory.ap_files[0], stop_before_pixels=True)
        phase_element = header.get((0x2001, 0x1017))
        try:
            phases = (int(phase_element.value) if phase_element is not None
                      else 25 if format_name == "classic" else 0)
            image_count = (len(inventory.ap_files) if format_name == "classic"
                           else int(header.NumberOfFrames))
            if phases > 0 and image_count % phases == 0:
                expected_slices = image_count // phases
                image_shape = [int(header.Rows), int(header.Columns), expected_slices]
        except (AttributeError, TypeError, ValueError):
            pass
    return {"dicom_counts": counts, "mask_counts": mask_counts,
            "pca_count": pca_count, "pca_directory": str(pca_directory),
            "expected_slices_from_headers": expected_slices,
            "image_shape_from_headers": image_shape}


def _dicom_count(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir()
               if path.is_file() and path.suffix.casefold() == ".dcm")


def _reuse_decision(
    case_directory: Path, output: Path, expected_slices: int, source: str,
    *, force_pca: bool = False,
) -> dict[str, Any]:
    """Only complete slice stacks may bypass segmentation or Mask preparation."""
    if expected_slices < 1:
        raise ValueError("病例必须至少有一层图像")
    if source not in {"raw", "msac"}:
        raise ValueError("速度来源须为 raw 或 msac")
    pca_count = _dicom_count(output / "results_PCA")
    mask_count = _dicom_count(case_directory / "mask")
    root_count = _dicom_count(case_directory / "root")
    reuse_pca = pca_count == expected_slices and not force_pca
    return {
        "expected_slices": expected_slices,
        "pca_count": pca_count,
        "mask_count": mask_count,
        "root_count": root_count,
        "reuse_pca": reuse_pca,
        "mask_ready": mask_count == expected_slices and root_count in (0, expected_slices),
        "root_mode": "zero" if root_count == 0 else "dicom" if root_count == expected_slices else "incomplete",
        "run_msac": source == "msac" or not reuse_pca,
        "pca_directory": str(output / "results_PCA"),
        "mask_directory": str(case_directory / "mask"),
        "root_directory": str(case_directory / "root"),
    }


def _load_mask_with_optional_root(
    case: LoadedDicomCase, inventory: MaskDirectoryInventory,
) -> MaskVolumeResult:
    """An empty root directory means a zero volume; complete root DICOMs keep subtraction."""
    if inventory.root_files:
        return load_mask_volumes(case, inventory)
    timer = RuntimeTimer()
    with timer.measure("读取 mask DICOM"):
        mask_source = _load_stack(inventory.mask_files, case.shape[:3], "mask")
    with timer.measure("root=0"):
        root_source = np.zeros(case.shape[:3], dtype=np.float64)
    with timer.measure("生成二值血管蒙版"):
        mask = np.asarray(mask_source != 0, dtype=np.uint16)
    return MaskVolumeResult(
        mask_source=mask_source, root_source=root_source, mask=mask,
        timings=timer.records, total_elapsed_seconds=timer.total_elapsed_seconds,
    )


def _preflight_zero_root_mask(
    case_directory: Path, expected_shape: tuple[int, int, int],
) -> dict[str, int]:
    """Reject a zero-root Mask that would turn the entire field of view into a mesh."""
    inventory = prepare_mask_directories(case_directory)
    mask_source = _load_stack(inventory.mask_files, expected_shape, "mask")
    foreground = int(np.count_nonzero(mask_source))
    background = int(mask_source.size - foreground)
    if background == 0:
        raise ValueError(
            "root=0 但 mask 的所有像素全部非零，整个图像都会被当成血管，"
            "不能用于建网格。请提供与 mask 配套的逐层 root DICOM，"
            "或重新导出背景为 0 的二值 mask。"
        )
    if foreground == 0:
        raise ValueError("mask 全部为 0，没有可用于建网格的血管区域")
    return {"foreground_voxels": foreground, "background_voxels": background}


def _require_output(case_directory: Path, output: Path) -> None:
    if output.name != "DICOM_4D_Qflow":
        raise ValueError("输出目录末级名称必须是 DICOM_4D_Qflow")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"输出路径不是文件夹：{output}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")


def _iso2mesh_tool(stem: str) -> Path | None:
    bin_directory = (
        Path(__file__).resolve().parents[3]
        / "main_AB4Dflow_matlab"
        / "iso2mesh-master"
        / "bin"
    )
    for filename in (f"{stem}_x86-64.exe", f"{stem}.exe"):
        candidate = bin_directory / filename
        if candidate.is_file():
            return candidate
    return None


def _select_iso2mesh_tool(value: str | None, stem: str, label: str) -> Path:
    if value is not None:
        path = Path(value).expanduser().resolve()
    else:
        path = _iso2mesh_tool(stem)
        if path is None:
            raise FileNotFoundError(
                f"未自动找到 {label}。请确认项目 iso2mesh-master/bin 中的程序存在。"
            )
        else:
            print(f"自动找到 {label}：{path}", flush=True)
    _require_file(path, label)
    return path


def run(args: argparse.Namespace) -> int:
    case_directory = _prompt_path(args.case, "选择外层 DICOM_4D_Qflow 病例目录")
    format_name = None if args.check_only and args.format is None else _choice(
        args.format, "DICOM 格式", ("classic", "enhanced")
    )
    inspection = _inspect(case_directory, format_name)
    if args.check_only:
        for label, value in (("CGALMesh", args.cgalmesh), ("TetGen", args.tetgen), ("DFW DLL", args.dfw_library)):
            if value is not None:
                _require_file(Path(value).expanduser().resolve(), label)
        inspection["auto_detected_tools"] = {
            stem: str(path) if (path := _iso2mesh_tool(stem)) is not None else None
            for stem in ("cgalmesh", "tetgen")
        }
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        print("只完成目录检查；没有读取像素或写入文件")
        return 0

    source = _choice(args.source, "解混叠使用的数据", ("raw", "msac"))
    flips = _flips(args.flip)
    output = _output_path(args.output, case_directory)
    _require_output(case_directory, output)
    print(f"结果写入：{output}；现有同名结果文件会被覆盖。", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report = ProgressReport(output, {
        "input_case": str(case_directory), "format": format_name,
        "source": source, "flip": list(flips), "output": str(output),
        "cgalmesh": args.cgalmesh, "tetgen": args.tetgen,
        "dfw_library": args.dfw_library, "save_arrays": args.save_arrays,
        "regenerate_pca": getattr(args, "regenerate_pca", False),
        "overwrite_same_named_outputs": True,
    })

    with report.step("00") as item:
        item["output"] = inspection
        shape = inspection.get("image_shape_from_headers")
        if (shape is not None and
                _dicom_count(case_directory / "mask") == shape[2] and
                _dicom_count(case_directory / "root") == 0):
            item["output"]["zero_root_mask_check"] = _preflight_zero_root_mask(
                case_directory, tuple(shape)
            )
    series_started: dict[str, float] = {}
    series_seconds: dict[str, float] = {}

    def record_dicom_progress(name: str, completed: int, total: int) -> None:
        now = perf_counter()
        if completed == 1:
            series_started[name] = now
        if completed == total:
            series_seconds[name] = round(now - series_started[name], 3)
            print(f"    DICOM {name}：{total} 张，{series_seconds[name]:.3f} 秒", flush=True)

    with report.step("01") as item:
        case = load_dicom_case(
            case_directory, is_enhanced=format_name == "enhanced",
            progress_callback=record_dicom_progress,
        )
        if case.cardiac_phases < 19:
            raise ValueError("最终后处理使用第 19 时相，当前病例不足 19 时相")
        item["output"] = {
            "shape": list(case.shape), "venc": case.venc,
            "voxel_size_mm": list(case.voxel_size_mm),
            "time_spacing_ms": case.time_spacing_ms, "timings": _timings(case),
            "classic_series_seconds": series_seconds,
        }
    reuse = _reuse_decision(
        case_directory, output, case.shape[2], source,
        force_pca=getattr(args, "regenerate_pca", False),
    )
    report.data["selections"]["existing_results"] = reuse
    report._write()
    print(f"已有结果检查：{json.dumps(reuse, ensure_ascii=False)}", flush=True)
    with report.step("02") as item:
        boundary = apply_phase_boundary_correction(case)
        corrected_case = boundary.corrected_case
        case_phases = case.cardiac_phases
        case_voxel_size = case.voxel_size_mm
        case_time_spacing = case.time_spacing_ms
        item["output"] = {
            "corrected_voxels_per_component": boundary.corrected_voxels_per_component,
            "ap": _array_info(corrected_case.ap), "timings": _timings(boundary),
        }
    del case, boundary
    with report.step("03") as item:
        velocity_result = convert_case_to_velocity(corrected_case)
        item["output"] = {
            "velocity": _array_info(velocity_result.velocity),
            "component_order": ["RL", "FH", "AP"],
            "orientation_signs": list(velocity_result.orientation_signs),
            "timings": _timings(velocity_result),
        }
        if args.save_arrays:
            item["output"]["checkpoint"] = report.save_array("03", "velocity", velocity_result.velocity)
    msac = None
    if reuse["run_msac"]:
        with report.step("04") as item:
            msac = _run_msac_low_memory(
                corrected_case, velocity_result, reencode_phase=source == "msac"
            )
            velocity_result = replace(velocity_result, velocity=msac.velocity)
            item["output"] = {"velocity": _array_info(msac.velocity), "timings": _timings(msac)}
            if args.save_arrays:
                item["output"]["checkpoint"] = report.save_array("04", "msac_velocity", msac.velocity)
    else:
        report.skip("04", "选择原始速度且已有完整 PCA，后续无需 MSAC")
    unwrap_case = corrected_case if source == "raw" else msac.corrected_case
    del corrected_case
    if reuse["reuse_pca"]:
        report.skip("05", "已有与病例层数一致的 PCA DICOM，不覆盖", {
            "directory": reuse["pca_directory"], "files": reuse["pca_count"],
        })
    else:
        with report.step("05") as item:
            segmentation = generate_segmentation_dicoms(
                unwrap_case, velocity_result, case_directory=output,
                peak_phase_number=min(15, case_phases),
            )
            exports = (
                segmentation.pcmra_gamma_02_export,
                segmentation.pcmra_gamma_05_export,
                segmentation.r1_export,
            )
            item["output"] = {
                "series": [{"directory": str(part.output_directory), "files": len(part.files)} for part in exports],
                "timings": _timings(segmentation),
            }
        del segmentation
    del velocity_result, msac
    # 分割图已导出；等待 Mask 时仅保留后续解混叠所需的相位数据。
    unwrap_case = replace(unwrap_case, classic_volumes=None, enhanced_data=None)
    while True:
        current = _reuse_decision(case_directory, output, reuse["expected_slices"], source)
        if current["mask_ready"]:
            report.input_ready("06")
            break
        prepare_mask_directories(case_directory)
        next_action = (
            "保持此窗口打开；完成外部分割后点击“继续”，会从 Mask 导入步骤接着运行。"
            if getattr(args, "wait_for_mask", False) else
            "命令行本次执行将结束；完成外部分割后重新运行，会重新读取原始 DICOM。"
        )
        message = (
            f"请在默认目录 {case_directory / 'mask'} 放入 {reuse['expected_slices']} 张对应切片的 DICOM；"
            f"{case_directory / 'root'} 可为空（按 0 处理）或放入相同张数。"
            f"当前 mask、root 分别为 {current['mask_count']}、{current['root_count']} 张。"
            f"{next_action}"
        )
        report.wait_for_input("06", message)
        print(f"分割参考图：{output / 'results_PCA'}", flush=True)
        print(f"逐步状态：{report.path}", flush=True)
        if not getattr(args, "wait_for_mask", False):
            print("命令行本次执行已结束；如需不中断地等待，可加 --wait-for-mask。", flush=True)
            return 3
        print("等待界面的继续指令；此时原始 DICOM 和速度保留在内存中。", flush=True)
        command = sys.stdin.readline()
        if not command:
            raise EOFError("等待 Mask 时界面与计算进程的连接已断开")
        if command.strip() != "continue":
            print("未收到有效的继续指令，仍在等待 Mask。", flush=True)
    cgalmesh = _select_iso2mesh_tool(args.cgalmesh, "cgalmesh", "CGALMesh")
    tetgen = _select_iso2mesh_tool(args.tetgen, "tetgen", "TetGen")
    dfw_library = (
        Path(args.dfw_library).expanduser().resolve()
        if args.dfw_library
        else Path(__file__).resolve().parent / "native" / "flow4d_dfwavelet.dll"
    )
    _require_file(dfw_library, "DFW DLL")
    report.data["selections"].update(cgalmesh=str(cgalmesh), tetgen=str(tetgen),
                                     dfw_library=str(dfw_library))
    report._write()
    with report.step("06") as item:
        mask_inventory = prepare_mask_directories(case_directory)
        mask_result = _load_mask_with_optional_root(unwrap_case, mask_inventory)
        if not np.any(mask_result.mask == 0):
            raise ValueError(
                "生成的 Mask 覆盖了整个图像，不能用于血管网格；"
                "请核对 root 扣除或改用背景为 0 的 mask。"
            )
        item["output"] = {
            "mask": _array_info(mask_result.mask),
            "foreground_voxels": int(np.count_nonzero(mask_result.mask)),
            "root_mode": "zero" if not mask_inventory.root_files else "dicom",
            "timings": _timings(mask_result),
        }
        if args.save_arrays:
            item["output"]["checkpoint"] = report.save_array("06", "mask", mask_result.mask)
    mask = mask_result.mask
    del mask_result
    with report.step("07") as item:
        unwrapped = run_unwrap_pipeline(
            unwrap_case, flip_rl="RL" in flips, flip_fh="FH" in flips,
            flip_ap="AP" in flips,
        )
        item["output"] = {
            "velocity": _array_info(unwrapped.velocity),
            "wrap_counts": {
                "RL": int(np.count_nonzero(unwrapped.wrap_counts_rl)),
                "FH": int(np.count_nonzero(unwrapped.wrap_counts_fh)),
                "AP": int(np.count_nonzero(unwrapped.wrap_counts_ap)),
            },
            "flipped": list(flips), "timings": _timings(unwrapped),
        }
        if args.save_arrays:
            item["output"]["checkpoint"] = report.save_array("07", "unwrapped_velocity", unwrapped.velocity)
    del unwrap_case
    with report.step("08") as item:
        surface_backend = CgalMeshSurfaceBackend(cgalmesh)
        tetra_backend = TetGenTetrahedralizationBackend(tetgen)
        mesh = run_mesh_pipeline(mask, unwrapped.velocity, case_voxel_size,
    surface_backend, tetrahedralization_backend=tetra_backend,
)

    surface_stl = _write_ascii_stl(
    output / "surface_mesh.stl",
    scale_tetrahedral_nodes(
        mesh.surface.nodes, case_voxel_size, refinement_scale=0.5
    ),
    mesh.surface.faces,
)
    item["output"] = {
        "crop_bounds": {
            name: getattr(mesh.preparation.bounds, name)
            for name in (
                "row_start",
                "row_stop",
                "column_start",
                "column_stop",
                "slice_start",
                "slice_stop",
            )
        },
        "mask_crop": _array_info(mesh.preparation.mask_crop),
        "refined_mask": _array_info(mesh.preparation.refined_mask),
        "surface_nodes": len(mesh.surface.nodes),
        "surface_faces": len(mesh.surface.faces),
        "surface_stl": str(surface_stl),
        "tetra_nodes": len(mesh.node_coordinates_mm),
        "tetra_elements": len(mesh.tetrahedral.elements),
        "sampling_boundary": _sampling_boundary_info(
            mesh.node_coordinates_mm,
            mesh.preparation.mask_crop.shape,
            case_voxel_size,
        ),
        "timings": _timings(mesh),
    }
    if args.save_arrays:
        item["output"]["checkpoints"] = {
                "refined_mask": report.save_array("08", "refined_mask", mesh.preparation.refined_mask),
                "nodes_mm": report.save_array("08", "nodes_mm", mesh.node_coordinates_mm),
                "tetrahedra": report.save_array("08", "tetrahedra", mesh.tetrahedral.elements),
            }
    del unwrapped, mask, mask_inventory
    with report.step("09") as item:
        smoothed = smooth_velocity_gaussian(
            mesh.preparation.velocity_crop, mesh.preparation.mask_crop
        )
        dfw = denoise_velocity_dfw_timeseries(
            smoothed.masked_velocity_smoothed, mesh.preparation.mask_crop,
            case_voxel_size, DfwNativeBackend(dfw_library),
        )
        item["output"] = {
            "gaussian_velocity": _array_info(smoothed.masked_velocity_smoothed),
            "dfw_velocity": _array_info(dfw.velocity_denoised),
            "gaussian_timings": _timings(smoothed), "dfw_timings": _timings(dfw),
        }
        if args.save_arrays:
            item["output"]["checkpoint"] = report.save_array("09", "dfw_velocity", dfw.velocity_denoised)
    nodes_mm = mesh.node_coordinates_mm
    tetrahedra = mesh.tetrahedral.elements
    del smoothed, mesh
    with report.step("10") as item:
        def report_final_substep(
            name: str, status: str, elapsed: float | None
        ) -> None:
            substeps = item.setdefault("substeps", [])
            if status == "running":
                record = {"name": name, "status": status,
                          "started_utc": datetime.now(timezone.utc).isoformat()}
                substeps.append(record)
            else:
                record = next(
                    step for step in reversed(substeps) if step["name"] == name
                )
                record["status"] = status
                if elapsed is not None:
                    record["elapsed_seconds"] = round(elapsed, 3)
            report._write()
            timing = "" if elapsed is None else f"，{elapsed:.3f} 秒"
            print(f"    [10] {name}：{status}{timing}", flush=True)

        final = run_final_postprocessing(
            dfw.velocity_denoised, nodes_mm,
            tetrahedra, case_voxel_size,
            case_time_spacing, connectivity_index_base=0,
            vtk_case_directory=output, vtk_overwrite=True,
            progress=report_final_substep,
        )
        item["output"] = {
            "node_velocity": _array_info(final.node_sampling.velocity_mm_per_second),
            "pressure_pa": _array_info(final.pressure.pressure_pa),
            "final_node_results": _array_info(final.final_node_results),
            "timings": _timings(final),
        }
        if args.save_arrays:
            item["output"]["checkpoint"] = report.save_array("10", "final_node_results", final.final_node_results)
    with report.step("11") as item:
        exported = final.vtk_export
        if exported is None or not exported.series_file.is_file():
            raise RuntimeError("最终后处理没有生成 VTK 时间序列索引")
        if len(exported.vtk_files) != case_phases or not all(
            path.is_file() for path in exported.vtk_files
        ):
            raise RuntimeError("VTK 时相文件数量不符合输入病例时相数")
        item["output"] = {
            "vtk_directory": str(exported.output_directory),
            "vtk_files": len(exported.vtk_files),
            "series_file": str(exported.series_file),
        }
    with report.step("12") as item:
        tecplot = export_matlab_dat_plt_series(
            final.node_sampling.variables_by_phase,
            final.final_node_results,
            tetrahedra,
            output,
            progress=lambda current, total: print(
                f"    [12] DAT/PLT 时相 {current}/{total} 已写入", flush=True
            ),
        )
        if len(tecplot.uvw_files) != case_phases or len(tecplot.plt_files) != case_phases:
            raise RuntimeError("DAT/PLT 时相文件数量不符合输入病例时相数")
        item["output"] = {
            "dat_directory": str(tecplot.output_directory),
            "uvw_files": len(tecplot.uvw_files),
            "fembrick_files": len(tecplot.plt_files),
        }
    print(f"整体流程结束逐步记录：{report.path}")
    return 0


def _write_ascii_stl(
    path: Path,
    nodes: np.ndarray,
    faces: np.ndarray,
    *,
    solid_name: str = "flow4d_surface",
) -> Path:
    """把现有三角表面网格原样导出为 ASCII STL。"""

    vertices = np.asarray(nodes, dtype=np.float64)
    triangles = np.asarray(faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"STL 节点必须为 (N, 3)，实际为 {vertices.shape}")

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"STL 三角面必须为 (M, 3)，实际为 {triangles.shape}")

    if not np.issubdtype(triangles.dtype, np.integer):
        if not np.all(np.equal(triangles, np.round(triangles))):
            raise ValueError("surface.faces 含有非整数索引")
        triangles = np.round(triangles).astype(np.int64)
    else:
        triangles = triangles.astype(np.int64, copy=False)

    n_nodes = len(vertices)

    # 同时兼容 Python 0-based 与 MATLAB/外部程序 1-based connectivity。
    face_min = int(triangles.min())
    face_max = int(triangles.max())

    if face_min >= 0 and face_max < n_nodes:
        triangles0 = triangles
    elif face_min >= 1 and face_max <= n_nodes:
        triangles0 = triangles - 1
    else:
        raise ValueError(
            "surface.faces 索引超出节点范围："
            f"min={face_min}, max={face_max}, nodes={n_nodes}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"solid {solid_name}\n")

        for face in triangles0:
            p0, p1, p2 = vertices[face]

            normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(normal))

            if norm > 0.0 and np.isfinite(norm):
                normal /= norm
            else:
                normal = np.zeros(3, dtype=np.float64)

            handle.write(
                f"  facet normal "
                f"{normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}\n"
            )
            handle.write("    outer loop\n")

            for point in (p0, p1, p2):
                handle.write(
                    f"      vertex "
                    f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n"
                )

            handle.write("    endloop\n")
            handle.write("  endfacet\n")

        handle.write(f"endsolid {solid_name}\n")

    return path


def _launch_gui() -> int:
    """Start the optional GUI without making PySide6 a CLI dependency."""
    try:
        from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
            QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
            QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
            QTextEdit, QVBoxLayout, QWidget,
        )
    except ImportError as error:
        print(
            "图形界面需要 PySide6。请先用当前 Python 执行："
            "python -m pip install PySide6；命令行 --run/--check-only 仍可使用。"
            f" 原始错误：{error}", file=sys.stderr,
        )
        return 2

    class FlowWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("4D Flow MRI 整体流程")
            self.resize(1060, 820)
            self.process: QProcess | None = None
            self.process_mode = ""
            self.started_at = 0.0
            self._last_status_time = 0.0
            self._checked_case: Path | None = None
            self._checked_format = ""
            self._checked_expected_slices: int | None = None
            self._active_case: Path | None = None
            self._active_output: Path | None = None
            self._waiting_for_mask = False
            self._waiting_expected_slices: int | None = None
            self._waiting_attempt = 0
            self._wait_attempt_sent = 0
            self._wait_started_at: float | None = None
            self._wait_seconds = 0.0
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            headline = QLabel(
                "先选择外层 DICOM_4D_Qflow 病例目录。程序会检查已有 PCA 与 mask/root；"
                "mask 自动从病例目录下的 mask 子文件夹读取。"
                "没有 Mask 时先生成分割参考图；制作 Mask 期间请保持窗口打开，导入后点继续。"
            )
            headline.setWordWrap(True)
            layout.addWidget(headline)

            paths = QGroupBox("病例和输出")
            self.paths_group = paths
            path_grid = QGridLayout(paths)
            self.case_edit = QLineEdit()
            self.case_edit.setPlaceholderText("外层 DICOM_4D_Qflow；包含 M/AP/FH/RL 四个目录")
            self.output_edit = QLineEdit()
            self.output_edit.setPlaceholderText("留空则写入病例目录；已有同名结果可能被覆盖")
            path_grid.addWidget(QLabel("病例目录"), 0, 0)
            path_grid.addWidget(self.case_edit, 0, 1)
            case_browse = QPushButton("浏览…")
            case_browse.clicked.connect(lambda: self._browse_directory(self.case_edit))
            path_grid.addWidget(case_browse, 0, 2)
            path_grid.addWidget(QLabel("输出目录"), 1, 0)
            path_grid.addWidget(self.output_edit, 1, 1)
            output_browse = QPushButton("浏览…")
            output_browse.clicked.connect(lambda: self._browse_directory(self.output_edit))
            path_grid.addWidget(output_browse, 1, 2)
            layout.addWidget(paths)

            options = QGroupBox("计算选择")
            self.options_group = options
            option_grid = QGridLayout(options)
            self.format_choice = QComboBox()
            self.format_choice.addItem("普通 DICOM（Classic）", "classic")
            self.format_choice.addItem("增强型 DICOM（Enhanced）", "enhanced")
            self.source_choice = QComboBox()
            self.source_choice.addItem("使用原始速度；已有 PCA 时跳过 MSAC", "raw")
            self.source_choice.addItem("使用 MSAC 校正结果", "msac")
            option_grid.addWidget(QLabel("DICOM 类型"), 0, 0)
            option_grid.addWidget(self.format_choice, 0, 1)
            option_grid.addWidget(QLabel("速度来源"), 1, 0)
            option_grid.addWidget(self.source_choice, 1, 1)
            self.flip_boxes = {}
            flip_row = QHBoxLayout()
            for component in ("RL", "FH", "AP"):
                checkbox = QCheckBox(f"反转 {component}")
                checkbox.setToolTip("仅在核对病例的解剖方向后勾选；默认不反转")
                self.flip_boxes[component] = checkbox
                flip_row.addWidget(checkbox)
            option_grid.addWidget(QLabel("速度方向"), 2, 0)
            option_grid.addLayout(flip_row, 2, 1)
            warning = QLabel("方向反转会影响后续压力和 Liutex；请根据原始图像方向与解剖位置核对。")
            warning.setWordWrap(True)
            option_grid.addWidget(warning, 3, 0, 1, 2)
            self.regenerate_pca = QCheckBox("即使 PCA 已完整也重新生成")
            self.save_arrays = QCheckBox("保存大型中间数组 .npy（会占用较多磁盘）")
            option_grid.addWidget(self.regenerate_pca, 4, 1)
            option_grid.addWidget(self.save_arrays, 5, 1)
            layout.addWidget(options)

            self.inventory = QLabel("尚未检查病例。")
            self.inventory.setWordWrap(True)
            layout.addWidget(self.inventory)
            buttons = QHBoxLayout()
            self.check_button = QPushButton("检查病例")
            self.run_button = QPushButton("开始／继续")
            self.mask_button = QPushButton("打开默认 mask 文件夹")
            self.check_button.clicked.connect(lambda: self._start_process("check"))
            self.run_button.clicked.connect(lambda: self._start_process("run"))
            self.mask_button.clicked.connect(self._open_mask)
            for button in (self.check_button, self.run_button, self.mask_button):
                buttons.addWidget(button)
            layout.addLayout(buttons)

            self.elapsed = QLabel("总耗时：尚未开始")
            layout.addWidget(self.elapsed)
            self.table = QTableWidget(len(STEPS), 4)
            self.table.setHorizontalHeaderLabels(("编号", "步骤", "状态", "耗时"))
            self.table.horizontalHeader().setStretchLastSection(True)
            for row, (number, name) in enumerate(STEPS):
                for column, value in enumerate((number, name, "未开始", "—")):
                    self.table.setItem(row, column, QTableWidgetItem(value))
            self.table.currentCellChanged.connect(self._show_guidance)
            layout.addWidget(self.table)
            self.guidance = QLabel()
            self.guidance.setWordWrap(True)
            layout.addWidget(self.guidance)
            self.table.selectRow(0)
            self.log = QTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log)
            self.timer = QTimer(self)
            self.timer.setInterval(500)
            self.timer.timeout.connect(self._tick)
            self.case_edit.textChanged.connect(self._preview_inventory)
            self.output_edit.textChanged.connect(self._preview_inventory)
            self.source_choice.currentIndexChanged.connect(self._preview_inventory)
            self.format_choice.currentIndexChanged.connect(self._preview_inventory)
            self.regenerate_pca.toggled.connect(self._preview_inventory)

        def _browse_directory(self, field: QLineEdit) -> None:
            selected = QFileDialog.getExistingDirectory(self, "选择目录", field.text())
            if selected:
                field.setText(selected)

        def _case_and_output(self) -> tuple[Path, Path]:
            case = Path(self.case_edit.text().strip().strip('"')).expanduser().resolve()
            output_text = self.output_edit.text().strip().strip('"')
            output = Path(output_text).expanduser().resolve() if output_text else case
            return case, output

        def _preview_inventory(self, *_: object) -> None:
            case, output = self._case_and_output()
            if not case.is_dir() or case.name != "DICOM_4D_Qflow":
                self.inventory.setText("请选择名称为 DICOM_4D_Qflow 的外层病例目录。")
                return
            pca = _dicom_count(output / "results_PCA")
            mask = _dicom_count(case / "mask")
            root_count = _dicom_count(case / "root")
            summary = (f"当前文件数：输出目录 PCA {pca} 张；默认 mask 子文件夹 {mask} 张、"
                       f"root {root_count} 张。")
            if (self._checked_case == case and
                    self._checked_format == self.format_choice.currentData() and
                    self._checked_expected_slices is not None):
                decision = _reuse_decision(
                    case, output, self._checked_expected_slices,
                    self.source_choice.currentData(),
                    force_pca=self.regenerate_pca.isChecked(),
                )
                omitted = []
                if not decision["run_msac"]:
                    omitted.append("04 MSAC")
                if decision["reuse_pca"]:
                    omitted.append("05 分割图导出")
                next_step = ("mask 已完整，root 为空或完整，可继续后续计算。" if decision["mask_ready"]
                             else "mask 不完整或 root 只有部分切片；先生成或复用 PCA，再补齐分割。")
                summary += (
                    f" 病例预计 {decision['expected_slices']} 层；"
                    f"可省略：{'、'.join(omitted) if omitted else '无'}。{next_step}"
                    " 请确认已有文件属于同一病例且切片顺序一致。"
                )
                if decision["root_mode"] == "zero":
                    summary += " root 为空时，mask 的背景像素必须为 0；开始计算时会先检查。"
            else:
                summary += " 点“检查病例”读取首张 DICOM 的头信息，判断是否可复用。"
            self.inventory.setText(summary)

        def _show_guidance(self, row: int, _column: int, *_: int) -> None:
            if row < 0 or row >= len(STEPS):
                return
            number, name = STEPS[row]
            purpose, required, produced = STEP_GUIDANCE[number]
            self.guidance.setText(
                f"{number} {name}｜做什么：{purpose}\n需要：{required}\n完成后：{produced}"
            )

        def _arguments(self, mode: str, case: Path, output: Path) -> list[str]:
            arguments = ["--check-only" if mode == "check" else "--run",
                         "--case", str(case), "--format", self.format_choice.currentData()]
            if mode == "run":
                arguments.append("--wait-for-mask")
                selected_flips = [key for key, box in self.flip_boxes.items() if box.isChecked()]
                arguments.extend(("--source", self.source_choice.currentData(),
                                  "--flip", ",".join(selected_flips) or "none",
                                  "--output", str(output)))
                if self.regenerate_pca.isChecked():
                    arguments.append("--regenerate-pca")
                if self.save_arrays.isChecked():
                    arguments.append("--save-arrays")
            return arguments

        def _start_process(self, mode: str) -> None:
            if self.process is not None and self.process.state() != QProcess.NotRunning:
                if mode == "run" and self._waiting_for_mask:
                    self._resume_mask()
                return
            case, output = self._case_and_output()
            if not case.is_dir() or case.name != "DICOM_4D_Qflow":
                QMessageBox.warning(self, "病例目录", "请选择外层 DICOM_4D_Qflow 病例目录。")
                return
            if mode == "run" and output.name != "DICOM_4D_Qflow":
                QMessageBox.warning(self, "输出目录", "输出目录末级名称须为 DICOM_4D_Qflow。")
                return
            if mode == "run":
                choice = QMessageBox(self)
                choice.setWindowTitle("确认开始计算")
                source_description = (
                    "原始数据（已执行相位边界校正）"
                    if self.source_choice.currentData() == "raw"
                    else "MSAC 背景校正数据"
                )
                choice.setText(
                    f"本次解混叠使用：{source_description}。\n"
                    "若需更改，请取消并在主页面调整速度来源。\n\n"
                    "使用原始数据且已有可复用 PCA 时会跳过 MSAC；需要重新生成分割参考图时，"
                    "仍会执行 MSAC。使用 MSAC 数据时会执行背景校正。\n"
                    "同名生成结果可能覆盖，请核对输出目录。"
                )
                start_button = choice.addButton("开始", QMessageBox.AcceptRole)
                cancel_button = choice.addButton("取消", QMessageBox.RejectRole)
                choice.setDefaultButton(cancel_button)
                choice.setEscapeButton(cancel_button)
                choice.exec()
                if choice.clickedButton() != start_button:
                    return
            self.process_mode = mode
            self.started_at = perf_counter()
            self._last_status_time = time()
            self._active_case, self._active_output = case, output
            self._waiting_for_mask = False
            self._waiting_expected_slices = None
            self._waiting_attempt = 0
            self._wait_attempt_sent = 0
            self._wait_started_at = None
            self._wait_seconds = 0.0
            self.check_button.setEnabled(False)
            self.run_button.setEnabled(False)
            self.paths_group.setEnabled(False)
            self.options_group.setEnabled(False)
            if mode == "run":
                for row in range(len(STEPS)):
                    self.table.item(row, 2).setText("未开始")
                    self.table.item(row, 3).setText("—")
            self.log.append("即将执行：" + " ".join(self._arguments(mode, case, output)))
            self.log.append("同名生成结果可能被覆盖；原始四方向 DICOM 与 mask/root 仅读取。")
            process = QProcess(self)
            process.setProcessChannelMode(QProcess.MergedChannels)
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONIOENCODING", "utf-8")
            environment.insert("PYTHONUNBUFFERED", "1")
            process.setProcessEnvironment(environment)
            process.setWorkingDirectory(str(Path(__file__).resolve().parent))
            process.readyReadStandardOutput.connect(self._read_output)
            process.finished.connect(self._finished)
            self.process = process
            process.start(sys.executable, [str(Path(__file__).resolve()),
                                           *self._arguments(mode, case, output)])
            self.timer.start()

        def _resume_mask(self) -> None:
            if (self.process is None or self._active_case is None or
                    self._waiting_expected_slices is None):
                QMessageBox.warning(self, "等待状态", "尚未收到 Mask 步骤的等待状态，请稍后重试。")
                return
            expected = self._waiting_expected_slices
            mask_count = _dicom_count(self._active_case / "mask")
            root_count = _dicom_count(self._active_case / "root")
            if mask_count != expected or root_count not in (0, expected):
                QMessageBox.warning(
                    self, "Mask 尚未备齐",
                    f"本病例的默认 mask 子文件夹需要 {expected} 张 DICOM；"
                    f"root 可以为 0 张或 {expected} 张。当前分别为 {mask_count}、"
                    f"{root_count} 张。请补齐后再点继续。",
                )
                return
            if self.process.write(b"continue\n") < 0:
                QMessageBox.warning(self, "继续失败", "无法向计算进程发送继续指令，请查看日志。")
                return
            self._wait_attempt_sent = self._waiting_attempt
            if self._wait_started_at is not None:
                self._wait_seconds += perf_counter() - self._wait_started_at
                self._wait_started_at = None
            self._waiting_for_mask = False
            self.run_button.setEnabled(False)
            self.run_button.setText("计算中…")
            self.log.append("mask 与可选 root 数量已核对；正在从阶段 06 继续，前面步骤不会重算。")

        def _read_output(self) -> None:
            if self.process is None:
                return
            message = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            if message:
                self.log.append(message.rstrip())

        def _tick(self) -> None:
            if self.process is not None and self.process.state() != QProcess.NotRunning:
                now = perf_counter()
                waiting = 0.0 if self._wait_started_at is None else now - self._wait_started_at
                compute_seconds = max(0.0, now - self.started_at - self._wait_seconds - waiting)
                state = "等待 Mask，不计入计算耗时" if self._waiting_for_mask else "正在运行"
                self.elapsed.setText(f"累计计算耗时：{compute_seconds:.1f} 秒（{state}）")
            if self.process_mode == "run":
                self._refresh_status()

        def _refresh_status(self) -> None:
            try:
                output = self._active_output
                if output is None:
                    return
                status_file = output / "overall_flow_status.json"
                if not status_file.is_file() or status_file.stat().st_mtime < self._last_status_time - 1:
                    return
                data = json.loads(status_file.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                return
            if self.process_mode == "run" and data.get("created_utc"):
                try:
                    created_at = datetime.fromisoformat(data["created_utc"]).timestamp()
                    if created_at < self._last_status_time - 0.5:
                        return
                except ValueError:
                    return
            labels = {"not_started": "未开始", "running": "运行中", "completed": "已完成",
                      "skipped": "已复用／跳过", "waiting_input": "等待 Mask",
                      "failed": "失败", "interrupted": "已中断"}
            for row, step in enumerate(data.get("steps", ())):
                if row >= len(STEPS):
                    break
                self.table.item(row, 2).setText(labels.get(step.get("status"), step.get("status", "")))
                seconds = step.get("elapsed_seconds")
                if step.get("status") == "running" and step.get("started_utc"):
                    try:
                        seconds = max(0.0, (datetime.now(timezone.utc) -
                                            datetime.fromisoformat(step["started_utc"])).total_seconds())
                    except ValueError:
                        pass
                self.table.item(row, 3).setText("—" if seconds is None else f"{seconds:.1f} 秒")
                if step.get("status") == "running":
                    self.table.selectRow(row)
                    self._show_guidance(row, 0)
                    substeps = step.get("substeps", ())
                    if substeps:
                        current = substeps[-1]
                        substep_seconds = current.get("elapsed_seconds")
                        if current.get("status") == "running" and current.get("started_utc"):
                            try:
                                substep_seconds = max(0.0, (datetime.now(timezone.utc) -
                                    datetime.fromisoformat(current["started_utc"])).total_seconds())
                            except ValueError:
                                pass
                        timing = ("" if substep_seconds is None else
                                  f"，已用 {substep_seconds:.1f} 秒")
                        self.guidance.setText(
                            self.guidance.text() + "\n当前子步骤：" + current.get("name", "")
                            + "（" + labels.get(current.get("status"), current.get("status", ""))
                            + timing + "）"
                        )
                elif step.get("status") in {"failed", "interrupted"}:
                    self.table.selectRow(row)
                    self._show_guidance(row, 0)
                    if step.get("error"):
                        self.guidance.setText(self.guidance.text() + "\n错误：" + step["error"])
            waiting = next((step for step in data.get("steps", ())
                            if step.get("status") == "waiting_input"), None)
            if waiting is not None:
                self.inventory.setText(waiting.get("reason", "请导入 mask/root DICOM 后继续。"))
                self.table.selectRow(6)
                self._show_guidance(6, 0)
                self._waiting_attempt = waiting.get("wait_attempts", 0)
                self._waiting_expected_slices = (
                    data.get("selections", {}).get("existing_results", {}).get("expected_slices")
                )
                if self._waiting_attempt > self._wait_attempt_sent:
                    self._waiting_for_mask = True
                    if self._wait_started_at is None:
                        self._wait_started_at = perf_counter()
                    self.run_button.setText("检查 Mask 并继续")
                    self.run_button.setEnabled(True)
            elif self.process_mode == "run" and self.process is not None and self.process.state() != QProcess.NotRunning:
                self._waiting_for_mask = False
                self.run_button.setEnabled(False)
                self.run_button.setText("计算中…")

        def _finished(self, code: int, _status: object) -> None:
            self._read_output()
            if self.process_mode == "run" and self._active_output is not None:
                if _mark_unfinished_run(self._active_output, code):
                    self.log.append("计算进程已退出，未完成的步骤已标为失败；请查看步骤错误。")
            self._refresh_status()
            self.timer.stop()
            self.check_button.setEnabled(True)
            self.run_button.setEnabled(True)
            self.run_button.setText("开始／继续")
            self.paths_group.setEnabled(True)
            self.options_group.setEnabled(True)
            now = perf_counter()
            waiting = 0.0 if self._wait_started_at is None else now - self._wait_started_at
            compute_seconds = max(0.0, now - self.started_at - self._wait_seconds - waiting)
            self.elapsed.setText(f"本次计算耗时：{compute_seconds:.1f} 秒")
            self._waiting_for_mask = False
            self._wait_started_at = None
            if self.process_mode == "check" and code == 0:
                case, _ = self._case_and_output()
                inspection = _inspect(case, self.format_choice.currentData())
                self._checked_case = case
                self._checked_format = self.format_choice.currentData()
                self._checked_expected_slices = inspection["expected_slices_from_headers"]
                self._preview_inventory()
                self.log.append("目录检查已结束；只读取首张 DICOM 的头信息，没有读取像素或写入结果。")
            elif self.process_mode == "check":
                self.log.append("目录检查未通过；请查看上方日志中的具体原因。")
            elif code == 3:
                self.log.append("命令行流程停在 Mask；图形界面原本应保持进程等待，请查看错误日志。")
            elif code == 0:
                self.log.append("流程结束。请查看上表、输出目录与逐步状态文件。")
            else:
                self.log.append("流程未完成；请查看最后一条错误和 overall_flow_status.json。")

        def _open_mask(self) -> None:
            case = self._active_case or self._case_and_output()[0]
            if not case.is_dir():
                QMessageBox.warning(self, "病例目录", "请先选择病例目录。")
                return
            directory = case / "mask"
            directory.mkdir(exist_ok=True)
            (case / "root").mkdir(exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
            self._preview_inventory()

        def closeEvent(self, event: Any) -> None:
            if self.process is not None and self.process.state() != QProcess.NotRunning:
                answer = QMessageBox.question(
                    self, "计算仍在运行", "关闭窗口会中断当前计算。确定关闭吗？"
                )
                if answer != QMessageBox.Yes:
                    event.ignore()
                    return
                self.process.kill()
                self.process.waitForFinished(3000)
            event.accept()

    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = FlowWindow()
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="仅扫描输入目录，不写文件")
    mode.add_argument("--run", action="store_true", help="执行完整计算和导出")
    mode.add_argument("--gui", action="store_true", help="打开 PySide6 按钮式界面")
    parser.add_argument("--case", help="外层 DICOM_4D_Qflow 病例目录")
    parser.add_argument("--format", choices=("classic", "enhanced"))
    parser.add_argument("--source", choices=("raw", "msac"), help="解混叠所用病例数据")
    parser.add_argument("--flip", help="none 或逗号分隔的 RL,FH,AP 反向分量")
    parser.add_argument("--output", help="输出目录；默认输入病例目录，允许覆盖同名结果")
    parser.add_argument("--cgalmesh", help=argparse.SUPPRESS)
    parser.add_argument("--tetgen", help=argparse.SUPPRESS)
    parser.add_argument("--dfw-library", help=argparse.SUPPRESS)
    parser.add_argument("--save-arrays", action="store_true", help="保存关键中间数组为 .npy")
    parser.add_argument("--regenerate-pca", action="store_true", help="忽略已有 PCA 并重新生成分割辅助图像")
    parser.add_argument("--wait-for-mask", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gui or (argv is None and len(sys.argv) == 1) or argv == []:
        return _launch_gui()
    try:
        return run(args)
    except (KeyboardInterrupt, EOFError):
        print("流程被中断；查看已写入的 overall_flow_status.json", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"流程失败：{type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
