# python_4Dflow

4D Flow MRI 血流定量分析的 Python 实现，迁移自 AB4Dflow MATLAB 工程。主流程包括 DICOM 读取、相位及背景校正、速度解混叠、血管网格生成、速度平滑与 DFW 去噪、节点压力和 Liutex 计算，以及 VTK / Tecplot 导出。

## 文件结构

```text
requirements.txt           Python 核心依赖
src/flow4d/                Python 主程序和计算模块
  overall_flow.py          命令行与可选图形界面入口
  final_pipeline.py        节点采样、压力和 Liutex 后处理
  backfill_tecplot.py      从已有 VTK 补导出 DAT / PLT
  native/                 DFW 运行 DLL 和外部网格程序说明
native/dfwavelet/          DFW C++ 源码、桥接接口及 CMake 配置
docs/                     流程与导出格式使用说明
```

仓库不包含本地测试脚本、病例影像、分割数据、计算结果、编译中间文件和个人学习记录。

## 本次更新（2026-10-09）

- **稀疏压力重构**：CGS 求解加入 Jacobi 对角预条件和上一时相初值，复用预条件器，并检查原方程残差；初值求解失败时从零初值重试。压力梯度公式、稀疏矩阵、单位、默认容差及历史列索引行为保持不变，有限容差下的压力结果可能与原版不同。
- **四维速度解混叠**：频域核只准备一次，由 RL/FH/AP 共用；消除每次 Laplacian 运算中对整幅数组的反复移位，改用 SciPy FFT。保留完整四维域、双精度相位、原核、时间权重及圈数取整规则。已验证的真实输入上圈数和恢复速度一致，各方向核心计算约提速 2.63–2.80 倍；该结果不代表所有病例或完整流程的加速幅度。
- **主页面与启动确认**：速度来源只在主页面选择，启动弹窗展示已选来源和覆盖提示，提供“开始／取消”，不再重复选择或覆盖设置；同时改善 Windows 短暂文件占用时的状态文件写入处理。

具体行为与验证范围见 [本次更新说明](docs/整体流程命令行入口.md#本次更新2026-10-09)。本次无需新增命令行参数或优化开关。

## 安装与运行

在项目根目录打开 PowerShell：

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m flow4d.overall_flow --help
python -m flow4d.overall_flow --check-only --case 'D:\your_case\DICOM_4D_Qflow'
```

`--check-only` 仅扫描输入目录，不读取完整像素、不写计算结果。完整计算示例：

```powershell
python -m flow4d.overall_flow --run --case 'D:\your_case\DICOM_4D_Qflow' --format classic --source raw --flip none --output 'D:\your_output'
```

完整运行会生成文件，并可能覆盖输出目录中的同名结果。可使用独立输出目录保存计算产物。

图形界面需要额外安装 PySide6；不带参数启动时默认进入图形界面：

```powershell
python -m pip install PySide6
python -m flow4d.overall_flow --gui
```

从已有 VTK 补导出 Tecplot 文件需要额外安装 `vtk`：

```powershell
python -m pip install vtk
python -m flow4d.backfill_tecplot --case 'D:\your_case\DICOM_4D_Qflow'
```

## 输入与运行依赖

病例目录名为 `DICOM_4D_Qflow`，四方向目录为 `4D_Qflow_M`、`4D_Qflow_AP`、`4D_Qflow_FH`、`4D_Qflow_RL`。`mask` 和 `root` 使用人工外部分割结果；程序不会自动完成血管分割。`root` 可为空，或与 `mask` 具有相同层数。当前后处理沿用 MATLAB 第 19 时相建立几何算子的约定，因此病例至少需要 19 个时相；Classic 入口当前按 25 时相读取。

CGALMesh 和 TetGen 可执行文件不随仓库提供。主入口尝试从相邻的 `main_AB4Dflow_matlab/iso2mesh-master/bin` 查找，也可通过 `--cgalmesh`、`--tetgen` 指定路径。

仓库保留 `src/flow4d/native/flow4d_dfwavelet.dll`，它是 DFW 计算实际使用的 Windows 64 位运行库。对应源码位于 `native/dfwavelet/`。DLL 与 Python 架构必须兼容，目标机器还需满足其运行时依赖。需要自行编译时：

```powershell
cmake -S native/dfwavelet -B native/dfwavelet/build -A x64
cmake --build native/dfwavelet/build --config Release
Copy-Item native/dfwavelet/build/Release/flow4d_dfwavelet.dll src/flow4d/native/flow4d_dfwavelet.dll
```

上述编译示例使用 Windows Visual Studio C++ 工具链和 CMake。编译目录已加入忽略规则。

## 验证边界

此项目是科研算法迁移实现。代码保留部分 MATLAB 历史行为；完整的 MATLAB 端到端数值等价性、跨机器部署和目标 Tecplot 软件可视化验收仍需进一步验证。上传仓库本身不构成这些验证的证据。

## 更多说明

- [整体流程命令行入口](docs/整体流程命令行入口.md)
- [UVW DAT 与 FEMbrick PLT 导出](docs/UVW_FEMbrick_导出方案与使用.md)
- [Tecplot 二进制导出](docs/tecplot_binary_export.md)
- [DFW DLL](src/flow4d/native/README.md)
- [CGALMesh 外部依赖](src/flow4d/native/cgalmesh/README.md)
- [TetGen 外部依赖](src/flow4d/native/tetgen/README.md)
