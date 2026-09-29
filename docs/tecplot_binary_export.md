# FINAL Tecplot 二进制导出

## MATLAB 活动语义

`main_for4DFlowMRI_FINAL_philips.m` 对每个时相构造一个 `tdata.FEvolumes`，再调用第三方 `mat2tecplot.m`：

- 变量固定为 FINAL 的 27 列节点变量；
- 所有变量均为 nodal float；
- 网格为四节点四面体；
- `strandID=1`；
- `solutiontime` 使用 MATLAB 的一基时相编号 `1..T`；
- 未显式提供 title 时，第三方 writer 使用 dataset title `tecplot data`；
- 未显式提供 `zonename` 时，单个 FEvolume 使用 zone name `FEVolume1`；
- 文件名保持历史名称 `FEMbrick_1.plt`、`FEMbrick_2.plt` 等；名称虽含 `brick`，实际 zone 是 `FETetra`；
- 原 writer 使用 `#!TDV112`。

## Python 替代边界

`flow4d.exporters.tecplot_binary` 只迁移上述真实活动子集，不复制未被 FINAL 调用的通用 `mat2tecplot.m` 能力。它保留默认 dataset/zone 名称，使用官方 PyTecplot 创建 `FETetra` zone，并指定 `BinaryFileVersion.Tecplot2009`，对应 v112 格式。

PyTecplot 和 Tecplot Engine 属于可选的外部运行依赖，不加入核心 `requirements.txt`。官方要求 64-bit Python、Tecplot 360、可用的 Tecplot 许可，以及具备 PyTecplot 权益的 TecPLUS 维护服务。独立 batch 模式会在第一次调用 PyTecplot API 时自动申请许可；connected 模式由调用者先启动 Tecplot 360、启用 TecUtil Server 并建立 `tecplot.session.connect()` 连接。本 exporter 不自行连接或停止外部 session。

`Frame.create_dataset()` 会替换并销毁该 frame 原有 dataset，因此 exporter 不在用户当前 active frame 上创建数据。每次写出会临时增加一个 Sketch frame，在该 frame 中建立导出 dataset；无论成功或失败，随后都删除临时 frame 并恢复原 active frame。这样在 connected session 中不会覆盖用户当前数据集。运行期间仍会短暂改变 Tecplot layout/active-frame 状态，因此不应与同一 Tecplot session 的其他并发 UI 或脚本操作交错执行。

未安装 PyTecplot、Tecplot Engine 初始化失败或无法取得许可时，核心计算、VTK 和 ASCII DAT 不受影响；只有明确请求 `.plt` 导出才会给出依赖或 Engine 错误。

顶层入口通过 `run_final_postprocessing(..., tecplot_binary_directory=...)` 启用。参数为 `None` 时不创建 `.plt` 文件。

## 外部验证边界

本模块需要在具有 Tecplot 许可和 PyTecplot 的环境中验证：

- 逐文件回读 27 个变量、节点数、四面体数和 connectivity；
- 确认变量类型为 float、位置为 nodal；
- 确认 strand 为 1、solution time 为 `1..T`；
- 确认 dataset title 为 `tecplot data`、zone name 为 `FEVolume1`；
- 与 MATLAB `.plt` 对照场值和拓扑语义。

二进制字节不要求与第三方 MATLAB writer 完全相同；验收对象是 Tecplot 数据集语义一致。
