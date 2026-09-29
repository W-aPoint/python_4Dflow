# UVW DAT 与 FEMbrick PLT 导出方案

## 目标与文件名

每个心动时相在病例目录的 `dat` 子目录保存两份文件，编号从 1 到时相总数：

- `UVW_1.dat` … `UVW_25.dat`：X/Y/Z 与 U/V/W，四节点四面体，ASCII Tecplot 格式。
- `FEMbrick_1.plt` … `FEMbrick_25.plt`：FINAL 的 27 列节点结果，Tecplot 112 二进制格式；虽然历史文件名含 `brick`，单元类型仍是四节点四面体。

## 数据来源与接入

`overall_flow.py` 完成 FINAL 节点采样、压力、Liutex 和 VTK 核对后，执行第 12 步导出。UVW 使用该次运行的 `node_sampling.variables_by_phase`；PLT 使用同一网格的 `final_node_results`。每处理完一个时相，界面日志打印一次进度，步骤结束后在 `overall_flow_status.json` 中记录文件数量与目录。

ASCII DAT 使用已有的 `tecplot_ascii.py`。PLT 使用新增的 `tecplot_v112.py`，仅实现本病例所需的单区域 FETetra、节点变量、float32 数据块和零基四面体连接，不依赖 PyTecplot/Tecplot Engine。参考 MATLAB `main_final.m` 的写出段与 `calLiutexfromFLUENTdat/mat2tecplot.m` 对应结构。此变更不触及速度、网格、压力和 Liutex 的计算公式。

## 已完成病例的补导出

已有完整 `vtk/NodeLiutexPressure_1.vtk` … 的病例可以只转换已有结果，避免重算：

```powershell
cd 'D:\PROJECT_CODE\暴哥代码\python_4Dflow'
$env:PYTHONPATH = 'src'
python -m flow4d.backfill_tecplot --case 'D:\PROJECT_CODE\暴哥代码\python_4Dflow\测试用例\DICOM_4D_Qflow'
```

补导出需要 Python `vtk` 包；它仅用于读取已有 VTK，正常全流程不需要这个包。默认不覆盖已有 DAT/PLT，确需重写时加 `--overwrite`。补导出不会把之前运行的 `overall_flow_status.json` 冒充为完成了第 12 步。

## 核验范围

小型网格测试覆盖二进制结构、变量顺序、四面体连接、相位文件名与 DAT 速度值。当前病例的 25 组文件已检查连续编号、27 个变量、节点与四面体数量，以及文件长度与 Tecplot 112 布局一致。MATLAB 历史 `.plt` 的固定结构开销与新写出器均为 1516 字节。当前 Python 网格为 100111 节点、604901 四面体，历史 MATLAB 结果为 99891 节点、603743 四面体，因此文件内容不会逐字节相同。仍需用目标 Tecplot 软件打开至少一帧做可视化验收；本机 PyTecplot Engine 当前无法启动。
