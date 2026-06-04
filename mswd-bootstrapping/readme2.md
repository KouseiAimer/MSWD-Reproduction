# mswd-bootstrapping 文件夹结构说明

本文件夹是对论文 *Two-sample distribution tests in high dimensions via max-sliced Wasserstein distance and bootstrapping* 的实验代码复现。论文提出用稀疏 max-sliced Wasserstein distance 构造高维两样本分布检验统计量，并通过 bootstrap 近似零假设分布；同时，该方法可以构造 simultaneous confidence intervals, SCI，用来定位显著不同的投影方向、边际变量或变量子集。

代码整体分为三条主线：

- `main.py`：仿真实验入口，对应论文 Figure 1 和 Table 1 一类的高维两样本模拟比较。
- `GBM_methylation.py` 与 `GBM/`：真实数据分析入口，对应论文第 5 节的 LGG/GBM DNA methylation 数据分析。
- `mswd/` 与 `pyCode/`：方法实现。`mswd/` 更像作者整理出的正式接口；`pyCode/` 是仿真实验中直接调用的底层函数和比较方法集合。

## 目录树

```text
mswd-bootstrapping/
|-- README.md
|-- readme2.md
|-- main.py
|-- GBM_methylation.py
|-- GBM_data_pre.R
|-- GBM/
|   |-- GBM_prognostic_low.txt
|   |-- GBM_prognostic_high.txt
|   |-- prognostic_glioma.json
|   |-- c5.go.bp.v2023.1.Hs.json
|   |-- idx_genes_cell_cycle.txt
|   |-- idx_genes_cell_cycle_process.txt
|   |-- idx_genes_cell_division.txt
|   |-- idx_genes_DNA_metabolic_process.txt
|   |-- idx_genes_mitotic_cell_cycle.txt
|   |-- idx_genes_mitotic_cell_cycle_process.txt
|   `-- idx_genes_mitotic_nuclear_division.txt
|-- mswd/
|   |-- Example.py
|   `-- mswd_bootstrapping.py
|-- pyCode/
|   |-- dataGenerate.py
|   |-- ED.py
|   |-- global_var.py
|   |-- maxSlicedWD_L0_L1Approx.py
|   |-- maxSlicedWD_L0_test.py
|   |-- maxSlicedWD_L1.py
|   |-- maxSlicedWD_L1_test.py
|   |-- MMD.py
|   |-- myProj.py
|   |-- permTest.py
|   |-- PWD.py
|   |-- pwdPerm.py
|   |-- sim.py
|   |-- sMMD.py
|   |-- WD_NNapprox.py
|   `-- __pycache__/
```

`pyCode/__pycache__/` 是 Python 运行后生成的缓存目录，理解和复现实验时通常不需要关注。

## 顶层文件

`README.md`  
原始说明文件，列出了主要依赖版本和非常简短的代码用途。关键依赖包括 `torch==1.13.1`、`numpy==1.23.5`、`POT==0.8.2` 和 `scikit-learn`。

`main.py`  
仿真实验主入口。它通过命令行参数设置模型、样本量、维度、显著性水平和需要比较的方法。默认设置包括 `n1=250`、`n2=250`、`sample_dim=500`、`nrun=500`、`alpha=0.05`。每轮实验会先用 `pyCode.dataGenerate.dataGenerate` 生成两组样本，然后运行论文提出的 L0 稀疏 max-sliced Wasserstein bootstrap 检验，并调用 `pyCode.sim.sim` 运行比较方法。

`GBM_data_pre.R`  
真实数据预处理脚本。它使用 R 包 `cgdsr` 从 cBioPortal 下载 TCGA lower-grade glioma 和 glioblastoma multiforme 的 methylation 数据，使用 `prognostic_glioma.json` 中的脑癌预后基因筛选变量，去除缺失值，并生成 GO term 相关基因子集的索引文件。

`GBM_methylation.py`  
真实数据分析脚本。它读取 `GBM/GBM_prognostic_low.txt` 和 `GBM/GBM_prognostic_high.txt`，调参后运行 L0 稀疏 max-sliced Wasserstein 检验，并继续计算单基因边际差异和若干 GO term 基因集合上的 max-sliced Wasserstein 统计量。该脚本对应论文第 5 节的 LGG 与 GBM DNA methylation 分析。

## GBM 数据目录

`GBM/GBM_prognostic_low.txt`  
LGG 组甲基化数据。本地文件维度为 `511 x 207`，即 511 个 LGG 样本、207 个预后基因。

`GBM/GBM_prognostic_high.txt`  
GBM 组甲基化数据。本地文件维度为 `150 x 207`，即 150 个 GBM 样本、207 个预后基因。

`GBM/prognostic_glioma.json`  
Human Pathology Atlas 中与 glioma prognosis 相关的候选基因信息。`GBM_data_pre.R` 从这里读取基因名。

`GBM/c5.go.bp.v2023.1.Hs.json`  
MSigDB C5 GO Biological Process 基因集文件。真实数据分析用它筛选与 cell cycle、DNA metabolic process 等生物过程相关的基因集合。

`GBM/idx_genes_*.txt`  
各 GO term 在 207 个预后基因中的索引文件。脚本中读入后会减 1，从 R 的 1-based index 转换为 Python 的 0-based index。本地文件包含的索引数量如下：

```text
idx_genes_cell_cycle.txt                    11
idx_genes_cell_cycle_process.txt             7
idx_genes_cell_division.txt                  5
idx_genes_DNA_metabolic_process.txt          6
idx_genes_mitotic_cell_cycle.txt             5
idx_genes_mitotic_cell_cycle_process.txt     4
idx_genes_mitotic_nuclear_division.txt       3
```

## mswd 方法接口目录

`mswd/mswd_bootstrapping.py`  
较完整的 proposed method 封装，面向直接使用。主要函数包括：

- `mswdtest(X, Y, lam, ...)`：运行两样本分布检验，返回统计量、p-value、拒绝/接受结果、最优投影方向、bootstrap 样本、选择的稀疏参数和 SCI bound。
- `mswd_sci_direction(X, Y, V, lam, ...)`：对给定投影方向构造 one-sided SCI。
- `mswd_sci_marginal(X, Y, idx, lam, ...)`：对给定边际变量集合构造 SCI。
- `tune_l0(...)`：用 k-fold cross-validation 选择 L0 稀疏参数。
- `maxSlicedWDL0_L1Approx(...)`：用 L1 约束近似求解 L0 稀疏 max-sliced Wasserstein 距离。
- `maxSlicedWDL0_L1Approx_bootstrap(...)`：生成 bootstrap 统计量并给出阈值。

`mswd/Example.py`  
正式接口的最小使用示例。脚本生成一个高维 Gaussian mean-decay 例子，调用 `mswdtest` 做两样本检验，再调用 SCI 函数检查指定方向和指定边际集合。

## pyCode 实验函数目录

`pyCode/dataGenerate.py`  
仿真数据生成器。包含 `null-gauss`、`null-mix_gauss`、`mean`、`mean-decay`、`var`、`var-decay`、`marginal`、`joint`、`high-order-normal-pois` 等模型，对应论文仿真中均值差异、方差差异、边际分布差异和联合分布差异等情形。

`pyCode/maxSlicedWD_L0_L1Approx.py`  
`main.py` 中 proposed L0 方法的核心实现。它先通过 `tune_l0` 做交叉验证选择 L0 稀疏参数，再用一系列 L1 约束近似 L0 约束下的 max-sliced Wasserstein 方向，并用 multinomial bootstrap 构造临界值。

`pyCode/maxSlicedWD_L1.py`  
L1 稀疏 max-sliced Wasserstein 距离的实现。包含基于分位数的 1-Wasserstein 计算、子梯度计算、投影梯度优化、L1 参数调参和 bootstrap。

`pyCode/myProj.py`  
投影工具函数。用于把方向向量投影到满足 L1/L2 约束的可行集合中，是 `maxSlicedWD_L1.py` 和 `maxSlicedWD_L0_L1Approx.py` 的关键辅助函数。

`pyCode/maxSlicedWD_L0_test.py` 与 `pyCode/maxSlicedWD_L1_test.py`  
对 L0/L1 max-sliced Wasserstein 检验和显著方向、显著变量检查的早期 wrapper。实际主实验主要调用 `maxSlicedWD_L0_L1Approx.py`、`maxSlicedWD_L1.py` 和 `sim.py`。

`pyCode/sim.py`  
仿真实验方法调度器。根据 `opts.methods` 决定是否运行 proposed L1 方法、permutation 比较方法、studentized MMD、projected Wasserstein permutation 和 neural-network Wasserstein approximation。

`pyCode/permTest.py`  
置换检验集合。一次性计算并置换比较 MMD、ED-L1、ED-L2 和 BG 统计量。

`pyCode/MMD.py`  
MMD 相关函数，包括 median heuristic 带宽选择、biased MMD 和 unbiased MMD。

`pyCode/ED.py`  
energy distance 计算函数，支持 L1 或 L2 距离矩阵。

`pyCode/sMMD.py`  
studentized MMD 实现，对应 Gao & Shao 类型的高维 MMD 检验。

`pyCode/PWD.py`  
projected Wasserstein distance 的优化实现，使用 POT 包 `ot.emd` 计算最优传输 coupling，并在 Stiefel manifold 上更新投影矩阵。

`pyCode/pwdPerm.py`  
基于 `PWD.py` 的 permutation test，代码中默认投影维数 `k=3`。

`pyCode/WD_NNapprox.py`  
用 spectral normalization 神经网络近似 Lipschitz 函数类，从而近似 Wasserstein distance，并提供 multiplier bootstrap。

`pyCode/global_var.py`  
全局设备设置。当前文件写死为 `device = 'cuda'`；如果在没有 GPU/CUDA 的机器上运行，需要先把这里改成 `device = 'cpu'`，或恢复文件中被注释掉的自动检测逻辑。

## 实验运行关系

仿真实验的大致调用链为：

```text
main.py
`-- pyCode.dataGenerate.dataGenerate
`-- pyCode.maxSlicedWD_L0_L1Approx.tune_l0
`-- pyCode.maxSlicedWD_L0_L1Approx.maxSlicedWDL0_L1Approx
`-- pyCode.maxSlicedWD_L0_L1Approx.maxSlicedWDL0_L1Approx_bootstrap
`-- pyCode.sim.sim
    |-- pyCode.permTest.permTest
    |-- pyCode.maxSlicedWD_L1.*
    |-- pyCode.sMMD.sMMD_test
    |-- pyCode.pwdPerm.pwdPerm
    `-- pyCode.WD_NNapprox.*
```

真实数据实验的大致调用链为：

```text
GBM_data_pre.R
`-- 生成或更新 GBM/ 中的 methylation 数据和 GO term 索引

GBM_methylation.py
`-- 读取 GBM/GBM_prognostic_low.txt 与 GBM/GBM_prognostic_high.txt
`-- tune_l0 选择稀疏参数
`-- maxSlicedWDL0_L1Approx 计算检验统计量和最优方向
`-- maxSlicedWDL0_L1Approx_bootstrap 计算 bootstrap 阈值和 p-value
`-- 对单基因边际和 GO term 子集继续计算统计量
```

## 运行提示

运行 Python 脚本时建议把工作目录设为 `mswd-bootstrapping/`，因为真实数据脚本使用了 `GBM/...` 这样的相对路径。例如：

```bash
python main.py --model mean-decay --sample_dim 500 --signal 0.8
python GBM_methylation.py
```

`mswd/Example.py` 位于子目录中，使用的是 `from mswd_bootstrapping import ...`，因此更适合在 `mswd/` 目录内运行：

```bash
cd mswd
python Example.py
```

仿真和 bootstrap 默认重复次数较多，例如 `main.py` 默认 `nrun=500`、`opts.nB=500`，运行时间会比较长。调试时可以先降低 `--nrun` 或关闭部分比较方法。
