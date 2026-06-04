# simulation 数值模拟说明

本文件夹用于复现论文 Figure 1 中的 `Model A, p = 500` 数值模拟结果。这里的脚本是基于项目根目录下 `main.py` 的实验逻辑单独整理出来的，不修改原始 `main.py`。

## 1. 实验模型

当前只复现论文中的 Model A，即均值偏移模型：

```text
X ~ N(0_p, Sigma)
Y ~ N(mu, Sigma)
Sigma_ij = 0.5^|i-j|
mu_j = 0.8 * beta * j^(-3)
```

默认参数为：

```text
n1 = 250
n2 = 250
p = 500
alpha = 0.05
beta = 0, 0.2, 0.4, 0.6, 0.8, 1
nrun = 50
```

其中 `beta = 0` 表示零假设情形，用于估计经验 size；`beta > 0` 表示备择假设情形，用于估计经验 power。

## 2. 比较方法

最终图中比较六种方法：

```text
Proposed
MMD-G
MMD-L
ED2
BG
PW
```

其中：

- `Proposed`：论文提出的 L0 sparse max-sliced Wasserstein distance + bootstrap。
- `MMD-G`：Gaussian kernel MMD。
- `MMD-L`：Laplace kernel MMD。
- `ED2`：基于 L2 距离的 energy distance。
- `BG`：Biswas-Ghosh 类型检验。
- `PW`：projected Wasserstein distance，代码中使用 `k = 3`。

`ED1`、`MSWD-L1`、`sMMD` 和 `WD_NNapprox` 不在本文件夹的 Figure 1 Model A 复现脚本中运行。

## 3. 文件说明

```text
simulation/
|-- simulation.py
|-- plot_result.py
|-- readme.md
`-- results/
    |-- model_A_raw_results.csv
    |-- model_A_summary.csv
    |-- figure1_model_A.png
    `-- figure1_model_A.pdf
```

`simulation.py`  
负责运行数值模拟。每完成一轮 Monte Carlo 实验，就立即把结果追加写入 `results/model_A_raw_results.csv`。

`plot_result.py`  
负责读取 `results/model_A_raw_results.csv`，按 `beta` 汇总每种方法的经验拒绝率，并绘制 Figure 1 风格的结果图。

`results/model_A_raw_results.csv`  
逐轮原始结果。每一行对应一个完整的 `(beta, run_id)`，保存了随机种子、统计量、阈值、p-value、各方法是否拒绝、运行时间等信息。

`results/model_A_summary.csv`  
汇总结果。每一行对应一个 `beta`，包含该 beta 下已完成轮数和六种方法的经验拒绝率。

`results/figure1_model_A.png` 与 `results/figure1_model_A.pdf`  
最终绘制的 Model A 对比效果图。

## 4. 如何启动实验

请先进入项目根目录：

```powershell
cd "D:\20 生物统计学\mswd-bootstrapping"
```

运行单个 beta，例如 `beta = 0`：

```powershell
python .\simulation\simulation.py --beta 0
```

运行单个 beta，例如 `beta = 0.2`：

```powershell
python .\simulation\simulation.py --beta 0.2
```

运行全部 beta：

```powershell
python .\simulation\simulation.py --beta all
```

默认每个 beta 跑满 50 轮。如果只想临时测试，可以指定较小的轮数：

```powershell
python .\simulation\simulation.py --beta 0 --nrun 2
```

## 5. 断点续跑机制

脚本支持断点续跑。运行时会先读取：

```text
simulation/results/model_A_raw_results.csv
```

如果某个 `(beta, run_id)` 已经存在且结果完整，就会自动跳过。比如：

```text
beta = 0.2 已经完成 30 / 50 轮
```

下一次运行：

```powershell
python .\simulation\simulation.py --beta 0.2
```

脚本会从缺失的第 31 轮继续运行，直到补满 50 轮。

如果某个 beta 已经跑满 50 轮，会提示：

```text
beta=0.2: already completed 50/50; no more runs needed.
```

## 6. 如何画图

当已经有部分或全部模拟结果后，可以运行：

```powershell
python .\simulation\plot_result.py
```

该命令会生成：

```text
simulation/results/model_A_summary.csv
simulation/results/figure1_model_A.png
simulation/results/figure1_model_A.pdf
```

如果所有 beta 都跑满 50 轮，最终图就是基于 `6 x 50 = 300` 次完整模拟得到的 Figure 1 Model A 复现图。

## 7. 常用参数

`--beta`  
指定要运行的 beta。可以传单个值，也可以传 `all`。

```powershell
python .\simulation\simulation.py --beta 0.6
python .\simulation\simulation.py --beta all
```

`--nrun`  
每个 beta 目标完成轮数，默认 50。

```powershell
python .\simulation\simulation.py --beta 0.6 --nrun 50
```

`--proposed-B`  
Proposed 方法 bootstrap 次数，默认 300，与原始 `main.py` 中 L0 proposed 部分一致。

`--perm-nperm`  
MMD-G、MMD-L、ED2、BG 的 permutation 次数，默认 500。

`--pw-nperm`  
PW 方法的 permutation 次数，默认 500。

`--tune-reps`  
Proposed 方法调参时的随机初始方向重复次数，默认 5。

`--reps`  
Proposed 方法正式优化时的随机初始方向重复次数，默认 10。

## 8. 注意事项

1. 实验运行时间较长，这是正常现象。每一轮包含 Proposed 方法的 bootstrap、多个 permutation 方法和 PW 方法。
2. 每轮完成后会立即保存，因此可以放心中断；下次运行会继续补齐未完成轮次。
3. 如果机器没有 CUDA，请检查 `pyCode/global_var.py`，必要时把 `device = 'cuda'` 改为 `device = 'cpu'`。
4. 不建议手动编辑 `model_A_raw_results.csv`。如果确实需要删除某个 beta 的结果，可以先备份该 CSV，再删除对应 beta 的行。
