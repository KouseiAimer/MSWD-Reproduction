# simulation-main 实验设置与说明

本文件夹用于在不修改根目录 `main.py` 的前提下，重复运行 `main.py` 来复现论文 Figure 1 中 Model A 的数值模拟结果，并增加结果保存与断点续跑机制。

## 1. 核心原则

`simulation-main.py` 只负责外层调度，不重写 `main.py` 的内部实验流程。

每次真正执行的逻辑等价于：

```powershell
python main.py --signal <0.8 * beta>
```

也就是说，外层脚本只向 `main.py` 传入 `signal` 参数。`main.py` 中其他默认参数全部保持原样，包括：

```text
nrun = 500
n1 = 250
n2 = 250
sample_dim = 500
alpha = 0.05
```

因此这里的唯一外部超参数是 `beta`，并按论文 Model A 的设定转换为：

```text
signal = 0.8 * beta
```

## 2. Model A 设置

论文 Figure 1 的 Model A 是均值衰减模型：

```text
X ~ N(0_p, Sigma)
Y ~ N(mu, Sigma)
Sigma_ij = 0.5^|i-j|
mu_j = 0.8 * beta * j^(-3)
```

本项目中 `dataGenerate(..., model="mean-decay")` 使用的是：

```text
mu_j = signal * j^(-3)
```

所以外层脚本使用 `signal = 0.8 * beta` 与论文设定对应。

默认 beta 网格为：

```text
beta = 0, 0.2, 0.4, 0.6, 0.8, 1
```

## 3. 对比方法

最终记录并用于画图的六种方法为：

```text
Proposed
MMD-G
MMD-L
ED2
BG
PW
```

其中 `Proposed` 对应 `main.py` 中的 L0 sparse max-sliced Wasserstein distance + bootstrap 方法。

## 4. 外层重复次数

`main.py` 自己内部默认已经运行 `nrun=500` 轮。

`simulation-main.py` 的参数 `--main-runs` 表示：每个 beta 要完整重复运行多少次 `main.py`。默认值为 10。

因此默认情况下，每个 beta 会保存 10 行结果；每一行都是一次完整 `main.py` 运行后得到的经验拒绝率，而不是单个 Monte Carlo 样本。

如果只想每个 beta 运行一次完整的 `main.py`，可以设置：

```powershell
python .\simulation-main\simulation-main.py --beta 0.2 --main-runs 1
```

## 5. 输出文件

结果保存到：

```text
simulation-main/results/model_A_main_outputs.csv
```

每一行对应一个完整的：

```text
(beta, main_run_id)
```

主要字段包括：

```text
beta
signal
main_run_id
seed
main_default_nrun
n1, n2, sample_dim, alpha
proposed_rate
mmd_g_rate
mmd_l_rate
ed2_rate
bg_rate
pw_rate
stdout_log
runtime_seconds
started_at
finished_at
```

其中 `*_rate` 是该次 `main.py` 内部 `nrun=500` 轮得到的经验拒绝率。

控制台输出日志会单独保存到：

```text
simulation-main/logs/
```

## 6. 断点续跑

脚本每完成一次完整 `main.py` 运行，就立即向 CSV 追加一行结果。

如果程序在某一次 `main.py` 运行中途停止，该次结果不会写入 CSV；下次运行同样命令时，会重新执行这个缺失的 `main_run_id`。

如果某个 beta 已经完成指定的 `--main-runs` 次完整运行，脚本会自动跳过，并提示该 beta 已经完成。

## 7. 启动方式

进入项目根目录：

```powershell
cd "D:\20 生物统计学\mswd-bootstrapping"
```

运行单个 beta：

```powershell
python .\simulation-main\simulation-main.py --beta 0.2
```

运行全部 beta：

```powershell
python .\simulation-main\simulation-main.py --beta all
```

指定每个 beta 完整运行几次 `main.py`：

```powershell
python .\simulation-main\simulation-main.py --beta all --main-runs 10
```

## 8. 结果解释

`beta = 0` 时，经验拒绝率应接近显著性水平 `alpha = 0.05`，用于观察 size 控制。

`beta > 0` 时，经验拒绝率越高，说明该方法对均值衰减差异的检测功效越强。

后续画图时，可以对同一 beta 下的多行 `*_rate` 再取平均，得到每种方法在该 beta 下的最终经验拒绝率曲线。
