# 01 WONN 机制与实验平台(零号积木)

> 本文件是一切的地基:WONN 是什么(原论文机制,公式完整推导)、我们的
> toy LM 实验平台长什么样、WONN 怎么塞进 transformer(WinfreeCore 全链
> 张量流)、全部实验轴的语义。**只读本文件即可获得复现所需的全部机制
> 知识**;实验怎么做、结果如何,见 `02_演化主叙事.md` 与
> `03_历史结果表格全录.md`。

---

## 0. 预备知识:术语与符号清单(前置)

| 术语 / 符号 | 定义 |
|---|---|
| 振荡子(oscillator) | 一个有相位的周期性单元,状态是圆周上的一个角度 |
| 相位 $\theta_i \in \mathbb{T}$ | 第 $i$ 个振荡子的角度,$\mathbb{T} := \mathbb{R}/(2\pi\mathbb{Z})$(实数模 $2\pi$,即圆周 $S^1$)——$0$ 与 $2\pi$ 是同一点 |
| 固有频率(natural frequency)$\omega_i$ / $\Omega$ | 没有耦合时振荡子的转速(漂移项);WONN 网络版里由输入决定,故是**内容载体** |
| sensitivity $S(\theta)$ | Winfree 交互的**接收方门控**——第 $i$ 个振荡子对平均场的敏感度随自身相位变化(三角版 $=\cos\theta_i$) |
| influence $I(\theta)$ | Winfree 交互的**发送方信号**——第 $j$ 个振荡子向平均场广播的量(三角版 $=\sin\theta_j$) |
| 耦合系数 $c_{ij}$ | 谁影响谁、影响多强 |
| $\operatorname{wrap}(\theta)$ | 角度回 $[-\pi,\pi)$;官方实现为 `atan2(sin θ, cos θ)`(数值稳定的取模) |
| 步长 $\gamma$ | 连续时间方程离散化的积分步长;原论文 Maze 实现是 $\gamma=0.1$ 的冻结标量(`requires_grad=False`) |
| 层数 $L$ / 步数 $T$ | Winfree 动力学层数 / 每层内相位迭代次数(权重跨 $t$ 共享) |
| 环面(torus)$(S^1)^m$ | $m$ 个圆周的乘积流形——$m$ 维振荡子群体的联合状态空间 |
| AKOrN | 前作「Artificial Kuramoto Oscillatory Neurons」(Miyato et al. 2025)——Kuramoto 型、状态是高维单位向量;WONN 的直接对照 |
| LM | language model,语言模型 |
| toy LM | 小规模语言模型(本实验自建,见 §2)——「toy」指刻意小,便于快速扫实验轴 |

---

## 1. WONN 是什么——原论文机制

**WONN**(Winfree-type Oscillator Neural Network,Dai & Song,
arXiv 2605.20922)把神经网络的一层换成**耦合振荡子群体的弛豫动力学**。
本节从统计物理的零号积木讲起,到论文的 Algorithm 1,公式不跳步。

### 1.1 零号积木:耦合振荡子为什么能「计算」

耦合振荡子的同步是统计物理/非线性动力学的经典课题。Winfree 1967 研究
生物节奏(萤火虫群同步闪烁、心脏起搏细胞、昼夜节律)时提出:每个振荡子
有自己的固有节奏 $\omega_i$,同时感受全群的「平均场」并被它牵引——
coupling 足够强时群体锁相(phase locking)。Kuramoto(1975/1984)给出最
著名的可解版本:全局耦合、成对相位差交互,耦合强度 $K$ 超过阈值 $K_c$
时序参量 $r = \left|\frac{1}{N}\sum_j e^{i\theta_j}\right|$ 从 $0$ 自发
变正(连续相变,$U(1)$ 对称自发破缺)。

**「同步 = 计算」的思想**:与其把表示学习看作一串静态特征变换(前馈
网络),不如把计算看作**相互作用动力学系统的集体演化**——解不是某层
算出来的,是振荡子群体在频率场与耦合的竞争下**弛豫**(relax)到的
吸引子。论文在 Maze-hard 迷宫求解任务上给出了可看的实证:早期多个
弥散的候选路径片段同时激活(互不一致、含非法转移),动力学演化中不相
容的片段被抑制、相容的互相强化,渐进合并成一条起终点连贯的路径——
不是逐步贪心构造,而是**并行候选 + 集体对齐**。

**为什么值得在 LM 上试**(假设,非论文主张):序列生成里有一类错误
正是「局部歧义 + 上下文消歧」——弱证据下该往哪个 token 收束。若把
「候选假设互相竞争、上下文锁定胜者」看作约束传播(constraint
propagation),振荡子同步天然是这种计算的连续松弛形式。另外 Winfree
迭代是**测试时可调的算力轴**($T$ 可以推理时加大),这给了 inference-
time scaling 一个不增参数的旋钮。诚实边界:WONN 论文自己**没做过任何
序列生成/语言任务**(论文附录 B.1 明说 language models 是 future
work)——这个假设正是本实验线要检验的东西,不是有文献支撑的预期。

### 1.2 Winfree 方程族:从 Kuramoto 到可分离交互

**直觉先行**:Kuramoto 的交互只看**相位差**($\theta_j - \theta_i$),
所以全体相位整体平移 $\alpha$ 动力学不变——全局 $U(1)$ 对称。这个对称
优美但限制了可达的集体行为。Winfree 把交互拆成两半:**接收方的敏感度
$S$**(我此刻的相位决定我多容易被带动)乘上**发送方的影响 $I$**(你
此刻的相位决定你广播什么)——两半各自是绝对相位的函数,乘积不再只
依赖相位差,对称破缺,行为更丰富。

**Kuramoto 模型**(Kuramoto 1984):

$$\dot{\theta}_i = \omega_i + \gamma \sum_{j=1}^{N} K_{ij}\sin(\theta_j - \theta_i)$$

逐符号:$\dot\theta_i$ 是第 $i$ 个振荡子相位的时间导数;$\omega_i \in
\mathbb{R}$ 固有频率;$\gamma \in \mathbb{R}_+$ 耦合强度;$K_{ij}$ 耦合
矩阵(谁连谁);$\sin(\theta_j-\theta_i)$ 把相位差卷到 $[-1,1]$ 的牵引
力——落后了($\theta_j > \theta_i$)被拉快,超前了被拉慢。

**广义 Winfree**(Manoranjani et al. 2023 对 Winfree 1967 的推广):

$$\dot{\theta}_i = \omega_i + \gamma \sum_{j=1}^{N} K_{ij}\big[\sin(\theta_j - \theta_i) + q\,\sin(\theta_j + \theta_i)\big]$$

$q$ 控制对称破缺程度:$q=0$ 精确退回 Kuramoto。**$q=1$ 的代数**(和差
化积 $\sin A + \sin B = 2\sin\frac{A+B}{2}\cos\frac{A-B}{2}$,取
$A = \theta_j - \theta_i$、$B = \theta_j + \theta_i$,则
$\frac{A+B}{2} = \theta_j$、$\frac{A-B}{2} = -\theta_i$,而
$\cos(-\theta_i) = \cos\theta_i$):

$$\sin(\theta_j-\theta_i) + \sin(\theta_j+\theta_i) = 2\cos\theta_i\,\sin\theta_j \;\;\Longrightarrow\;\; \dot{\theta}_i = \omega_i + 2\gamma\cos\theta_i \sum_j K_{ij}\sin\theta_j$$

这正是**可分离形式**(Winfree 原始模型):$S(\theta_i) = \cos\theta_i$、
$I(\theta_j) = \sin\theta_j$——它们来自交互函数的一阶 Fourier 分量。
WONN 的三角参数化直接用这一对。

### 1.3 能量结构与拓扑阻碍(论文附录 D)

取三角交互、对称耦合 $c_{ij} = c_{ji}$,定义交互能量:

$$E_{\mathrm{int}}(\Theta) = -\tfrac{1}{2}\sum_{i,j} c_{ij}\sin\theta_i\sin\theta_j$$

对 $\theta_i$ 求偏导($\partial \sin\theta_i/\partial\theta_i =
\cos\theta_i$;交叉项在对称耦合下相等,$\frac12 \times 2 = 1$):

$$\frac{\partial E_{\mathrm{int}}}{\partial \theta_i} = -\cos\theta_i\sum_j c_{ij}\sin\theta_j$$

于是动力学可**分解**为 $\dot\theta_i = \omega_i - \gamma\,\partial
E_{\mathrm{int}}/\partial\theta_i$——**漂移 + 梯度流**。沿轨迹求能量
变化率(链式法则 $\frac{dE}{dt} = \sum_i \frac{\partial E}{\partial
\theta_i}\dot\theta_i$):

$$\frac{dE_{\mathrm{int}}}{dt} = \underbrace{\langle \nabla_\Theta E_{\mathrm{int}},\, \omega\rangle}_{\text{漂移项,不定号}} \;-\; \underbrace{\gamma\,\|\nabla_\Theta E_{\mathrm{int}}\|_2^2}_{\text{梯度项,}\le 0}$$

三个推论(论文附录 D 逐条证明):

1. **$\omega = 0$ 时纯梯度流**:$E_{\mathrm{int}}$ 是 Lyapunov 函数,
   单调下降到局部极小——同步态 = 能量极小,振荡子群体「下坡」。
2. **$\omega \neq 0$ 时无全局势函数**——拓扑论证:若纯漂移
   $\dot\theta_i = \omega_i$ 是某个全局 $E$ 的负梯度,则局部解出
   $E = -\sum_i\omega_i\theta_i + \mathrm{const}$;但环面上
   $\theta_i \sim \theta_i + 2\pi$,绕基本闭路一圈 $E$ 变化
   $-2\pi\omega_i \neq 0$,非单值,矛盾。等价说法:漂移场沿闭路的环流
   $\oint \omega_i\, d\theta_i = 2\pi\omega_i \neq 0$,而梯度场环流必为
   零。(物理直觉:有旋度的地方没有标量势——磁矢势绕 Aharonov–Bohm
   相位同类。)
3. **含义**:固有频率(= 网络里的输入内容)阻止系统塌缩到平衡——漂移
   与下坡的竞争产生极限环、锁频等非平衡定态,**内容驱动 + 耦合整形**
   的动力学。这正是「$\Omega$ 是内容载体」的物理根据。

### 1.4 WONN 架构:Algorithm 1

**总览**:输入 $x$ 编码成初始频率场 $\Omega$,相位 $\Theta$ 随机初始化;
经过 $L$ 层 Winfree 动力学(每层 $T_l$ 次权重共享的迭代 + 一次层间
转换),最后从 $\Theta$ 读出预测。**输入只进 $\Omega$,不进 $\Theta$**
——内容全部由频率携带,相位从无结构的随机态出发、被动力学组织起来。

**双状态与两时间尺度**:
- $\Theta \in (S^1)^{d}$:相位(**快变量**)——层内 $T_l$ 次迭代反复重组织;
- $\Omega \in \mathbb{R}^{d}$:频率(**慢变量**)——每层只更新一次,跨层
  缓慢演化(携带内容)。论文明说这个分离「decouples transient
  oscillatory computation from cross-layer feature transformation」
  (瞬态振荡计算与跨层特征变换解耦)。

**初始化**(官方 Maze 实现实锚):

```python
omega_base = self.f_init(inp)                        # 可学习查表
theta = self.wrap_pm_pi(0.1 * torch.randn_like(omega))  # Θ ~ wrap(0.1·N(0,1))
```

$\Theta$ 初始化为 $0.1\times$ 标准正态再 wrap——**小相位抖动**(接近
全同相位起跑,避免初始随机性主导动力学)。

**层内迭代**(论文 Eq. 5 / Alg. 1;每层 $t = 1,\dots,T_l$):

$$\theta_i^{(l,t+1)} = \operatorname{wrap}\Big(\theta_i^{(l,t)} + \gamma\big[\omega_i^{(l)} + S(\theta_i^{(l,t)})\cdot \big(\textstyle\sum_j c_{ij}\, I(\theta_j^{(l,t)})\big)\big]\Big)$$

**官方实现的逐算子对照**(Maze 版,point-wise、三角 $S/I$、attention
coupling):

```python
def winfree_step(self, theta, omega):
    theta = self.wrap_pm_pi(theta)           # wrap = atan2(sin, cos)
    sensitivity = torch.cos(theta)           # S(θ) = cos θ,零参数
    influence = torch.sin(theta)             # I(θ) = sin θ,零参数
    field = self.coupling(influence)         # C:Sequential(Attention, GroupNorm, ReLU)
    dtheta = omega + sensitivity * field     # Winfree 更新:Ω + S ⊙ C(I)
    energy_int = influence * field           # E_int 的逐元素被加项(监控用)
    return dtheta, energy_int
```

耦合算子 $\mathcal{C}$:输入只有 `influence`($\sin\theta$,不是原始
$\theta$);`StandardAttention`(标准多头自注意力:$W_{qkv}$ 线性投影 →
分头 → 2D RoPE → softmax 加权 → $W_o$ 出投影)后接 `GroupNorm` + `ReLU`
——场是非负半空间截断的,这不在论文公式里(实现细节)。attention 是
**双向的**(softmax over 全序列)——这是序列化任务必须改 causality 的
地方(我们全部实验的 WONN 挂点都用**因果**耦合,见 §3)。

**层间转换**($L>1$ 才激活;Maze/Sudoku 用 $L{=}1$,转换层是 Identity):

$$\Theta^{(l+1)} = \texttt{ThetaUpdate}(\Theta^{(l,T_l)}), \qquad \Omega^{(l+1)} = \texttt{OmegaUpdate}(\Omega^{(l)}, \Theta^{(l,T_l)})$$

- `ThetaUpdate`(相位刷新,替换式):$\Theta_{\mathrm{new}} =
  \operatorname{atan2}\big(\mathrm{Conv}(\sin\Theta),\
  \mathrm{Conv}(\cos\Theta)\big)$——同一个 conv 权重分别作用于
  $\sin\Theta$ 和 $\cos\Theta$(weight-tied;论文 Eq. 8 的解读:线性算子
  作用于复表示 $e^{i\Theta}$ 后取辐角——**在圆周上做线性组合的正确
  方式是直角坐标组合再取辐角**,而不是把角度直接平均)。
- `OmegaUpdate`(频率刷新,残差式):`omega = omega_base +
  OmegaUpdate(theta, omega)`——$\Theta$ 的 embedding($[\sin\theta,
  \cos\theta]$ 过 depthwise 1×1 conv)与 $\Omega$ 拼接过 conv + BN +
  ReLU 得增量;**`omega_base`(初始编码)每层都加回来**——输入内容的
  锚定不丢。这是官方实现里**唯一**的 $x + f(x)$ 残差结构。

**输出头**:从**末层末步相位** $\Theta^{(L,T_L)}$ 读出($\Theta$ 的
$[\sin,\cos]$ embedding → 投影出 logits),不从 $\Omega$——计算的答案
沉淀在相位组织里,频率只是驱动。

### 1.5 论文证据与边界

**结果**(3 seeds 均值):

| 任务 | WONN 最佳 | 参数量 | 对照 | 超参(L/T/N) |
|---|---|---|---|---|
| CIFAR-10 | 95.26% | 11.86M | ViT-S 93.43% / 21.34M | 6/3/2 |
| CIFAR-100 | 76.20% | 12.04M | ViT-S 71.18% / 21.38M | 6/3/2 |
| ImageNet-1K | 76.78% | 12.28M | ResNet-50 76.89% / 25.56M(一半参数打平) | 6/3/2 |
| Maze-hard | 76.2%(energy voting 80.1%) | **0.396M** | HRM 74.5% / 27M(1.5% 参数) | **1/24/1** |
| Sudoku | **100.0%** | 1.58M | Transformer 98.6% / 16.87M | **1/16/1** |

规律(论文附录 C.2 消融):**感知任务用多层浅迭代**($L{=}6, T{=}3$),
**推理任务用单层深迭代**($L{=}1, T{=}16\sim24$)——约束传播要的是长
弛豫时间,不是深堆叠。$T$ 与 $L$ 是两个独立的算力轴。

**内部现象——相位双峰**(论文 §3.2 Fig. 4):末层末步 $\theta$ 的直方
图呈双峰,两个 dominant modes **相隔 ~π**(反相双集群);membership 映
射回图像空间后两群分工到互补视觉结构。物理解读:反相双集群 = 耦合振
荡子经典解;$\theta$ 与 $\theta+\pi$ 的 sin/cos 全反号,两群是 **± 两条
信号信道**(BPSK 式二相调制)。

**AKOrN 对比**(理解 WONN 定位用):
- 几何:AKOrN 振荡子是高维**单位向量**(欧氏嵌入,周期性是软约束);
  WONN 直接用**相位角**(环面是硬几何,周期性精确);
- 交互:AKOrN 是 Kuramoto 型成对相位差(全局 $U(1)$ 对称);WONN 是
  Winfree 可分离 $S\cdot I$(对称破缺,可达动力学更丰富);
- 算力:ImageNet-1K 上 WONN 750 vs AKOrN 950 GPU·h,显存 35GB vs 110GB。

**边界(诚实清单,全部要改造才能上 LM)**:
1. **没做过序列生成/LM**(论文 B.1 原文明说 language models 是 future
   work)——本实验线正是这个方向的第一手数据;
2. attention coupling **双向**(无 causal 掩码)——自回归解码必须改;
3. $\gamma$ 固定 0.1 不学习;$\Theta$ 每 forward 随机初始化——没有跨
   样本/跨时间步的状态持久性;
4. 任务头是分类/逐格预测(非自回归 teacher forcing);输入是整图/整盘
   (无流式)。

---

## 2. 我们的 toy LM 实验平台

### 2.1 研究问题与策略

**问题**:WONN 作为 transformer 的组件,在语言建模上是否有益(相对
FFN/attention 等标准组件)?

**策略**:不直接上大模型——先在**刻意小**的 toy LM 上把实验轴扫透。
理由:① 单卡小时级可跑完一个 run,一晚能扫一个 $10^2$ 量级的矩阵;
② 机制问题(有没有用、怎么挂、什么形态)在小尺度就能显形;③ 大模型
迁移是后话(原始 TODO 里的大参数量路线,见 02 尾声)。

### 2.2 平台规格(全实验线恒定)

| 项 | 值 | 说明 |
|---|---|---|
| 层数 | 48 | 全 dense decoder |
| hidden 维 $d$ | 512 | |
| FFN 中间维 $d_{ff}$ | 1408 | SwiGLU 激活 |
| 参数量 | 232.5M | |
| **Loop 段** | 第 16-31 层(16 层) | WONN 挂载区——只在这 16 层上做文章;前 16 层(Prelude)与后 16 层(Coda)是纯标准层 |
| 词表 | 151,669(训练态 pad 到 151,672) | Qwen tokenizer |
| 位置上限 | 1024 | |
| 数据 | 四域混合 752M tokens | Orca-Math(数学 CoT,真名 `microsoft/orca-math-word-problems-200k`)+ TinyStories V1(叙事)+ TinyGSM(数学,Python 解题码)+ ProofWriter(逻辑证明)——train 2,224,862 行 / 776.79M tokens(名义 752M,实际 +3.3%);val 8,000 行(四域各 2,000)。**构造全链(下载/prep/shuffle/运行时)见 §4.1** |
| 序列 | max_length 1024,截断策略 = error(超长样本在数据准备期已被过滤 0.13%) | |
| batch 结构 | global batch 256 | dataloader 每 step 出 256 行(`data.train_batch_size=256`),trainer 内切 16 个 micro(`data.micro_batch_size_per_gpu=16`,每 micro 16 行 × 1024 tok)做梯度累积——两键均 run 脚本 hydra 实配;单卡 nproc=1 跑(非数据并行) |
| 每 epoch 步数 | 8,690 | 2,224,862 行 ÷ 256 ≈ 8690.87,sampler drop_last 后 8690 步 = 恰 1 epoch;数据顺序所有 run bit 级相同(为什么,见 §4.1 运行时层) |
| val 在线打点 | test_freq=100 | 每 100 步打一次在线 val(final step 无条件补跑,8690 % 100 = 0 → 收官档末点 = step 8690);口径与换代见 §4.2 |
| ckpt 存档 | save_freq=500 | 每 500 步存一次(run 脚本 `trainer.save_freq=500` 实配,save_contents = model/optimizer/extra);8690 非 500 倍数,最后一个常规存点 = 8500——**C′ val 用 @8500 权重的来源**(03 §0.3);final step 另有上游原生 is_last_step force save(fit 循环内置) |
| 优化器 | **Muon**(三组;分组详解见档案 1 的 04) | 2D 矩阵 → Muon / embedding+head → AdamW / 1D(bias/LN/标量)→ AdamW;385 个 Muon 参数(175.77M)+ 2 个 head-AdamW 组(78.18M)+ 386 scalar-AdamW(0.20M) |
| lr | 1e-3(默认家族) | cosine 退火到 0,分母 8,690;warmup 434 步(= 8690 × 5%)。**换代**:rwi34 起各批改 3e-4(同形纯 scale ×0.3,warmup/cosine 形状不变——换代事件见 02 §6.3)——读跨批表格先对 lr 代 |
| seed | 42 / 1234 / 2345(主三 seed;+3456 备用) | |
| 训练长度 | 1 epoch = 8,690 步,**完整退火跑满** | 中途不早停——「我们好像从 annealing 里面 benefit 很大」(实跑多次证实:完整退火前 WONN 档常高于 baseline,末段才反超或收敛) |
| 判读窗 | **w100**:train loss @8690 ± 100 步均值(且 lr < 1.1e-7,确认已退火到底) | |

### 2.3 关键工程事实:STEPS 双属性解耦

训练框架(verl SFT trainer)有两个步数属性,**各喂一个消费方**:
`total_steps` = len(dataloader) × epochs = 8690,喂 **lr scheduler 的
cosine 分母与 warmup**;`total_training_steps` 可被挂单 env(STEPS=8690)
覆盖,喂**停止判据**。挂单时 STEPS 只写 8690(与分母一致)即语义
自洽;曾有一批(供奉七批 B 第一版)把 STEPS=4000 当「跑 4000 步的
批」挂出,结果 cosine 分母仍是 8690(warmup 434 而非设计 200,停点
cosine 只走到 ~46%)——审计后拍板「改文档追实装」保留原始曲线语义。
详见档案 1 的 04(训练基建讲义)。

---

## 3. WONN 塞进 transformer:WinfreeCore

### 3.1 全链张量流

所有实验的核心组件是 `WinfreeCore`(源码
`scripts/di/v2_modeling/modeling_toy_wonn.py`,档案 1 的 `scripts/`
内有脱敏原文)。**读法 B core** 的张量流:

```
输入 u [B, N, d_in]
   │
   ├─→ Ω = W_in·u            [B, N, m]     (per_layer 模式:每层新鲜投影;
   │                                        three_source 模式:见 §3.4)
   ├─→ Θ 初始化               [B, N, m]     (random / proj / carry 三档,见 §3.4)
   │
   ├─→ T × Winfree step(权重跨 t 共享):
   │       θ ← wrap(θ)
   │       S = cos θ,  I = sin θ
   │       field = c_ij(I)                 (耦合:c_ij 可 dense Linear /
   │                                        banded / 块对角 [G,m,m],见 §3.4)
   │       dθ = Ω + S ⊙ field
   │       θ ← θ + γ·dθ                    (γ = 0.1)
   │
   ├─→ readout:r = W_out·[cos θ ‖ sin θ]   [B, N, d_out]
   │                                        W_out **零 init**(no-op 起步)
   └─→ r 回注主干(挂点决定注哪里,见 §3.3)
```

形状记号:$B$ = batch,$N$ = 序列长,$d_{in}/d_{out}$ = 主干接口维
(512),$m$ = 振荡子数(默认 256)。

**参数账**($m{=}256$, $d_{in}{=}d_{out}{=}512$):$W_{in}$ 131K +
$c_{ij}$ 65.5K(dense $m{\times}m$ Linear)或 8.4K(banded $k{=}16$)+
$W_{out}$ 262K(+ $W_a/W_b$ 2×131K 当 `theta_init=proj`)。
`three_source` 模式:$W_{in}$ 换成 $w_{\omega}$ $[d_{in}+m+2m
\rightarrow m]$ 327.7K 零 init(ΔΩ 投影,输入 concat $[h ‖
\Omega_{prev} ‖ \cos/\sin \Theta_{prev}]$)。

### 3.2 T 步递推的精确形式(eager 参考实现)

`WinfreeCore.forward` 的 T 步循环本体(源码直译):

```python
for _ in range(self.T):
    theta = wrap_pm_pi(theta)
    sens, infl = torch.cos(theta), torch.sin(theta)
    if self.coupling == "dense":
        field = self.c_ij(infl)            # Linear(m, m):Σ_j c_ij·sin θ_j
    # (banded / 块对角分支见源码)
    dtheta = omega + sens * field          # Ω + cos θ ⊙ field
    theta = theta + self.gamma * dtheta    # γ = 0.1
```

**与官方 Maze 实现的对应**:同一个 Winfree step($\operatorname{wrap}$
→ $S/I$ → coupling → $d\theta = \Omega + S \odot C(I)$ → 累加),差异
只在耦合算子(官方是 attention+GN+ReLU,我们是 Linear / banded / 块
对角)与因果性(官方双向,我们全部**因果**——序列任务要求)。

### 3.3 挂点谱系(十四形态,形态轴 `wonn_form`)

WONN 挂在 transformer 的哪个位置?第一轮形态扫描(L12 时代设计,
L48 上沿用到后续批次)的十四个形态:

| # | 形态名 | 挂点语义 |
|---|---|---|
| 1 | `baseline` | 无 WONN(纯 FFN 对照) |
| 2 | `ffn_wide` | FFN 加宽($d_{ff}$ 2816,仅 Loop 层)——参数对齐臂 |
| 3 | `per_layer_t3` | 每层独立 WONN,$T{=}3$ |
| 4 | `segment_t16` | 段级深弛豫:Loop 段末每 token 一局,Ω 一次注入冻结 + Θ 随机 16 步弛豫(官方推理任务 L=1 T=16 模式的平移) |
| 5 | `banded` | 耦合带宽化(每振子只连 $k{=}16$ 个邻居,$c_{ij}$ 参数 8.4K) |
| 6 | `②b head 板` | 振荡子按 head 分组(耦合在 head 维组织) |
| 7 | `head-only` | 只在 head 段挂 |
| 8 | `moe_v2` | 振荡子 MoE(路由到多套 WONN) |
| 9 | `swish_gate` | WONN 输出做 FFN 的 SwiGLU gate |
| 10 | `stress_token_time` | recurrent cell:token 轴与弛豫时间轴合并(K=3 圈,相位选专家 MoE——带动力学记忆的 router) |
| 11 | `a2_hybrid` | 振荡器版状态空间模型(SSM)——与 Mamba/RWKV 同族的设计 |
| 12 | `dual_stream` | 双流完全体:$r$ 侧流与 $h$ 主干流并行 |
| 13 | `attn_gate_parallel` | WONN 与 attention 并行,gate 汇合 |
| 14 | `ffn_gate_cross` | **跨层 gate**:WONN(l) 吃上一层 MLP 的中间激活 gate_act(l−1),产出加性调制项给本层 MLP——设计语义「l−1 层 gate 内容 → WONN 平行跑 → l 层 gate 消费」 |

后续批次的主力挂点是 **`ffn_gate_cross`**(★ 档基座)+ θ init
`proj`(Θ 从输入投影 atan2 定相)——它在多轮判读中表现最稳,成为
RWI 系批次(`wonn_form=baseline` + `wonn_gate=ffn_gate_cross` 两键制)
的默认宿主。

### 3.4 实验轴总语义表(全批次共用词汇)

后续批次在 WinfreeCore 上叠加了一族正交的实验轴。**每个轴的 config
键、取值、语义**——这是读 02/03 全部批次表格的词汇表:

**θ 初始化(`wonn_theta_init`)**:
- `random`:Θ ~ wrap(0.1·N(0,1))(官方同款,每 forward 重采样);
- `proj`:Θ = atan2(W_b·u, W_a·u)(从输入投影定相,前向确定)。

**Ω 模式(`wonn_omega_mode`)**:
- `per_layer`:Ω = W_in·u 每层新鲜投影(**无激活**——`omega_act`
  键对它不生效);
- `three_source`:Ω = Ω_init 底座(段首快照常驻)+ W_Ω·[h ‖ Ω_prev ‖
  Θ_emb] 零 init 增量——跨层接力链(ΔΩ 要过梯度);
- **ΔΩ 激活(`wonn_omega_act`)**:three_source 的 ΔΩ 过 SiLU
  (`silu_split` 档,v5 星档到 ts16 全链实配;Ω_init 底座不带激活
  ——`omega_act` 只在 three_source 的 ΔΩ 分支被消费)。**换轨
  警告**:g8_old 侧四代批(g4/g16/group8/sw8)被 prep 显式压回
  `per_layer`——这四代批既无跨层 Ω 也无 SiLU(与星档语义不同,
  判读污染声明见 02 §9.2)。

**RS 矩阵四旋钮(ABCS,RS trial 1 批引入)**:
- **A**(`wonn_rs_a`,跨 L Θ residual):`off`(A0,每层独立定相)/
  `delta`(A1,Θ 跨层 carry:θ_in 非空时 wrap(Θ_prev + W_δ·u));
- **B**(`wonn_rs_b`,跨 T 耦合场累积):`off`(B0,field 即算即弃)/
  `accum`(B1,h_{t+1} = h_t + field_t,dθ 由累积场驱动);
- **C**(`wonn_rs_c`,主干汇回方式):`off` / `add`(C1,加性:r 预乘
  标量 w 直加进 MLP 流——不走乘性 gate);
- **S**(`wonn_rs_s`,参数共享档):`per_layer`(S0,Loop 段 16 层每层
  一套实例,参数 ×16)/ `shared`(S1,全段共享单实例)。
主力实配 **A0 B0 C1 S0**。

**分组(`wonn_group_g` / `wonn_group_g1`)**:把 $m$ 个振子分成 $G$ 组
(两级版:$m \cdot G_1 \cdot G_2$ 层级)。$G{>}1$ 时 $c_{ij}$ 变块对角
`c_ij_g [G, m, m]`(组内稠密、组间零);$G{=}1$ 走旧 Linear 路径
(bit 级不变)。

**λ 组间混合(`wonn_lambda_mode`)**:
- `off`:组间零交互(4 组完全独立,边界 W_in/W_out 仍稠密通信);
- `static`:裸 Parameter $[G,G]$ 对角 mask 乘零(λ 参数,Kuramoto
  群间强度 $K_{gh}$ 语义;对角永久归零——「场的平均不自耦合」);
- `dynamic`:per-position λ(x) = s_λ·tanh(LN(u)@W_λ)。
三档 init 全零起步(no-op 纪律:差异纯在参数化,不在起点)。
两级版(group8 批):λ1 $[G_2,G_1,G_1]$ 小组间 + λ2 $[G_2,G_2]$
大组间,mask 同款对角归零。

**λ 形式(`wonn_lambda_form`)**——组间平均场的合成方式:
- `mean_sin`:ψ = mean_j sin(θ_j)(g4 批默认);
- `sin_psi`:ψ = sinψ_h,其中 ψ_h = atan2(s̄_h, c̄_h) 是组内相位圆
  均值(s̄ = Σsinθ, c̄ = Σcosθ;两级折叠:小组 ψ1 → 大组 ψ2);
- `phase_diff`:正/反向项差(u·cosθ − v·sinθ)。
ψ 的 Jacobian(clamp 语义精确版):$\partial\sin\psi/\partial s =
1/d_{\mathrm{eff}} - s^2 \cdot \mathrm{mask}/d_{\mathrm{eff}}^3$,
$d_{\mathrm{eff}} = \sqrt{s^2+c^2}.\mathrm{clamp\_min}(10^{-6})$
——clamp 段梯度截断(对跖 batch:一半 θ=0 一半 θ=π 时 s=c=0,
无 clamp 必 NaN)。

**读出(`wonn_readout_pn`)**:`off` / `pn`(各组过 PolyNorm 再
add——PolyNorm = 归一化 + 低秩投影的读出变体,+46.1M 容量,是容量
confound,判读时要注意)。

**支路初始化(`wonn_branch_init`)**:
- 零 init(no-op 起步,W_out 全零——起点与 baseline bit 一致);
- `rwi`(random weight init,RWI 批):「主干怎么 init 支路就怎么
  init」——w_out 走 scaled 档 N(0, 0.02/√96)、w_delta/w_omega 普通
  档 N(0, 0.02)、rs_add_w 标量 1.0。起点非 no-op(支路从第一步就
  参与)。

**实现档(`wonn_impl`)**:`eager`(纯 PyTorch,correctness 参考)/
`triton`(T 步链 fused kernel)/ `vec`(banded 向量化)/ `fast` /
`graph`(CUDA graph)。数值契约:eager 是 ground truth,triton 与
eager 的 bf16 容差先例 3.9e-3(autocast rounding 域)。

**loop 宿主(`loop_repeat`)**:Loop 段每层跑几遍。`loop2` = 每层
2 遍(16 层 × 2 = 32 次 WONN 调用/step)——主干循环复用,与 WONN
正交的容量/算力轴;`loop1` = 现状 1 遍。

### 3.5 baseline 与对照臂族

每个批次的判读都靠对照臂:
- **baseline**:纯 FFN 宿主(无 WONN),三 seed;
- **ffnmatch**(FFN FLOPS match):FFN 加宽到参数/FLOPS 对齐 WONN 档
  (「baseline 不能只是纯 baseline 啊。还要加上一个 FFN FLOPS
  match」——判读 WONN 增益是「有侧支」还是「加了容量」的必要对照);
- **no-op 检查**:零 init 档的起点 logits 与 baseline bit 一致
  (W_out 零 init 的直接推论)——每个新形态实装的验收判据。

---

## 4. 数据与判读

### 4.1 数据构造全链(f_combo 四域混合)

**为什么四域**:单域(Shakespeare 时代)会过拟合遮蔽架构差;四域
混合给「通用 LM 信号」(完整选型决策链见 02 数据选型节)。本节是
**操作层**——从 HF repo 到训练消费顺序的每一步,对着 prep 代码
(`scripts/prep_f.py`)与训练框架代码回答;只读本节即可重建出
**bit 级同序**的数据集。

#### 4.1.1 四域来源与下载(HF repo → 本地)

| 域 | HF repo id(实锚) | 原始形态 | 本地落位 |
|---|---|---|---|
| orca(真名 **Orca-Math**) | `microsoft/orca-math-word-problems-200k` | train.parquet 84MB,200,035 题(question + answer 两列;GPT-4 Turbo 合成 GSM8K 风格数学 word problems,answer = 自然语言 step-by-step CoT) | `data/orca_math/train.parquet` |
| stories(**TinyStories V1**) | `roneneldan/TinyStories`(ungated;GPT-3.5/4 合成儿童故事) | `TinyStories-train.txt` 1.92GB 单文件(2,119,489 篇)+ `TinyStories-valid.txt`(21,990 篇);story 间分隔标签 = hex `3c7c656e646f66746578747c3e`。**repo 另有 V2-GPT4 版,没用——复现要拿对 V1** | `data/tinystories/TinyStories-{train,valid}.txt` |
| gsm(**TinyGSM**) | `TinyGSM/TinyGSM`(ungated,MIT;survey 里 `peiyi9979/TinyGSM` 是错误 id) | 17 个 train shards(共 11,846,109 行;question + code 两列——code 是 Python 解题源码,非自然语言 CoT);**无官方 val split** | `data/tinygsm_raw/data/train-000{00..16}-of-00017.parquet` |
| pw(**ProofWriter**) | `tasksource/proofwriter` | train 2 shards(585,552 行)+ validation + test;theory / question / answer 列 | `data/proofwriter_raw/data/train-0000{0,1}-of-00002-*.parquet` + validation-*.parquet |

**下载方式**:orca/stories 早期批直接从 HF 拉取;gsm/pw 走
`scripts/download_stage_e_data.sh`(ungated LFS 逐文件显式
`hf download`,6 并行 xargs,全程约 6 分钟)。

#### 4.1.2 prep 层构造链(逐域)

**orca(经一层早期物化)**:200,035 题 → qwen tokenize,过滤
「q+a+eos ≤ 1024」→ keep 199,767(drop 268 = 0.13% 超长)→
**rng(42) 洗牌后 split:train 197,767 / val 2,000**(动机:
不洗牌直接切会撞数据集潜在排序,洗牌保 iid)→ 落
`data/orca_qwen/{train,val}.parquet`;f_combo 时代**原样并入**
(prompt=question / response=answer / domain=orca,**无域内 shuffle**)
——注意 orca 的 val 2,000 在这一层就定了,rng 42;f_combo 时代的
seed 43 抽样(下述三域)不作用于 orca。

**stories**:txt 读入 → 按分隔标签 split → 2,119,489 篇(空篇
230 丢弃)→ 逐篇 tokenize(`add_special_tokens=False`)+ **篇间
插 eos `151645`**(Qwen `<|im_end|>`;不沿用语料自带标签的 pad
`151643`——pad 语义会与 SFTDataset 的填充位冲突,eos 则与其他域
response 尾接一致)拼成 token 流 → 总 451,699,812 tok(story 正文
449.58M + eos 2,119,488 个)→ **切 1023-token 定长窗**:train
441,544 个整窗 + 尾窗 300 tok(≥ TAIL_MIN 256,保留)= 441,545 窗;
valid 21,990 篇 → 4,447 窗(尾窗 867 tok ≥ TAIL_MIN 256,同样保留)→ **seed 43 抽 2,000**。**窗长是
1023 不是 1024 的原因**:verl SFTDataset 会对 response 尾接一个
eos → 1023 + 1 = 1024 = max_length;若窗开 1024,加 eos 后 1025
必炸(run 脚本 `data.max_length=1024` + truncation=error)。
**stories 行的 prompt 留空**(tokenize 空串 = 0 token →
prompt_length=0 → loss_mask 不 mask prompt 区)——**纯 LM 全位置
loss**(只有尾接 eos 一个非 loss 位),不是 instruction tuning。

**gsm**:17 shards 全量读入 = 11,846,109 行 → **域内全局 shuffle
(`df.sample(frac=1, random_state=42)`,四域里唯一在域内洗过牌
的)**——动机:不洗直接取 repo head 可能撞生成批次聚集 → 前
1,000,000 行 train + 第 1M-1.02M 行(20,000 行)val holdout →
tokenize 过滤「q+a+1 ≤ 1024」:train drop 2 行超长 → **999,998
行**;val holdout 过滤后 seed 43 抽 2,000 → prompt=question /
response=code(Python 源码)。

**pw**:train 2 shards **原样读入(无域内 shuffle)**585,552 行 →
**label-only 口径**:prompt = `theory + "\n" + question` /
response = answer(**True/False/Unknown 三分类,不喂 allProofs**
——完整证明链是答案侧不参与训练)→ 过滤超长(drop 0)→
validation split 85,468 行同口径过滤 → **seed 43 抽 2,000**。

#### 4.1.3 merge 与行数 token 账

**merge**:四域 train 按 **[orca, stories, gsm, pw] 顺序** concat
= 2,224,862 行 → **全局 shuffle(`df.sample(frac=1,
random_state=42)`)** → 单文件 `train.parquet`(行序固定,所有 run
共用同一文件同一顺序)。val 四域各 2,000 行按 [orca, stories, gsm,
pw] 顺序拼接**不洗**(域等权平均与顺序无关)。终检:train 抽 10k
(`random_state=0`,只影响抽检不影响产物)+ 全 val 复算 seq_len
全 ≤ 1024。

**注意**:merge 的全局 shuffle 会洗掉一切域内顺序——域内顺序只
对 gsm 的「取前 1M」选择起过作用。

parquet 格式 = verl SFT 格式(prompt / response / domain 列;
verl 0.6.0 的 sft_dataset 只消费 prompt_key/response_key 两列)。

| 域 | train 行 | train tok | val 行 | val tok | token 占比 |
|---|---|---|---|---|---|
| orca | 197,767 | 69,706,539 | 2,000 | 707,200 | 8.97% |
| stories | 441,545(窗) | 452,141,357 | 2,000 | 2,047,844 | 58.21% |
| gsm | 999,998 | 185,602,443 | 2,000 | 370,128 | 23.89% |
| pw | 585,552 | 69,343,121 | 2,000 | 236,074 | 8.93% |
| **合计** | **2,224,862** | **776,793,460** | **8,000** | **3,361,246** | 100% |

(名义 752M 实际 776.79M = +3.3%;token 占比 = 各域 train_tokens
归一化,也是 §4.2 C′ val 的 train_mix 权重。)

#### 4.1.4 运行时层(训练框架怎么消费这份 parquet)

- **train 顺序为什么所有 run bit 级相同**:train 用
  `DistributedSampler(shuffle=True, drop_last=True)` 且**不传
  seed 参数** → torch 实现内部每次迭代造进程本地 Generator 并
  `manual_seed(epoch)` → **shuffle seed = epoch 号**;训练
  total_epochs=1 且 8690 < 8690.87(drop_last)→ 全程停留在
  epoch 0 → 不论模型 seed(WONN_SEED)是多少,**数据顺序对每个
  run 都 bit 级相同**(WONN_SEED 只作用于 model init,见
  §2.2);resume 走 StatefulDataLoader 样本级接续,epoch 0 内
  顺序一致;
- **batch 切法**:dataloader 每 step 出 256 行
  (`train_batch_size=256`),trainer 内部把这 256 行切 16 个
  micro(`micro_batch_size_per_gpu=16`)做梯度累积;
- **val 消费**:`DistributedSampler(shuffle=False, drop_last=
  True)` + dataloader batch = micro 16 → 8,000/16 = 500 batch
  整除无丢行;触发条件 = `is_last_step or (test_freq > 0 and
  step % test_freq == 0)`——**final step 无条件跑 val**(8690 %
  100 = 0,收官档末点 = step 8690);
- **超长样本**:max_length 1024、truncation=error——超长样本在
  prep 期已过滤(§4.1.2 各域 drop 数),运行期不会触发截断报错。

**数据侧复现核对清单**(与本节对照即可验证你建的数据集是对的):
① 四域行数 = 表内五个数;② train parquet 总行 2,224,862 且
全局 shuffle seed 42;③ val 8,000 行域序 [orca, stories, gsm, pw]
各 2,000 不洗;④ stories 行长全 = 1024(prompt 空 + response
1023 + eos 1);⑤ 8690 步/epoch(2,224,862 ÷ 256 drop_last)。

### 4.2 判读口径(全批次统一)

- **主口径**:train loss @8690,w100 窗(8690 ± 100 步均值 + lr <
  1.1e-7 确认退火到底);
- **三 seed**:42/1234/2345;跨 seed 方差是判读的噪声底,小于它的
  差异不当结论;
- **对照**:同批 baseline 三 seed 均值为差值基线;
- **val 曲线**:监测训练健康(过拟合/发散早发现)用,实验结论以
  train loss 完整退火末段为准(实验哲学:pretrain loss 的降低已是
  及格线之上的意外之喜,WONN 主战场在下游/生成质量,但下游评测是
  后话,见 02 尾声)。**val 有两代口径**:训练中在线打点
  (test_freq=100,四域等权 batch-mean,~0.55 量级)与 **C′ 独立
  val loop**(forward-only 重跑,verl SFTDataset 原类 bit 级同
  口径 tokenize + bf16 autocast,8,000 行全前向;表格 val 列取
  **train token 配比加权合成键**(stories 58.21% / gsm 23.89% /
  orca 8.97% / pw 8.93%,~0.85 量级,与 train loss 域加权同构)
  ——两代口径数值不可直接比;权重全 @8500 存点,唯 baseline
  s1234/s2345 两格 @8690(本地续跑收官权重);
- **窗口径三代与 Δ 锚换代**(table500 → w25/w100/w200 → merger
  系;bl2 → bl3)的完整表述见 **03 §0**——读任何结果表格先过
  03 §0 再对数。

### 4.3 常见判读陷阱(实跑踩过的)

- **过拟合 U 型**(小数据时代):val loss 中途最低然后回升——判读
  必须看 train + 完整退火;
- **发散**:ε-gate 族在 lr 1e-3 下发散(loss 跑飞)——lr 与形态的
  交互是真实约束(催生 mid-flight decay 批,见 02 时代 3);
- **完整退火前勿下结论**:WONN 档常在退火末段才与 baseline 拉开或
  反超——「最后才低于 baseline 的本来!我们好像从 annealing 里面
  benefit 很大」;
- **容量 confound**:任何带新参数的读出/汇回变体(如 readout_pn)
  都要先问「增益是机制还是容量」——ffnmatch 臂就是干这个的。

---

## 5. 复现 checklist(一个 run 的最小要素)

1. **环境**:PyTorch(2.11 级)+ verl 0.6.0 + transformers + triton;
   GPU(单 run 单卡,micro 16×1024);
2. **数据**:按 **§4.1 全链**重建四域 f_combo parquet(repo id →
   下载 → prep → merge,含核对清单五条——数据顺序与行数要
   bit 级对上,复现实验数字才有效);
3. **模型**:`scripts/di/v2_modeling/modeling_toy_wonn.py`
   (transformers `AutoModelForCausalLM` 接口,config 驱动全部轴);
4. **物化**:用 `scripts/di/prep_v2.py`(V2 系)或 `scripts/di/
   prep_rwi.py`(RWI 系,含 rwi/g4/phase_mean/group8/sw8/ts16 全部
   批次的档名解析)从 backbone 底座物化各档 ckpt(改 `*_V2_ROOT`
   等路径指向你自己的底座树);
5. **训练**:单 run 用 `scripts/di/v2_run_one.sh`(ENTRY 可换
   `v2_sft_entry.py` 系);多 run 批量用 `scripts/di/gen_runs.sh`
   (GEN=批名,tsv 驱动,详见档案 1 的 04);
6. **判读**:各批的 build_*/plot_* 脚本(族谱见档案 1 的 05)拉 metrics 出
   合表。

---

*(机制讲义完。下一步:`02_演化主叙事.md`——九个时代的完整决策链。)*
