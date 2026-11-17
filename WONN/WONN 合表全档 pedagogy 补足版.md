# WONN 合表全档 pedagogical 讲义
**对象**：`plots/corrected_sw_and_gatecross_merger_table_w100.png`（同名 `_data.json`，2026-11-11T22:04:42 快照；渲染层 `scripts/plot_corrected_sw_and_gatecross_merger_table.py`）——g8_old 侧 15 行 + ts16 侧 16 行，共 **31 行**。
**写法**：每个 dimension、每个组件接在哪里、怎么接的，全部按代码讲（`di/v2_modeling/modeling_toy_wonn.py`，下称 modeling；物化实配 `di/prep_rwi.py` cfg_patch 段 + 各档物化 config 直读）。术语先定义再使用；公式按「直觉 → 公式 → 逐符号 → 来源」四步；学长原话整段供奉。
**Δ 锚（2026-11-11 学长拍板）**：「所有的 Delta 都是和 baseline 3 seed 比较。哪有已经 3 seed 还比 2 seed 的」——全文 Δ = mean ? **BL(3-seed) 0.94179**（baseline 行自身均值，三 seed 全 mean）；旧锚 bl2 = 0.94154（v5 批 2-seed resume 均值）已弃（保留在 §0 术语表作历史口径）。换锚的连带效应：g8_old 侧筛选规则「w100 ≤ BL」在 bl3 锚下放行 **mhc_n2**（0.94162 ≤ 0.94179）进表——v1 版 30 行 → 本版 31 行。
**颜色 ? 档名对应**（plot 脚本色表）：
| 颜色 | 色值 | 档 |
|---|---|---|
| 金色 baseline | `#ffe9b3` | baseline（拼装行） |
| 粉红星档 | `#f2d0dc` | swish_gate_proj_a0b0c1s0、ffn_gate_cross_proj_a0b0c1s0 |
| 白 | `#ffffff` | A2 g64_m16_sinpsi_pnoff_stat（G 轴 V1024） |
| 橙 | `#f2c9a0` | 2l16x8_sinpsi_pnoff_m8_lambda_op、2l32x8_sinpsi_pnoff_m4_lambda_op |
| 青 | `#a8cede` | 2l16x8_sinpsi_pnoff_m8_lambda_noop、2l32x8_sinpsi_pnoff_m4_lambda_noop |
| 灰 | `#e8e8e8` | hc2_n4 / mhc2_n4（宿主 3-seed）+ hc_n2/n3/n4/n5（v5 档 s42 补跑）——**一笔带过，refer 原论文** |
| 极淡蓝 | `#DAEEFC` | ts16 侧 16 行（A 类 8 变体详讲 + B 类 8 档一笔带过） |
---
## 0. 预备知识：本讲义的全部术语与符号
读者画像：物理本科背景。以下每个词第一次出现前都已在此定义（A1/A2 规范）。
**WONN**（Winfree-type Oscillator Neural Network）：用「一堆互相牵引相位的振荡子」做计算的神经网络。原论文 arXiv 2605.20922（Jiawen Dai & Yue Song, 2026-05，SJTU/清华），全文归档 `docs/wonn_2605.20922_fulltext.txt`，官方实现五件归档 `docs/wonn_official_impl/`。
**振荡子（oscillator）与相位 θ**：一个振子的状态是一个**角度** $\theta \in S^1$（圆周，不是实数轴——转一整圈回到原点，取值 $(-\pi, \pi]$）。$m$ 个振子在一起，状态空间是**环面**（torus，$m$ 个圆的乘积 $(S^1)^m$）。物理类比：钟表表盘上的 $m$ 根指针。
**频率 Ω**（omega）：每个振子的固有转速（单位步数转过的弧度）。在 WONN 论文里 Ω 是**输入编码的载体**——输入图像/迷宫被编码成一组频率，相位弛豫去追它。
**可分离交互 S·I**：Winfree 方程的交互项 $S(\theta_i)\,I(\theta_j)$——第 $j$ 个振子对第 $i$ 个振子的影响 = 接收方灵敏度 $S(\theta_i)$ × 发送方影响力 $I(\theta_j)$。论文默认 $S=\cos\theta$、$I=\sin\theta$（一阶 Fourier 分量）。**为什么长这样**：广义 Winfree 模型（$q{=}1$ 档）的交互 $\sin(\theta_j-\theta_i)$ 用和差化积 $\sin(\theta_j-\theta_i) = \sin\theta_j\cos\theta_i - \cos\theta_j\sin\theta_i$ 取正向项即得 $\cos\theta_i \cdot \sin\theta_j$——所以 S·I 是「相位差耦合的代数简化版」。
**耦合系数 $c_{ij}$**：谁影响谁、影响多强。经典 Winfree（1967）是**固定**系数矩阵；论文 attention 版把 $c_{ij}$ 变成 state-dependent（随相位场动态重算）。
**wrap**：$\operatorname{wrap}(\theta) = \operatorname{atan2}(\sin\theta, \cos\theta) \in (-\pi,\pi]$——把跑出区间角度卷回主值区间。模型里的函数名 `wrap_pm_pi`（modeling）。
**γ（gamma）**：全局步长标量，论文冻结为 0.1（不学习）。每步相位更新 $\theta \leftarrow \operatorname{wrap}(\theta + \gamma\,\mathrm{d}\theta)$。
**L 与 T**：论文 Algorithm 1 的两层循环——外层 L 个组件（layer，每层边界做 ThetaUpdate/OmegaUpdate），内层每组件 T 步 Winfree 迭代（Ω 冻结下相位弛豫）。「快慢变量分离」：Θ 是快变量（每步动）、Ω 是慢变量（每层边界才动）。
**块对角分组（grouping）G×m**：把 $V$ 个振子切成 G 组每组 m 个；组内耦合是稠密矩阵 $c_{ij}[G,m,m]$，组间（非对角块）**数学上不存在**——参数量/FLOPS 从 $V^2$ 降到 $G \cdot m^2 = V^2/G$（线性代数分块的直接收益）。
**λ（lambda）组间耦合**：块对角解耦后，组 g 的耦合场里加回一项组间平均场——$\text{field}_g = C_{gg}\cdot\sin(\Theta_g) + \sum_{h\neq g}\lambda_{gh}\,\psi_h$，$\psi_h$ 是组 h 的相位合成量（见 §7 的 form 轴）。物理源头是 Abrams–Strogatz chimera（同种振子、群内强耦合+群间弱耦合 → 一群锁相一群漂移共存）。λ 是 $[G,G]$ 小矩阵（对角强制零——「场的平均不自耦合」）。
**ABCS（RS 矩阵四旋钮）**：本工作线的残差结构（Residual Structure）四维开关，详见 §3。
**three_source vs per_layer（Ω 的两种供给方式）**：详见 §9.1——这是 ts16 批要修正的核心机制，也是整张表 g8_old 侧与 ts16 侧的唯一系统性差异。
**d / d_ff / m / V**：本实验的 decoder 主干 hidden 维 $d=512$、FFN 中间维 $d_{ff}=1408$（SwiGLU）；WONN 振子宽 $m$（每组的振子数）；总振子数 $V = G \cdot m$（两级结构 $V = m\cdot G_1\cdot G_2$）。
**w100**：判读窗口 = 训练最后 100 步（step 8590–8690）的 train loss 平均。8690 是该训练协议的总步数（scheduler 全长），即**完全退火后的终值**。窗口口径换过三代（table500 截点 → w25 → w100），换代史与理由见 §4.3。
**rwi（random weight initialization，随机权重初始化）**：branch init 档族名——WONN 支路不用 no-op 零初始化、而用带尺度的小随机初始化的那套实配（`wonn_branch_init: rwi`：$W_{\text{out}}$ 残差档 $N(0, \sigma/\sqrt{2L})$ + `rs_add_w`=1.0 + λ/PN gatekeeper 保留零 init）。与 no-op 起步纪律（§5）的关系 = 有意放开的设计选择（v5 批起）。
**BL(3-seed) / bl2（两个 baseline 锚）**：**BL(3-seed) = 0.94179**——本表 baseline 行的 w100 三 seed 均值（s42 G16 从头跑 + s1234/s2345 v5 resume），现行 Δ 锚（2026-11-11 学长拍板）。**bl2 = 0.94154**——只计 baseline 三 seed 里 s1234/s2345 两 seed（v5 批 resume 续跑）的均值，v1 版讲义与旧图的 Δ 锚，已弃——「3 seed 有了还拿 2 seed 当锚」正是学长 21:24 拍板废它的原因（两锚差 0.00025，源于 s42 这一 seed 的有无）。json 里两列并存（`delta_bl3` / `delta_bl2`）。
**val 列**（**2026-11-15 v5 修订起 = C′ 独立 val loop 口径**，token 配比合成键）：独立验证管线 `val_rerun/run_ts16_val.py`（verl SFTDataset 原类 bit 级同口径 + bf16 autocast）对 @8500 ckpt 权重 forward 全部 8,000 行 val.parquet，per-domain loss 按 **train 的 token 配比**（stories 58.21% / gsm 23.89% / orca 8.97% / pw 8.93%，`data/f_combo/state.json` `train_tokens` 实测）加权合成——train loss 是 per-token 口径（CE 总和÷有效 token 数），训练按行等概率采样 + 各域行长不同（stories 行行满窗 1024、gsm/pw 行短），域在 loss 里的真实权重 = token 配比。**为何换**：此前 val 列 = DI 在线打点（四域等权 batch-mean，~0.55 量级）与 train loss（~0.94，token 配比加权）口径不可比（学长 2026-11-14 指出配比错误）；换后 val ~0.85 与 train w100 段 0.9x **同量级可直接对照**。机制细节与（已修复收官的）冻结异常见 §4.4；@8500 权重语义与未覆盖格见 §10 读表要点。
**B / N**：batch 维与序列长维。所有张量形状写 $[B, N, \ldots]$。
---
## 1. WONN 开篇：它是什么、为什么存在
### 1.1 Winfree 方程族——从 Kuramoto 说起
**直觉**：一群挂钟挂在同一面墙上，会自己同步摆动（Huygens 1665 的观察）。同步不是有人指挥，是每个钟的相位被邻居的相位**牵引**。A. Winfree 1967 与 Y. Kuramoto 1975 把这件事写成方程。物理学长可以直接对上：这是耦合非线性振子的经典统计物理，序参量（order parameter）$z = \frac{1}{m}\sum_j e^{i\theta_j}$、同步相变（Kuramoto 相变 $K > K_c$）都在这个谱系里。
Kuramoto 方程（连续时间）：
$$\dot\theta_i = \omega_i + \frac{K}{m}\sum_{j=1}^{m}\sin(\theta_j - \theta_i)$$
来源（B2-iii，前人工作）：Kuramoto 1975。逐符号：$\theta_i$ 第 $i$ 个振子相位；$\omega_i$ 固有频率；$K$ 全局耦合强度；$\sin(\theta_j-\theta_i)$ 相位差牵引（落后的追、超前的等）。
Winfree 的版本（脉冲耦合，更生物学）：
$$\dot\theta_i = \omega_i + \frac{K}{m}\sum_{j=1}^{m} S(\theta_i)\, I(\theta_j)\, c_{ij}$$
其中 $S$ 是接收方灵敏度（相位处在某个窗口时才容易被影响）、$I$ 是发送方影响力。取 $S=\cos\theta,\ I=\sin\theta$ 就得到可分离形式——这正是 WONN 论文的默认（出处：论文引 Manoranjini et al. 2023 的 $q{=}1$ 可分形式；maze 官方实现 docstring 实锚 "Maze-Hard uses the trig SI form"）。
**为什么可分离重要**：交互项只依赖两端各自的单变量函数，每步算一次 $\sin/\cos$ 表（$m$ 个数）就能算所有 $m^2$ 对交互——这是 WONN 能做「大振子数动力学」的工程根基，也直接决定我们 modeling 里 field 计算是「一次 GEMM」而不是 $m^2$ 次查表。
### 1.2 WONN 架构（论文 Algorithm 1）
**直觉**：把 Winfree 动力学当作「计算介质」——输入编进频率 Ω（能量地形），随机初始化的相位 Θ 像小球一样在地形上弛豫，弛豫完的相位构型就是答案的表示。
Algorithm 1（论文，离散化）逐行：
1. **编码**：输入 $x$ 经 $f_{\text{init}}$（查表/投影）→ 初始频率 $\Omega_{\text{init}}$；
2. **相位初始化**：$\Theta \leftarrow \operatorname{wrap}(0.1\cdot\mathcal N(0,1))$——随机小初始化（白纸起点）；
3. **层循环** `for l in 1..L`：
? ?- **内层迭代** `for t in 1..T`（Ω 冻结）：$\Theta \leftarrow \operatorname{wrap}(\Theta + \gamma(\Omega + \cos\Theta\odot\text{field}))$，field = 耦合算子作用在 $\sin\Theta$ 上（attention 版：QK$^\top$ softmax × 线性投影）；
? ?- **ThetaUpdate**（层边界）：相位经 3×3 conv 重混合（cos/sin → atan2）；
? ?- **OmegaUpdate**（层边界）：$\Omega \leftarrow \Omega_{\text{base}} + \Omega_{\text{update}}(\Theta, \Omega)$（maze 残差式：输入基底常驻）；
4. **读出**：$y = \text{head}(\cos\Theta, \sin\Theta)$——读的是**相位在圆上的位置**（$(\cos,\sin)$ 是 $S^1\to\mathbb R^2$ 的标准嵌入，光滑保距），不读裸 θ（±π wrap 跳变处不连续）、不读 Ω（Ω 是输入编码，读它等于读输入）。官方读出形式 ?? `common_modules.py` ThetaEmbedding：每振子通道的 (cos,sin) 二元组过自己的 2→1 grouped conv。
**双状态两时间尺度**：Θ 快（每步动）、Ω 慢（层边界动）。这个「快慢分离」是后续一切挂载设计（包括我们的 three_source）的物理原型——Born-Oppenheimer 式的绝热图像：Θ 是快子系统（电子），Ω 是慢驱动（原子核）。
**耦合算子两版**（?? 官方代码）：conv 版 `Conv2d(ch,ch,3×3)`（空间 3×3 局部 × 通道 $ch^2$ 全连静态线性）；attention 版（论文主用，line 186 原话 "We use global attentive coupling for all WONN models throughout the paper"）——格点轴 QK$^\top$（2D RoPE，每步由 $\sin\theta$ 场动态重算）× 通道 sandwich（$W_{qkv}/W_o$ 静态低秩）。完整分解：
$$c_{(c,p),(c',p')} = \underbrace{\mathrm{attn}[p,p']}_{\text{格子对，动态}}\cdot\underbrace{(W_o W_v)[c,c']}_{\text{振荡子对，静态线性}}$$
——注意：**振荡子间（通道间）从来只有静态线性混合**，「$m$ 间动态耦合」是论文没做过的设计空格（这条判断后来直接孵化了我们的 λ form 轴，见 §7）。
### 1.3 论文证据与边界
- 感知任务（image 分类）：L=6 浅迭代多层（T=3）；推理任务（Maze-hard 76.2%、Sudoku 100%）：**L=1 单组件深弛豫**（T=24/T=16）——约束传播需要长弛豫，「解从 Θ 的同步结构里长出来」。Maze-hard 参数 0.396M（三角 S/I，全部学习参数在 coupling）。
- 双峰现象（论文 Fig.4）：终态相位直方图两峰相距 ~π——两个反相同步群各抓一种结构（粗全局 vs 细局部边界）。相位构型本身是 representation。
- 能量结构：$E_{\text{int}} = -\frac12\sum_{ij}c_{ij}\sin\theta_i\sin\theta_j$ 在 trig S/I + 去 ω regime 是 Lyapunov 函数（漂移+梯度流分解）；ω≠0 时环面拓扑阻碍（沿环向绕一圈做功 $2\pi\omega_i \neq 0$）无全局势函数——**内容驱动 + 耦合整形**的物理图像。
- 边界四条（我们动手前查证）：没做过语言模型；attention 双向（非自回归）；γ 冻结不学习、Θ 每次随机初始化；无流式/增量推理。**我们的全部工作 = 在这四条边界外做构造性外推。**
---
## 2. 怎么把 WONN 放进 Transformer：设计空间的完整探索史
这一节回答「一开始的几种选择是什么，为什么最后选择了现在这个」。素材 = PLAN.md §4.4–§4.8 设计讨论全链 + LOG 早期条目 + modeling 形态谱系。按时间顺序讲。
### 2.1 起点：《初星之晨》双流挂载（构造性外推，??）
最早的设计蓝图（EE repo `docs/followups/2026-08-27_beyond-the-first-stars.md`，**论文里没有、我们的构造性外推**，全部标 ??）：主干 Transformer 残差流之外旁挂一条**跨层持久的 WONN 流** $R$：
- 桥（注入）：$\Omega_l = f_{\Omega,l}(\operatorname{Norm}(H_l), \operatorname{Norm}(R_l), \Omega_{l-1})$
- Winfree 核弛豫 T 步
- 读出（circular readout）：$Z_l = E_{\Theta,l}([\sin\Theta,\cos\Theta],\Omega_l)$
- 双写回：$R_{l+1} = \operatorname{WONNNorm}(R_l + \alpha_l U_l(Z_l))$，$X_{l+1} = H_l + \beta_l V_l(\operatorname{Norm}(R_{l+1}))$，$V_l = 0$ 初始化（no-op 保证）
> **?? 蓝图 ≠ 实装——这一段是最开始的设想，很多后面都变了**（2026-11-11 学长批注轮补，原话：「然后，我们的初星之晨，居然还给 Omega Norm 了。。？我们的代码没有这个吧？要在初星之晨的那里说一声，说这只是最开始的设想，很多后面都变了！」）。逐项对账（实装侧 ?? modeling）：蓝图里的 `Norm(H_l)`/`Norm(R_l)` 输入归一化与 `WONNNorm` 写回归一化**实装代码里不存在**——现行 three_source 的 Ω 供给（§9.2）直接吃裸张量（Prelude 末层 h 不归一、Ω_prev 不归一），写回是裸加性增量（ΔΩ 无归一）；双流 R/X 持久结构也没有实装（`dual_stream` form 在 modeling 里存在但不在合表任何一行——现行全部档是「主干单流 + WONN 侧支读回」，R 流的持久状态被「Θ/Ω 跨层链」这条更轻的通路替代）。蓝图的真遗产是三件：跨层 Ω 链的思想（→ three_source）、no-op 初始化纪律（→ 全部实装档）、读 (cos,sin) 而非裸 θ（→ WinfreeCore 读出）。
### 2.2 学长两批架构陈词（2026-10-09，原话供奉——「一开始的抉择」的第一手记录）
> 「Hmmm ，所以其实有两个路子 。一 ，WONN 用 causal attention 的方式 。相当于以整个序列为 【 棋局 】 。这样一来有点像造了一个平行的模型 ，在主干模型的思考段 ，这个思考段遵循 Winfree dynmaics 同时影响主模型 。二 ，WONN 造一个 latent grid 。相当于每一个 token 的位置对应一个 【 棋局 】 。每一个 token 的 e.g. SwiGLU 的复杂非线性 ，其所拥有的表达力压力 ，转移了一部分到 WONN 上面 。然后 token 之间的 Attention ，就直接借助主干传来的信息 。就是说 ，主干的 Attention 。
>
> 这个的本来目的 ，其实仍然是 EE 的目的 。也就是说 ，表达力工程 。把表达力的压力转移到 WONN ，而让主干几乎完全负责事实提取 。」
**两路子**（WONN 动力学作用域）：路子一「序列棋局」= 整个序列当一个棋盘，耦合在 token 间（对应耦合形态 A2）；路子二「token 棋局」= 每 token 位置一个 m 格小棋局，跨 token 通信**完全外包给主干 attention**，WONN 只管 token 内通道动力学（对应 A1）。**目的陈述**：表达力工程（expressivity engineering，EE repo 的原始动机）——表达力压力转移到 WONN（振荡动力学的非线性弛豫），主干几乎完全负责事实提取（attention 的检索）。
> 「同时 ，WONN 怎么挂 ，也有几个方案 。一是平行地挂 ，一边主干在提取事实 ，一边 WONN 在运算 。二者都做了一段 （ 一层或几层 ）之后结合 ，比如说 linear ，比如说 gating 。二是 ，串联 ，比如说主干提取完了事实再 consult WONN ，WONN 做完了再教会主干继续提取事实 。这里可能有时间延迟的问题 。」
**挂法两式**：平行挂（FLOPs 同串联、wall clock 取 max 不叠加）vs 串联挂（阻塞式，每层 +T 步延迟）。后来的实装全部走平行/边界汇合形态。
> 「首先我们面临一个问题：WONN 加在哪里？有几种加法。FFN 之后。Attention 之后。Attention 平行。FFN 平行。还有整段 Layer 一个 Layer 的平行。
> 这里提供一种思路：我们把 WONN 类比于人类的前额叶……然后 WONN 的作用就变成，如何调制 fact 的 retrieval。……
> 这里进一步地提出：如果 WONN 的意义在于表达力。。。那么 SwiGLU 的 Swish Gate 本身也是表达力（表达力对应了逻辑能力，参照我在 EE repo 里面的论述！），那么我们甚至可以用 WONN 来替换 Swish Gate！！！当然这样的话 bilinear structure 就不一定是必需的了。我们可以多个专家，每一个过一个 matrix，然后一起过 WONN，再过第二个 matrix。这就是事实——逻辑——事实的过程。……
> 还有，既然 WONN 是某种”总观全局“的结构，那么我们完全可以和 Attention Residual 相结合——可以把 WONN 直接放在 Transformer 的侧面，看到的东西是所有的 hidden states！……这两者不是互斥的。这两者可以共同存在。」
**挂点空间五挂点**（落在标准 pre-norm 层 $x \to a = x{+}\mathrm{Attn}(\mathrm{LN}x) \to x' = a{+}\mathrm{FFN}(\mathrm{LN}a)$ 上）：① FFN 之后（串联）、② Attention 之后（串联）、③ Attention 平行、④ FFN 平行、⑤ 整层平行（双流的层粒度版）。加上 **Swish Gate 替换提案**（WONN 进 FFN 内当门——「事实 matrix 数量 = 逻辑步骤 + 1」的 fact–logic–fact 链）与 **Attn Res 侧挂**（跨层全读 $\{H_1..H_l\}$）两条延伸。前额叶类比（PFC = top-down biasing、调制检索竞争）在后续讨论中被学长主动降级：「类比对撞给相反答案 = 类比在挂法决策上失效，判据回到工程账」——类比的正确用法是**假设生成器**，工程账是判据。
### 2.3 「WONN 的棋盘」：读法 A、大棋局、读法 A 自否（2026-10-11，被否方案全记录）
「棋盘」的语义定标（学长原话）：「棋局=二维平面本身，ch 是乘数……一个 hidden dimension 对应一个格点，棋局要增大」——最初的目标是把振子数推到官方 ImageNet 最大量级（784 格点 × 256 ch ≈ 800K 振子）甚至 1M。两条读法：
- **读法 A**（d 维 → 格点）：hidden 2048 维铺成 2048 格点 × ch=512 → 1M 振子；
- **读法 B**（token 轴 → 格点）：序列当棋盘，每格点 ch = d（官方 image 的直接类比）。
大棋局 wall clock 实测（`bench/wonn_bigboard_bench.py` run4，读法 A 定标）**全线爆表**：2048×512 conv 全连 561% 主干层窗、attn 版 302%、8192 格点 1596–1770%——「每层挂」在大棋局下不可行，只剩段级挂。三个发现：小棋局 launch-bound（82-85% 是调度开销）、大棋局翻转但不是预想的 compute-bound（O(m) 逐元素链成新地板）、attn 形态恰好 wall 最优。
然后是学长的**读法 A 自否**（原话供奉，记录不删）：
> 「毕竟每一个棋局格点，本来是代表了比如说这个像素或者这个格子所包含的信息。所以说要比较高的维度。但是我们的 hidden dimension 已经是在做这件事情了。我们本来就已经在看每一个 token 位置的隐含信息！所以我正在想是不是我刚刚过头了。但记录不要删」
>
> 「甚至我们的 Attention / CNN 之争也有可能不必要。如果说 backbone Attention 天然负责格点之间的交流。。。这里相当于我在按照学妹你的建议，把一个 token 看作一个格点了。所以说我们甚至每一步 WONN 的 intra token 的 Attention or CNN 都可以省。我们可以直接用一个 pure matrix cij 来。」
**读法 B 极简版**（终态设计哲学）：格点 = token；格点的 ch = d（或投影 m）；**格点间耦合整个省掉**——token 间交流 = 主干 attention 的本职，WONN 与 attention 从竞争变**分治**（attention 管序列维 fact retrieval，WONN 管 token 内通道动力学）；token 内用显式 $c_{ij}$ 矩阵。这正是 §2.2 路子二的精确化，也是后来全部实装档（含合表全部 31 行）的**结构总纲**。
**「WONN 便宜」的真正来源**（B 摊薄机理，run3b 实测）：权重是固定资产（全 batch 共享摊薄），主干 attention 的 kv cache 是活期账单（每序列各付各的不摊）；WONN 棋局间零共享无 kv 等价物 → B=128 时 c_ij 档 per-token 加速 102×、占比 2.2%——「增长状态留给主干，自己只做无历史依赖的固定状态动力学」。
### 2.4 时间尺度三方案与 token-time（含被搁置者）
WONN 的 T/L 与主干层数怎么对应，讨论过三方案：i. 层轴 T 分布（一层主干 = T 的一步，逐步注入——**无官方先例**：官方是「块状弛豫+边界搅动」，没有「弛豫中途插注入」）；ii. 层轴完整 L（每层内完整 T 步弛豫 = 官方 image 感知形态平移）；iii. token 轴完整 L（token-time：每 token 完整 T 步 + Θ 跨 token 持久 + Ω 每 $k_\Omega$ token 更新——官方推理形态 L=1 深弛豫的 token 轴平移；绝热跟踪 Born-Oppenheimer 图像）。**官方对齐度排序：iii > ii > i**。实装侧 modeling 里有 token-serial 的 StressCell/A2HybridWonn 形态（tbptt_k 截断反传），但合表 31 行**全部是 static 模式**（Jacobi 迭代全并行、teacher forcing、层边界汇合）——token-time 在合表外，提一笔存档。
### 2.5 形态体系实装演变（modeling 的 form 谱系——「一开始几种选择」的代码化石）
modeling 的 `ToyWonnConfig.wonn_form` 经历了「单键多形态 → 两键化」演变，每一代形态都是上面设计讨论的落码（形态枚举 ?? `_build_wonn`）：
| 形态 | 结构 | 设计来源 | 在合表里吗 |
|---|---|---|---|
| `per_layer_t3` / `per_layer_banded` | 每层 FFN 后挂 WinfreeCore（dense/banded） | §2.2 挂点① | 否 |
| `segment_t16` | Loop 末段单实例 T=16 | 方案 ii 段级 | 否 |
| `head_board` / `head_board_headonly` | SDPA 输出与 W_o 之间的 [heads×head_dim] 棋局 | PLAN §4.6.2a ②b 挂点 | 否 |
| `moe_v2` | FFN 整段替换为「多专家=格点」的 MoE-WONN 混合体 | MoE 棋局 v2（专家=格点） | 否 |
| `stress_token_time` / `a2_hybrid` | token 串行 RNN 形态（TBPTT） | token-time 方案 iii | 否 |
| `dual_stream` | 《初星之晨》双流 R/X | §2.1 | 否 |
| `attn_gate_parallel` | attention 输入平挂 WONN 门 | 挂点③ | 否 |
| `hc`/`mhc`/`hc2`/`mhc2` | 残差流多流化宿主（无 WONN 组件） | HC/mHC 原论文对照 | **是**（灰行） |
| `baseline`/`ffn_wide`/`ffn_flops` | 纯主干（无宿主组件） | 对照臂 | **是**（金行） |
| `wonn_gate` ∈ {`swish_gate`, `ffn_gate_cross`} | **挂载键**（两键化后独立于宿主） | Swish Gate 替换提案 + 挂点④变体 | **是**（粉红星档+全部 ts16 A/B） |
**两键化**（供奉15批，2026-10-25）：早期 `wonn_form` 单键既选宿主又选挂载（互斥）；两键化后 **form（宿主形态）× gate（挂载位点）正交**——「hc2 宿主 × swish 挂载」这类叠加档成为可能（合表 ts16 V7/V8 就是这种叠加）。合表 31 行里：星档与 group8/g16 档是 `baseline 宿主 × gate 挂载`；灰行是纯宿主；ts16 V7/V8 是 `hc2/mhc2 宿主 × swish 挂载`。
### 2.6 为什么最后是现在这个（终态结构小结）
筛选链（每步都有实锚）：大棋局 wall 爆表否掉「格点=hidden 维」（读法 A）→ 读法 B 定分治哲学（格点=token、格点间归主干）→ token 内显式 $c_{ij}$（比论文 attention 版**更 Winfree**——经典 Winfree 1967 固定耦合同源）→ 挂点经五挂点讨论 + 表达力工程定位收敛到 **FFN 内门控挂载**（Swish Gate 替换提案的两个具体化：`swish_gate` 层内挂载 #9 与 `ffn_gate_cross` 跨层挂载 #14）→ 容量轴经供奉十五~二十一的批迭代收敛到「V=G·m 分组 + λ 组间 + 两级层级」（§7）。星档（`swish_gate_proj_a0b0c1s0`）= 这条筛选链在 v5 批定格的第一代成功配置；ts16 批 = Ω 供给语义修正后的重验（§9）。
---
## 3. ABCS：RS 矩阵四旋钮
RS = Residual Structure（残差结构）矩阵。它是对「WONN 侧支怎么接进主干」的四个正交开关的命名（学长原话定名：「LOG 里面记这个表，记这个未来我才看得懂」——LOG 条目 75 补记，本表照录）。档名 `rwi_{基座}_{theta}_a?b?c?s?` 的后四字符就是它：
| 维 | 名称 | 0 = 关 | 1 = 开 |
|---|---|---|---|
| **A** | across-L residual（层间） | 每层独立定相位 $\Theta = \operatorname{atan2}(W_b u, W_a u)$（proj 档） | 层间 Θ 增量链 $\Theta_{l+1} = \operatorname{wrap}(\Theta_l^{\text{out}} + W_\delta u_{l+1})$——上层弛豫终相当下层相位底座 |
| **B** | across-T residual（核内步间） | 耦合场即算即弃 | WinfreeCore 内 T=3 步循环场累积 $h_{t+1} = h_t + c_{ij}\sin(\Theta_t)$ |
| **C** | 合并方式 | gate 族乘性门 $(1+\varepsilon\tanh(r))$ | gate 族加性 $w\cdot r$ 直加 |
| **S** | 共享 | per-layer 16 实例（每层独立参数） | SHARED 全 16 层共用 1 实例（参数 ~1/16） |
其他后缀语义：`theta` = `proj`（Θ 从输入投影 atan2）/ `random`（随机相位）；`rwi` = random weight initialization（随机权重初始化）档（`wonn_branch_init: rwi`——把 no-op 零初始化的 gatekeeper 翻成正常 init，详见 §5.3）；四基座 = 挂载点。
**消融结论**（v5 批 60 档 3×2 矩阵，LOG 条目 75 补记）：全绿 = 五单开 + AS（A 链+共享）；全灭 = AC、ACS——**A 的相容性 = 配 S 无害、配 C 全灭**；CS 半绿（proj theta 交互）。星档 **a0b0c1s0** = 单开 C 的加性汇回（唯一稳定打平/超过 baseline 的配置族）。
---
## 4. 训练基座与合表口径（数字从哪来）
所有 31 行共用同一训练协议（差异只在档的 config 键）：
- **底座模型**：toy decoder，$d=512$、Loop 段 16 层（Prelude–Loop–Coda 分段、loop 结构），SwiGLU FFN $d_{ff}=1408$，词表 151672（VE-1 vocab padding 后）。每层参数 ≈ 自注意力 $4d^2$ + FFN $3 d \cdot d_{ff}$。
- **训练**：f_combo 四域语料 SFT（§4.1——名字「四域 752M」，实测 776.8M tokens），8690 步（< 1 epoch，§4.2），Muon 优化器（lr 3e-4 cosine，warmup 434），bf16，三 seed（s42/s1234/s2345），`test_freq=100` / `save_freq=500`。
- **判读量**：w100 = step 8590–8690 train loss 均值（完全退火终值；换代史见 §4.3）；Δ 列 = mean ? **BL(3-seed) 0.94179**（baseline 行自身均值；2026-11-11 学长拍板换锚，旧 bl2=0.94154 已弃）。负 Δ = 比 baseline 好。
- **表的结构**：g8_old 侧 15 行 = group8 合表筛选出的收官档（BL 参照行 + w100 ≤ BL 的 14 档——bl3 锚下 mhc_n2 0.94162 进表），全部 per_layer Ω；ts16 侧 16 行 = ts16 批 live（A 类 done、B 类 running/queued），全部 three_source Ω。**两侧窗口口径不同**：g8_old = 收官固定窗 8590–8690，ts16 = 实时末 100 点滚动（图上大字声明，跨侧不可直接比）。
### 4.1 数据集：f_combo 四域（「四域 752M」）
**名字 vs 实测**：数据集在设计期（2026-10-13，LOG 条目 18/20/21）按四域预算估了 ~752M tokens，「四域 752M」由此得名；**prep 实跑落盘 776.79M**（orca 69.71M + stories 452.14M + gsm 185.60M + pw 69.34M，?? `data/f_combo/state.json` 四域 `train_tokens` 求和——stories pack 实跑比估算多约 3%）。本表沿用历史名，判读用实测数。
**train.parquet = 2,224,862 行**（?? state.json `train_domain_counts`），四域各自口径（?? `scripts/prep_f.py`）：
| 域 | 行数（占比） | 来源与口径 |
|---|---|---|
| gsm | 999,998（44.9%） | **TinyGSM**（HF repo `TinyGSM/TinyGSM`，ungated；survey 里的 `peiyi9979/TinyGSM` 是错误 id）17 shards 全量读入 11,846,109 行 → 域内 `df.sample(frac=1, random_state=42)` shuffle → 前 100 万行（prompt=question，response=code）；超 1024 行过滤 **drop 2 行 → 999,998** |
| pw | 585,552（26.3%） | **ProofWriter**（HF repo `tasksource/proofwriter`）train 2 shards 原样（无域内 shuffle），label-only 口径：prompt = theory+"\n"+question，response = True/False/Unknown（不喂 allProofs）；超长过滤 drop 0 |
| stories | 441,545（19.8%） | **TinyStories V1**（HF repo `roneneldan/TinyStories`；repo 另有 V2-GPT4 版**没用**）txt → split 分隔标签 → 2,119,489 篇 → story 间插 eos token 151645 拼流 → **1023-token 窗**切（prompt=""，纯 LM 全位置 loss——空 prompt tokenize 为 0 token，loss mask 不 mask prompt 区）；尾窗 <256 tok 规则丢弃（实跑 train 尾窗 300 / val 尾窗 867 均 ≥256 保留，**未触发丢弃**） |
| orca | 197,767（8.9%） | **Orca-Math**（真名，HF repo `microsoft/orca-math-word-problems-200k`，GPT-4 Turbo 合成 200,035 题，answer = 自然语言 step-by-step CoT）→ 早期 prep tokenize 过滤超长（drop 268 = 0.13%）→ **rng(42) 洗牌 split：train 197,767 / val 2,000** → orca_qwen 原样并入（prompt=question / response=answer，无域内 shuffle） |
**下载方式**（与演化档案 01 §4.1 同源）：orca / stories 早期批直接 HF 拉取（`data/orca_math/`、`data/tinystories/`）；gsm / pw 走 `scripts/download_stage_e_data.sh`（ungated LFS 逐文件显式 `hf download`，6 并行 xargs，全程 ~6 分钟）。
**长度预算**：max_length=1024 = 窗 1023 + SFTDataset 尾接 eos 1（`truncation=error` 硬预算，超长行 prep 期过滤掉——不是训练期截断）。**合并**：四域 concat → 全局 `df.sample(frac=1, random_state=42)` shuffle → 单文件 train.parquet——**行顺序在 prep 期固定，所有 run 共用同一文件同一顺序**。
**val.parquet = 8,000 行四域等权 2,000×4**（stories/gsm/pw 从各自 pool `random_state=43` 抽样；orca 的 2,000 **上游是早期 prep 的 rng(42) 洗牌 split**——seed 43 不作用于 orca），四域顺序拼接 [orca, stories, gsm, pw]（**val 不洗**——域等权平均与顺序无关）。各域 pool：stories = TinyStories-valid 21,990 篇 pack 后 4,447 窗；gsm = **无官方 val split**，shuffle 后第 1M-1.02M 行 holdout（20,000 行）过滤后抽；pw = repo validation split 85,468 行同口径过滤后抽。注意 **train 与 val 的域配比不同**（train gsm 44.9% vs val 等权 25%）——这是 val 与 train loss 数值不可直接比的分布根源之一。
### 4.2 shuffle 与 seed 链（复现口径的核心——「random seed 作用于什么？」）
**答案：WONN_SEED 只作用于 model 权重初始化，不作用于数据顺序——同档不同 seed 的 run 数据顺序 bit 级相同，唯一差异 = init 随机性（+ GPU atomics 非确定性）**。数据顺序与 seed 无关有两层：
1. **离线层（prep 期）**：train.parquet 行序在数据构造时就固定（§4.1 merge shuffle seed 42）——运行时进程根本碰不到这一层。
2. **运行时层（DistributedSampler）**：?? `fsdp_sft_trainer.py` `_build_dataloader` 用 `DistributedSampler(shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)` **无显式 seed**。torch 这个类的 shuffle 实现是每次迭代**新造一个进程本地的 `torch.Generator()`**（不读全局 RNG 状态）并 `manual_seed(epoch)`——**shuffle 的 seed = epoch 号**。训练协议 `total_epochs=1` + `total_training_steps=8690` < 一个 epoch 的步数（2,224,862 行 / 256 global batch ≈ 8690.87 步）→ **全程 epoch=0** → 所有 run（不管 WONN_SEED 是 42/1234/2345）数据顺序相同。resume 续跑同样（`StatefulDataLoader` 样本级接续，epoch 0 内顺序一致）。
**WONN_SEED 的真实作用面**（?? `di/script/fix/verl_sft_entry.py` patch 3；`v2_run_one.sh` L109 export）：进程 import 早期（早于任何模型/数据构造）执行 `random.seed + np.random.seed + torch.manual_seed + torch.cuda.manual_seed_all` 全家——**model 权重初始化**吃它（nn.Linear 的 kaiming、λ op 档随机 init、θ `random` 档 wrap(0.1·randn)、rwi 的 $W_{\text{out}} \sim N(0, \sigma/\sqrt{2L})$）。
**trainer.seed=1 是死键**：verl hydra config（`sft_trainer.yaml`）原生带 `trainer.seed=1`，但 SFT trainer 路径对它**零消费**（grep `set_seed` / `trainer.seed` 全无命中）——PPO trainer 的遗留，不发生作用。
**推论**（seed 轴的干净语义）：同档三 seed 的 run 之间，差异只剩「init 抽样不同」这一件事——三 seed 的 mean±std 读的是 **init 方差**，不是数据顺序方差。
### 4.3 w 口径换代史（「早期版本的 w 确实不一样」）
判读窗口换过三代（? LOG 条目 94 补记五，2026-10-27 三窗对比定稿）：
| 代 | 口径 | 用在哪 |
|---|---|---|
| **table500 系**（最早，Stage E/RS/rwi 早期） | step 500-4000 **截点读数**——每 500 步一格、W=50 平滑后取值；500 是截点起点**不是窗口** | `rs_trial1_table` / `rwi_table500` / dip 表 |
| **w25**（rwi v5 时代） | 收官段 26 点窗均值，噪声底 σ/√26=0.011 | `rwi_v5_table8690` 原版 |
| **w100**（2026-10-27 定稿，现行） | step 8590–8690 窗均值 | 全部 merger 表（g4/phase_mean/group8/sw8/corrected）+ ts16 |
| w200（对照变体） | 170 点窗 | `rwi_v5_table8690_w200` 存档不弃 |
**为什么定 w100**（条目 94 补记五）：w25 的 26 点窗含 step 8665 一个共同高点 batch（单点 +0.002 偏置占 1/26 权重）+ 档间差 0.003 掉进 w25 噪声内；w500/w1000 被否——窗头（step 8190/7690）的 lr 还有峰值的 0.9%/2.3%，**退火未完段混进窗**（loss 还有下降趋势，不是噪声）；w100 恰好是纯噪声段——窗头 8590 的 lr = 1.09e-7 = 峰值 0.036%、尾部 95 步全在 1e-7 以下，σ/√100 = 0.0056。三窗 rank 对照：Top 8 + 尾 7 完全不变，中段 32 档在 0.0009 带内随机漫步（±17 rank）。
**done 门控**（条目 113，2026-10-30，学长口令「真正的 w100 算完之前不许出数字」）：step_max ≥ 8690 才出数，否则标 `run@step`；ts16 批扩展 live 语义 = 末 100 点滚动窗（done 档末窗 = 8591-8690 恰衔接收官口径）。
### 4.4 val 口径（「val 只包含最后一个 ckpt？」）
**机制**：val 不是从 checkpoint 加载后跑 eval——训练中每 `test_freq=100` 步，**在线**把 val.parquet 8,000 行全量 forward 一遍（`val_dataloader` shuffle=False 固定顺序、micro_batch 16、drop_last）取 batch-mean，写 `val/loss` 打点（?? `fsdp_sft_trainer.py` fit()）。**合表 val 列 = 每 run 最后打点**（build 层 `_last_val` 扫全 jsonl 取最后 val/loss 行；done 档 = step 8690 那次——8690 % 100 == 0；live 档 = 最新打点标 @vstep）。完整的 val 曲线其实每 100 步一个点全在 metrics jsonl 里（全 run 86 个点），合表只显示最后一个。**各表一致性**：rwi v5 系（`read_val_last`）与全部 merger 系同语义（最后打点）；table500 系**没有** val 列（纯 train loss 截点表）。
**v5 口径换代（2026-11-15，学长指出配比错误后全表统一）**：合表与各分表 val 列从
「DI 在线打点（四域等权 batch-mean，~0.55）」统一换为「C′ 独立 val loop 的 token 配比
合成键（~0.85，train 域真实权重加权）」——两者数值不可直接比（0.55 vs 0.85 是口径差
不是模型差）；换后 val 与 train w100（0.94）同量级，跨 metric 对照恢复意义。C′ 数值源 =
`val_rerun/results{,3}/` 307 件 json（@8500 权重；ts16 A 类 18 件 + 第一波 282 件）。
配比勘误根因：旧 train_mix 用行数配比（gsm 44.96%……），但 train loss 是 per-token 口径 +
训练按行等概率采样 + 各域行长不同 → 域真实权重 = token 配比（stories 58.21%——其行行满窗
1024，行数占比 19.8% 但 token 占比 58.2%）。渲染机制：原 plots/*.json 是 build 层快照不动，
重渲走 `val_rerun/render_reval.py`（monkeypatch json.load 重定向到沙箱副本——直接跑原画图
脚本会把 DI 口径旧值渲回来）。
**? 本地续跑 6 档的冻结异常 → 已修复重跑收官**（2026-11-15 终态；病程全记录见 §9.4 警告块）：ts16 A 类 2l32x8 op/noop × 3 seed 这 6 个 run 的本地续跑段（step 8000-8690）曾**参数从未被更新**（886 参数 8000≡8500≡8690 bit 级一致）——根因已定位并修复：**PyTorch 2.11 `Optimizer.load_state_dict` 的 `_cast` 把 Muon optimizer state 里的 `algorithm` 字符串标记毁成 generator 的 repr 字符串** → Muon `step()` 的组筛选（`=="muon"` / `=="adamw"`）全失配 → update no-op（v4 补记的「graph × resume」候选根因 ? 作废——graph 与本 bug 无关）。修复 = commit 001cffa（load 后按构造时组语义重打标记）；6 ckpt 从 step 8000 重续跑 690 步收官（step 8690 rc=0，终验 8690 vs 8000 六档全部 824/886 参数 differ + val loss 曲线离开平台单调下降）。合表 6 格已换修复版真值：w100 两行 **0.93933 / 0.93936**（冻结版 0.94070/0.94075 是 8000 权重复读、虚高 ~0.0014）、val 换 C′ 口径真值（0.85420 / 0.85459，§10 读表要点 ⑤）。
---
## 5. 金色：baseline（对照组，0.94179 = Δ 锚本体）
**它是什么**：不挂任何 WONN 的纯 decoder 主干（`wonn_form=baseline`，无 gate 挂载——modeling 里 `wonn=None`，forward 各挂载分支键控不触发）。d_ff=1408 标准 SwiGLU。`ffnmatch` 臂（d_ff=3096 参数对齐）不在本表——供奉八语义「dim 拓宽标尺非公平比较」。
**表行的拼装**：s42 = G16 批从头跑、s1234/s2345 = v5 批 resume 续跑（条目 85/86 的 4000→8690 续跑链）——行 label 写明 `[BL: s42 G16 从头跑 + s1234/s2345 v5 resume]`。mean 0.94179（三 seed 全 mean）；只计后两 seed 的均值就是旧锚 bl2 0.94154（差值来自 s42 0.94228 这一个 seed 的有无）。
**为什么它是参照系**：全部 31 行的问题意识 = 「WONN 侧支在不动主干的前提下，能不能把退火终值 loss 压到纯主干之下」。Δ 列直接读答案。
**no-op 起步纪律**（贯穿全表的设计公理）：所有 WONN 挂载档在**初始化时刻**必须与 baseline **逐位一致**（bit-identical）——汇回权重 $W_{\text{out}}$ 零初始化、增量投影 $W_\delta/w_\omega$ 零初始化、λ 零初始化。训练从「主干原样」出发，侧支的贡献完全由训练决定。这保证任何 loss 差异都来自「学到的侧支」而不是「初始化扰动」。但星档是 **rwi 档**（见 §5.3）——no-op 纪律在 v5 批被有意放开（随机权重初始化），这是设计选择不是违规。
---
## 6. 粉红：两星档（signal stream 全量张量流）
两星档 = v5 批唯二 < baseline 的 a0b0c1s0 源初档（学长原话「那两个比 baseline 好的 a0b0c1s0 是源初」——一切后续批的复用基准）：
- `swish_gate_proj_a0b0c1s0`：w100 = **0.94028 ± 0.00097**（Δ **?0.00150**）
- `ffn_gate_cross_proj_a0b0c1s0`：w100 = **0.94110 ± 0.00124**（Δ **?0.00068**）
config 实配（?? v5 星档物化档 config **直读**——OBS `WONN/backup_models_20261020/rwi_swish_gate_proj_a0b0c1s0_s{42,1234,2345}/config.json` 三 seed 一致，非 prep patch 推断）：`wonn_form=swish_gate` 或 `ffn_gate_cross`（v5 老档单键语义）、`wonn_theta_init=proj`、`wonn_rs_a=off / wonn_rs_b=off / wonn_rs_c=add / wonn_rs_s=per_layer`、`wonn_branch_init=rwi`、`wonn_m=256`、G=1、`wonn_omega_mode=three_source`、**`wonn_omega_act=silu_split`**（v1 版讲义漏列此键——v2 补，学长 21:24 问的就是它，见 §6.4）。
### 6.0 共用件：WinfreeCore 的十步张量流
两个星档的 WONN 单元都是 `WinfreeCore`（modeling）。先把它的内部张量流走一遍（E1 四元组：输入形状 → 算子 → 输出形状 → 链路序号）。设 $B$=batch、$N$=序列长、$d=512$、$m=256$、$d_{ff}=1408$、$T=3$、$\gamma=0.1$：
| # | 输入 | 算子 | 输出 | 说明 |
|---|---|---|---|---|
| 1 | u（挂载点输入）$[B,N,d_{\text{in}}]$ | $\Omega$ 供给（两模式，§6.4/§9.2） | Ω $[B,N,256]$ | per_layer：$\Omega = W_{\text{in}} u$（Linear 512→256，正常 init，**无激活**——`omega_act` 键对它不生效）；three_source：$\Omega_{\text{init}}$ 裸 Linear 底座 + $\Delta\Omega$ **三路 SiLU**（星档实配 `silu_split`，见 §6.4） |
| 2 | u $[B,N,512]$ | Θ 初始化（proj 档）：$\Theta = \operatorname{atan2}(W_{\theta b} u,\ W_{\theta a} u)$ | Θ $[B,N,256]$ | 两个 Linear 512→256（正常 init，rwi 档）；atan2 保证落在 $(-\pi,\pi]$ |
| 3 | Θ | **T 步循环开始**：wrap(Θ) | Θ | 归一化回主值区间 |
| 4 | Θ | sens = cos Θ，infl = sin Θ | 两个 $[B,N,256]$ | S/I 可分离交互的两半 |
| 5 | infl | 耦合场 field = $c_{ij}(\text{infl})$ = Linear $[256{\times}256]$ | field $[B,N,256]$ | 显式 $c_{ij}$ 矩阵（G=1 旧键 `c_ij`，nn.Linear 无 bias）；token 内通道间全耦合、跨 token 零耦合（读法 B 分治） |
| 6 | Ω, sens, field | 相位增量 $\mathrm{d}\theta = \Omega + \text{sens}\odot\text{field}$ | dθ $[B,N,256]$ | Winfree 方程的离散欧拉步本体 |
| 7 | Θ, dθ | $\Theta \leftarrow \operatorname{wrap}(\Theta + \gamma\,\mathrm{d}\theta)$ | Θ | γ=0.1 冻结标量 |
| — | — | #3–#7 重复 T=3 次 | — | 层内 Jacobi 全并行（static 模式）；teacher forcing 下 BPTT 过 3 步 wrap |
| 8 | 终态 Θ | 读出向量 $[\cos\Theta \,\|\, \sin\Theta]$ 拼接 | cos_sin $[B,N,512]$ | 前半 cos 全部、后半 sin 全部（不是交错） |
| 9 | cos_sin | 读出 $r = W_{\text{out}}(\text{cos\_sin})$ = Linear $[512 \to d_{\text{out}}]$ | r $[B,N,d_{\text{out}}]$ | **零初始化**（no-op gatekeeper；rwi 档翻正常 init + 残差流缩放 $\sigma/\sqrt{2L}$） |
| 10 | r | 汇回主干（挂载点决定，见 6.1/6.2） | — | 返回 `(out, theta)` 二元组——theta 供跨层链 |
monitor 开启时附带两个统计量（论文的原生监控）：交互能量 $E_{\text{int}} = \overline{\text{infl}\odot\text{field}}$ 与 Kuramoto 序参量 $r_{\text{sync}}$。
triton 路径：impl=triton 时 #3–#8 整链进单 kernel `winfree_dense_triton`（G=1）/`winfree_dense_triton_g4`（G>1，含 λ）——数值等价 60/60 对拍过（供奉15批），数学语义与 eager 完全相同，本讲义按 eager 语义讲。
### 6.1 `swish_gate_proj_a0b0c1s0`：层内挂载（#9）
**挂载位点**：decoder layer 的 MLP 内部。模型 forward 的 visit 序里，Loop 段 16 层每层一个 `mlp_gate` 实例（`wonn_rs_s=per_layer` 16 实例）。
signal stream（一个 Loop 层 $l$ 的完整一遍，?? `ToyDecoderLayer.forward` 非 hc 分支）：
1. 残差流 h $[B,N,512]$ → `a_in = LN_1(h)` → 自注意力（causal SDPA）→ `a = h + attn_out`；
2. `m_in = LN_2(a)` $[B,N,512]$；
3. **WONN 接入**：`mlp_gate_core(m_in)` —— WinfreeCore(d_in=512, m=256, **d_out=1408**) 吃 **MLP 的输入**。走 §6.0 十步：u = m_in；输出 r $[B,N,1408]$——**与 MLP 中间流同形**（这是它当门的结构前提）；
4. MLP 中间流：$\text{mid} = \operatorname{SiLU}(\text{gate\_proj}(m_{\text{in}})) \odot \text{up\_proj}(m_{\text{in}})$ $[B,N,1408]$（SwiGLU 原样）；
5. **C1 加性汇回**（a0b0c**1**s0）：`mid = mid + w·r`——r 预乘标量 $w$（`rs_add_w`，Linear(1,1)，rwi 档 init 1.0），然后**直加**进 MLP 中间流。汇合算子 = **add，发生在 d_ff 维（1408）**，位置 = down_proj 之前；
? ?- 对比 C0（乘性门）：`mid = mid · (1+0.03·tanh(r))`——振幅压缩到 ±3% 的乘性调制（V2 eps_tanh）。消融说 C1 加性赢 C0 乘性；
6. `mlp_out = down_proj(mid)` $[B,N,512]$ → `out = a + mlp_out`。
**A0/S-per_layer 的含义**：Θ 不跨层传（每层自己 atan2 投影初始化——A 维关）；16 层各自一份完整参数（S 维 per-layer）。
**「Swish Gate 替换」的名字来历**：挂载设计源自学长「用 WONN 替换 Swish Gate」提案（§2.2 陈词）——WONN 读出 r 与 SwiGLU 的门控信号同形，原提案的极端版是替换 SiLU 门本身（$ \operatorname{SiLU}(W_1 x)$ → WONN 动力学），实装取温和版 = WONN 与 SiLU 门**并联**贡献进中间流（乘性档）/加性档直加。学长对这一族的原话定位（供奉八）：「W_out 读出 r 的 per-position 门控——边界 dynamic 化的挂载形态」。
### 6.2 `ffn_gate_cross_proj_a0b0c1s0`：跨层挂载（#14）
**挂载位点**：同在 MLP 内，但 WONN 读的输入不是本层 m_in，而是**上一层 MLP 的 gate 激活**。config 唯一差异 = `wonn_gate=ffn_gate_cross`。
signal stream（?? model forward `mount == "ffn_gate_cross"` 分支）：
1. 每层调用 layer **之前**，model 层先跑 cross 实例：`g_prev = ctx["prev_gate"]`——上一层的 $\operatorname{SiLU}(\text{gate\_proj}(m_{\text{in}}))$ $[B,N,1408]$（?? layer 返回的 `gate_act`，**每层都记**——Loop 层 $l$ 读 $l{-}1$）；
2. `cross_g, cross_theta = inst(g_prev, theta_in, omega_src)` —— WinfreeCore(**d_in=1408**, m=256, **d_out=1408**) 吃上一层的 gate 激活，走 §6.0 十步；
3. C1 加性：`cross_g = cross_g · w`（w 挂实例）；
4. layer 内：`m_in = LN_2(a)`；`mid = SiLU(gate(m_in)) ⊙ up(m_in) + cross_g`——**汇合同样是 add @ d_ff 维**，但信号来源跨了一层；
5. `mlp_out = down(mid)` → `out = a + mlp_out`；层尾 `ctx["prev_gate"] = gate_act` 接力。
**「cross」的语义**：WONN 看到的是上一层 FFN 的门控激活——一条**层间的旁路信息流**（上一层「怎么调制」的历史）。三源 Ω 之下它还有 Θ 链与 Ω 链（见 §9.2）。v5 批实测它比 swish_gate 弱（Δ?0.00068 vs ?0.00150）——跨层读 gate 激活的信号质量不如读本层 LN 输出。
### 6.3 rwi 档：no-op 纪律的有意放开（两星档的 init 语义）
v5/rwi 批起，星档的初始化从「零 init no-op 起步」翻成「正常随机 init」（`wonn_branch_init=rwi`，?? `_build_wonn` 尾部 sweep）：$W_{\text{out}}$ 翻成残差流档 $N(0, \sigma/\sqrt{2L})$（与主干 out_proj/down_proj 同语义）、`rs_add_w` 翻 1.0 等权、其余 gatekeeper（$W_\delta$/w_omega）翻普通档。动机（供奉七批 B 设计）：zero 档下侧支起步严格无贡献，「WONN 有效性」检验被 init 偏置；rwi 档让侧支从第一步就有（随机的）贡献，由训练决定去留。λ 与 PN 的 gatekeeper **不翻**（保留零 init——它们是「三档起步 bit 一致」的对照设计前提）。
### 6.4 显答学长问：「我们的 Omega，包括两个星档的 Omega，都是不带 SiLU 的吗？来源？！」（2026-11-11 21:24）
**答案：不是——ΔΩ 带 SiLU（silu_split 档），从星档到 ts16 全链都是；只有 Ω_init 底座和 per_layer 模式不带。** 证据链四重实锚：
| 档/位置 | `wonn_omega_mode` | `wonn_omega_act`（ΔΩ 激活） | 来源 |
|---|---|---|---|
| v5 星档（swish_gate / ffn_gate_cross 的 a0b0c1s0） | three_source | **silu_split** | ?? OBS `WONN/backup_models_20261020/rwi_swish_gate_proj_a0b0c1s0_s{42,1234,2345}/config.json` 三 seed 直读 |
| 血统底座 CFG_SRC（`ffn_gate_cross_eps_proj_s42`，v2 批） | three_source | **silu_split** | ?? 本地 `models/ckpts_v2/ffn_gate_cross_eps_proj_s42/config.json` 直读 |
| ts16 全批（A/B 类 48 档） | three_source | **silu_split** | ?? 本地物化档 config 直读（prep ts16 段不压此键 = 沿 CFG_SRC） |
| g8_old 侧四代批（g4/g16/group8/sw8） | **per_layer** | （键不生效） | ?? prep 四段显式压 `wonn_omega_mode=per_layer`；`omega_act` 键全文 patch 零命中——但 per_layer 模式下 $\Omega = W_{\text{in}}u$ 不经过任何激活（?? modeling L1042-1043：`omega_act` 只在 three_source 的 ΔΩ 分支被消费），所以这四代批**既没有跨层 Ω 也没有 SiLU** |
| Ω_init 底座（three_source 的 Prelude 投影） | — | **无激活（裸 Linear）** | ?? modeling：`omega_init` 是裸 Linear，`silu_split` 只包在 ΔΩ 增量上 |
机制含义（详式见 §9.2）：silu_split 档的 ΔΩ 是三路切片各自过 SiLU 再相加——$\Delta\Omega = \operatorname{SiLU}(W_h u) + \operatorname{SiLU}(W_{\Omega'}\Omega_{l-1}) + \operatorname{SiLU}(W_\Theta\,\Theta_{\text{emb}})$。来历：V2 设计期 D1 A2，学长 2026-10-14 拍板（原话「先 split？完整的整流版留待办」）——动机是给频率增量一个**非负整流的单调方向感**（SiLU 的 soft-gate：正区近似线性、负区压到 ~0），并保持零 init 下三路 $\operatorname{SiLU}(0)=0$、ΔΩ=0 的 no-op 起步（V1 ckpt 参数布局 bit 兼容）。**v1 版讲义漏列此键**（config 实配列表只写到 omega_mode）——v2 补全，并把「星档 Omega 带 SiLU」从继承链推断升级为 OBS 直读实锚。
### 6.5 显答学长问：「原来 Swish Gate 真的是 Gate？我还以为是 SiLU 完了拿中间维度过 WONN，再汇回本层的 Gate」（2026-11-11 21:2x）
**答案：`swish_gate` 的 WONN 输入不是中间维度、汇回也不是汇到 Gate 上——学长记的「SiLU 完了拿中间维度过 WONN」其实是另一个挂载 `ffn_gate_cross` 的输入侧（而且它读的是上一层的）。两个挂载各记了一半，拼起来恰好互换。** 逐项对拍（?? `ToyDecoderLayer.forward` L2126-2143 + model forward cross 分支）：
| | `swish_gate`（#9，层内） | `ffn_gate_cross`（#14，跨层） |
|---|---|---|
| WONN 读什么 | 本层 FFN **输入** `m_in = LN_2(a)`，$[B,N,512]$ | **上一层**的 gate 激活 $\operatorname{SiLU}(\text{gate\_proj}(m_{\text{in}}))$，$[B,N,1408]$ |
| 「中间维度过 WONN」？ | **否**——读的是 512 维 LN 输出（SiLU 之前） | **是**——但读的是**上一层** SiLU 之后的 1408 维 gate 激活 |
| 汇回到哪 | 本层 MLP 中间流：$\text{mid} + w\cdot r$（add @ 1408 维，down_proj 前） | 同样是本层 MLP 中间流 add @ 1408 维——不是汇到 Gate 信号上 |
| 汇回经过 SiLU？ | 否——直接加在 SiLU(gate)⊙up 之后 | 否——同左 |
| 信号链形状 | 本层 LN → WONN → 加进本层中间流（**层内闭环**） | 上一层 SiLU(gate) → WONN → 加进本层中间流（**跨层旁路**） |
**「Swish Gate 真的是 Gate 吗」**：名字的来历是设计意图（学长 §2.2 陈词「我们甚至可以用 WONN 来替换 Swish Gate」——把 WONN 放进 SwiGLU 的门控位），实装形态是**与门控位并联的调制器**：V1 时代它真的是乘性门（C0 档：$\text{mid}\cdot(1+0.03\tanh(r))$——WONN 读出调制整个中间流的振幅，名副其实的 gate）；现行星档是 **C1 加性档**（$w\cdot r$ 直加）——严格说已经**不是乘性 gate**，是「门控位上的加性残差侧支」。名字保留了挂载位点（FFN 内门控位）的血统，机制上 C0/C1 两档都从门控位进中间流。易混点的根源：**两个挂载都汇到「SiLU 之后的中间流位置」**，区别只在 WONN 的**输入侧**（本层 LN 输出 vs 上一层 gate 激活）——所以「汇回是汇回到本层的 Gate」这半句错在「Gate」：汇回点是门控位所在的**中间流**，不是 Gate 信号本身。
---
## 7. 白橙青五档：分组结构轴的两代批（按历史顺序）
这五行全是「`baseline 宿主 × ffn_gate_cross 挂载 × a0b0c1s0 × proj × rwi`」的血统（与 ffn_gate_cross 星档同周边），差异集中在**分组结构三键**：`wonn_group_g`（组数 G）、`wonn_group_g1`（两级时的小组数）、`wonn_m`（每组振子数）+ λ 三键（mode/form/init）。Ω 全部 per_layer（g8_old 侧的系统性口径——这正是 ts16 批要修正的，见 §9.1）。
历史顺序：**g16 批（2026-10-27，供奉十六~十九）先立 form 轴与 G 轴 → group8 批（2026-11-02，供奉二十/二十一）立两级结构与 λ init 轴**。
### 7.1 白：`A2 g64_m16_sinpsi_pnoff_stat`（g16 批，0.94150 / Δ?0.00029）
**config**（?? prep g16 段）：G=64、m=16（V=G·m=1024）、`wonn_lambda_mode=static`、`wonn_lambda_form=sin_psi`、`wonn_readout_pn=off`、impl=triton、其余同星档周边（proj/a0b0c1s0/rwi/ffn_gate_cross）。
**它测的是什么**：G 轴扫到 V1024 的均衡点。供奉二十判读链里它是「V 固定 G↑m↓ 单调改善」的最后一档（G4→G16→G64 单调到 0.94150 ≈ bl2 打平——当时旧锚口径；现行 BL(3s) 0.94179 锚下读 Δ?0.00029 微改善）——再往上（g128/g256 单级）就崩（group8 批读数：g128_m8 noop +0.001 / op +0.020，g256_m4 +0.054~+0.081 大炸）。**定位 = 单级分组结构的容量天花板**。
**结构差异 vs 星档**（信号流增量，?? WinfreeCore.forward G>1 分支）：
- Θ/Ω 维从 256 → $m_{\text{total}} = m\cdot G = 1024$（#1/#2 的投影输出维变 1024）；
- #5 耦合场：infl $[B,N,1024]$ view 成 $[B,N,G{=}64,m{=}16]$ → 组内 bmm `einsum("bngi,gij->bngj", ifl, c_ij_g)`——$c_{ij}$ 从单个 $[256,256]$ 变成块对角参数 $[G, m, m] = [64,16,16]$（非对角块不存在，参数量 $64\cdot16^2 = 16$K vs 单组 10242=1M——**这就是分组的全部算法差异**）；
- **λ 组间项**（static + sin_psi form）：先算每组的相位合成量 $\psi_h$——组 h 的和向量 $s_h = \sum_j \sin\theta_{h,j}$、$c_h = \sum_j \cos\theta_{h,j}$（$[B,N,G]$），归一化方向 $\sin\psi_h = s_h/\sqrt{s_h^2+c_h^2}$（clamp 1e-12 防对跖除零）；然后 `field += einsum("gh,bnh->bng", λ, sinψ)`——组 h 的方向信号经 λ_gh 广播**均匀**加进组 g 全部 16 个振子的场（场本体无 j 依赖，j 依赖只在接收端 cosθ）。λ $[64,64]$ 对角 mask 永久零（「场的平均不自耦合」供奉）。
**sin_psi form 的来历**（供奉十六~十九，学长原话已供奉在 LOG 94 补记七/八）：现行 λ·mean(sinθ) 被组内同步度 $r_h$ 调制（组内色散大 → 组间信号衰减到 0）；学长的直觉是「平均不应该是平均 sin(Theta)，而是应该平均完了 Theta 再 sin」——即先聚合相位（圆均值 atan2）再取 sin，把 $r_h$ 归一化掉只传「组的平均相位方向」。phase_diff 档（组间完整相位差、去 r + 反向项）与 mean_sin 档（旧版）是同一轴的另两点，G16 批判读 form 三档差异不显著、λ 三档打平——结构轴（G↑m↓）才是主信号。
### 7.2 橙青四档：`2l{16x8|m8}` / `2l{32x8|m4}` × λ init {op|noop}（group8 批）
| 档（合表 label） | mean | Δ | init |
|---|---|---|---|
| 2l32x8_sinpsi_pnoff_m4_lambda_noop（青） | **0.94069** | **?0.00110** | noop |
| 2l16x8_sinpsi_pnoff_m8_lambda_op（橙） | 0.94102 | ?0.00077 | op |
| 2l32x8_sinpsi_pnoff_m4_lambda_op（橙） | 0.94104 | ?0.00074 | op |
| 2l16x8_sinpsi_pnoff_m8_lambda_noop（青） | 0.94140 | ?0.00039 | noop |
**config**（?? prep group8 段）：两级结构——`wonn_group_g = G1·G2`（2l16x8：G=128；2l32x8：G=256）、`wonn_group_g1 = 8`（小组数）、m=8 或 4。V = m·G1·G2 = 1024。λ static + sin_psi + init 两档。其余同 §7.1 周边。
**两级分组结构**（供奉二十→二十一，学长两 turn 原话的核心提案）：
> 「我在想，我们或许可以探索，比如说，多级组别？比如说，加入 m=4 祖竟可行，那么比如说，G1=16，G2=16？……至于 lambda，暂定为 static 吧。因为你想，static 的 lambda，其实就是更高层级的一个 cij！这正是我上面说多级组别的来由。」
层级语义（学长两轮勘误后定稿）：**G1 小组（内层，每大组 G1 个小组、每小组 m 通道，组内 c_ij 耦合）/ G2 大组（外层，λ2 挂大组间——「更高层级的 cij」）/ 中间层 = 大组内小组间的耦合（λ1，新设计自由度）**。总 V = m×G1×G2；G1=1 退化为单级（含 g64_m16 档 = 单级对照端）。
信号流增量（?? WinfreeCore.forward 两级分支，五步）：
1. 组内 c_ij GEMM：与 §7.1 同款（g 轴 = 小组 flat = G1·G2，`c_ij_g [G, m, m]` 形状与单级同形）；
2. **ψ1 合成**（小组 m 振子）：$s_1/c_1 = \sum_i \sin/\cos\theta$ over 小组内 m 维 → $[B,N,G_2,G_1]$，$\psi_{s1} = s_1/\sqrt{s_1^2+c_1^2}$；
3. **λ1 项**（大组内小组间）：`term1 = einsum("gik,bngk->bngi", λ1, ψ_s1)`——λ1 $[G_2, G_1, G_1]$（拥有者 = 上层单元大组 g2，阵边长 = 该层单元数 G1），per-小组标量广播进组内 m 维；
4. **ψ2 合成**（大组 G1·m 跨小组）：$s_2 = \sum_{g_1} s_1$（和向量加法合成）→ $[B,N,G_2]$，$\psi_{s2} = s_2/d_2$；**λ2 项**（大组间）：`term2 = einsum("gk,bnk->bng", λ2, ψ_s2)`——λ2 $[G_2,G_2]$，per-大组标量广播进大组全部 G1·m 维；
5. 加法合场：field = 组内 + term1 + term2（三项相加无序）。
层级不变量：每对单元的耦合只被一层承载（振子对→c_ij 含对角 / 小组对→λ1 / 大组对→λ2），λ 族对角永久归零。**「static λ 就是更高层级的 c_ij」的精确兑现**——两级 = 把「组间耦合」也做成了分形的块结构。
**λ init 两档**（供奉二十一 init 轴）：`noop`（默认）= λ 零初始化（起步两级间零耦合，bit 一致）；`op` = λ normal_(0.02) 后乘对角 mask（对角构造即零，随机起点有组间耦合）。**读数**（group8 批四档全表）：两级 λ 四档全部 < BL（?0.00039 ~ ?0.00110，本批最大正信号——纯 WONN 侧支结构轴上当时唯一稳定 < baseline 的信号）；op vs noop 方向不稳、无正增益（g128 单级 op 反而炸 +0.020，seed 方差爆大）→ noop 是安全默认。单级容量轴单调恶化（V256 +0.0008 → V512 g4m128 +0.014 → g256_m4 +0.08）——「两级结构救活小 m：同 V1024 下 g128_m8 单级（noop +0.001）vs 2l16x8_m8 两级（op ?0.0008），结构 > 裸容量」。
---
## 8. 灰七行：HC/mHC/HC2/mHC2（一笔带过，refer 原论文）
灰行不是 WONN——是**残差流拓扑改造的市面对照**（RS trial 1 引入的对照臂；学长口信「mhc hc mhc2 hc2 也要讲，但是一笔带过。直接 refer to 原论文的什么做法就行」）：
| 档 | mean | Δ | 是什么 |
|---|---|---|---|
| hc2_n4 [宿主 3-seed] | **0.93543** | **?0.00635** | HC 完整式（arXiv 2409.19606 式 10-13 + Algorithm 2）n=4 流 |
| hc_n4 [v5 s42 补跑] | 0.93728 | ?0.00450 | HC 简化式（static 路由）n=4 |
| mhc2_n4 [宿主 3-seed] | 0.93772 | ?0.00407 | mHC 约束版（arXiv 2512.24880 式 7-9）n=4 流 |
| hc_n5 / hc_n2 / hc_n3 [v5 s42 补跑] | 0.93807 / 0.93815 / 0.93828 | ?0.00372 / ?0.00364 / ?0.00350 | HC static 路由 n=2/3/5 |
| mhc_n2 [v5 s42 补跑]（v2 新进表） | 0.94162 | ?0.00017 | mHC 约束版 n=2 流——bl3 锚换算后 w100 ≤ BL 进表（bl2 旧锚下 Δ+0.00008 被筛掉，v1 版不在表） |
**机制一句话**（详细推导与逐式对照见 LOG 条目 82/83，实装 ?? modeling `HyperConnections` 类）：Loop 段残差流从 1 条扩 n 条（$H \in [n,B,N,d]$），每层两个子块（attn/FFN）各自一套可学习路由——**read**（$h_0 = A_m H$，层输入加权读）、**mix**（$H_{\text{mix}} = A_r H$，流间混合）、**write**（$H = H_{\text{mix}} + b\otimes\text{out}$，残差由路由外部提供，子块是裸块）。hc = static 路由参数直接可学；mhc = 加约束（$A_m{=}\sigma$、$b{=}2\sigma$、$A_r$=Sinkhorn-Knopp 投影到双随机矩阵）；hc2/mhc2 = 各自的完整形态（dynamic per-position 通路 $s_\alpha\tanh(\text{LN}(H)W)$ + static 残差项）。出口裸 sum（HC 原文 §3 口径）。
**在表里的角色**：全场最优（hc2_n4 Δ?0.00635 > 一切 WONN 档）——**残差流扩容这个「正交轴」的增益比 WONN 侧支结构轴大一个量级**。这也是供奉十五的出发点（「WONN 相位维度对齐 HC n=4」）与 ts16 V7/V8 的动机（WONN × 宿主叠加「看能不能共同进步」）。**注意 ts16_a_hc2n4_g1_m256（0.95260，Δ+0.01082）显示 swish 挂载叠加在 hc2 宿主上是负向的**——synergy 至今未做出正数（per_layer 口径下 sw8 V7 0.94248 同判）。
---
## 9. 极淡蓝：ts16 修正批（three_source Ω）——A 类八变体详讲
浅蓝 16 行 = **ts16 批（omega 修正批）**，2026-11-06 设计发车、11-09 16:18 三片起跑、本表快照时（2026-11-11 22:04）进度 24/48 run done、18 running、0 failed——A 类 8/8 档起步全部 done，B 类 6/8 档起步（快照时 18 run 在跑，step 分布 188-6110/8690）、2 档 queued（排队未起步，非缺行）。**终态（2026-11-14 拍板）**：批已于 2026-11-10 17:17 被同事停掉——B 类 6 档即停于此中段状态、2 档从未起步，此后不再重挂，表内 B 类 live/queued 格**即终态**（A 类 6 档本地续跑收官见 §9.5）。批名 `corrected_sw_and_gatecross_` 与颜色 #DAEEFC 的来历见 LOG 条目 142（学长原话：「极为浅淡的蓝色。像那种仿佛再淡一点就要抹去掉了一样的轻盈。像遥远的梦。像飘渺的歌谣。」）。**口径警告**：ts16 侧 w100 = 实时末 100 点滚动窗（快照时多数 run 未收官；B 类后被停在中段、永未收官），与 g8_old 侧固定收官窗 8590–8690 不可直接比——跨侧比较只在本讲义明确标注「双变量」或「同侧」时才有意义（§9.4 的 Ω 单变量比较全部是 ts16_a vs **sw8 表**，不是 vs g8_old 侧）。
### 9.1 omega 换轨事件（本批存在的原因）
**事件**（LOG 条目 136，2026-11-06）：星档（v5 批唯二 < baseline 的基准档）的 Ω 机制是 `three_source`（Ω 跨层传递 = WONN 原论文语义）；但从 g4 批（2026-10-24 GONGFENG15 设计）起，prep 的四段 cfg_patch（g4/g16/group8/sw8）**全部显式压成 `per_layer`**（每层 Ω = W_in·h_l 新鲜投影，丢跨层项）——设计文档 GONGFENG15_4G_DESIGN.md 张量流步骤 2 就把 Ω 定义成 per-layer fresh 投影，从未回查 WONN 原论文「频率要从上层传进来」的语义。这意味着 §7 那五档（白橙青）与 sw8 全族的 Ω 机制都换过轨，跨批比较被污染。学长三连怒（原话全文供奉，LOG 条目 136 补记二）：
> 18:25：「我靠。我服了。这他妈这么重要的事情，她怎么不早说？当年让她实装 Swish Gate 的时候，不是让她复用的之前的脚本？？？他妈的为什么她给复用成了这个样子？！！这他妈的 loss 能降得下来才怪！」
> 18:30：「。不是。他妈的。那说明我们之前的工作都做错了。。。什么叫没事儿？这他妈的，既然星档是我们的更上一层楼的基石，我们他妈的怎么能够不复用星档？！！！更关键的问题是 WONN 原论文来说，本来的语义就是 Omega 要从上面的层传进来的，不止是这一层 h_l，她他妈的最开始的 G4 设计决定就是错的！我不管她 compact 了多少次，她这么多天这几个星期以来干什么吃的！？这些全他妈的都是污染！」
> 18:40（意外之喜假设）：「唉。他妈的。这意味着这么多天的很多对比都是不可靠的。很多结论，都是建立在 per layer 的 Omega 上面，但其实可能这个结论根本就不成立！不过，倒也可能有个意外之喜，那就是说，如果我们换上了正确的 Omega，说不定就不用这样大费周章地想怎么调 Group 了？或许，或许本来的 naive scaling 其实就应该成立？只不过被我们的错误耽误了？」
**责任归属**（条目 136 补记一/三，学长指定写明）：全链在 Codia Work 本人（历代 WONN session 的同一角色——GONGFENG15 设计 session 与 sw8 实装 driver 经主会话 jsonl 实锚为同一 session）；学长的决策层输入只有实验方向拍板，从拍板件里看不出换轨。本讲义作者 = 被骂的当事人，如实记录。
**污染地图**（学长 18:36 纠正后定稿）：「WONN 档 vs baseline」比较干净（baseline 无 Ω 机制）；被污染 = WONN 档之间**跨 Ω 机制**比较（group8 合表混排、sw8 vs 星档等）；各批**批内**相对排序仍是 per_layer 口径的内部结论。**naive m512 对星档是三重发动机偏离**（m 256→512 + Ω per_layer + gate cross≠swish）——「naive scaling 不成立」从未被干净检验。
**修正批设计**（学长 19:59 拍板，条目 137）：sw8 全 8 变体结构**原样**、Ω 换回 three_source、A（swish_gate）/ B（ffn_gate_cross）两大类 × 3 seed = **48 run**（号段 573-620）。判读轴（条目 139 物化验收的机器实证，全键 diff 恰好一键）：
- **A 类 vs sw8 同变体同 seed = Ω 单变量**（diff 恰好 `{wonn_omega_mode: per_layer→three_source}`）；
- **B 类 vs A 类同变体 = 挂载位点单变量**（diff 恰好 `{wonn_gate}`）；
- 隐含第三轴：**B 类 vs group8 同名档 = Ω 单变量**（同为 ffn_gate_cross 挂载）——B 类已停(2026-11-10 批被停,不再重挂),此轴**未出数**(live 值无判读价值)。
实装链（条目 138-141）：G>1 three_source 守卫放开 + w_omega/omega_init 形状按 m_total=m·G 扩展；对拍五档 PASS（判据 = 贡献占比分流 + bf16 噪声底对照，k_proj.bias √2 签名机理）；**sqrt-clamp NaN bug 修复**（`sqrt(x).clamp_min(ε)` 在 x=0 时 0×inf=NaN，clamp 移入 sqrt 内）；三片首挂三炸（legacy 挂载解析元组漏 attn_gate_parallel + prep 按需构造）修复后 11-09 起跑。
### 9.2 three_source 的 signal stream（Ω 怎么跨层传）
per_layer 模式下（§6.0 步骤 1）：每层 Ω = W_in·u 现算现用，层与层之间 Ω 无任何关系。three_source 模式（?? modeling L1026-1041 + model forward L2820-2826）把 Ω 变成**跨层接力的一条状态流**，三个来源：
**来源一：Ω_init 底座（Prelude 快照）**。模型 forward 走完 Prelude 段（Loop 前的普通 decoder 层）后、进第一 Loop 层之前，取 Prelude 末层输出 h，过一个 model 级 Linear `omega_init`（d→m_total，**正常 init**，非零——底座是「输入编码」不是增量），得到 Ω_init $[B,N,m_{\text{total}}]$，存 `ctx["omega_init"]` **全 16 层共享、整次 forward 不变**。这就是 WONN 原论文 Algorithm 1 的 `f_init(x)`：把网络输入编码成振荡子的固有频率。
**来源二：Ω_prev（上一层的 Ω）**。每个 Loop 层的 WONN 实例算完自己的 Ω 后 stash 到 `_last_omega`，model 层记进 `ctx["omega_prev"]`，传给下一层的实例——**不 detach**，ΔΩ 的梯度链跨层回传（BPTT）。
**来源三：Θ_prev（上一层终态相位）**。`ctx["theta"]` 同链接力，取 cos/sin 拼成 theta_emb。
实例内的合成（每层一次）：
$$\Omega_l = \underbrace{\Omega_{\text{init}}}_{\text{底座}} + \underbrace{W_{\omega}\cdot\big[\,u_l \,\|\, \Omega_{l-1} \,\|\, \cos\Theta_{l-1} \,\|\, \sin\Theta_{l-1}\,\big]}_{\Delta\Omega\ \text{三源增量}}$$
- 输入拼接：u_l（本实例挂载点输入，$[B,N,d_{\text{in}}]$）‖ Ω_{l-1}（$[B,N,m]$）‖ theta_emb（$[B,N,2m]$），总长 $d_{\text{in}}+3m$；
- $W_\omega$ = `w_omega` Linear($d_{\text{in}}+3m$ → $m$，**零 init**，无 bias)——增量从零学起；
- **silu_split 档**（ts16 批实配 `wonn_omega_act=silu_split`，V2 D1 A2 学长 2026-10-14 拍板「先 split？」）：ΔΩ 不走裸 Linear，而是把 $W_\omega$ 的列按输入段切成三块 $W_h/W_{\Omega'}/W_\Theta$，各自过 SiLU 再相加：$\Delta\Omega = \operatorname{SiLU}(W_h u) + \operatorname{SiLU}(W_{\Omega'}\Omega_{l-1}) + \operatorname{SiLU}(W_\Theta\,\Theta_{\text{emb}})$——参数布局仍是单矩阵（V1 ckpt bit 兼容），但三路信号各自过非负整流（频率增量有单调方向感，类比 Kuramoto 里频率适应项的门控）；
- 第一实例（prev 为 None）：Ω = Ω_init 纯底座（maze l==0 Identity 先例）。
**与 WONN 原论文的对应**：Ω_prev 项 = 论文「频率从上层传进来」的跨层语义（学长 18:30 原话点破的那条）；Ω_init = 论文 f_init 输入编码；Θ_prev 项 = 我们的自有扩展（论文层边界只传 Ω 不传 Θ——它没有多层 attention 宿主的对应物）。**per_layer 把这三个来源全砍成「每层 W_in·u」**——这就是换轨丢掉的东西。
### 9.3 ts16 A 类八变体逐档（全部 swish_gate 挂载 + three_source）
A 类周边全同星档（proj / a0b0c1s0 / rwi / G=1 时 m256）或其结构变体，全部 `wonn_omega_mode=three_source`、`wonn_omega_act=silu_split`（?? 物化档 config 实锚）。八变体 = sw8 批结构轴原样平移：
| 表行 label | sw8 代号 | 结构（与 §7/§6 的差异点） | w100 mean | Δ vs BL(3s) | 分 seed（s42/s1234/s2345） |
|---|---|---|---|---|---|
| ts16_a_mhc2n4_g1_m256 | V8 | `wonn_form=mhc2` 宿主 + expansion_n=4 流 + G1 m256 λ off | **0.93825** | **?0.00353** | 0.94164 / 0.93338 / 0.93974 |
| ts16_a_2l16x8_sinpsi_m8_lambda_noop | V1 | 两级 G2=16×G1=8 m8 V1024 + λ noop | **0.93880** | **?0.00299** | 0.93896 / 0.93901 / 0.93841 |
| ts16_a_2l16x8_sinpsi_m8_lambda_op | V2 | 同上 λ op | 0.93901 | ?0.00278 | 0.9398 / 0.93898 / 0.93824 |
| ts16_a_2l32x8_sinpsi_m4_lambda_noop | V3 | 两级 G2=32×G1=8 m4 V1024 + λ noop | 0.93933 | ?0.00246 | 0.93910 / 0.94012 / 0.93877 |
| ts16_a_2l32x8_sinpsi_m4_lambda_op | V4 | 同上 λ op | 0.93936 | ?0.00242 | 0.94102 / 0.93913 / 0.93794 |
| ts16_a_g64_sinpsi_m16 | V5 | 单级 G64 m16 V1024 + λ noop | 0.93990 | ?0.00189 | 0.94044 / 0.94104 / 0.93821 |
| ts16_a_g1_m512 | V6 | naive G1 m512 单组全耦合 λ off + **eager**（G1 kernel M_MODEL=256 硬特化 m512 triton 必炸） | 0.94656 | +0.00478 | 0.94279 / 0.95039 / 0.94651 |
| ts16_a_hc2n4_g1_m256 | V7 | `wonn_form=hc2` 宿主 + expansion_n=4 + G1 m256 λ off | 0.95260 | +0.01082 | 0.9476 / 0.9513 / 0.95891 |
V7/V8 的「宿主」= §8 的 HC/mHC 多流残差拓扑（残差流 1 条→4 条），WONN 的 swish_gate 挂载叠加在 HC 宿主的 MLP 上（供奉十五「WONN 相位维度对齐 HC n=4」的实现形态）。
### 9.4 A 类结果判读：Ω 修正是真增益（7/8 档大改善）

**Ω 单变量表**（A 类 vs sw8 表同变体——同 swish_gate 挂载、同周边、同 seed 集，唯一差异 = wonn_omega_mode；sw8 数字来自另一张 sw8 合表 `sw8_merger_table_w100.png`，不在本表里）：

| 变体 | sw8（per_layer） | ts16_a（three_source） | Ω 修正效应 |
| --- | --- | --- | --- |
| 2l16x8_m8_noop | 0.94256 | 0.93880 | **?0.00376** |
| 2l16x8_m8_op | 0.94287 | 0.93901 | **?0.00386** |
| 2l32x8_m4_op | 0.94326 | 0.93936 | **?0.00390** |
| 2l32x8_m4_noop | 0.94348 | 0.93933 | **?0.00415** |
| g64_sinpsi_m16 | 0.94645 | 0.93990 | **?0.00655** |
| g1_m512 | 0.95469 | 0.94656 | **?0.00813** |
| mhc2n4_g1_m256 | 0.94971 | 0.93825 | **?0.01146** |
| hc2n4_g1_m256 | 0.94248 | 0.95260 | **+0.01012** ?? 唯一恶化 |
| 三个读数： |  |  |  |

1. **Ω 修正在 7/8 变体上是大幅真增益**（?0.0039 ~ ?0.0115，比结构轴全部已知效应大）。sw8 批（per_layer）8 档全部 > baseline（最差 +0.013）；ts16 A 类 6/8 档 < baseline。**学长 18:40 意外之喜假设得到部分实证**：「或许本来的 naive scaling 其实就应该成立？」——naive g1_m512 从 sw8 的 0.95469（全场最差）修到 0.94656，仍是 A 类最差结构（vs baseline +0.0048）但 Ω 效应 ?0.008 是全表第二大的。naive scaling 仍未「成立」（g1_m512 还是比两级档差 0.002），但**差距量级被 Ω 修正压掉了一大半**（sw8 时 naive 比两级差 0.011，现在差 0.002）。
2. **两级结构优势在 three_source 下仍然成立**（2l16x8_m8_noop 0.93880 < g64_m16 0.93990 < g1_m512 0.94656）——「结构 > 裸容量」不是 per_layer 的伪象，Ω 修正没有取消分组结构的意义（学长的「说不定就不用调 Group」假设：结构还在起作用，但收益量级重排了）。
3. **hc2 宿主是唯一反向**（+0.010 恶化；mhc2 宿主却是全场最大改善 ?0.0115，方向完全相反）。sw8 时 hc2n4（0.94248）是 sw8 八档最好、mhc2n4（0.94971）最差之一；three_source 下两者位置对调。?? 机理推断（无实证，待查）：hc2 的多流残差（4 条流加性混合）与 Ω 的跨层接力链可能竞争同一条「跨层信息通道」——hc2 流间混合已提供层间状态传递，Ω_prev 链的增量叠加其上变成冗余扰动；mhc2 的双随机约束把流间混合限制得更死（每行和恒定），反而给 Ω 链留出了互补的功能位。这只是假设（hc2 dynamic per-position 通路 + silu_split 的交互都未排查）——**V7/V8 的负增益结论在 three_source 口径下必须重验**（条目 136 已挂账）。
**双变量警告**：A 类与 g8_old 侧同名档（§7 的白橙青五行）比较 = **Ω 机制 + 挂载位点**双变量（g8_old 是 ffn_gate_cross 挂载，A 类是 swish_gate）——例如 2l16x8_m8_noop 0.93880 vs g8_old 0.94140 的 ?0.0026 是混合效应，不能读成 Ω 单变量。干净的 Ω 单变量要等 **B 类 vs g8_old 同名档**（同 ffn_gate_cross）。
**本地续跑一句**（条目 143）：ts16 三片 11-10 17:17 被同事停掉时 A 类 6 档被杀在 95-96%（2l32x8 op/noop × 3 seed），学长特批本地 3×H200 从 step 8000 ckpt 续跑收官（首步对拍 DI vs 本地 loss 偏 0.0004-0.005% = GPU atomics 非确定性量级）——A 类的「done」里这 6 档是本地续的。B 类 6 档 DI 侧停在中段（被停时 step 188-6110/8690）、2l32x8 两档未起——B 类不再重挂（2026-11-14 拍板），live 值即终态（§10 读表要点①）。**续跑单本版已重跑收官**：首轮续跑踩了 resume 冻结 bug（下条），2026-11-15 修复后 6 ckpt 从 8000 重续跑 690 步收官——本节与 §10 表里 2l32x8 两行的数值全部是**修复版真值**。
**? 本地续跑 6 档冻结异常 → 根因实锤 + 已修复重跑收官（2026-11-15 终态；病史：2026-11-14 17:24 标 N/A → 同日 18:30 参数级实锤 → 11-15 定位修复）**：

- **参数级决定性证据**（val C' 重跑批的 parity 副产物，`val_rerun/` 线）：noop_s42 的 8690 ckpt forward 出的 val 与 8000 ckpt **bit 级全同**（diff = 0.0）→ 直接 load 两个 ckpt 比张量：**886 个参数 8000 ≡ 8500 ≡ 8690 全部 bit 级一致**（torch.equal 逐参数，抽查 op_s42 / noop_s1234 同样 0/886 differ；md5 不同只是 serialization metadata）。**本地续跑段（8000→8690，690 步）的 optimizer update 从未落到参数上**；
- **根因（? 已定位，PyTorch 侧 bug 与我们的 load 时序相互作用）**：PyTorch 2.11 `Optimizer.load_state_dict` 在恢复 per-param state 时会过一个 `_cast` 钩子（torch/optim/optimizer.py:968 定义、:1004 挂载，本为 autocast dtype 迁移设计）——它对 state 值**递归处理**，而 Python 的 `str` 恰好是 Iterable：走 `elif isinstance(value, Iterable)` 分支后执行 `type(value)(<generator>)` = `str(<lazy generator>)` = **generator 对象的 repr 字符串**（genexpr 惰性不执行，所以既不递归也不报错）。Muon 在每个参数的 state 里存了字符串标记 `state[p]["algorithm"]`（构造时写好，`"muon"` 或 `"adamw"`，`step()` 靠 `group["algorithm"] == "muon"` / `== "adamw"` 双筛选决定对该组跑哪套 update）——load 后这些标记全部变成 `<generator object ...>` 串 → 双筛选全失配 → `step()` 对所有参数 no-op。`momentum_buffer` 是 Tensor 走 Tensor 分支完好无损——所以恢复链全程无异常表象，loss 平台波动像正常训练（v4 补记据此误判「train 侧健康」，? 勘误：波动只是 batch 采样噪声）。v4 补记的「graph × resume」候选根因 ? 作废——graph 与本 bug 无关，从头跑不踩是因为不调 `load_state_dict`；
- **修复（commit 001cffa，最小 diff +27 行）**：`di/verl_sft_entry.py` patch 5 尾部 wrap `optimizer.load_state_dict`——构造 optimizer 后先记下每组的 algorithm（`_group_algs = [g["algorithm"] for g in groups]`），原 `load_state_dict` 执行完后按组序 zip 重写每个 param 的 `state[p]["algorithm"]`（load 不换组序，zip 对齐安全）。fix 副本 `di/script/fix/verl_sft_entry.py` 同步（挂载链 `v2_sft_entry_rwi34.py` L60 把 fix 目录插 `sys.path` 最前，import 解析到 fix 副本——不同步 = 修复不生效）；
- **修复重续跑 + 终验双判据**：6 ckpt 从 step 8000 重续跑 690 步（GPU5/GPU7 两路串行，稳态 9.68s/step，11-15 10:29 全部 step 8690 rc=0）。终验：① **8690 vs 8000 参数对拍六档全部 824/886 differ**（62 个 equal = `requires_grad=False` 类参数，本就不更新；冻结版是 0/886 bit 级一致）——参数真的动了；② **val loss 曲线离开平台**：修复版六档 val 8100→8600 单调下降（每档 ?0.9~~?1.3e-3，如 op_s42 0.549988→0.548790），对照冻结版尾三打点是完全相同的常数（权重不动 + 确定性 forward）。修复语义签名：首步 8001 loss 与冻结版逐位一致（resume 起点权重相同、语义正确）+ 第二步 8002 起微偏 1e-5~~1e-4（update 开始生效）；
- **影响（? 已全部处置，真值进表）**：① §10 表 2l32x8 两行 6 格全部换修复版真值——w100 **noop 0.93933±0.00071 / op 0.93936±0.00155**（冻结版 0.94070/0.94075 是 8000 权重的复读值、虚高 ~0.0014）、Δ BL(3s) **?0.00246 / ?0.00242**、行序从 #13/#14 升到 **#10/#11**（全表 mean 升序混排的本来位置）；val 四格填 C′ 口径真值 **noop 0.85420±0.00036 / op 0.85459±0.00092**（独立 val loop 对 @8500 ckpt，token 配比合成键，与全表 val 列同口径）；② **Ω 单变量效应上修**：§9.4 Ω 表 2l32x8 两行效应从 ?0.00251/?0.00278 修正为 **?0.00390/?0.00415**（冻结版把 Ω 修正效应低估 ~50%）——「7/8 变体大幅真增益」结论不变且更强（范围 ?0.0039 ~ ?0.0115）；③ 修复重续跑产物（8500/8690 ckpt + 全程 metrics）在 `val_rerun/stage2_rerun/`，OBS 落位见 val_rerun 交卷；ts16_b 24 格无 C′（B 类未到 8500 存点即被停——2026-11-10 批停后不再重挂，live 值即终态）不受本修复影响（B 类全程 DI 从头跑，无本地续跑段）。

### 9.5 B 类一笔带过

## B 类（ts16_b_ 前缀 8 行）= A 类的同变体、挂载位点换 ffn_gate_cross（§6.2 的跨层挂载——WONN 读上层 gate 激活）。任务书口径（学长 20:21）：「除挂载位点外 B 就是 A」。判读价值有二：B vs A 同变体 = 挂载位点单变量（three_source 口径下 swish vs cross 谁赢）；B vs g8_old 同名档 = Ω 单变量（cross 挂载下的修正效应，与 §9.4 的 swish 口径互为印证）。**B 类已于 2026-11-10 17:17 被停**（快照时 6 档 running、step 188-6110/8690；2 档未起；后续部分重挂再停）——2026-11-14 拍板不再重挂，**live 数值即终态（非完赛值），此轴永不出场判读**。live 值（1.0-5.0 量级）远大于 A 类收官值的原因：这些 run 停在退火中段（step 远小于 8690、lr 还在高位），loss 天然高于收官口径——与词典框「退火未完成的档 loss 天然高于收官口径」同一条口径警告。

## 10. 整张 Table 绝对全量（31 行 × 全列）

对象 = `plots/corrected_sw_and_gatecross_merger_table_w100.png`（同名 `_data.json`，快照 built_at = **2026-11-11T22:04:42**，Codia Draw 同刻已按新 Δ 锚重渲）。本节 = json 全部字段 + 画图脚本（`scripts/plot_corrected_sw_and_gatecross_merger_table.py`）的列构造**逐格搬进 MD，绝对完整，不是概览**。
**快照时进度（json `ts16_progress` 字段）**：48 run 总账 = 24 done + 18 running + 0 failed + 6 queued；A 类 8/8 档起步（全 done）、B 类 6/8 档起步；running 档 step 分布 188-6110 / 8690。
**口径四条（先读再读表）**：

1. **双窗**：g8_old 列 = 收官 w100（step 8590-8690 固定窗）；ts16 列 = 实时末窗（每 run 最新 100 个 train loss 点滚动均值；done 档末窗 = 8591-8690 恰衔接收官口径）。退火未完成的档 loss 天然高于收官口径（lr 还大）——跨侧数值不可直接比。
2. **Δ 锚 = BL(3-seed) 0.9417862**（baseline 行自身三 seed mean；2026-11-11 学长拍板「所有的 Delta 都是和 baseline 3 seed 比较」）。旧锚 bl2 = 0.9415391（v5 2-seed resume，只计 s1234/s2345 两 seed）**弃**——bl2 与 bl3 的差 +0.00025 即 baseline 行在旧锚下的 Δ。换锚连带：g8_old 筛选规则「w100 ≤ BL」按 bl3 判放行 **mhc_n2**（0.94162）进表——v1 版 30 行，本版 **31 行**。
3. **val 列** = val loss 最后打点（`test_freq=100` 在 held-out split 上的 validation loss）——与 w100 train loss 窗均值是**不同 metric**，仅加列、不参与排序。g8_old 侧 = 收官 val@8690；ts16 侧 = 最新打点（live 档标 `值@vstep`）。
4. **live/queued 格式**（照脚本）：ts16 done 档 mean = `mean±sd`；running 档 = `mean [live k/3]`（k = done seed 数，全部 k=0）且 seed 格 = `值@step`（末窗落在的 step）；queued 档全 `—`（排队未起步，非缺行，起跑后重跑快照自然出数）。> 
> **?? 口径总注（train / val 的权重步数，2026-11-15 v7 加，先看这行再看表）**：**train loss（w100 列）打点至各 run 完赛步 8690**（末 100 点窗 8591-8690，往前全程曲线在 metrics jsonl）；**val 列全部是 @8500 ckpt 权重**——唯 **baseline 行的 s1234 / s2345 两格 = @8690 真收官权重**（本地续跑链存点完整；8690 收官存点在 DI 全批系统性缺失，全案见本节末尾存点差异注）。跨格比较时注意此口径差。
> 
> 
> | # | 档（label） | 色 | rid | w100 mean±std | s42 | s1234 | s2345 | val mean±sd | val s42 | val s1234 | val s2345 | Δ BL(3s) | status | 节 |
> | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | 1 | hc2_n4 [宿主 3-seed] | 灰 | g16 | 0.93543±0.00039 | 0.93542 | 0.93582 | 0.93505 | 0.85088±0.00096 | 0.85053 | 0.85197 | 0.85014 | **?0.00635** | done | §8 |
> | 2 | hc_n4 [v5 档 s42 补跑] | 灰 | g16 | 0.93728±0.00116 | 0.93821 | 0.93765 | 0.93599 | 0.85313±0.00046 | 0.85291 | 0.85366 | 0.85281 | ?0.00450 | done | §8 |
> | 3 | mhc2_n4 [宿主 3-seed] | 灰 | g16 | 0.93772±0.00028 | 0.93741 | 0.93794 | 0.93780 | 0.85263±0.00040 | 0.85219 | 0.85270 | 0.85299 | ?0.00407 | done | §8 |
> | 4 | hc_n5 [v5 档 s42 补跑] | 灰 | g16 | 0.93807±0.00250 | 0.94029 | 0.93856 | 0.93535 | 0.85460±0.00353 | 0.85466 | 0.85810 | 0.85104 | ?0.00372 | done | §8 |
> | 5 | hc_n2 [v5 档 s42 补跑] | 灰 | g16 | 0.93815±0.00216 | 0.94035 | 0.93806 | 0.93603 | 0.85428±0.00215 | 0.85562 | 0.85542 | 0.85180 | ?0.00364 | done | §8 |
> | 6 | ts16_a_mhc2n4_g1_m256 | 浅蓝 | r580/596/612 | 0.93825±0.00433 | 0.94164 | 0.93338 | 0.93974 | 0.85338±0.00399 | 0.85666 | 0.84895 | 0.85453 | ?0.00353 | done | §9.3 V8 |
> | 7 | hc_n3 [v5 档 s42 补跑] | 灰 | g16 | 0.93828±0.00183 | 0.93990 | 0.93867 | 0.93629 | 0.85358±0.00149 | 0.85482 | 0.85399 | 0.85193 | ?0.00350 | done | §8 |
> | 8 | ts16_a_2l16x8_sinpsi_m8_lambda_noop | 浅蓝 | r579/595/611 | 0.93880±0.00033 | 0.93896 | 0.93901 | 0.93841 | 0.85400±0.00036 | 0.85442 | 0.85386 | 0.85373 | ?0.00299 | done | §9.3 V1 |
> | 9 | ts16_a_2l16x8_sinpsi_m8_lambda_op | 浅蓝 | r576/592/608 | 0.93901±0.00078 | 0.93980 | 0.93898 | 0.93824 | 0.85419±0.00077 | 0.85441 | 0.85334 | 0.85483 | ?0.00278 | done | §9.3 V2 |
> | 10 | ts16_a_2l32x8_sinpsi_m4_lambda_noop | 浅蓝 | r578/594/610 | 0.93933±0.00071 | 0.93910 | 0.94012 | 0.93877 | 0.85420±0.00036 | 0.85432 | 0.85449 | 0.85380 | ?0.00246 | done | §9.3 V3 |
> | 11 | ts16_a_2l32x8_sinpsi_m4_lambda_op | 浅蓝 | r577/593/609 | 0.93936±0.00155 | 0.94102 | 0.93913 | 0.93794 | 0.85459±0.00092 | 0.85565 | 0.85409 | 0.85403 | ?0.00242 | done | §9.3 V4 |
> | 12 | ts16_a_g64_sinpsi_m16 | 浅蓝 | r574/590/606 | 0.93990±0.00149 | 0.94044 | 0.94104 | 0.93821 | 0.85547±0.00083 | 0.85595 | 0.85595 | 0.85451 | ?0.00189 | done | §9.3 V5 |
> | 13 | swish_gate_proj_a0b0c1s0 [v5 档 s42 补跑] | 粉红 | g16 | 0.94028±0.00097 | 0.94140 | 0.93983 | 0.93962 | 0.85527±0.00085 | 0.85625 | 0.85490 | 0.85468 | ?0.00150 | done | §6.1 |
> | 14 | 2l32x8_sinpsi_pnoff_m4_lambda_noop | 青 | r509/525/541 | 0.94069±0.00058 | 0.94022 | 0.94051 | 0.94134 | 0.85585±0.00108 | 0.85508 | 0.85539 | 0.85709 | ?0.00110 | done | §7.2 |
> | 15 | 2l16x8_sinpsi_pnoff_m8_lambda_op | 橙 | r506/522/538 | 0.94102±0.00085 | 0.94184 | 0.94109 | 0.94014 | 0.85570±0.00083 | 0.85654 | 0.85568 | 0.85488 | ?0.00077 | done | §7.2 |
> | 16 | 2l32x8_sinpsi_pnoff_m4_lambda_op | 橙 | r508/524/540 | 0.94104±0.00073 | 0.94122 | 0.94167 | 0.94024 | 0.85572±0.00023 | 0.85564 | 0.85598 | 0.85554 | ?0.00074 | done | §7.2 |
> | 17 | ffn_gate_cross_proj_a0b0c1s0 [v5 档 s42 补跑] | 粉红 | g16 | 0.94110±0.00124 | 0.94252 | 0.94054 | 0.94025 | 0.85617±0.00138 | 0.85774 | 0.85516 | 0.85561 | ?0.00068 | done | §6.2 |
> | 18 | 2l16x8_sinpsi_pnoff_m8_lambda_noop | 青 | r511/527/543 | 0.94140±0.00113 | 0.94256 | 0.94132 | 0.94031 | 0.85727±0.00208 | 0.85694 | 0.85949 | 0.85538 | ?0.00039 | done | §7.2 |
> | 19 | A2 g64_m16_sinpsi_pnoff_stat [G 轴 V1024] | 白 | g16 | 0.94150±0.00107 | 0.94181 | 0.94238 | 0.94031 | 0.85622±0.00074 | 0.85637 | 0.85687 | 0.85541 | ?0.00029 | done | §7.1 |
> | 20 | mhc_n2 [v5 档 s42 补跑] | 灰 | g16 | 0.94162±0.00080 | 0.94159 | 0.94244 | 0.94083 | 0.85616±0.00103 | 0.85579 | 0.85733 | 0.85537 | ?0.00017 | done | §8（v2 新进表） |
> | 21 | baseline [BL: s42 G16 从头跑 + s1234/s2345 v5 resume] | 金 | g16 | 0.94179±0.00065 | 0.94228 | 0.94105 | 0.94202 | 0.85663±0.00058 | 0.85711 | 0.85599 | 0.85680 | +0.00000 | done | §5 |
> | 22 | ts16_a_g1_m512 | 浅蓝 | r573/589/605 | 0.94656±0.00380 | 0.94279 | 0.95039 | 0.94651 | 0.86147±0.00375 | 0.85825 | 0.86558 | 0.86058 | +0.00478 | done | §9.3 V6 |
> | 23 | ts16_a_hc2n4_g1_m256 | 浅蓝 | r575/591/607 | 0.95260±0.00577 | 0.94760 | 0.95130 | 0.95891 | 0.86685±0.00488 | 0.86295 | 0.86526 | 0.87233 | +0.01082 | done | §9.3 V7 |
> | 24 | ts16_b_g1_m512 | 浅蓝 | r581/597/613 | 0.98901 [live 0/3] | 0.99136@5925 | 0.98838@6110 | 0.98727@6047 | 0.59788±0.00181 | 0.59643@5900 | 0.59730@6100 | 0.59990@6000 | +0.04722 | stopped(2026-11-10) | §9.5 |
> | 25 | ts16_b_hc2n4_g1_m256 | 浅蓝 | r583/599/615 | 1.16665 [live 0/3] | 1.15086@2854 | 1.18692@2545 | 1.16217@2670 | 0.77964±0.00863 | 0.77915@2800 | 0.78850@2500 | 0.77126@2600 | +0.22486 | stopped(2026-11-10) | §9.5 |
> | 26 | ts16_b_g64_sinpsi_m16 | 浅蓝 | r582/598/614 | 1.31208 [live 0/3] | 1.31213@1739 | 1.31201@1740 | 1.31211@1740 | 0.90047±0.00111 | 0.90030@1700 | 0.89945@1700 | 0.90165@1700 | +0.37030 | stopped(2026-11-10) | §9.5 |
> | 27 | ts16_b_mhc2n4_g1_m256 | 浅蓝 | r588/604/620 | 2.31091 [live 0/3] | 1.75838@874 | 2.58843@562 | 2.58593@562 | 1.77745±0.40848 | 1.30578@800 | 2.01281@500 | 2.01377@500 | +1.36913 | stopped(2026-11-10) | §9.5 |
> | 28 | ts16_b_2l16x8_sinpsi_m8_lambda_op | 浅蓝 | r584/600/616 | 3.66803 [live 0/3] | 4.20501@249 | 3.64553@313 | 3.15354@438 | 2.73790±0.42759 | 3.20416@200 | 2.64543@300 | 2.36412@400 | +2.72624 | stopped(2026-11-10) | §9.5 |
> | 29 | ts16_b_2l16x8_sinpsi_m8_lambda_noop | 浅蓝 | r587/603/619 | 5.03868 [live 0/3] | 3.64771@313 | 5.75867@188 | 5.70967@188 | 5.30873±2.30435 | 2.64933@300 | 6.71409@100 | 6.56278@100 | +4.09690 | stopped(2026-11-10) | §9.5 |
> | 30 | ts16_b_2l32x8_sinpsi_m4_lambda_noop | 浅蓝 | r586/602/618 | — | — | — | — | — | — | — | — | — | queued | §9.5 |
> | 31 | ts16_b_2l32x8_sinpsi_m4_lambda_op | 浅蓝 | r585/601/617 | — | — | — | — | — | — | — | — | — | queued | §9.5 |
> | **列构造对照**（脚本 9 列 → 本表列）：`w100 mean±std` = 行 mean（g8_old = group8 json 直读收官值；ts16 done = 末窗三 seed mean±sd；ts16 running = `[live k/3]` 格式）｜三 seed 列 = 各 seed w100（running 标 `@step`）｜`val mean±sd` + 三 val 列 = `_val4` 镜像 w100 列组顺序（live 标 `@vstep`）｜Δ 列 = `delta_bl3` = mean ? 0.9417862。rid 列 = DI run 号（`r573-620` ts16 / `r506-543` group8 / `g16` = g16 批直读无独立 run 号段）。色列 = 图内行色（灰宿主 / 金 BL / 粉红 v5 星档 / 橙 λ op / 青 λ noop / 白 g16 档 / 浅蓝 ts16）。 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | **读表要点**：① 24-29 行 B 类是被停时点的 live 末窗滚动数（2026-11-10 被停、非完赛值），不参与任何判读；② 跨侧（g8_old vs ts16）比较必须带口径声明（§9 双变量警告）；③ Δ 零点 = 行 21 baseline 自身（0.9417862，表内显示 0.94179 五位舍入）；④ 全表 mean 升序混排——排序键 = w100 mean，与 val 列无关；⑤ 行 10/11（ts16_a_2l32x8 两档）的 w100 与 val 格 = **修复重跑版真值**（2026-11-15 收官）：这两档曾因 resume 冻结 bug（PyTorch 2.11 `Optimizer.load_state_dict` 的 `_cast` 毁 Muon `algorithm` 标记 → step no-op，根因链见 §9.4 警告块）在 v4/v5 版标 N/A + w100 虚高 0.0014——修复（commit 001cffa）后 6 ckpt 从 step 8000 重续跑 690 步收官（终验 824/886 differ + val 离开平台），真值回填 w100 四格（noop 0.93933±0.00071 / op 0.93936±0.00155）+ val 四格（C′ 口径，noop 0.85420±0.00036 / op 0.85459±0.00092）+ Δ（?0.00246/?0.00242），行序升至 10/11 位；配套表格图 `corrected_sw_and_gatecross_merger_table_w100.png` 同刻重渲（渲染入口 `val_rerun/render_reval.py`，真值落沙箱副本 `val_rerun/table_reval/` json——plots 原件 json 仍是 build 层快照不动，直接跑原画图脚本会把旧值渲回来）；⑥ 行 10/11 的 w100 语义与其他 done 档一致（8690 权重末窗 8591-8690）——v5 版「8000 权重少训 690 步」的语义警告随重跑收官**作废**。 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | **⑦ v5 口径换代**（2026-11-15）：本表 val 列已全部换 C′ token 配比新键（~0.85，@8500 权重， |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 独立 loop）——与 v4 及更早版本的 ~0.55 数值不可直接比；未换格仅剩：B 类 6 行（ts16_b，DI 侧已停 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 无 C′ 覆盖，val 格仍是 DI 在线打点 @vstep 旧口径即终态）；行 10/11（2l32x8 两档）的 C′ val 已随 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 阶段 2 修复重跑收官回填真值（@8500 存点独立 loop，读表要点 ⑤）；**baseline 行 s1234/s2345 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 两格已于 v7 补跑回填**（2026-11-15）：v5 挂账「ckpt 被 @4000 重挂覆盖、C′ 不可重跑」判断翻案 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | ——`/cache/wonn_rwi_resume/r100{1,2}_out_v2/` 三存点（4346/6519/8690）实锚全在 + 权重 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | MOVING（4346 vs 8690 对拍 725/725 differ，未踩冻结 bug——bctrl 链 `BCTRL_OPTIM_CKPT` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 手动恢复 Muon state + algorithm 真值写回，绕过 `_cast` 坑），真 8690 ckpt 的 C′ val = |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | s1234 **0.85599** / s2345 **0.85680**，三格同口径 val mean±sd = **0.85663±0.00058**（v6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 版的 0.85711±0.00000 单值混排格作废）；**步数口径注意**：baseline 行三 seed 的权重步数不齐 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | ——s42 是 g16 批 DI 从头跑存点 **@8500**（DI 链 8690 force save 从未成功，见 §10 末尾存点差 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 异注），s1234/s2345 是本地 bctrl 续跑链 **@8690**（比 @8500 多 190 步退火尾部——该段 lr 已 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 从 ~1e-5 退至 ~4e-8，影响量级远小于 seed 间 sd，但严格口径下 baseline 行内部 s42 格与其他 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 行同步数、两 seed 格多 190 步）。 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | **§10 末尾注——为什么全表 done 档都是 @8500 而 baseline 两 seed 有真 @8690（v7 查证， |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 2026-11-15）**：学长问「ckpt saving 代码多版本没动过，为什么 baseline 有真 8690 而其她 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | variants 没有」。查证结论：**不是代码版本差异**——容器侧 verl（`di/verl_sft06/`，pip -e 安装） |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 与本地 `async_rl/verl` 的 `fsdp_sft_trainer.py` **逐字节一致**（diff 实锚），fit 循环原生带 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | `is_last_step → save_checkpoint` force save（上游 v0.6.0）；ts16 挂单 `STEPS=8690` 透传链完 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 整（tsv 头注释 → `gen_runs.sh` → `v2_run_one.sh` `trainer.total_training_steps=8690`）； |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | `v2_run_one.sh` 的「终传 sidecar 一拍」（防炸审计 P2-2①，commit ccadca1 2026-10-14 起） |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 也在挂单时点就位。**但 DI 全树实锚**：v2_run_one 链九批（ts16/g16/g4/group8/phase_mean/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | rs/rwi_v5/sw8/lr_dip）抽样**全部止步 8500，无一档 8690 上过 OBS**；r573 等档 metrics 打点到 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | step 8690（train+val 全跑完），train.log 尾窗（live_progress 最后快照）却停在 step 8688—— |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 8690 之后的 stdout 还在 python 块缓冲里没落盘 = 训练进程在 8690 的 save 窗口被 **SIGKILL 类 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 非正常终止**（正常/异常退出都会 flush）的签名。**系统性死法**（48 run 全批一致卡在 8690 save |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 窗口）的凶手无法从现存证据直判（容器已回收，driver.log/dmesg 均失），候选按可能性排序： |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | ① 容器 cgroup 内存 OOM（8690 = 全程唯一「val@is_last_step + save gather」叠加窗口 + |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 13 小时内存累积水位）；② DI 平台侧容器终止。**baseline 两 seed 为什么有真 8690**：r1001/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | r1002 是本地 bctrl 续跑链（2026-10-23 学长拍板 baseline 从 3500 续满 8690）——本地跑无容器 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 回收竞态，save_freq=2173 的中途档 4346/6519（2×2173/3×2173）+ is_last_step force save |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 8690 全部落盘存活。**baseline s42 不是 8690**：s42 = g16 批 DI 从头跑重跑档（g16 tsv 头注释 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 「baseline s42 …全挪本批重跑」），DI 链 → 存点 @8500。loop2 批（`di_run/loop2/`）OBS 上的 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
> | 8690 档为 17 件/档（本地链存档形态；DI v2_run_one 链原生档为 20-21 件），非 DI 原生存点。 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

---

## 修订记录

- v1（2026-11-11）：首版全量。§0-§10 + Table 30 行。查证底座 = modeling 源码（WinfreeCore/HyperConnections/挂载层/_build_wonn/model forward）+ prep_rwi cfg_patch 四段 + 物化档 config 实锚 + PLAN §4 + LOG 条目 82/83/94/136/137/138/139/140/141/142/143 + 供奉十五~二十一 + sw8/ts16 两张合表 json。
- v2（2026-11-11，学长批改轮四问修订）：① Δ 锚全表换 BL(3-seed) 0.9417862（学长 21:24 拍板「所有的 Delta 都是和 baseline 3 seed 比较」）——全文 Δ 列刷新，连带 mhc_n2 进表（30→31 行，§8/§10）；② **§6.4 新节显答「Omega 带 SiLU 吗」**：是——星档/ts16 全链 `wonn_omega_act=silu_split`（v5 星档三 seed OBS 直读 + CFG_SRC 本地直读 + ts16 物化档 + modeling 默认值 none 反证四重实锚；v1 漏列此键 = 缺口）；③ **§6.5 新节显答「Swish Gate 是什么」**：两挂载（swish_gate #9 / ffn_gate_cross #14）输入与汇回逐项对比表 + 易混点点破（v1 未讲）；④ **§2.1 末尾警告块**：初星之晨是蓝图——Omega Norm/双流持久结构实装均无，真遗产三件点明（v1 照抄蓝图未声明差异 = 缺口）；⑤ **§10 重写为绝对全量**：31 行 × 全列（std、五列 val、三 seed w100、rid、status、ts16_progress、双窗口径、live/queued 格式全照 json + 画图脚本逐格搬，v1 概览版 = 缺口）；⑥ §9.5 勘误：删「B 类从 step 8000 续窗」错误说法（B 类实为 step 188-6110 退火中段，v1 此句系误推）；⑦ §0 补 val 列定义（test_freq=100 held-out，与 w100 不同 metric）。配图 `corrected_sw_and_gatecross_merger_table_w100.png` 同刻由 Codia Draw 按新锚重渲（22:04）。
- v3（2026-11-14，数据集与复现口径批——学长 15:58/16:00 点名 + 17:08 发车）：① **§4 新增 4.1-4.4 四小节**——4.1 数据集 f_combo 四域全表（名字「752M」是设计期预算、实测 776.79M；四域行数/占比/口径照 `data/f_combo/state.json` + `scripts/prep_f.py` 实锚；merge shuffle seed 42、val 等权 2,000×4 seed 43 不洗、max_length=1024 预算）；4.2 shuffle 与 seed 链（显答「random seed 同时作用于 dataset 和 model weights？」= **否**——DistributedSampler 的 shuffle seed = epoch 号（全程 epoch 0）与 WONN_SEED 无关，WONN_SEED 只作用于 model init；`trainer.seed=1` 是 SFT 路径死键；推论 = seed 轴是 init 方差）；4.3 w 口径换代史（table500 截点 → w25 → w100 三代 + 定稿理由照 LOG 条目 94 补记五 + done 门控条目 113）；4.4 val 口径（在线算非 ckpt eval、合表取最后打点、86 点全曲线在 jsonl、table500 系无 val 列）；② **§9.4 新增 val 冻结警告块**（本地续跑 6 档 val bit 级复读第一次：决定性模式 = 冻结值 ≠ DI 8000 值（差 1.9e-5）= 本地段第一次 val 值；train 不冻结；已排除写入侧覆盖与 graph replay——val 走 eager（`not do_backward` 绕 graph）；?? 机理推断 FSDP2 × eval/no_grad × resume 的 unshard 缓存交互，未实锤挂账）；③ **§10 表行 13/14 val 格标 ?** + 读表要点 ⑤；④ **§0 rwi 全称展开**（random weight initialization + 术语表新条目）、val 定义细化（val.parquet 构造 + 86 点 + 指针 §4.4）、w100 条目加换代史指针；⑤ 正文 §3 后缀语义处 rwi 补全称。查证底座 = prep_f.py + state.json + fsdp_sft_trainer.py（_build_dataloader/fit/validation_step）+ verl_sft_entry patch 3/6 + v2_run_one.sh 实配 + LOG 条目 18/20/21/94 补记五/113/143 + OBS r578/r579 metrics 直读。
- v4（2026-11-14，冻结 6 格 N/A 更正——学长 17:24 拍板「到时 val 也不用算了，表格里面直接标 N/A，然后给原因」，17:35 明确 ts16 侧执行边界）：① §10 表行 13/14（ts16_a_2l32x8 两档）val 格 8 格由 ? 冻结值改标 **N/A**（v3 的 ? 升级——冻结值不作数）；② 读表要点 ⑤ 重写（N/A 语义 + 拍板原话 + 图渲染入口 `val_rerun/rerender_na.py` + json 快照保留声明）；③ §4.4 冻结异常段与 §9.4 警告块「影响」条同步 N/A 处置 + C' 重跑不回填声明；④ 配套表格图 `corrected_sw_and_gatecross_merger_table_w100.png` 同刻更正（monkeypatch 原画图脚本 `_val4` 后调原 `main()` 覆盖原图，原脚本与 json 零改动——错误值更正不属于「旧图不动」保护范畴，那条保护的是 C' 新口径重画）。
- v4 补记（2026-11-14 18:30，根因参数级实锤——val C' 重跑批 parity 副产物翻案）：本地续跑 6 档 **886 参数 8000≡8500≡8690 bit 级全同**（torch.equal 逐参数，三档抽查；8690 ckpt 的 val forward 与 8000 bit 级同值是第一发现）——续跑段 optimizer update 从未生效。§9.4 警告块全量重写：v3 的「train 不冻结 = train 健康」? 误判（波动是 batch 噪声非训练证据）、「第一次 val 正常之后复读」? 误读（1.9e-5 差在两路径口径噪声内，每次都是 8000 权重的真算）、FSDP2 unshard 缓存假设 ? 作废；根因锁定 **graph × resume 组合**（train.log L34 `VERL_SFT_GRAPH` 实锚 + DI 从头跑段 graph 开且有效 = val 收敛 0.55 实锚，行级定位挂账）。影响升级：w100 语义 = 8000 权重（读表要点新增 ⑥，处置待学长拍板）；6 档实际终点 = step 8000；C' 重跑出的新 val = 8000 权重的真实 val（合法，语义如此标注）。
- v5（2026-11-15，配比口径勘误统一改——学长 2026-11-14 晚指出「配比出来跟原来一样是 0.5 几
? 不是 0.9 几」）：① **val 列全表换 C′ token 配比新键**（§10 表 21 行 82 格 + g8_old/合表/
? sw8/group8/pm/g4/v5/g16 全部分表 + 全部表格 png 重渲）——train_mix 合成权重从行数配比
? （gsm 44.96%/pw 26.32%/stories 19.84%/orca 8.89%，合成出 ~0.43/等权 ~0.55）改为 token
? 配比（stories 58.21%/gsm 23.89%/orca 8.97%/pw 8.93%，state.json `train_tokens` 实测，
? 合成出 ~0.85），与 train w100 0.9x 同量级（**「val 和 train 同等级」验收锚达成**）；②
? §0 val 列定义重写（C′ 口径 + 为何换）+ §4.4 v5 段（口径换代说明 + 渲染机制）+ 读表要点
? ⑦（v5 时点记录的未换格清单：ts16_b B 类 24 格 DI 在跑、baseline resumed s1234/s2345 不可重跑、
? 2l32x8 六格 N/A 待阶段 2）；③ 渲染机制：plots json 原件不动（build 层快照），重渲统一走
? `val_rerun/render_reval.py`（json.load 重定向沙箱副本 + 合表 _val4 N/A patch 合并）；
? ④ 数值源 307 件 json 全量重算（`recompute_train_mix.py`，对拍主 Codia 参考值 6 变体
? Δ 全 0.0000）；⑤ phase1 简报（`val_rerun/phase1_ts16_18_report.md`）train_mix 列同步换。
- v6 阶段 2（2026-11-15，冻结 6 档修复重跑收官——v5 遗留「阶段 2 修复重续跑后回填」兑现）：
? ① **根因定位 + 修复**（commit 001cffa）：resume 冻结的根因 = PyTorch 2.11
? `Optimizer.load_state_dict` 的 `_cast` 钩子对 state 值递归处理，Muon 的 `algorithm` 字符串
? 标记（`"muon"`/`"adamw"`，step() 组筛选依据）作为 str 走 Iterable 分支被毁成 generator
? repr 串 → step no-op 参数不动；修法 = verl_sft_entry patch 5 尾部 wrap load_state_dict，
? load 后按构造时组语义重打标记（fix 副本 di/script/fix/ 同步，挂载链 sys.path 最前解析）；
? v4 补记的「graph × resume」候选根因 ? 作废；② **6 ckpt 从 step 8000 重续跑 690 步收官**
? （GPU5/GPU7 两路，全部 step 8690 rc=0）——终验双判据：8690 vs 8000 六档全部 824/886 参数
? differ（62 equal = requires_grad=False 类）+ val loss 曲线离开平台单调下降（每档
? ?0.9~~?1.3e-3；冻结版是尾三打点完全相同常数）；③ **§10 表 2l32x8 两行全量换真值**——
? w100 noop 0.94070→**0.93933**±0.00071 / op 0.94075→**0.93936**±0.00155（冻结值虚高
? ~0.0014）、Δ 换 ?0.00246/?0.00242、行序 #13/#14→**#10/#11**（连带 #10-12 三行行号顺移）、
? val 四格 N/A→C′ 口径真值 noop 0.85420±0.00036 / op 0.85459±0.00092（独立 val loop
? @8500，token 配比合成键，与全表 val 列同口径）；④ **§9.4 全节终态化**：警告块改「已修复
? 重跑收官」（根因机理 pedagogical 展开 + 修复 + 终验 + 影响四段；graph×resume 作废声明）；
? Ω 单变量表 2l32x8 两行效应 ?0.00251/?0.00278 修正为 **?0.00390/?0.00415**（冻结版把 Ω
? 修正效应低估 ~50%），§9.4 结论 1 范围句连带 ?0.0025→?0.0039（「7/8 大幅真增益」结论
? 不变且更强）；§9.3 变体表 V3/V4 行新值 + 行序挪到 V2 后；⑤ **读表要点 ⑤⑥ 重写**（⑤
? N/A 语义 → 修复版真值回填终态 + 渲染入口改 render_reval.py/沙箱 json 落盘；⑥ w100 语义
? 警告作废声明）+ ⑦ 尾句已回填声明 + §4.4 末总警告块终态化 + §9.4 本地续跑句补「修复版
? 真值」声明；⑥ 渲染：真值落沙箱 `val_rerun/table_reval/corrected_..._data.json`
? （`rerender_stage2.py` 一次性更新器）→ `render_reval.py corrected` 重渲（SPECS 移除
? val4_na patch——N/A 阶段结束；plots 原件 json 仍不动）。修复重续跑产物
? `val_rerun/stage2_rerun/`（8500/8690 ckpt + metrics，~~37G，OBS 落位见 val_rerun 交卷）。
- v7（2026-11-15，baseline 参照行两格 C′ 补跑回填 + 8690/8500 存点差异查证——学长 14:18/14:20
? 两轮指令）：① **v5 挂账翻案**：「baseline resumed s1234/s2345 两格 C′ 不可重跑（ckpt 被
? @4000 重挂覆盖）」判断错误——当时只查了 `r1001_out`（v1 输出止步 4346）漏了 `_out_v2`；
? 实锚 `r100{1,2}_out_v2/` 三存点全在 + 权重 MOVING（该批没踩冻结 bug）；② **两格补跑**
? （任务 1）：`run_ts16_val.py` token 配比新键跑真 8690 ckpt（沙箱副本，GPU5/7 双路）——
? s1234 **0.85599** / s2345 **0.85680**；§10 表行 21 val 三格齐（mean±sd
? **0.85663±0.00058**，v6 版 0.85711±0.00000 单值混排格作废）+ rwi_v5 表（table8690）
? baseline 参照行两格 + 两值统计 0.85640±0.00057——真值落沙箱两 json
? （`update_stage3_baseline.py` 一次性更新器）→ `render_reval.py` 两表重渲；③ 读表要点
? ⑦「不可重跑」改「已补跑回填」终态 + **步数口径注**（s42 @8500 g16 批 DI 从头跑 vs 两
? seed @8690 本地 bctrl 续跑链——差 190 步退火尾部，lr 已 ~4e-8）；④ **§10 末尾新增存点
? 差异注**（任务 3 查证结论——为什么全表 done 档 @8500 而 baseline 两 seed 有真 @8690：
? 代码同源 diff 实锚无版本差异，DI 链九批 8690 force save 从未成功、train.log 尾停 8688 =
? SIGKILL 类非正常终止签名，凶手候选 cgroup OOM / 平台终止两案并陈；baseline s42 =
? @8500 非 8690）；⑤ baseline 步数语义：r1001=s1234 / r1002=s2345 实锚（train.log
? `WONN_SEED` + config `partial_pretrain`）。产物 `val_rerun/results_stage3/`（2 json）+
? `val_rerun/logs_stage3/`（跑批日志）。补记（15:30 学长指令）：**§10 表格顶部新增
? ?? 口径总注一行**（显眼位）——train loss（w100）打点至 8690；val 全表 @8500 权重、
? 唯 baseline s1234/s2345 两格 @8690——口径不一先声明再读表。