# 一个更加正式的 Neo PolyNorm Upcycling 汇报 （ draft ！）

## 目录

* [Overall](#sec-1)
* [方法（PolyNorm 2M MLP side branch）以及整体上的效果](#sec-2)
* [更多的说明（关于 benchmark 的结果）](#sec-3)
* [关于早期实验的说明（Best 4 layers）](#sec-4)
* [未来的工作](#sec-5)
* [RL rollout 成本 · 部署 · 可持续扩展](#sec-6)
* [开放性问题之关于 PolyNorm 架构本身的启发](#sec-7)
* [【 关于算力的讨论 】](#sec-8)
* [总结](#sec-9)

<a id="sec-1"></a>

## Overall

这是关于当前的 RL 加速技术的探索研发的一个汇报总结 。

**文字都是手搓 ；如果偶尔有 AI 风格 ，可能是因为跟 AI 交流久了 ，入乡随俗 （ 233 ） 。哦 ，不过图片都是 AI 画的 ，本质上 。图片的信息表达基本准确 。如有脑雾感请见谅 。**

整体结论是 ：**I propose 一种可以把 RL 加速 ≥ 2 倍的技术** （ 在 benchmark 的提升速度的意义上 ，包括绝大多数 In Domain 和 Out Of Distribution 的 benchmarks ；如果是 reward 的提升 ，则整体上在 2 倍左右或略超过 2 倍 ） 。  
这里没有在尝试去新增一个什么什么 PO 作为一个 GRPO 的 “ 如何如何考虑到 long horizon agentic task 或者 OPSD 或者教训自我总结或者 RSI 或者什么什么的 ” 变体 。  
同时 ，也不是一个 RSI harness 的什么工作 。这跟什么什么 PO 无关 ，也跟 harness 无关 。虽然当前仅在一种 Asynch RL method based on GRPO ( more specifically , Flash REINFORCE ) 上面验证 ，更多的严谨验证需要更多的算力验证更多的什么什么 PO ，不过 ，我将详尽阐述这一方法 ，从而提供一种可能的直觉与判断 ，也即 ，这种加速方法并不是一种传统意义上的 ” 算法 “ ， 同时也与所有传统意义上的 “ 算法 ” 是正交的 。也即 ，这个方法在尝试基于 **任何 的 RL setting 下** 获得 ≥ 2 倍的提升 。（ 注意 ，是 × 2 ，不是 × ( 2 + 1 ) ；局部是有 × 3 （ or maybe more ？ ） 的加速 ，但是这个不是中位数 ） 。

<a id="sec-2"></a>

## 方法（PolyNorm 2M MLP side branch）以及整体上的效果

具体来说 ，这个方法是 ：  
我们知道有一种 （ 相对 ） 小众的 Activation ，叫做 PolyNorm 。本质上是用一个多次多项式 ，其中各系数可学习 ，来代替常规的形状给定好的 Activation 。PolyNorm 原作者的意思是 ，PolyNorm 哪怕用 2-Matrix 的传统 MLP 构型 ，只在中间加一个 PolyNorm ，而不用 Gated MLP ，也能够在 ISO PARAMS 的前提下打赢 SwiGLU ，做到更低的 loss 和更高的 downstream benchmarks 效果 。当然 ，已知的基于 PolyNorm 的相对大规模的模型训练 ，来自一个韩国的开源大模型 Motif-3 ，这里的 PolyNorm ，仍然相对保守地采取了 Gated MLP 的形式 。这里不去争论这种原始的 PolyNorm 的架构的 ablation 或者选择 ，谁对谁错 ，只报告我的发现 。  
我们同时知道 ，有一种方法叫做 Latent MoE ，本质上是用 low-rank bottleneck 来逼出 higher-level representation ，当然这种技巧不仅局限于 MoE 的设计 。

【我发现】 ：  
基于 Qwen3-8B 模型 ，基于 《 Is One Layer Enough ？》 论文的 Math 数据集与 benchmarks 的选择 ，以及对于 OOD benchmarks 的选择 ，  
*（ 这里注意 ，之前其实被这个论文给坑了一下 。这个论文的结论 ，本质上是在想说 ，RL 模型的其中特定一层 ，或根据 RL 特定一层的结果选择性地 RL 特定数层 ，其 RL 效果要超过 RL 所有层 。可能如此结论在其所使用的 GRPO 及其变体里面成立 ，不过 ，在我这个 Flash REINFORCE 的 Asynch RL 设定下并不成立 。也有可能是 lr 的设置问题 ，比如说我的 lr 是 1e-6 ，their 可能是 5e-6 。不过 ，这个不应实质上影响结论的方向 ，因为 their lr scanning 下其实更低的 lr 是 favour them 的 。不过 ，这里不过多讨论 their 论文的结论是否在更广泛的条件下成立与否的问题 。这个论文本身提供了一定的研究背景与研究设定 ，同时 ，我的工作的一些早期实验基于该论文测得的 Qwen3-8B 的 Best 4 layers 展开 ，所以和 Full Base 同时 RL 的 baseline 不具备可比性 ，但是在 B4 的内部具备可比性 ，并且这种可比性在扩展到 Full Base 的时候观察到了可信的迁移 —— 虽然 ，更完整的迁移 ，由于当前的算力限制 ，不再有时间展开 。 之后可能还会提到这篇论文对早期实验的影响 ，所以这里姑且赘述一番 。 ）*
【我们可以观察到】 ，如果我们把一个 2-Matrix 的 PolyNorm MLP ，其中这跟 MLP 的 middle dimension 不为 Qwen3-8B 原 SwiGLU 的 middle dimension 也即 model dimension 的 3 倍 ，而是取 model dimension 的四分之三 ，也即 3072 ；以 main + delta 其中 delta = epsilon × tanh(MLP) ，其中 epsilon 取 0.03 ，以这样的方式把一个 PolyNorm MLP 加到模型 backbone 的 SwiGLU 的并行处 ，作为一个 side branch ；并且 ，初始化遵循 PolyNorm 2-M MLP 的第二个 Matrix 全 0 ，第一个 Matrix 是随机的较小值 （ 即严格控制初始的幅值 ） ；以这样的方式把 Qwen3-8B 所有的层都挂上这样的 PolyNorm 2M MLP ；并且 RL 的时候 ，全参数同时参与训练 ，也即 ，包括 Attention ，SwiGLU ，PolyNorm 2M MLP —— 那么我们可以得到 ，相比于同样作为全量训练 ，但是只有原来的 backbone 没有 PolyNorm 2M MLP 作为一个 limited-magnitude 的 side branch 参与进来的时候 ，**我们实现了绝大多数 benchmarks 上面的 ≥ 2 倍加速的 RL 训练 ！**

!\[PolyNorm 2M MLP side branch 加法机制图：单层视角，SwiGLU 主路（main）与 side branch（lin1 → PolyNorm → lin2 → PolyNorm → tanh → ×ε=0.03 = delta）并行，y = main + delta 汇合；图例含 PolyNorm 公式展开与 no-op 初始化](pn\_side\_branch\_architecture.png)

<a id="sec-3"></a>

## 更多的说明（关于 benchmark 的结果）

首先 ，这里可以看到 ，确实是不同于传统意义上的 “ 某种 ” 神奇的 RL 算法了 。我合理地谨慎期待 ，这种方法具有一定的普适性 。

其次 ，关于 “ 绝大多数 ” benchmarks 。具体可以看图 。Math 的 benchmarks 上面 ，都是无一例外地领先 ，并且无论是我们作为代理分数的 average ，还是各个 benchmarks 各自的分数 ，都是 ≥ 2 倍的加速 （ GSM8K 属于在极度前期就达到了饱和平台期 ，所以这个加速速率是最高的 ，不过这是最简单的 benchmark ，所以不过多描述 。 ） 。不过 ，我们最关心的那些最难的 benchmarks ，也就是我们最希望模型能够提升最快的 —— 恰恰是相对来说提升的更好的 。在 OOD 方面 ，我们在 coding 这个重要的 OOD 层面 ，我们的所有的 coding benchmarks 上面都是 PolyNorm 2M MLP side branch 的方案领先 ，并且这里最难的 benchmark 也就是 Live Code Bench ，实现了显著的超越 2 倍的提升 。我们的 knowledge 层面 ，MMLU-Pro 的提升速度约为 2 倍 ，CEval 被略微反超 。我们的 GPQA-Diamond ，衡量多学科推理能力 ，并且借由 GLM-5.3 Flash 进行打分的 ，我们的 PolyNorm 方案略输一筹 。在衡量指令遵循的 IFEval 上面同理 。

具体来说 ，这样的结果的结论是 ，在 In Domain 上面提升非常明显 ，在 OOD 方面 ，在最直接的 coding 的泛化层面非常优秀 ，在知识与多学科推理的任务上面各有千秋 ，但是也没有明显严重的灾难性遗忘 ，在真实的大规模 RL 中 ，完全可以通过 Mix RL 和 CPT 等手段来弥补 。

然后 ，这样的 side branch 的加法 ，令人想起常规的增加专家的方法 i.e. Upcycling 。不过 ，该工作确与传统的 Upcycling 存在诸多不同 ，因此实现了传统的 Upcycling 达到不到的效果 。我们姑且称之为 **Neo PolyNorm Upcycling** 。具体效果见下 。

第一张图为 Math benchmarks 上面的加速对比 。第二张图是 OOD 。GPQA 评分分开来是因为 GPQA 的评分拿 GLM-5.3-Flash 起的 ，这两天 rollout 刚出来 ，评分还没评完 ，所以先拿别的 。
第三四五张图都是 reward 曲线 ，仅供参考 。不过注意 ，reward 曲线这里可以看到 （ 其实包括 benchmark eval 也是 ） ，PolyNorm 2M MLP side branch 的方案 ，到第一个 epoch 了以后 ，出现了 \*\*\*\*\*\***reward 还有 eval 评分下降的情况** 。目前尚没有足够的算力去调查 ，这个是出于该数据集下模型 over train 了 RL 的必然结果 ，还是要叠加 Asynch RL 的效果 ，还是 PolyNorm 2M MLP side branch 特有的 。如果是特有的 ，**应对起来倒也不难** ，实践中观察到明显的 reward 回转下降 ，则及时早停 ，然后进入 MOPD 和再加化的轮回就行 。具体见 【 RL rollout 成本 · 部署 · 可持续扩展 】 一章节 。

我判断有可能是数据集和问题特有 ，因为早年用 Qwen3-1.7B 进行特定 layer 的 RL 实验时 ，当时包括 RL specific layer of base model 然后一点 side branch 不加 ，也遇到了多 epoch 的情况下 reward 曲线和 eval 曲线来回反复的情况 。当时还是普通的 GRPO ，不是 Flash REINFORCE 。所以推测可能是该情境的固有问题 。

!\[A ep1 高原带 vs FB ep2 同水平带逐档 Math 四件（math500 / gsm8k / olympiadbench / amc23）](add36r\_fullbaser\_math.png)

!\[OOD 七件 + GPQA 逐档（coding / knowledge / GPQA-Diamond）](add36r\_fullbaser\_ood.png)

!\[A add36 vs Full Base 同数据同步数 ep1 直接对照（2-epoch 窗口）](add36\_fb\_ep1\_reward.png)

!\[resume 三键重放 vs 原线 4-panel（重放线加粗，贴合度 sanity）](add36r\_fb\_resume\_reward.png)

!\[combo 三案 rolling-5 reward 四面板主参考图：C 4l add / A add36 / B add36tri 全参联训（紫）vs fullbase（橙）vs ctrl 4 层真 baseline（灰）](combo3\_reward\_rolling5mean\_main.png)

<a id="sec-4"></a>

## 关于早期实验的说明（Best 4 layers）

关于早期实验的说明 （ 这跟已有实验的 implications 和未来实验的方向相关 ） ：  
早期实验基于 B4 layers 进行 。得出的结论很有趣 ：这 side branch ，用 SwiGLU 不行 。不止是原 dimension 的 SwiGLU 不行 ，low-rank bottleneck 的 SwiGLU 也不行 。用 low-rank bottleneck 的 PolyNorm Gated MLP 也不行 。都没有这样的加速效果 。只有 low-rank bottleneck 的 PolyNorm 2M MLP 才可以 。所以说 ，PolyNorm 的表达力可能是一方面 ，low-rank bottleneck 可能也有其原因 ，但是 ，作为一个与 SwiGLU 这样一个 Gated MLP 异质化的构型 ，可能这也是一个重要原因 。这就解释了这篇工作的一个 （ 并不起眼但也略施巧思从而难以被直接想到 ） 的创新点 。也即 ，直接按照传统的 Upcycling 的思路 ，去加一样的形状的专家 ，是收效甚微的 ！所以 ，我们认为 ，更多的自由参数固然拓展了优化问题的空间维度 ，拓展了 loss landscape 的丰富程度 ，但也不是任何形式的丰富都能够达到同样良好的效果的 。这里仍然存在一定的选择性 ！

早期 B4 实验还得出 ：加两个 side branches 略高于加一个 side branch ，但是存在 diminishing marginal return 。同时 ，对于第二个 side branch ，是 add 到主干 ( main + delta ) ，还是 mult 回主干 main × ( 1 + delta  ) ，效果提升差不多 。

最早的 B4 实验是单纯的一个 side branch mult 回主干 ，这个效果有 ，但是没有两倍加速那么显著 。但是有一点 ：这个效果 ，是完全的净增益的 ，没有副作用 ，在包括 GPQA-Diamond 这样的 OOD benchmark 上面都是增益 。这是最早的表达力工程的试点 ，不过 ，由于 additive 的 side branch 的表现太过于优秀 ，所以研究临时转向了这种看起来有点像传统的 Upcycling 的方法 。  
我还没有试过第二个 branch add 回还是 mult 回相比于 OOD benchmarks 的影响 。但是因为这个之前都是 B4 layers 的老黄历 ，我打算之后直接用 Full Base 的基础来做 。

<a id="sec-5"></a>

## 未来的工作

这里主要是只针对一个 dense 的 Qwen3-8B 进行了训练 。我们更多的是要对 MoE 模型进行训练 。因此需要在 MoE 模型上面进行这类方案的探索 。最关键的一个问题是 ：我们要增加多少的 side branches ？我们是增加一个就足够 ？还是说要增加相对于每一层的总激活专家数量对应的专家 ？还是要增加相当于每一层的专家总数量的专家呢 ？我的直觉认为是第二种情况 ，也即激活专家数量的 PolyNorm 2M MLP 就足够 。因为这些增加的 PolyNorm 2M MLP 是常驻的 （ 这也是与常规的 Upcycling 不同的地方 ，那里是要加到专家池里面 ） 。  
之前的 B4 layers 的实验 ，没有在 Full Base 上面做的 ，要搬到 Full Base 上面来做一次 。都有现成的代码 ，这样做最快 。  
另外 ，为了工作的完整性 ，我们也需要进行多个领域的 PolyNorm 2M side RL ，然后把这几个模型给 MOPD 回没有 side branch 的主模型 。这个 MOPD 之后会提 。  
当然 ，还有一些超参数的探索 ，比如说 ，初始化的幅度 ，epsilon 的幅度 ，以及 lr 的幅度 。

<a id="sec-6"></a>

## RL rollout 成本 · 部署 · 可持续扩展

在真实的 RL 中的成本问题 ，和训练完成之后的部署问题 ，以及可持续扩展的问题 ：  
加 side branches ，无疑是增加 forward 成本的 。在当前的大模型 MoE Sparsity 很高的前提下 ，只增加激活的专家总数的专家数量 ，并且 model dim 有 low-rank bottleneck 从而节省参数 ，并且用小矩阵组合成大矩阵和 batched matrix multiplication 等的前提下 ，大量节约每一层的额外显存 ，额外计算负担 ，额外总墙钟时长 。这样 ，rollout 时期 forward 的成本提升在 15% 左右 （ 具体 15% 怎么来的 ，见 额外.txt ），但是换来的是 RL 的极大倍率的加速 ！  
当然 ，我们不希望部署推理的时候也一直带着 ，更不希望每一次继续 RL 都要这样扩一遍 ，面多了加水 ，水涨船高 。我们同时也注意到 ，如果是激活专家数量的 PolyNorm 2M 数量 ，那么 ，我们可能在单一 domain 的 RL 以及其临近 domain 的泛化 （ 比如说这一次的 Math \& coding ） ，能够取得不错效果 ，但是 ，遇上多 domain 齐上 ，我们可能面临增加的专家总容量不足的情况 。  
所以这里我们提出 ，在超大模型的大规模 RL 过程中 ，对各自的 domain 进行分别的 RL ，或者进行 Mix RL 但是把题库分散给多个同一模型的复制品然后各复制品独立地进行 RL ，待训练一段时间 ，再 MOPD 回主模型 （ MOPD 耗时相对整个后训练进程极少 ，效率高 ） 。MOPD 以后 ，再基于新的主模型 ，进行 PolyNorm 2M side branches 的添加 ，如此往复 。直至达到相对满意的效果 。在每一次 MOPD 之后 ，可以考虑对主模型进行一小段时间的单独 RL ，将获得的知识进行整合吸收 。

<a id="sec-7"></a>

## 开放性问题之关于 PolyNorm 架构本身的启发

一个更加开放性的问题 ：针对已经采取 Latent MoE 从而有 low-rank bottleneck 的大模型 ，比如说 Kimi K3 系列 ，我们的 add 策略还有效吗 ？还保持一样吗 ？由于 Kimi K3 仍然是 SwiGLU ，我倾向于有效 。  
不过 ，如果未来出现已知采用 PolyNorm 2M MLP 作为本体的大模型呢 ？这种情况下 ，我们是直接加 ？还是要为了异质化专家的原则去换个加法呢 ？  
或者说 ，这里引出的核心问题是 ：到底为什么 PolyNorm 2M MLP 这么有效 ？明明 PolyNorm Gated MLP 无效 ？是 PolyNorm 的表达力特性要求只有 2M 的 MLP 才能够行吗 ？  
这份工作的本质 ，一方面是基于已有的已知的全天下采用 SwiGLU 作为每一个专家的结构的大模型 ，给出一种切实可行的 RL 加速办法 。另一方面 ，也是在尝试发出这样一个追问 ：或许 PolyNorm 2M MLP 的真实性能 ，其作为相对小的参数量却有着大作用的特性 ，非得到后训练阶段才能够完全地施展出来 ，而不应仅仅盯着架构 paper 的那一点点 perplexity 或者说 loss ？是否有对此实行中等规模的训练实验的可能性和必要性 ？因为注意 ，如果 PolyNorm 2M MLP 以 2M MLP 达到了 Gated MLP 的效果 ，这简直是直截了当的 1/3 的参数量节省 ！同时我们还可以得到 ，在同样的参数量预算下 ，我们可以获得显著地多的 expert number 从而更多的 granularity ！  
当然 ，我们要时刻注意 ，这里的 PolyNorm 2M MLP side branch 是经过了一个 tanh 的限制的 ，所以输出的幅值不是无限的到正负无穷 。虽然 ，PolyNorm 的 Norm 部分也已经做出了一定的 magnitude 限制 。不知道 tanh 本身在大规模训练中是否有必要 。如果证明有必要 ，那么这可能是一个类似于 Kimi 的 SiTU 的设计 ，也即在 SwiGLU 的中间加上 tanh 。SiTU 的最后一个 matrix 后面有没有加我忘了 。可能有吧 ！

<a id="sec-8"></a>

## 【 关于算力的讨论 】

<a id="sec-9"></a>

## 总结

这个方法 ，到最后开发出来 ，我并不会称之为很巧妙 。只是恰好鲜有人往这方面想而已 。不具备什么架构或者算法上面灵机一动令人拍案叫绝的美感 。不过 ，目前看来 ，如果结论可以 safely extrapolate ，那么 ，这个或许会是一个很有用的方法 。能够帮助我们所有的超大模型的大规模 RL 进行大量的提速 。

