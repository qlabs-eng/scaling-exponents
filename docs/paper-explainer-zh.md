# How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents

这是一份面向组内 pre 的论文讲解稿。目标是让第一次接触这篇论文的人听懂三件事

1. 论文把什么问题放进了同一套实验框架
2. 论文的数据支持到哪一步
3. 哪些地方还不能下结论，以及我们怎样补实验

## 论文信息

| 项目 | 内容 |
| --- | --- |
| 题目 | How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents |
| 作者 | Zixi Chen、Akshay Vegesna、Samip Dahal、Andrew Gordon Wilson |
| 版本 | arXiv v2，2026 年 9 月 17 日 |
| 论文 | [arXiv:2609.19107](https://arxiv.org/abs/2609.19107) |
| 代码 | [qlabs-eng/scaling-exponents](https://github.com/qlabs-eng/scaling-exponents)，本文分析对应提交 `47ae5d7` |
| 讨论重点 | loop、模型增长、计算深度、边界算子，以及中途从 1 次 loop 增长到 2 次 loop 的补实验 |

## 开场怎样讲

可以先给听众一个具体问题

> 如果训练预算固定，我们把更多计算放在后期的深度上，模型的 loss 随计算量下降的速度会不会改变？

传统 scaling law 通常把架构当作固定条件。计算量增加以后，模型参数和训练 token 一起增加，验证 loss 近似服从幂律

\[
L(C)=E+A\left(\frac{C}{C_0}\right)^{-\gamma}。
\]

这里的 \(C\) 是训练 FLOPs，\(E\) 是 loss floor，\(A\) 是常数，\(\gamma\) 是 scaling exponent。常数变小，表示整条曲线下移。\(\gamma\) 变大，表示计算量增加以后曲线下降得更快，规模越大时优势越明显。

论文的核心主张是，架构干预可以改变 \(\gamma\)。作者用 looped transformer 做实验，因为 loop 允许重复执行一段共享权重的 core。训练过程中增加 loop 次数，就能增加有效计算深度，同时不增加存储参数量。

这句话要带上限定条件。论文比较的是每种架构各自调过的 compute-optimal scaling recipe，结果主要来自 FineWeb 的单 epoch 训练和一个固定数据池的多 epoch 训练。它没有证明所有任务、所有数据和所有残差结构都会得到相同收益。

## 先把网络结构看懂

论文把模型拆成三段

- **Prelude，前段 P** 负责把输入 token 变成 core 能处理的表示 \(e=P(s)\)
- **Core，中段 R** 是可以重复执行的主体
- **Coda，后段 D** 把 core 的状态映射到输出分布

一次前向可以写成

\[
h_0=e,\qquad h_k=R_{\theta_k,\phi}(h_{k-1},e),\quad k=1,\ldots,K。
\]

共享 loop 时，所有遍次使用同一组 \(\theta\)。每一遍仍然要重新计算 attention、MLP 和反向传播。共享权重节省的是参数存储，不能把重复执行的 FLOPs 当成免费计算。

```mermaid title="论文的 prelude-core-coda 结构"
flowchart LR
  X[输入 token s] --> P[Prelude P]
  P --> E[表示 e]
  E --> R1[Core R 第 1 遍]
  R1 --> R2[Core R 第 2 遍]
  R2 --> RK[Core R 第 K 遍]
  RK --> D[Coda D]
  D --> Y[输出分布]
  E -. 边界注入 .-> R2
```

图中的边界算子会对残差状态做归一化，并把前段表示重新注入后续计算。为了分析 loop 本身，我们需要把带边界算子的结果和普通 pre-norm 残差结果分开。

### 图 1 中的六个模型

| 模型 | core 执行次数 | 权重关系 | 训练中是否增长 | 论文用来隔离的因素 |
| --- | ---: | --- | --- | --- |
| Vanilla | 1 | 每层独立 | 否 | 标准残差基线 |
| Operator-1 | 1 | 每层独立 | 否 | 边界算子 |
| Loop-2 | 2 | 两遍共享 core | 否 | 固定 loop 和参数共享 |
| Untied-2 | 2 | 两遍各有 core | 否 | 固定深度下的权重共享 |
| Loop-Grow | 2→4 | 四遍共享 core | 是 | 共享权重的模型增长 |
| Untied-Grow | 2→4 | 增长时复制并独立训练 | 是 | 独立权重的模型增长 |

以论文图 1 的示意模型为例，Prelude、Core、Coda 各有两层。Vanilla 的执行深度是 6。Loop-2 的存储层数仍是 6，但 core 执行两遍，执行深度变成 8。正式 d8 配置使用 2/3/3 的分段，因此 K=1、2、4 的执行深度分别是 8、11、17。

模型增长发生在训练过程中。Loop-Grow 在切换前执行两遍，切换后执行四遍。Untied-Grow 会复制已有 core，复制出来的块拥有独立参数。作者把新启用的遍次看作增加计算深度的方式，不把它当作普通的 checkpoint 续训技巧。

## 论文的主要结果

### 单 epoch 结果

作者在不同模型规模上分别调节模型大小、token 数和训练超参数，然后用最终验证 loss 对总 FLOPs 做拟合。图 3 给出的拟合指数约为

| 架构 | 拟合的 \(\gamma\) |
| --- | ---: |
| Vanilla | 0.111 |
| Operator-1 | 0.114 |
| Loop-2 | 0.114 |
| Untied-2 | 0.114 |
| Loop-Grow | 0.116 |
| Untied-Grow | 0.117 |

在 Vanilla 预算约 \(10^{20}\) FLOPs 对应的 loss 上，论文报告 Loop-Grow 的 compute multiplier 约为 1.36，Untied-Grow 约为 1.55。这个数字的含义是，达到 Vanilla 在该预算下的 loss，Vanilla 需要的计算约是 Loop-Grow 的 1.36 倍，约是 Untied-Grow 的 1.55 倍。由于 multiplier 随规模增加，作者据此认为增长架构带来了 exponent 层面的收益。

图 3 还显示 Operator-1 的 multiplier 约为 1.25。它的参数量和 FLOPs 与 Vanilla 对齐，差异来自边界算子。这个结果说明深度相关的信号传播设计可能影响 scaling curve，loop 并不是唯一因素。

### 多 epoch 结果

作者固定 100M 个独特 FineWeb token，训练 10 个 epoch。这个设置模拟高质量数据有限、同一 token 会被重复看到的情况。

在固定 weight decay 的比较中，最优 loop 数会随计算预算增加，从约 1.4 增到约 6.7。多 loop 给模型增加计算深度，同时保持参数量不变，因而能在重复数据上降低过拟合。论文在标记点报告了约 2.2 倍的 compute efficiency gain。

这里必须记录数据口径。单 epoch 结果比较的是新鲜 token 和 compute-optimal 的模型与数据配比。多 epoch 结果比较的是固定数据池和不同曝光次数。一个实验中的 loop 优势不能直接搬到另一个实验。

### 作者给出的解释

论文用 computational depth 解释结果。作者认为，固定计算预算下，有用的深度决定了模型能组合多少步计算。普通深模型会遇到 curse of depth，后面的残差更新相对于整体 residual stream 变小，部分层对最终预测的影响减弱。边界归一化、早期表示注入和 loop 都可能改善信号传播。

这个解释目前更像统一视角。实验能证明不同架构的 loss 曲线不同，尚不能证明有效深度就是唯一因果机制。KL effective depth 是论文使用的代理指标，适合辅助分析，不能取代 loss、梯度和实际计算成本。

## 图和数字怎样读

### 图 2 和图 3

图 2 左侧把单 epoch scaling law 和多 epoch 的 loop 曲线放在一起。图 3 的横轴是完整训练的总 FLOPs，每个点通常对应一次从头训练结束后的结果。它不等于一次训练过程中的实时曲线。

如果要判断模型增长瞬间发生了什么，应当查看训练中的 validation loss、梯度范数和 hidden-state RMS，并把增长点标到曲线上。仅凭最终点无法判断切换是否造成 loss 跳变，也无法判断模型用了多少 FLOPs 恢复到原有趋势。

### compute multiplier

假设架构 A 和 B 的拟合曲线分别为 \(L_A(C)\) 和 \(L_B(C)\)。在 A 达到某个 loss 时，B 需要的计算量记为 \(\hat C_B\)。论文报告的 multiplier 可以写成

\[
\pi(C)=\frac{C_{\mathrm{Vanilla}}(L)}{C_{\mathrm{variant}}(L)}。
\]

\(\pi>1\) 表示 variant 达到相同 loss 所需计算更少。比较时必须在两条曲线都实际覆盖的 loss 区间内插值。把远离观测区间的 power law 外推到很大的预算，结论会强烈依赖拟合 floor 和指数。

### 参数量、深度和计算量

共享 core 只减少存储参数。执行两遍 core 仍要付两遍 attention 和 MLP 的 FLOPs，激活保存和通信也会改变真实 wall-clock 时间。比较时至少要同时记录

- 理论训练 FLOPs，说明是否包含 optimizer 和重计算
- 实测 GPU 时间，包含增长时的编译和 kernel 变化
- 唯一 token 数和 token 曝光次数
- 峰值显存、激活内存和吞吐
- 推理时的延迟和显存

## 论文没有回答的地方

下面这些问题适合作为批判性讨论。每一点都要区分论文已经展示的事实和我们提出的解释。

### 1→2 的中途增长没有系统对照

论文主实验采用 2→4。附录图 13 搜索了增长时机，Loop-Grow 的简化方案在约 80% 的 token 进度启用新增遍次。这个结果不能直接当作 1→2 的最优时点。

作者选择从 2 开始有一个可理解的实验理由。新鲜数据上的固定次数搜索把最优值放在 1 和 2 之间，因此固定模型采用 2，再从 2 比较增长到 4、6、8。论文没有系统报告先用 K=1、训练中途改为 K=2 的路径。这里可以提出一个明确假设

> 早期训练使用 K=1，可以把计算用在更多新 token 上。后期使用 K=2，可以用更深的计算处理更复杂的模式。

这个假设可能成立，也可能因为输入分布突然改变、优化器动量失配和适应时间不足而失败。只有配对实验能判断。

### 增长收益和架构收益没有完全拆开

Loop-Grow 同时包含共享权重、增长时机、边界算子以及论文为各架构单独调出的超参数。Loop-Grow 和 Vanilla 的最终差异不能全部归因于 loop。

Deep Vanilla Grow 的指数约为 0.114，高于 Deep Vanilla 的约 0.111，这个对照支持增长本身有价值。它使用独立权重复制，仍然缺少普通残差下的共享 1→2 对照。后续实验应固定残差形式，分别改变 loop 次数、是否共享和增长时机。

### scaling law 的统计不确定性需要更完整报告

指数差异只有千分位量级。不同模型规模的训练噪声、loss floor、数据顺序和超参数拟合都会影响斜率。三个种子可以作为最低复核，接近噪声的差异需要更多种子。拟合时应报告参数协方差、bootstrap 区间、拟合区间和去掉单个点后的敏感性。

比较 compute multiplier 时，目标 loss 必须落在两种架构的观测交集内。目标未达到就报告未达到，避免把外推结果写成实测结果。

### 数据和任务覆盖仍然有限

单 epoch 主实验使用 FineWeb。多 epoch 只固定 100M 个独特 token 并重复训练。还需要在 FineWeb-Edu、代码和更长上下文上复核，观察 loop 对记忆、复制和长程依赖的影响。CORE 分数和 GPT-3 13B 的比较提供了一个外部锚点，但它不能替代不同任务的系统评测。

### 训练成本和部署成本可能指向不同结论

共享权重降低参数存储，重复执行 core 却会增加激活、反向和推理延迟。论文图 15 报告 wall-clock multiplier，但实际部署还涉及 kernel 融合、动态切换和编译开销。若 1→2 在理论 FLOPs 上有优势，仍需测量真实 GPU 时间，确认节省没有被低利用率抵消。

### 机制证据还不够直接

论文用 KL effective depth 解释后层是否仍影响预测，也讨论边界算子和残差流的信号传播。下一步应保存切换前后每一遍的 hidden-state RMS、梯度范数、logit lens 和 attention 统计，并在增长后同时评估 K=1 与 K=2。如果 K=2 的第二遍只带来很小的增益，深度解释需要重新审视。

### mHC 和 Attention Residuals 尚未与 loop 做统一实验

mHC 通过多条残差流和近似双随机的流间混合改善信号传播。Attention Residuals 让当前层从历史表示中选择信息。它们和 loop 可以组合，但跨遍次是否保存历史表示会改变计算图和 FLOPs。论文没有提供这两类残差结构与中途增长的组合实验，不能预先断言 mHC 一定会放大 loop 收益。

## 我们建议先做的 1→2 实验

### 实验问题

首轮只回答一个问题

> 在普通 pre-norm 残差、相同累计 FLOPs 和相同数据顺序下，训练中途从 K=1 切换到 K=2，是否比全程 K=1 或全程 K=2 更低 loss？

关闭额外边界归一化和前段注入，让 K=1 与 Vanilla 的前向尽量一致。另跑一组论文原配方的 2→4，检查代码和数据管道是否与论文对齐，两组结果分开报告。

### 推荐模型和预算

第一轮使用 d8、宽度 1024、8 个 attention heads，Prelude、Core、Coda 按 2/3/3 划分。第 3 至第 5 层作为连续 core，重复时把这 3 层整体执行第二遍。

令 \(F_1\) 为 K=1 的每 token 训练 FLOPs，\(F_2\) 为 K=2 的每 token 训练 FLOPs，令 \(r=F_2/F_1\)。前 \(\tau\) 比例 token 用 K=1，后面用 K=2，则

\[
C_{\mathrm{late}}=T[\tau F_1+(1-\tau)F_2],\qquad C_{\mathrm{always}}=TF_2。
\]

同 token 数下，相对全程 K=2 的节省比例是

\[
\tau\left(1-\frac{1}{r}\right)。
\]

采用公开代码的 d8 FLOPs 估算，K=1 约为 \(1.026\times10^9\) FLOPs/token，K=2 约为 \(1.295\times10^9\) FLOPs/token，\(r\approx1.262\)。首轮预算 \(C_0\) 取 K=1 模型训练 2048 步的估算值，约为 \(1.102\times10^{18}\) FLOPs。下表按整步向下取整，gate 的逐元素开销尚未计入，正式运行以 profiler 和累计计数为准。

| 方案 | K=1 更新步数 | K=2 更新步数 | 总 token 约数 | K=2 启用时的 token 进度 |
| --- | ---: | ---: | ---: | ---: |
| 全程 K=1 | 2048 | 0 | 1.074B | 不启用 |
| 全程 K=2 | 0 | 1622 | 0.850B | 0% |
| 25% \(C_0\) 启用 | 512 | 1217 | 0.906B | 29.6% |
| 50% \(C_0\) 启用 | 1024 | 811 | 0.962B | 55.8% |
| 70% \(C_0\) 启用 | 1433 | 487 | 1.007B | 74.6% |
| 85% \(C_0\) 启用 | 1740 | 244 | 1.040B | 87.7% |

这里的切换点按累计 FLOPs 定义，不能直接把 50% FLOPs 写成 50% token。K=2 每 token 更贵，所以 50% FLOPs 对应约 55.8% token 进度。

### 初始化和切换

直接切换时复用 P、R、D 和优化器状态。共享的 R 不新增参数。为了观察切换本身的影响，建议比较三种方式

1. 直接把第二遍打开
2. 用 1% \(C_0\) 线性门控
3. 用 5% \(C_0\) 线性门控

门控可以写成

\[
h_2=h_1+g\,[R_\theta(h_1)-h_1],\qquad 0\leq g\leq1。
\]

切换瞬间令 \(g=0\)，随后随累计 FLOPs 线性增到 1。这样第二遍刚出现时输出与旧模型连续，完成后得到标准 K=2。只要第二遍实际执行，就按完整 K=2 的 FLOPs 记账，即使门控值还很小。

不建议把共享 R 的 attention 或 MLP 输出投影清零，因为这会同时破坏第一遍已经学到的函数。独立权重的对照可以把 R 复制成 R 和 R′，保留旧参数动量，新参数状态从零开始。另加一个新增分支输出为零的恒等初始化，用来区分复制初始化和函数保持初始化。

### 学习率和位置对照

主对照让所有方案沿累计 FLOPs 使用同一套学习率进度，不在切换点重启学习率。若测试论文中的 40 步 re-warmup，需要加入一个只改变学习率、不改变 loop 的 K=1 对照，避免把学习率收益误判为深度收益。re-warmup 应恢复到当前调度的学习率，不能回到训练初始峰值。

时机筛选结束后，固定最佳时机，再比较三种 loop 位置

- 前段第 1 至第 3 层
- 中段第 3 至第 5 层
- 后段第 6 至第 8 层

首选中段是一个设计假设，目的是保留稳定的输入处理和输出处理区域。需要保持重复层数和 FLOPs 一致。重复必须接收前一遍的 hidden state，不能每遍重置为前段输出。RoPE 的 token 位置保持原序列位置，反向传播覆盖全部遍次。

### 首轮验收指标

每次运行保存验证 loss 对累计 FLOPs、token 数和实际 GPU 时间的曲线，并标出切换点。切换前、刚切换和门控完成后使用同一份验证集，记录

- loss 跳变和恢复所需 FLOPs
- 梯度范数、hidden-state RMS 和每遍输出差异
- K=1 与 K=2 的额外验证 loss
- 理论 FLOPs、GPU 时间、吞吐和峰值显存
- 唯一 token 数、token 曝光次数和推理延迟

每个方案先跑 3 个配对种子。筛出最佳时机后，再与全程 K=1、全程 K=2 做相同 token 数的配对训练。这样可以分别回答固定 FLOPs 和固定 token 数两个问题。

如果首轮结果显示延迟 1→2 有优势，再在 \(C_0\)、\(3C_0\)、\(10C_0\) 上重新搜索 d6、d8、d10、d12 等模型大小。每个预算都给 Vanilla、Loop-2 和延迟 1→2 相同的调参机会，才能比较各自最优曲线。

固定 FLOPs 时报告

\[
\Delta L(C)=L_{\mathrm{late}}(C)-L_{\mathrm{Vanilla}}(C)。
\]

固定目标 loss 时，在两条实测曲线的交集内插值计算

\[
\frac{C_{\mathrm{Vanilla}}(L^*)}{C_{\mathrm{late}}(L^*)}。
\]

比值大于 1 表示延迟增长达到相同 loss 所需计算更少。若目标 loss 没有被某一方法达到，就报告未达到。

## 和 mHC、Attention Residuals 怎样接起来

这几类改动作用在不同位置，需要分开控制

| 方法 | 改变了什么 | 和 loop 组合时要控制什么 |
| --- | --- | --- |
| 普通残差 | 每层用残差分支累加输出 | 作为第一轮参照 |
| mHC | 使用多条残差流，并对流间混合施加近似双随机约束 | 流是否跨遍次传递，以及混合的显存和通信开销 |
| Attention Residuals | 从历史层表示中选择要累加的内容 | 历史是否包含前一次 loop 的表示 |
| 中途增加 loop | 改变不同训练阶段的计算深度和算力分配 | 权重共享、执行次数、启用时机和切换开销 |

mHC 的双随机约束只作用于流间混合矩阵，不能直接推出整个 Transformer 都是恒等映射。Attention Residuals 如果跨遍次保存历史表示，计算量也不再是简单的 K 倍。

第二轮可以做普通残差、mHC、Block AttnRes 三种架构，每种架构比较全程 K=1、全程 K=2、最佳延迟 1→2，共九种组合。每种残差结构先调好自己的 K=1，再给 loop 方案相同调参预算。所有额外混合、历史读取和 kernel 开销都计入 FLOPs 与 GPU 时间。

## pre 的建议顺序

| 时间 | 要讲的内容 | 听众应该记住什么 |
| ---: | --- | --- |
| 2 分钟 | 问题和 scaling exponent | 论文问的是架构能否改变 loss 曲线的斜率 |
| 4 分钟 | Prelude、Core、Coda 和六种模型 | loop 用共享权重增加执行深度，growth 改变训练中的深度 |
| 4 分钟 | 图 2、图 3 和多 epoch 结果 | Loop-Grow、Untied-Grow 和 Operator-1 都有论文报告的收益 |
| 3 分钟 | 计算深度解释 | 这是统一解释，仍需更多直接机制证据 |
| 4 分钟 | 局限和 1→2 补实验 | 2→4 已有结果，1→2 的时机、切换和公平算力比较仍待验证 |
| 2 分钟 | mHC、AttnRes 和后续路线 | 先固定普通残差做因果实验，再扩展残差结构 |

最后留下三句话就够了

- 论文最有价值的结果是，模型增长和边界算子可能改变 scaling exponent，收益随计算规模扩大
- loop 的收益来自计算深度、权重共享和训练数据条件的共同作用，不能只看参数量
- 我们下一步先用普通残差做 1→2 的中途增长实验，在固定 FLOPs、固定 token 和实测 GPU 时间三个口径下分别验收

## 参考资料

1. Chen et al.，[How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents](https://arxiv.org/html/2609.19107v2)，重点查看图 1、图 2、图 3、图 5、图 13，章节 3、4.1、4.2、5、6 和附录 A.2.3、A.4.3。
2. [qlabs-eng/scaling-exponents](https://github.com/qlabs-eng/scaling-exponents/tree/47ae5d70d30513abac66be0bab1068053c1275ac)，重点查看 `train.py`、`models/transformer.py`、`ladder_scripts/loop_grow.sh`。
3. [mHC](https://arxiv.org/html/2512.24880v2)，用于残差流混合和双随机约束的背景。
4. [Attention Residuals](https://arxiv.org/abs/2603.15031)，用于历史层表示选择的背景。
