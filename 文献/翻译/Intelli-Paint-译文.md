---
title: "Intelli-Paint: Towards Developing More Human-Intelligible Painting Agents"
short: Intelli-Paint
source: "[[Intelli-Paint]]"
pdf: "[[Intelli-Paint.pdf]]"
tags:
  - 译文
  - 全文
status: 全文
---

# Intelli-Paint · 全文译文

对应笔记：[[Intelli-Paint]]。要点：[[Intelli-Paint-要点]]。原文：[[Intelli-Paint.pdf]]。

> [!abstract] 这篇怎么用
> 按原论文章节用中文写全。标题编号尽量跟原文一致。
> **读笔序时抓住两条：**（1）**先背景、后前景**——第一层只关心非显著区域，画完场景再在更高层按显著性递减补主体；（2）**先粗后细**——在选中的物体框内，局部注意力窗口随时间缩小，笔触被限制在窗口里。
> 后文 [[ProcessPainter]] 把它当作 Image-to-Painting-Process 的笔画渲染对照之一；本文输出的是**矢量笔触序列 + 可微渲染**，不是 8 帧 RGB 关键帧视频。
> **先背景、后前景**是场景分层作画习惯，不是素描课里的构图—结构—排线—收细四阶段。
> 正文 Fig. 1–4 与表 1–2 已裁在 `附件/图/Intelli-Paint/`；公式、表号跟原文。
> 库内 PDF 是 arXiv v1，首页标题作 *Human-like Painting Agents*。正式 ECCV 2022 题是 *More Human-Intelligible Painting Agents*。译文按库内 PDF 写，不另译会议改稿。

## 摘要

做出设计良好的作品往往很费时，也要求画家有相当功底。为减轻负担，已有大量工作教机器「像人一样画」，再把训练好的智能体当作绘画辅助工具。但不少方法依赖**渐进式网格划分**：智能体把整图划成越来越细的格子，再**并行**往各格子里填笔触。这样得到的作画顺序往往很「机械」，人眼很难顺着看懂。

为此，作者提出 Intelli-Paint：在生成目标画面的同时，让过程更像人画。管线包含三部分：（1）**渐进分层**——智能体先画自然的**背景场景**，再逐层、逐个地把**前景物体**画上去；（2）**顺序笔触引导**——在语义感知的方式下，把注意力在图像不同区域之间切换；（3）**笔触正则（推理阶段）**——在观感几乎不变的前提下，总笔触数大约可少 **60%～80%**。定性与定量实验表明：智能体在成图效率上更好，作画风格也更自然，便于用户用数字方式表达想法。

## 1 引言

绘画是人类表达想法与情绪的重要媒介，但表达是否顺畅，常受画家水平限制；要画细、画好，也往往耗时很多。

一条出路是训练**自主作画智能体**，帮人在更短时间里把想法落到画布上。教机器「怎么画」的工作很多：例如 Huang 等用深度 [[强化学习]] 做无监督笔触分解；Zou 等（Stylized Neural Painting）用梯度下降优化整段轨迹上的笔触参数；Liu 等的 Paint Transformer 把学习作画表述为前向的笔触集合预测。这些方法成图质量不错，但普遍**缺少对图像语义的理解**，且大量依赖**由粗到细的网格、并行填色**，过程呈层次化「自底向上」堆细节，不像人在教你怎么画，也难直接当辅助工具。

Intelli-Paint 从三方面模仿人的习惯：

1. **渐进分层**：不一上来整幅画一起画，而是**先铺背景，再一层层加前景**。
2. **局部顺序注意力**：画家会盯着局部画；本文用一串**由粗到细的局部注意力窗口**，让智能体在区域间切换，而不是一次看全图或固定网格块（对比 Paint Transformer、Optim 等）。
3. **推理时笔触正则**：以往常固定笔触预算，容易重叠、浪费（见原文 Fig. 4），过程也不自然；本文在推理阶段去掉冗余笔触，**总笔数大约可减 60%～80%**，过程仍让人读得懂。

**贡献概括：**

- 渐进分层：像人一样分多层画同一场景。
- 顺序笔触引导：用学到的由粗到细局部窗口，把注意力在不同区域间移动。
- 推理时笔触正则：分解效率大约提升 60%～80%，序列更易被人理解。

## 2 相关工作

**笔画渲染（SBR）。** 用离散笔触、点彩等再现非真实感图像。传统方法常要贪心搜索、启发式优化或用户指定每笔位置与形状；较新方法用 RNN 分解笔触，但依赖**密集人工笔触标注**。Zhao 等用条件 VAE 合成「重画目标图」的延时摄影，需要真实画家延时视频，且分辨率低于本文的高分辨率序列。

**无监督笔触分解。** 近年多用 [[强化学习]] 与对抗训练，或像 Optim 那样对最优传输类损失做梯度下降。Paint Transformer 把问题做成前向笔触集合预测。这些工作成图强，但生成过程仍多依赖**网格划分、并行绘制**，效率与**可解释的过程**都受影响——网格化步骤对人不友好。

## 3 方法

![[图/Intelli-Paint/fig2.png]]

Fig. 2：目标图 $I$ 经 Sequential Planner 先铺背景、再按检测框与局部窗口 $\mathcal{G}_t$、$\mathcal{W}_t$ 补前景；初始笔序再经 Stroke Regularization 去冗余。

Intelli-Paint（原文 Fig. 2）采用**两阶段混合优化**：**Sequential Planner（顺序规划器）** 与 **Stroke Regularizer（笔触正则器）**。

- 第一阶段：Sequential Planner 预测粗但「像人」的初始笔触序列 $s_{\mathrm{init}}$。
- 第二阶段：基于梯度下降的 Stroke Regularizer 去掉冗余笔触、微调参数，得到更高效的分解 $s_{\mathrm{pred}}$。

两阶段写为：

$$
s_{\mathrm{init}} = \mathrm{Planner}(C_{\mathrm{init}}, I_{\mathrm{target}}),
$$

$$
s_{\mathrm{pred}} = \mathrm{StrokeReg}(s_{\mathrm{init}}, I_{\mathrm{target}}),
$$

其中 $C_{\mathrm{init}}$ 为空白画布，$I$ 为目标图。下文分模块说明。

### 3.1 Sequential Planner

#### 3.1.1 强化学习表述

Sequential Planner 建模为深度 [[强化学习]] 智能体，策略 $\pi$ 根据当前状态 $s_t$ 预测向量化笔触参数 $a_t$。状态为

$$
s_t = (C_t,\, I,\, t,\, G_t,\, W_t,\, S_I,\, l),
$$

其中 $C_t$ 为画布；$I$ 为目标图；$S_I$ 为显著性图；$l$ 为当前**作画层**（见 3.1.2）；$G_t$、$W_t$ 分别为**粗**、**细**局部注意力窗口（算法细节见附录 C.1）。

画布用**可微神经渲染器**光栅化 $a_t$，得到 alpha 图 $S_\alpha(a_t)$ 与颜色图 $S_{\mathrm{color}}(a_t)$，更新为：

$$
C_{t+1} = C_t \odot (1 - S_\alpha(a_t)) + S_{\mathrm{color}}(a_t).
$$

（$\odot$ 为逐元素乘，与原文合成方式一致。）

#### 3.1.2 渐进分层（先背景，后前景）

![[图/Intelli-Paint/fig1.png]]

Fig. 1：左列对比 Paint Transformer / 人 / 本文的成画过程；右列示意渐进分层、由粗到细注意力，以及笔触正则后约 $N=250$ 对 $4.8\mathrm{k}$ 笔的成图效率。

人的作画常是**多阶段、多层**的：不是一次涂满，而是**先铺背景层，再把前景物体一层层叠上去**（原文 Fig. 1、Fig. 2）。若只最小化画布 $C_t$ 与目标 $I$ 的像素距离，很难学出这种顺序。

**渐进分层**让画布在多个连续层里演化：

- **第 0 层（背景层）**：目标是在**非显著（背景）区域**上画出可信的场景。显著区域（例如树上的鸟）在这一层也会被涂到，但策略是**借周围背景来画**——例如先围绕枝叶去填，而不是先把鸟当独立主体抠出来画（Fig. 4 下半部鸟例、早期帧可见这一策略）。
- **第 1 层及以后（前景层）**：背景画完后，按**显著性从高到低**，逐个补前景物体。

训练主文取 **$L=2$**（背景 + 前景）；$L>2$ 的掩码扩展见附录 A.3、式 (16)。

整段 episode 按层合成输出：

$$
C_{\mathrm{out}} = \sum_{l=0}^{L-1} \sum_{t=1}^{T/L} C_t^l \odot (1 - S_\alpha(a_t^l)) + S_{\mathrm{color}}(a_t^l),
$$

$T$ 为 episode 长度；$C_0^{l=0}$ 从空画布开始；$C_0^{l=1}$ 用上一层结束时的画布 $C_{T/L}^{l=0}$ 初始化（原文 Fig. 2 的两层示意：先地面、天空等背景，再在第二层加前景物体）。

**层掩码与奖励。** 给定 $C_t$、$I$ 与前景显著性 $S_I$，第 $l$ 层通过掩码 $M_I(l)$ 约束「这一层该优化哪一块」：

$$
M_I(l) = 1 - S_I \odot (1 - l).
$$

（主文 $L=2$ 时，$l=0$ 侧重背景侧，$l=1$ 侧重前景侧；实现上与显著性图配合使用。）

层奖励用条件 Wasserstein GAN 判别器得分 $D(\cdot,\cdot)$（与 Huang 等 RL 作画一致）：

$$
r_{\mathrm{layer}}(l) = D\bigl(I \odot M_I(l),\, C_{t+1} \odot M_I(l)\bigr) - D\bigl(I \odot M_I(l),\, C_t \odot M_I(l)\bigr).
$$

**和 ProcessPainter 的对照（读笔序时用）：** ProcessPainter 从**人类关键帧或合成视频**学「抽象→完整」的 **8 帧 RGB**；Intelli-Paint 不依赖画家录像，而是用 **U²-Net 显著性 + 检测框** 把「先背景后前景、先大物体后小物体」写进 RL 的层与窗口里，输出仍是**笔触参数序列**。

#### 3.1.3 顺序笔触引导（粗窗口 → 细窗口 → 落笔）

人画画时注意力是**局部、顺序**的，不是整图或固定网格同时刷（对比 RL 全局分解、Paint Transformer / Optim 的网格块）。本文在 RL 智能体上增加**由粗到细**的窗口序列 $\{W_0,W_1,\ldots,W_T\}$，分三步：

**（1）前景物体选择（粗窗口 $G_t$）。** 智能体预测粗全局窗口 $G_t$ 的坐标。设图中有 $N$ 个前景物体，检测框为 $B_i \in \mathbb{R}^4$，则

$$
G_t = \sum_{i=0}^{N} \alpha_t^i B_i,\quad \sum_i \alpha_t^i = 1,\;\; \alpha_t^i \geq 0,
$$

$\alpha_t$ 由 RL 在时刻 $t$ 预测。$B_0$ 表示**整幅画布**，用于把焦点切回**背景区域**（画背景层时尤其重要）。

**（2）局部窗口 $W_t$（由粗到细）。** 在 $G_t$ 内，再用 Markov 更新缩小、平移细窗口 $W_t = (x_t^L,y_t^L,w_t^L,h_t^L)$。设归一化时间 $\tilde{t}\in[0,1]$，最小宽高 $(w_{\min},h_{\min})$，RL 预测增量 $\Delta W_t$，则（与原文式 (8)–(11) 一致）：

$$
x_{t+1}^L = x_{t+1}^G + (x_t^L + \Delta x_t)\, w_{t+1}^G,
$$

$$
y_{t+1}^L = y_{t+1}^G + (y_t^L + \Delta y_t)\, h_{t+1}^G,
$$

$$
w_{t+1}^L = \bigl(\max(1-\tilde{t}, w_{\min}) + \Delta w_t\bigr)\, w_{t+1}^G,
$$

$$
h_{t+1}^L = \bigl(\max(1-\tilde{t}, h_{\min}) + \Delta h_t\bigr)\, h_{t+1}^G.
$$

随 $\tilde{t}$ 增大，窗口相对 $G_t$ 缩小，形成**同一物体上由粗到细**的笔触分布。

**（3）笔触参数调整。** 用 $W_t$ 改写预测笔触 $a_t^l$，把落笔**限制在局部窗口内**：

$$
a_t^l \leftarrow \mathrm{ParamAdjustment}(a_t^l, W_t).
$$

实现上假设每笔是二次 Bézier 曲线参数（位置、形状、透明度与 RGB）；把控制点与线宽按窗口左上角与宽高缩放（附录 Algorithm 3）。

**读过程时的顺序：** 先根据 $\alpha_t$ 选「当前画哪块（背景整图 / 哪个框）」→ 再在框内用越来越小的 $W_t$ 扫细节 → 每笔只改窗口内的参数。这比「全图一起最小化 $L_2$」或「固定网格并行」更接近人眼能跟着走的笔序。

#### 3.1.4 与人一致性的惩罚

- **空间惩罚 $r_{\mathrm{spatial}}$：** 相邻粗窗口 $G_t$ 不应乱跳：$r_{\mathrm{spatial}} = -\|G_{t+1}-G_t\|_F$。
- **颜色过渡惩罚 $r_{\mathrm{color}}$：** 相邻时刻笔触颜色应相近：$r_{\mathrm{color}} = -\|(R,G,B)_{t+1}-(R,G,B)_t\|_F$。

Markov 细窗口已保证相邻 $W_t$ 空间接近；$G_t$ 切换仍可能振荡，故加空间项。

### 3.2 笔触正则（Stroke Regularization）

许多系统对简单图与复杂图都用**几乎固定的笔触预算**，导致重叠、冗余（原文 Fig. 4），过程也不自然。

**推理阶段**，在 Sequential Planner 给出 $s_{\mathrm{init}}$ 之后，为每笔引入重要性 $\beta_t^l \in [0,1]$，修改合成：

$$
C_{\mathrm{out}} = \sum_{l,t} C_t^l \odot (1 - \beta_t^l S_\alpha(a_t^l)) + \beta_t^l S_{\mathrm{color}}(a_t^l),
$$

其中 $\beta_t^l = \mathrm{Sign}(x_t^l)$，$x_t^l \sim \mathcal{N}(0, 10^{-3})$ 随机初始化。对 $a_t^l$ 与 $x_t^l$ 做梯度下降，最小化

$$
L_{\mathrm{total}}(a_t^l, x_t^l) = L_2(I, C_{\mathrm{out}}) + \gamma \sum_{l,t} \|\beta_t^l\|_1,
$$

$\gamma$ 平衡重建与**少用笔触**；$\partial \beta_t^l / \partial x_t^l$ 经 sigmoid 回传。优化后得到精简序列 $s_{\mathrm{pred}}$（附录 Algorithm 4）。主文 Fig. 1 右侧示例：相对 Paint Transformer，笔触数可少到约 **1/20**（同例约 4800 对 250），成图仍够细。

## 4 实现细节

**神经渲染器。** 主要采用 Huang 等 PixelShuffleNet 可微渲染；相比 Paint Transformer、Optim 的不透明笔触，采用 Huang 等更自然混合的笔触表示，更接近人画习惯。

**分层训练。** 每层策略在**上一层输出的画布**上继续训练；训练时两层策略**分批连续**训练以省算力；训练固定 $L=2$，推理可通过改显著性排序扩展到 $L>2$（附录 A.3）。

**显著性与框。** 显著性用预训练 **U²-Net**；边界框取 **YOLOv5** 与显著性外接框的**并集**。顺序引导依赖这些预训练模块，检测失败时该块会更多落在背景层处理（见 5.3）。

**整体训练。** Sequential Planner 用 **model-based DDPG**（同 Huang 等设定）。总奖励：

$$
r_{\mathrm{overall}}(l) = r_{\mathrm{layer}}(l) + \mu r_{\mathrm{gbp}} + \eta r_{\mathrm{spatial}} + \gamma r_{\mathrm{color}},
$$

$r_{\mathrm{gbp}}$ 来自 Singh 等 Semantic-RL 的 guided backprop 聚焦奖励；$\mu=10$；$\eta,\gamma$ 从 $10^{-4}$ **线性增到** $0.1$。共训练 **500 万** 次迭代，batch **128**。

## 5 与最先进方法对比

与 RL [12]、Semantic-RL [26]、Optim [36]、Paint Transformer [21] 对比：先讲**有限笔触下的成图效率**（5.1），再讲**过程像不像人**（5.2），最后讨论假设与鲁棒性（5.3）。

### 5.1 作画效率（成图质量 vs 笔触预算）

![[图/Intelli-Paint/fig3.png]]

Fig. 3：各方法在约 300 笔预算下的成图对比（车、湖边小屋、双鹅、山村四例）；本文在细结构处更完整。

**定性（Fig. 3）。** 每幅约 **300** 笔（Paint Transformer、Optim 因网格形式分别约 **360**、**330** 笔）。本文在车、小屋、双鹅等**细结构**上更完整；对手常缺乏「把有限笔触分到该分的地方」的机制。Paint Transformer 推理快，但**笔触很少时**往往不如 Optim。

![[图/Intelli-Paint/table1.png]]

表 1：Stanford Cars 与 CUB-Birds 上约 300 笔/幅的 $L_{\mathrm{pixel}}$、$L_{\mathrm{pcpt}}$；本文两数据集均最低。

**定量（表 1）。** 指标为与目标图的像素 $L_2$ 距离 $L_{\mathrm{pixel}}$ 与感知损失 $L_{\mathrm{pcpt}}$ [15]（约 300 笔/幅）。

| 方法 | Stanford Cars $L_{\mathrm{pixel}}$ | $L_{\mathrm{pcpt}}$ | CUB-Birds $L_{\mathrm{pixel}}$ | $L_{\mathrm{pcpt}}$ |
| :--- | ---: | ---: | ---: | ---: |
| RL [12] | 97.85 | 0.67 | 96.14 | 0.76 |
| Semantic-RL [26] | 79.98 | 0.55 | 68.46 | 0.55 |
| Optim [36] | 76.52 | 0.54 | 67.90 | 0.53 |
| Paint Transformer [21] | 87.78 | 0.57 | 82.43 | 0.56 |
| Ours | 56.92 | 0.44 | 50.94 | 0.45 |

在 **CUB-Birds** 上，$L_{\mathrm{pixel}}$ 相对 RL、Semantic-RL、Optim、Paint Transformer 分别约降 **47.1%**、**25.6%**、**24.9%**、**38.2%**（原文相对降幅表述；Paint Transformer 一行原文误标 [20]，实为 [21]）。

### 5.2 与人类作画风格的相似度（过程能否跟着读）

![[图/Intelli-Paint/fig4.png]]

Fig. 4：各方法用不同总笔数（框内数字）使重建损失大致相当，取约 10%、40%、60%、100% 进度帧；上半部为车景、下半部为鸟例。本文先铺背景再以由粗到细补前景，更接近人；对照多呈网格并行或低层特征乱铺。PDF 里 Fig. 4 是**整页宽的大对比图**，不是单行细条。

**定性（Fig. 4）。** 各方法用**不同笔触数**，使像素重建损失大致相当；取 episode 约 **10%、40%、60%、100%** 的帧对比。本文在**分层演化**与**局部注意力**上更接近真人：例如先 sky、山、河、地面等**背景**，再以由粗到细方式画前景**车**；Paint Transformer、Optim、RL 等更早按低层特征（如车身的红、鸟头）在整图乱铺，呈**自底向上**网格感；Semantic-RL **前景背景并行**，缺少「先场景后主体」的语义节奏。

![[图/Intelli-Paint/table2.png]]

表 2：MTurk 两两比较中，受试者偏好本文成画过程的比例（对 RL、Semantic-RL、Optim、Paint Transformer 四次研究）。

**定量（表 2，用户研究）。** 50 名 Amazon Mechanical Turk 受试者，100 段随机序列（CelebA [18] 与 CUB-Birds [30]）；每题两段 GIF（各 10 秒），选「更像人画」的一条。下表为**受试者偏好 Intelli-Paint 的样本占比**：

| 对比对手 | 偏好 Intelli-Paint 的比例 |
| :--- | ---: |
| RL [12] | 83.11% |
| Semantic-RL [26] | 69.09% |
| Optim [36] | 75.41% |
| Paint Transformer [21] | 86.50% |

四次两两比较里，多数情况下用户更选 Intelli-Paint。界面见附录 Fig. 8；过滤规则见附录 D。

### 5.3 局限与假设的鲁棒性

1. **只模仿一般习惯**，未对齐每位画家的细粒度风格；但表 2 表明多数人仍觉得更「像人」。
2. **依赖 U²-Net 显著性**；漏检某显著物体时，该区域更可能在**背景层**被填掉，最终成图质量仍可接受。
3. **需要真实图像上的自监督 RL 训练**；Paint Transformer 可在人工数据上自训练，Optim 可不训练。训练完成后跨域泛化尚可：**Fig. 3、4 全部用只在 CUB-Birds 上训练的模型**生成。

## 6 结论

自主作画系统的价值，不应只看**最终画布**，还要看**过程能否被真人读懂**。Intelli-Paint 用**渐进分层**让画布演化更像人（**先背景、后前景**），用**由粗到细的局部注意力窗口**组织笔序，并用**有限笔触**画细场景。相对先前 SOTA，成图更高效，风格也更贴近用户熟悉的作画顺序——这也是 [[ProcessPainter]] 等在反推过程任务里把它列为笔画渲染对照的原因。

## 附录

### A 渐进分层分析

#### A.1 艺术化演化（人脸，Fig. 5）

相对一味压像素距离，人脸域上会出现更像人的阶段：**粗略轮廓（a）→ 带明暗的中间稿（b）→ 再补眼鼻唇等细节（c）**。不是一上来用白笔涂嘴区等低层 trick。

#### A.2 应用：前景移除（Fig. 6）

分层机制也可在**非标准 mask** 下做前景移除：在前景区域用「学习作画」式 inpainting，中间背景比 Yu 等 GAN inpainting 更自然。

#### A.3 扩展到 $L>2$ 层（Fig. 7）

用排序显著性图 $S_I[r_k]$，层掩码推广为：

$$
M_I(l) = 1 - \bigvee_{k=1}^{L-l} S_I[r_k].
$$

Fig. 7 三层示例：第一层背景 → 第二层最显著（白车）→ 第三层较不显著（牛）。

### B 消融

三模块：**渐进分层**、**顺序笔触引导**、**笔触正则**。因效果主要体现在过程可视化，作者把各模块消融动画放在[项目主页](https://1jsingh.github.io/intelli-paint)。

### C 算法细节

- **C.1** 顺序引导：Algorithm 1 选 $G_t$，Algorithm 2 Markov 更新 $W_t$，Algorithm 3 窗口内改笔触参数（与正文 3.1.3 一致）。
- **C.2** 笔触正则：Algorithm 4。
- **C.3** 推理总流程：Algorithm 5（逐层、逐步：策略 → 选 $G_t$ → 更新 $W_t$ → 调整 $a_t$ → 渲染更新 $C_t$ → 最后 `StrokeReg`）。

### D 用户研究细节

MTurk；HIT 接受率 >90%；学历本科及以上；**重复题**检验认真度，不一致答卷丢弃（Fig. 8 界面）。

### E 推理时间（表 3）

单卡 **Nvidia V100**，用各作者官方实现测得：

| 方法 | 推理时间 (s) |
| :--- | ---: |
| RL [12] | 2.317 |
| Semantic-RL [26] | 2.631 |
| Optim [36] | 416.7 |
| Paint Transformer [21] | 1.154 |
| Ours（含 StrokeReg） | 72.21 |
| Ours（不含 StrokeReg） | 0.948 |

**Ours（w/o StrokeReg）** 只跑 Sequential Planner：因总笔触常远少于 Paint Transformer / RL / Semantic-RL，反而**更快**。含 StrokeReg 时虽也要梯度优化，但初值来自 Planner，比 Optim 从随机或启发式初值起步**快很多**。
