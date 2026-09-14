---
title: "DoodleFormer: Creative Sketch Drawing with Transformers"
short: DoodleFormer
source: "[[DoodleFormer]]"
pdf: "[[DoodleFormer.pdf]]"
tags:
  - 译文
  - 全文
status: 全文
---

# DoodleFormer · 全文译文

对应笔记：[[DoodleFormer]]。要点：[[DoodleFormer-要点]]。原文：[[DoodleFormer.pdf]]。项目页：https://ankanbhunia.github.io/doodleformer ；代码：https://github.com/ankanbhunia/doodleformer

> [!abstract] 这篇怎么用
> 按 ECCV 2022 正文与补充材料的节号写全。公式、表号、图号跟 PDF（共 20 页：正文约第 1–14 页，补充材料约第 15–20 页）。
> 不要整段粘英文。正文图与关键表已裁进 `附件/图/DoodleFormer/`。
>
> **核心流程（先框后细）：** 第一阶段 **PL-Net** 根据用户给的初始笔画，预测各身体部件的**包围盒**，相当于先把鸟/兽的「框」摆好；第二阶段 **PS-Net** 在这些框里**画线**，得到 128×128 栅格创意简笔。这和人类涂鸦里「先定整体构图、再填细节」一致。
>
> **和 [[StrokeFusion]] 的对照：** StrokeFusion 全文译文里把 DoodleFormer 列为栅格创意草图代表——两阶段、部件级、输出是**像素简笔**而非矢量笔画序列；StrokeFusion 则用笔画–UDF 编码 + 潜空间扩散，目标是**可编辑矢量**与 FaceX 等数据集上的结构稳定。读 DoodleFormer 时重点看**部件框 + GAT + GMM 多样性**；读 StrokeFusion 时重点看**笔画集合与去噪步**，不要把 PL/PS 两阶段误读成「绘画阶段扩散」。

## 摘要

创意简笔（doodling）是一种表达活动：画的是日常物体**想象化、以前没见过的组合**。创意简笔**图像**生成很难，要在多样性和真实感之间平衡，还要拼出训练里少见的部件布局。

我们提出 **DoodleFormer**：**由粗到细**的两阶段框架，把任务拆成「先定**粗构图**」和「再补**细线条**」。编码器用**图感知 [[Transformer]]（GAT）**，同时利用部件间的**全局动态关系**和**局部静态连接**。粗阶段解码器用**高斯混合模型（GMM）**显式建模各部件位置与尺度的变化，以提高多样性。

实验在 **Creative Birds** 与 **Creative Creatures** 两个创意简笔数据集上进行。定性、定量与用户 study 均优于先前方法。Creative Creatures 上 [[FID]] 比当时最优绝对降低约 **25**。我们还展示了**文本→创意简笔**、**简笔补全**和**户型布局生成**等扩展。代码见上文链接。

## 1 引言

人类擅长用简笔传达抽象想法与情绪。简笔由多笔组成，每笔可看作点列。常规**简笔图像生成**追求可辨认、贴近真实概念的**典型**画法。

![[图/DoodleFormer/fig1.png]]

**创意简笔生成**（Ge et al., ICLR 2021）不同：要画更想象、更少见 everyday 概念的组合（正文 Fig. 1(a)），常以外部给的**随机初始笔画**为条件。自动创意简笔可辅助创作，例如帮用户解读起稿。但比「模仿真实场景」更难。

![[图/DoodleFormer/fig2.png]]

Ge 等提出 **DoodlerGAN**：按部件用 GAN 逐个生成，再与随机输入顺序拼成整图。它没有显式保证部件相对位置，易出现**拓扑错误**（多头、断连等，Fig. 2）和**多样性不足**。

我们模仿自然涂鸦：**先画整体粗结构，再填细节**。粗结构先定各部件**位置与大小**，再画细线，有利于消除拓扑与连通问题，并用 GMM 建模部件框变化以提升多样性。

### 1.1 贡献

- 提出两阶段编码–解码框架 **DoodleFormer**：先建**整体粗构图**，再注入**细部**，得到最终简笔图像。
- **GAT 编码器**：在动态自注意力里融入**静态邻接图**，编码部件间局部结构关系。
- **概率粗解码器**：用 GMM 采样各部件框位置，提升输出多样性（Fig. 2）。
- 在 Creative Birds / Creative Creatures 上全面评估；用户 study 中相对 DoodlerGAN，约 **86%** 认为更像人画、**85%** 初始笔画融合更好、**82%** 更有创意；[[FID]] 在 Creatures / Birds 上分别约 **+25**、**+23** 的绝对增益。

![[图/DoodleFormer/fig8.png]]

- 扩展：**文本条件**、**不完整简笔补全**、**气泡图→户型**（Fig. 1(b)(c)、Fig. 8）。

## 2 相关工作

**标准简笔生成**研究很多：SketchRNN 用序列 VAE 生成矢量简笔；多类生成、可微渲染、注意力、[[强化学习]]等均有探索；也有补全、分类与检索工作。

**创意简笔**较新。DoodlerGAN 基于 StyleGAN2，**每个部件单独训练**一个 GAN，推理时顺序生成，计算开销大，且存在拓扑、连通与多样性问题。本文针对这三点改进。

## 3 方法

### 动机：两个设计目标

**整体部件构图：** DoodlerGAN 按部件生成，但不保证相对位置，拓扑与连通易出错（Fig. 2）。应**显式**建模部件的整体排布。

**细粒度且多样的生成：** 创意简笔在外观上变化大。DoodlerGAN 容易「吃掉」噪声输入，只靠平移部分草图的启发式增多样性，仍不够。我们主张在框架里做**显式概率建模**。

### 整体框架

![[图/DoodleFormer/fig3.png]]

两阶段 **DoodleFormer**（Fig. 3）对应上述两点：

1. **Part Locator（PL-Net）**  
   条件：外部随机初始笔画点列 $\mathcal{C}$（矢量形式）。输出：各身体部件的**包围盒**，即简笔的**粗结构**（ holistic part composition）。含 GAT 编码器 $E_b$、$E_c$ 与**概率粗解码器**（GMM 预测框；位置头 $\mathcal{H}_{xy}$、尺寸头 $\mathcal{H}_{wh}$）。作者称这是创意简笔生成里**首次**用 GAT 块编码器。

2. **Part Sketcher（PS-Net）**  
   输入：PL-Net 预测的框 + $\mathcal{C}$。输出：栅格化高质量简笔 $\bar{\bm{I}}_{im}$。同样用 GAT 编码器 $\bar{E}_b$、$\bar{E}_c$，以及卷积编解码 $\mathcal{R}_E$、$\mathcal{R}_D$ 与 **mask regressor**。

![[图/DoodleFormer/fig4.png]]

Fig. 4 从单阶段基线起，逐步加入两阶段、GAT、GMM，展示「先框后细」与多样性如何改善。

### 3.1 Part Locator Network（PL-Net）

PL-Net 以 $\mathcal{C}$ 为条件，返回描述目标简笔**整体部件布局**的粗结构。编码器用 GAT 块；解码器用 GMM 做框预测以增多样性。

#### 图感知 Transformer 编码器

两个 GAT 编码器：

- $E_b$：编码粗结构 $\mathcal{B}$（各部件框 $(x_t,y_t,w_t,h_t)$）与部件身份嵌入 $\bm{v}_t \in \mathbb{R}^d$ 的拼接。
- $E_c$：编码条件笔画 $\mathcal{C}$（经线性层）。

序列首加 **cls token**；用固定位置编码。cls 输出作为整序列的上下文向量。

#### 图感知 Transformer（GAT）块

![[图/DoodleFormer/fig5.png]]

结构见 Fig. 5(c)：在标准多头自注意力（MHSA）前/中融入**邻接图**上的谱图卷积（Kipf & Welling, ICLR 2017）。

标准注意力（Fig. 5(b)）：

$$\bm{\alpha} = \mathrm{softmax}\left(\frac{\bm{Q}\bm{K}^T}{\sqrt{d}}\right). \tag{1}$$

节点 $i$ 的邻居集 $\mathcal{N}_r(i)$ 由邻接矩阵 $\bm{A}$ 给出。边权：

$$e_{ij} = \bm{W}_b^{T} \mathrm{ReLU}\left(\bm{W}_a \left[\bm{n}_i, \bm{n}_j\right]\right), \quad j \in \mathcal{N}_r(i); \quad e_{ij}=0 \text{ 否则}. \tag{2}$$

谱图卷积（第 $l$ 层）：

$$\bm{n}_i^{(l+1)} = \mathrm{ReLU}\left(\bm{n}_i^{(l)} + \sum_{j \in \mathcal{N}_r(i)} e_{ij}\bm{W}_c\bm{n}_j^{(l)}\right). \tag{3}$$

**直觉：** 邻接图是**静态、稀疏、对称**的局部连接；自注意力是**动态、可稠密、可非对称**的全局关系。二者在注意力权重上融合：对 $j \in \mathcal{N}_r(i)$，

$$\alpha_{ij} = \frac{e_{ij}\exp(\varphi_{ij})}{\sum_{j \in \mathcal{N}_r(i)} e_{ij}\exp(\varphi_{ij})}, \quad \varphi_{ij} \in \frac{\bm{Q}\bm{K}^T}{\sqrt{d}}. \tag{4}$$

$E_c$ 的 cls 输出送入 **Prior-Net**（条件先验）；$E_b$ 与 $E_c$ 的 cls 送入 **Recog-Net**（变分后验）。二者均为 MLP 参数化高斯。训练时从变分分布采样潜变量 $\bm{z}$，再送入概率粗解码器。

**邻接如何建（实现细节见第 4 节）：** 粗结构上**框与框重叠**则连边；初始笔画上**同笔相邻点**连边。

#### 概率粗简笔解码器

两个子模块：**位置预测** $\mathcal{H}_{xy}$（中心 $x_t,y_t$）与**尺寸预测** $\mathcal{H}_{wh}$（$w_t,h_t$）。均含多层 self-attention 与 encoder–decoder attention（key/value 来自 $E_c$ 输出）；部件嵌入 $\bm{v}_t$ 作 query 侧位置编码。各层输出特征 $\bm{f}_t^{xy}$、$\bm{f}_t^{wh}$，再映射为框分布参数。

框参数不用确定性回归，而用 **GMM**（$M$ 个高斯分量，权重 $\pi_k$、参数 $\theta_k$）：

$$p(\bm{b}_t \mid \mathcal{C}, \bm{z}) = \sum_{k=1}^{M} \pi_{k,t}\,\mathcal{N}(\bm{b}_t; \theta_{k,t}), \quad \sum_k \pi_{k,t}=1. \tag{5}$$

对 $P$ 个部件最小化负对数似然：

$$\mathcal{L}_b = -\frac{1}{P}\sum_{t=1}^{P}\log\left(\sum_{k=1}^{M}\pi_{k,t}\mathcal{N}(\bm{b}_t; \theta_{k,t})\right). \tag{6}$$

四元分布拆成两个二元 GMM：$p(\bm{b}_t \mid \mathcal{C}, \bm{z}) = p(x_t,y_t \mid \mathcal{C}, \bm{z})\,p(w_t,h_t \mid x_t,y_t,\mathcal{C}, \bm{z})$。线性层还在 $\bm{f}_t^{xy}$、$\bm{f}_t^{wh}$ 上预测**部件是否存在**，用二元交叉熵 $\mathcal{L}_c$。

#### PL-Net 损失

$$\mathcal{L}_{PL} = \mathcal{L}_{rec} + \lambda_{KL}\mathcal{L}_{KL}, \quad \mathcal{L}_{rec} = \mathcal{L}_b + \mathcal{L}_c. \tag{7}$$

$\mathcal{L}_{KL}$ 拉近 Recog-Net 与 Prior-Net；$\lambda_{KL}=1$（实验设置）。

PL-Net 输出的粗框 $\bar{\mathcal{B}}$ 送入 PS-Net 画最终简笔。

### 3.2 Part Sketcher Network（PS-Net）

PS-Net 含 GAT 编码器 $\bar{E}_b$（框 $\bm{b}_t$）、$\bar{E}_c$（$\mathcal{C}$），特征拼接后经线性层得 $\bm{u}_t$。

$\mathcal{C}$ 光栅化为 $\bm{I}_C$，经卷积编码器 $\mathcal{R}_E$ 得空间特征 $\bm{g}=\mathcal{R}_E(\bm{I}_C)$。解码：

$$\bar{\bm{I}}_{im} = \mathcal{R}_D\left(\bm{g}, \{\bm{u}_t\}_{t=1}^{P}\right). \tag{8}$$

$\mathcal{R}_D$ 以 ResNet 为骨干。为增多样性，在送入解码器前给 $\bm{g}$ 加零均值单位方差噪声。**Mask regressor**（上采样卷积 + sigmoid）为每个框预测辅助 mask，缩放到框尺寸，用于解码器归一化层里的**实例级、结构感知仿射**参数，便于细形状。

训练为标准 GAN：生成器后接三个判别器 $\mathcal{D}_{im}$、$\mathcal{D}_{part}$、$\mathcal{D}_{app}$，对应图像级、部件级、外观对抗损失：

$$\mathcal{L}_{PS} = \mathcal{L}_{im} + \lambda_p \mathcal{L}_{part} + \lambda_a \mathcal{L}_{app}, \quad \lambda_p=\lambda_a=10. \tag{9}$$

GAT 提升真实感；PL-Net 的 GMM 提升多样性。合起来即「**先框身体各 part，再在框内Sketcher 填线**」。

## 4 实验

### 数据集

**Creative Birds**：8067 张鸟创意简笔；**Creative Creatures**：9097 张各类生物。均有**部件标注**和每条样本的**自然语言短语**（Ge et al. 数据集）。

### 实现细节

- 每个 GAT 编码器 **6** 层，8 头；粗解码器里 $\mathcal{H}_{xy}$、$\mathcal{H}_{wh}$ 各 **3** 层 attention。
- 嵌入维 **$d=512$**；矢量简笔做小幅仿射增广，光栅 **128×128**。
- **阶段一：** 用部件标注得到 GT 框，归一化到 $[0,1]$，训 PL-Net。
- **阶段二：** 用光栅图与 GT 框训 PS-Net。两阶段条件输入均为矢量初始笔画。
- batch **32**，学习率 **1e-4**。

### 4.1 定量与定性对比

![[图/DoodleFormer/table1.png]]

Tab. 1 对比 SketchRNN、StyleGAN2、DoodlerGAN 与 DoodleFormer。[[FID]] 与 **GD（generation diversity）** 用 QuickDraw3.8M 上训练的 Inception 特征（与 DoodlerGAN 一致）。

| 方法 | Birds FID↓ | Birds GD↑ | Birds CS↑ | Creatures FID↓ | Creatures GD↑ | Creatures CS↑ | Creatures SDS↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集 | — | 19.40 | 0.45 | — | 18.06 | 0.60 | 1.91 |
| SketchRNN | 82.17 | 17.29 | 0.18 | 54.12 | 16.11 | 0.48 | 1.34 |
| StyleGAN2 | 130.93 | 14.45 | 0.12 | 56.81 | 13.96 | 0.37 | 1.17 |
| DoodlerGAN | 39.95 | 16.33 | 0.69 | 43.94 | 14.57 | 0.55 | 1.45 |
| **DoodleFormer** | **16.45** | **18.33** | 0.55 | **18.71** | **16.89** | 0.56 | **1.78** |

DoodleFormer [[FID]] 更低、GD 更高。**CS（characteristic score）** 衡量生成图被 Inception 判成鸟/生物的比例；只画「很典型」的图 CS 也会高，不一定代表创意。**SDS（semantic diversity score）** 看生物类别语义多样性，更可靠；Creatures 上 DoodleFormer SDS 明显优于 DoodlerGAN。

![[图/DoodleFormer/fig6.png]]

Fig. 6(a) 为随机初始笔画条件下的视觉对比。

### 4.2 用户 study

100 名参与者，每对图一来自 DoodleFormer、一来自对比方法（与 DoodlerGAN 设置类似）。每对回答 Fig. 7 图例中五个问题 (a)–(e)：

(a) 谁更有创意？(b) 谁更像鸟/生物？(c) 谁更像人画的？(d) 谁更好融合初始笔画？(e) 总体更喜欢谁？

![[图/DoodleFormer/fig7.png]]

Creative Birds 上相对 DoodlerGAN，DoodleFormer 在 (a)(c)(d) 约 **82% / 86% / 85%** 胜出；五题上整体优于 DoodlerGAN，且与**人类创意数据集**原图相当。Creatures 上趋势一致（Fig. 7）。Fig. 7 柱状图为五题偏好比例：深蓝对 DoodlerGAN，浅蓝对数据集原图；纵轴为 DoodleFormer 胜出百分比。

### 4.3 消融实验

![[图/DoodleFormer/table2.png]]

Tab. 2（Creative Birds）：

| 设计 | 方法 | FID↓ | GD↑ |
| :--- | :--- | ---: | ---: |
| 单阶段 | baseline*（Transformer 编解码，decoder 顺序出部件再拼） | 46.45 | 16.87 |
| 两阶段 | baseline（无 GAT、无 GMM，PL 用 L1） | 20.14 | 17.05 |
| 两阶段 | + GAT | 16.89 | 17.16 |
| 两阶段 | + GAT + GMM（完整 DoodleFormer） | **16.45** | **18.33** |

两阶段相对 baseline* [[FID]] 约 **+26.3**；GAT 再降约 3.25 但 GD 几乎不变；GMM 使 GD 约 **+1.17**，对应 Fig. 4 中姿态、尺寸、朝向更多样。完整模型相对 baseline* / 两阶段 baseline 的 [[FID]] 绝对增益约 **30.0 / 3.7**。

![[图/DoodleFormer/table3.png]]

Tab. 3（GAT 设计，Birds）：

| 编码器 | FID↓ | GD↑ |
| :--- | ---: | ---: |
| 纯 GCN | 27.45 | 17.23 |
| 标准 Transformer | 20.14 | 17.05 |
| Mesh Graphormer 式串联 | 20.34 | 16.78 |
| **GAT（本文）** | **16.45** | **18.33** |

### 4.4 相关应用

**文本→创意简笔：** 文本作为 $E_c$（及 PS-Net 侧 $\bar{E}_c$）条件；PS-Net 去掉 $\mathcal{R}_E$，cls 特征直接进 $\mathcal{R}_D$。80/20 划分。对比 StackGAN、AttnGAN。Creative Birds 上 [[FID]] / GD：StackGAN **53.1 / 16.7**，AttnGAN **45.2 / 16.5**，DoodleFormer **18.5 / 17.3**（Fig. 6(c)）。

**创意简笔补全：** 不完整输入 → PL-Net 预测缺失部件框 → PS-Net 生成缺失区域再与输入融合。Birds 上 DoodlerGAN vs DoodleFormer：[[FID]] **44.2 vs 18.3**，GD **15.1 vs 17.8**（Fig. 6(b)）。

**户型布局生成：** 只用 PL-Net 架构：气泡图（节点=房间类型，边=邻接）→ $E_c$ 编码 → 概率解码出各房间轴对齐框 → 后处理（提取框边、合并共线段、对齐成闭合多边形）。数据 LIFULL HOME；按房间数分五组（1–3、4–6、…、13+），与 House-GAN 相同设定：训练时排除同组样本，测试每组随机气泡图生成 10 样本。指标 [[FID]] 与 **Compatibility**（输出布局反建气泡图与输入的图编辑距离）。

![[图/DoodleFormer/table4.png]]

Tab. 4 摘要（FID 越低、Compatibility 越低越好）：相对 Ashual、Johnson、House-GAN，本文在各组 [[FID]] 与 Compatibility 多数更优；Fig. 8 为定性对比。

## 5 结论

DoodleFormer 用**由粗到细**两阶段做创意简笔生成：GAT 编码器融合全局动态与局部静态部件关系；概率粗解码器用 GMM 建模各部件变化以保多样性。在两个创意简笔数据集与用户 study 上验证有效，并在文本生成、补全、户型布局上展示可迁移性。

## 附录（补充材料）

### 1 由粗到细的生成过程（Supp. Fig. 1）

补充 Fig. 1 并排展示 PL-Net 的**粗框**与 PS-Net 的**最终简笔**（眼、喙、身体、头、腿、嘴、尾、翅等部件）。强调：**先 holistic 粗结构，再填 fine-details**，与正文 Fig. 3–4 一致。

### 2 评价指标补充说明

与 DoodlerGAN 相同：QuickDraw3.8M Inception 特征。从未见过的初始笔画随机采样，生成 **10 000** 张简笔，缩到 **64×64** 过 Inception，算 [[FID]]、GD、CS、SDS。

### 3 更多定性结果

- Supp. Fig. 2：Birds / Creatures 上相对 DoodlerGAN 与数据集真值的三列对比。
- Supp. Fig. 3：文本条件，每条文本 3 个样本。
- Supp. Fig. 4：补全 challenging 局部输入。
- Supp. Fig. 5：同一初始笔画下 GMM 采样的**多组框布局**，外观/尺寸/朝向/姿态更分散。
- Supp. Fig. 6：与 SketchRNN 的创意鸟简笔对比。

### 4 用户 study 补充

100 人；DoodleFormer 对 DoodlerGAN 与对人类数据集原图的五题偏好与正文 Fig. 7 一致。五题原文：(a) more creative (b) looks like bird/creature (c) likely drawn by human (d) initial strokes integrated (e) like the most overall。

**致谢：** 芬兰 Academy of Finland 项目 USSEE（345791）与 Aalto Science-IT。

---

## 节标题与 PDF 页码（`pdftotext -layout`）

| 节 | 起始页 |
| :--- | ---: |
| 摘要、Fig. 1 | 1 |
| Fig. 2、**1 引言** | 2 |
| **1.1 贡献**、**2 相关工作** | 3 |
| **3 方法**（动机、Fig. 3） | 4 |
| **3.1 PL-Net**、Fig. 4–5 | 5–8 |
| **3.2 PS-Net**、**4 实验**（数据集、实现） | 9 |
| **4.1** 定量对比、Tab. 1、Fig. 6 | 10 |
| **4.2** 用户 study、Fig. 7 | 11 |
| **4.3** 消融、Tab. 2–3 | 12 |
| **4.4** 相关应用、Tab. 4、Fig. 8 | 13 |
| **5 结论** | 14 |
| 补充材料 §1–2 | 15–17 |
| 补充材料 §3–4、参考文献 | 18–20 |
