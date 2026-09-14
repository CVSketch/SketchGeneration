---
title: "VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation"
short: VQ-SGen
source: "[[VQ-SGen]]"
pdf: "[[VQ-SGen.pdf]]"
tags:
  - 译文
  - 全文
status: 全文
---

# VQ-SGen · 全文译文

对应笔记：[[VQ-SGen]]。要点：[[VQ-SGen-要点]]。原文：[[VQ-SGen.pdf]]。项目页：https://enigma-li.github.io/projects/VQ-SGen/VQ-SGen.html

> [!abstract] 这篇怎么用
> 核心链路：**Stage 1** 把每笔光栅笔画压进形状码本、位置码本（向量量化词典）；**Stage 2** 用级联 [[Transformer]] 自回归地抽码、拼回整图。按原文章节写全，方便日后建要点页链进来。
> 主文 PDF 共 9 页正文 + 参考文献；第 11–15 页为补充材料。Fig 号跟原文。

> [!note] 与 [[StrokeFusion]]、FaceX 怎么对照
> **VQ-SGen（本文）** 在 Creative Birds / Creative Creatures 上主评，**没有** FaceX 实验。做法是：每笔当成独立实体 → 形状与位置解耦 → 各自 VQ 码本（默认 **8192×512**）→ Gen-Transformer 依次预测部件标签、形状码、位置码，再解码成 **256×256** 光栅笔画并拼成终稿。
> **[[StrokeFusion]]** 在相关工作里把 VQ-SGen 当作「栅格 + 离散笔画 token + 自回归」的代表；自身输出[[矢量图]]，把笔画当**无序集合**联合建模位置与形状。StrokeFusion 在 **FaceX** 人脸草图上 [[FID]] **7.27**（SketchRNN 约 155、SketchKnitter 约 157），强调五官空间对齐；VQ-SGen 未在同一基准上对比。两条线都优化**成品草图分布**，Stage 1/2 **不是**人类起稿、排线的作画阶段。

**正文标题与起页（[[VQ-SGen.pdf]]）**

| 标题 | 页 |
| :--- | ---: |
| Abstract | 1 |
| 1 Introduction | 1 |
| 2 Related Work | 2 |
| 3 Method | 3 |
| 3.1 Vector-quantized Stroke Representation | 3 |
| 3.2 Autoregressive Generation | 4 |
| 4 Experiments | 5 |
| 4.1 Comparison | 5 |
| 4.2 Ablation Study | 6 |
| 4.3 User Study | 7 |
| 5 Discussion and Application | 7 |
| 6 Conclusion | 8 |
| References | 9 |
| Supplementary Material | 11 |
| A Data Preprocessing and Augmentation | 11 |
| B Training and Inference Details | 11–12 |
| C Location Codebook Probing | 12 |
| D Creativity Comparison with Diffsketcher | 12–13 |
| E Further discussions | 13–14 |
| F Detailed Network Configuration | 14–15 |

## 摘要

![[图/VQ-SGen/fig1.png]]

Fig. 1：左列 DoodleFormer、DoodlerGAN、SketchKnitter 与真值；中间 Stage 1 把单笔收成形状 VQ token，Stage 2 Gen-Transformer 自回归抽码；右下「Ours」终稿。图中 Stage 1/2 指**码本学习 + 自回归解码**，不是人类起稿、排线的作画阶段。

本文提出 VQ-SGen，一种高质量**创意草图生成**算法。近年工作多把任务做成整图像素生成或按部件生成，忽略了笔画之间的形状与远近空间关系，局部易糊、笔画易散（如 Fig. 1 里 DoodleFormer、DoodlerGAN 鸟头）。本文把草图里**每一笔当作实体**，引入**向量量化（VQ）笔画表示**，做细粒度生成。

方法分两阶段：**阶段一**解耦每笔的形状与位置，让 VQ 表示优先学形状；**阶段二**把紧凑离散码喂给自解码 [[Transformer]]，联合语义标签、位置与形状自回归生成。token 化笔画后，生成保真度高，并支持文本/类别条件生成与草图补全。在 CreativeSketch 数据集上全面实验，指标与视觉结果均优于现有方法。

## 1 引言

草图是人类传达想法、表达创意的直观方式。相关研究涵盖检索、语义分割与生成等。本文聚焦 **creative sketch generation**（Ge et al. 2020）：要的是多样、复杂、好看的**想象化**造型，而不是「标准鸟图」那种常规形态，对生成能力要求更高。

DiffSketcher 等扩散方法可做文本到草图，但大模型多在真实图像上预训练，结果偏写实、少创意，仍需要专门做创意草图的方法。

现有创意草图方法（DoodlerGAN、DoodleFormer 等）多在像素域整图或分部件生成，**没有**把单笔笔画的关系建模清楚，近处、远处笔画的形状与相对位置都被弱化，常出现局部模糊、笔画孤立。SketchKnitter 用扩散把笔画点从乱序整理成连贯形，但**没有**「笔画实体」概念，复杂创意鸟上效果仍差（Fig. 1）。

本文在**笔画级**处理：每笔独立实体，用 **VQ** 得到紧凑离散码，减轻冗余、突出形状变化；码空间里还能观察到**语义聚类**，便于采样新笔画。定制生成器同时用形状、语义与位置，在压缩的 VQ 空间里采样，并推理笔画间结构关系。

**VQ-SGen 两阶段框架（读法：词典 + Transformer，不是作画阶段）**

1. **阶段一（VQ-Representation）**：解耦形状与位置；形状走码本 $D_s$，位置走码本 $D_l$，让形状码专注笔画形变，少被坐标干扰。
2. **阶段二（Gen-Transformer）**：自回归 [[Transformer]] 依次融入语义标签、形状码、位置码，利用邻域上下文，使形状与位置跟部件语义一致。

在 CreativeSketch 上，对比与消融表明优于当时 SOTA；用户研究里 VQ-SGen 也整体更受偏好。

**贡献概括**

- 提出笔画级创意草图方法 VQ-SGen。
- 提出 VQ 笔画表示：每笔一实体，紧凑编码形状，作为生成基础。
- 提出级联生成器，联合形状、语义与空间位置。
- 系统实验与用户研究验证有效性。

## 2 相关工作

**草图表示学习。** 按数据形式可分为：图像式、序列式、图式。图像式用光栅与绝对坐标，难学结构与笔画顺序；图式难刻画笔内细节；序列式用相对坐标点列，笔内结构好，但笔间邻近关系弱。ContextSeg 提出**单笔实体**用于语义分割；SketchXAI 把单笔解耦为形状、位置、顺序嵌入做分类可解释性。本文同样以单笔为原子单位并解耦形状与位置，且为生成任务训练 **VQ 离散码空间**。

**向量量化 VAE。** 离散表示利于推理与规划；StrokeNUWA 用 VQ-VAE 表示矢量图再用大模型生成。本文对**光栅化单笔**学 VQ 表示，期望紧凑码提升生成能力。

**创意草图生成。** 与传统草图生成不同，创意任务强调想象化描绘。DoodlerGAN 为每个身体部件单独训 GAN，开销大。DoodleFormer 两阶段 [[Transformer]] 偏粗粒度部件关系，易出现不同区域笔画「串味」。本文用 VQ token + 细粒度自回归，减轻 Fig. 1 类伪影。

## 3 方法

![[图/VQ-SGen/fig2.png]]

Fig. 2（PDF 第 3 页）：输入草图拆成笔画序列 → 阶段一解耦形状/位置并得到离散码 → 阶段二 Gen-Transformer 自回归预测 $\{(I_i, b_i, l_i)\}$ → 合并、重定位得到重建或生成草图。

**生成目标。** 输出仍为三元组序列，按形状与位置拼成终稿（Fig. 1）。

### 3.1 向量量化笔画表示

三元组 $(I_i, b_i, l_i)$ 让生成时可分别控制形状、位置、语义；下一步是把三元组编码成适合生成的紧凑形式。受 VQ-VAE 启发，在**整草图上下文**里压缩笔画，并基于解耦三元组提高形状与语义感知。

**草图与笔画。** 草图 $S$ 由笔画序列 $\{s_i\}_{i=1}^N$ 与标签 $\{l_i\}_{i=1}^N$ 组成，顺序对应默认绘制顺序。每笔是 **256×256** 光栅图；$l_i$ 为 $C$ 维 one-hot 部件标签。

**解耦形状与位置。** 对每笔光栅图求轴对齐包围盒与中心 $(x_i,y_i)$。位置 $b_i=(w_i/2, h_i/2, x_i, y_i)$ 由框宽高与中心表示。把框平移到与草图画布中心对齐、尺度不变，得到只含形状的 $I_i$。于是 $S=\{(I_i,b_i,l_i)\}_{i=1}^N$。

**笔画潜嵌入。** 用 2D CNN 自编码器 $E^s,D^s$ 编码 $I_i$，瓶颈展平得 $e_i^s=\mathrm{flat}(E^s(I_i))$。重建损失：

$$
\mathcal{L}_{\mathrm{recons}}=\left\|I_i-D^s(E^s(I_i))\right\|^2 \tag{1}
$$

为更好学形状，网络加 CoordConv，训练加距离场监督（同 ContextSeg 等）。

**笔画 token 化（收成词典）。** 形状与位置各建离散码空间——位置与语义相关（如鸟头大小、位置相近）。对形状：码本 $D_s$，整图笔画形状潜码 $\{e_i^s\}$ 经 1D CNN 编码器 $E^f$ 得 $\{z_i^s\}$，最近邻量化：

$$
v_i^s=\operatorname{argmin}_{j\in[0,V)}\left\|z_i^s-c_j\right\| \tag{2}
$$

量化特征序列送解码器 $D^f$ 重建潜嵌入。VQ 训练用码本、commitment、重建项（VQ-VAE 标准形式，权重 $\alpha$，$\mathrm{sg}$ 为 stop-gradient）：

$$
\mathcal{L}_{\mathrm{VQ}}=\frac{1}{N}\sum_{i=1}^N \alpha(\|z_i^s-\mathrm{sg}[c_{v_i^s}]\|_2^2+\|\mathrm{sg}[z_i^s]-c_{v_i^s}\|_2^2)+\|z_i^s-D^f(c_{v_i^s})\|_2^2 \tag{3}
$$

**位置码本**用同样流程，$E^l,D^l$ 建 $D_l$，从 $b_i$ 重建位置。单笔紧凑表示为 $(v_i^s, v_i^l, l_i)$，对应码向量 $c_{v_i^s}, d_{v_i^l}$ 与标签。

生成阶段 [[Transformer]] 里直接用码本索引与嵌入，不再喂原始光栅与四元组位置。

### 3.2 自回归生成

在 VQ 表示上估计 $p(S)$ 并采样新草图。因 $S$ 含形状、位置、标签，生成分两步：先预测标签，再在该标签条件下预测形状码与位置码。链式法则：

$$
p(S)=\prod_{i=1}^N p(I_i,b_i,l_i)=\prod_{i=1}^N p(v_i^s,v_i^l\mid l_i)\cdot p(l_i) \tag{4}
$$

用两个级联 **decoder-only Transformer**（参数 $\theta_1,\theta_2$）实现：

$$
\begin{aligned}
p(l_i)&=p(l_i\mid l_{<i}, v^s_{<i}, v^l_{<i};\theta_1),\\
p(v_i^s,v_i^l\mid l_i)&=p(v_i^s,v_i^l\mid l_{\leq i}, v^s_{<i}, v^l_{<i};\theta_2).
\end{aligned} \tag{5}
$$

![[图/VQ-SGen/fig3.png]]

Fig. 3（PDF 第 4 页）：

- **标签 Transformer $T^l$**：输入为 $(c_{v_i^s}, d_{v_i^l}, l_i)$ 各自嵌入后**相加**（各 512 维），经注意力得 $F_{\mathrm{fuse}}$，全连接 + softmax 预测下一笔 $l_{i+1}\in\mathbb{R}^C$。
- **码 Transformer $T^c$**：以预测的 $l_{i+1}$ 嵌入为条件，与 $F_{\mathrm{fuse}}$ 相加，两路分支各输出 $V$ 维 softmax，采样形状码、位置码索引；取 $c_{v_{i+1}^s}, d_{v_{i+1}^l}$ 作为下一笔 $T^l$ 的输入。

**损失。** 最小化负对数似然，对标签 one-hot、形状码索引、位置码索引做监督：

$$
\mathcal{L}_{\mathrm{gen}}=-\log p(S) \tag{6}
$$

**训练与测试策略。** 顺序三阶段：① 训笔画潜嵌入自编码器；② 固定后训 VQ 与码本；③ 固定前述模块，训级联生成器（补充材料用 scheduled sampling 缓解 exposure bias）。

**推理。** 从特殊 **STR** token 起自回归生成，直到 **END**。码条目解码为位置四元组或形状潜码，再还原光栅单笔，三元组合并成草图（Fig. 2 左侧）。采样时先取累积概率超过阈值 $p_n$ 的最小码集合，再按相对概率随机抽，兼顾多样与稳定。

## 4 实验

### 4.1 对比实验

**数据集。** 沿用 CreativeSketch：Creative Birds（CB）8067 张、Creative Creatures（CC）9097 张，带部件标注，统一 **256×256**。笔画潜嵌入自编码器用两集笔画联合训练 $E^s,D^s$；其余网络在 CB、CC 上分别训练。预处理与增强见附录 A。

**指标。** 用 QuickDraw3.8M 上训练的 Inception 模型评估：

- **FID**：两集合在 Inception 特征空间的 Fréchet 距离。
- **GD（Generation Diversity）**：子集间 Inception 特征欧氏距离均值，衡量多样性。
- **CS（Characteristic Score）**：生成图被分成 Creative Bird / Creative Creature 的比例。
- **SDS（Semantic Diversity Score）**：按生物类别衡量的语义多样性（CC 报告）。

**对手。** DoodleFormer、DoodlerGAN（像素级）；SketchKnitter（笔画点序列级）。均用默认超参在相同数据上训练。

![[图/VQ-SGen/table1.png]]

**表 1：Creative Birds / Creative Creatures 统计对比**

| 方法 | CB FID↓ | CB GD↑ | CB CS↑ | CC FID↓ | CC GD↑ | CC CS↑ | CC SDS↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training Data | — | 19.40 | 0.45 | — | 18.06 | 0.60 | 1.91 |
| SketchKnitter | 74.42 | 14.23 | 0.14 | 64.34 | 12.34 | 0.42 | 1.32 |
| DoodlerGAN | 39.95 | 16.33 | 0.69 | 43.94 | 14.57 | 0.55 | 1.45 |
| DoodleFormer | 17.48 | 17.83 | 0.57 | 20.43 | 16.23 | 0.53 | 1.68 |
| Ours | **15.78** | **18.92** | 0.53 | **17.61** | **17.42** | 0.57 | **1.86** |

![[图/VQ-SGen/fig4.png]]

Fig. 4：Creative Birds / Creative Creatures 上四列对比（真值、本文、DoodleFormer、DoodlerGAN），每格一张终稿草图；DoodlerGAN 易断笔错位，DoodleFormer 易局部糊。

CB 上 FID 比最强对手低 **1.7**（相对 DoodleFormer 17.48），GD 高 **1.09**；CS 不是最高，但高于训练集参考 0.45，且 DoodlerGAN / DoodleFormer CS 更高往往对应更「好认」、略欠创意。CC 上四项指标均为最佳（FID 低 2.82、GD 高 1.19 等）。Fig. 4（PDF 第 6 页）可见 DoodlerGAN 断笔错位、DoodleFormer 局部糊，本文细节与多样性更好。SketchKnitter 两集都差，视觉对比未列入。

### 4.2 消融实验

![[图/VQ-SGen/table2.png]]

**表 2：消融与形状码本规模（名如 8192×512 表示码本大小 × 特征维）**

| 方法 | CB FID↓ | CB GD↑ | CB CS↑ | CC FID↓ | CC GD↑ | CC CS↑ | CC SDS↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training Data | — | 19.40 | 0.45 | — | 18.06 | 0.60 | 1.91 |
| 2048×512 | 26.23 | 15.34 | 0.43 | 43.21 | 14.51 | 0.48 | 1.52 |
| 4096×512 | 16.92 | 18.27 | 0.50 | 18.44 | 16.14 | 0.54 | 1.63 |
| 4096×1024 | 16.67 | 18.15 | 0.51 | 18.12 | 16.43 | 0.55 | 1.69 |
| w/o $T^l$ | 16.51 | 18.35 | 0.51 | 19.21 | 16.14 | 0.55 | 1.74 |
| w/o VQ | 48.53 | 13.34 | 0.46 | 54.56 | 14.02 | 0.45 | 1.36 |
| w/o Decouple | 17.14 | 18.12 | 0.57 | 19.42 | 16.42 | 0.56 | 1.79 |
| Ours (8192×512) | **15.78** | **18.92** | 0.53 | **17.61** | **17.42** | 0.57 | **1.86** |

变体说明：

- **w/o VQ**：不用码本，连续潜嵌入直接训生成器，FID 最差，说明**离散词典压缩**关键。
- **w/o Decouple**：形状位置共用一个码本，FID 升至 17.14 / 19.42，解耦让形状码专注意形变、位置码学笔间布局。
- **w/o $T^l$**：去掉标签 Transformer，性能略降但仍第二梯队；标签提供语义补充，也证明无部件标注数据集可去掉 $T^l$（见第 5 节 QuickDraw）。

![[图/VQ-SGen/fig5.png]]

Fig. 5：消融里形状码本规模（$2048×512$ 至 $8192×512$）对**单笔重建**的影响；码本越大断笔与模糊越少，极大形变仍可能盖不住。

### 4.3 用户研究

![[图/VQ-SGen/fig6.png]]

Fig. 6：五道偏好题（a）–（e）上「本文胜出比例」；深蓝对 DoodleFormer，浅蓝对创意真值集；CB 上（b）略低于 50%，其余多数高于 DoodleFormer。

50 名被试，每对图为一幅本文生成 vs DoodleFormer 或真值集。五道单选题：（a）更有创意？（b）更像鸟/生物？（c）更像人画的？（d）笔画更融合？（e）总体更喜欢？除 CB 上（b）与 CS 一致略输外，其余多数题本文更受欢迎；与创意数据集几乎打平。

## 5 讨论与应用

![[图/VQ-SGen/fig8.png]]

Fig. 8：形状码本 $D_s$ 的 UMAP；外围簇对应 Beak、Eye、Tail、Wing、Leg 等典型笔画，中心大簇混有 Head、Body——**未加语义监督**仍呈语义聚类。

**码空间探索。** 对形状码本 $D_s$ 做 UMAP，可见 Beak、Eye、Tail、Wing、Leg 等簇，中心混有 Head、Body，利于在压缩空间里按语义采样。

![[图/VQ-SGen/fig7.png]]

Fig. 7：（a）类别标签 → 多样草图；（b）文本描述 → 想象化鸟形；（c）给定首笔 → 补全，对比 DoodlerGAN / DoodleFormer。

**类别条件生成。** 取 QuickDraw 20 类子集（**无**笔画标签）：去掉 $T^l$，用类别 index token 替换 STR，重训生成器。(a) 显示可生成多样类别草图。

**文本到草图。** CB/CC 带文本描述：用 CLIP 嵌入文本，替换 STR token，(b) 如 “Hug me”“Smiling” 等想象化结果。

**草图补全。** 给定首笔，从该笔起自回归补全。(c) 中 DoodlerGAN 结构乱，DoodleFormer 易糊与鬼影笔画，本文更连贯。

与 DiffSketcher 的创意/效率对比见附录 D（本文约 **0.86 s** vs DiffSketcher **153 s** 量级，且更偏创意鸟而非写实背景）。

## 6 结论

本文提出创意草图生成的 VQ 笔画路线：紧凑 VQ 表示 + 自回归 [[Transformer]]，外观与结构更一致。作者认为 stroke VQ 可迁移到其他草图任务。

致谢：Adobe 资助；感谢校对与 DiffSketcher 对比协助。

## 附录

补充材料从 PDF 第 11 页起；以下为正文外有用细节摘要。

### A 数据预处理与增强

Fig. A1：去掉带 `details` 标签的笔画（太阳、地面、装饰点）；删掉过长笔画；合并过短且相连的笔画（如翅膀多条并一条）。笔画级随机旋转/平移/缩放；草图级整体旋转或随机删笔模拟残缺输入。

### B 训练与推理细节

- 优化器 Adam；VQ 模块学习率 **$10^{-4}$**，Gen-Transformer **$10^{-5}$**；VQ 用 step decay（步长 10）。
- **2× NVIDIA V100**；VQ 训练约 **20 h**（batch 64）；生成器至收敛约 **10 h**（batch 8）。
- 最大笔画数：CB **$N=20$**，CC **$N=35$**；$e_i^s\in\mathbb{R}^{256}$；码本大小 **$V=8192$**，码维度 **512**；类别数 **$C=8$**（CB）、**17**（CC）；VQ 平衡权重 **$\alpha=0.8$**。
- 推理采样：累积概率阈值 $p_n$ 截断候选码再随机抽（补充材料 B 节）。

### C 位置码本探测

形状码本消融见主文 Tab. 2 / Fig. 5；位置码本扫 **2048/4096/8192 × 512/1024**。Tab. A1：最终 **8192×512** 包围盒 IoU 最高（CB **0.963**，CC **0.957**），生成 FID/GD 亦最佳。Fig. A2 为框重建可视化（非生成结果）。

### D 与 Diffsketcher 的创意对比

文本驱动任务目标相近但 DiffSketcher 偏写实、常带背景线。Fig. A3–A4：调 stroke 数与 CFG 仍难同时抽象与创意；本文 “Just dance”“Smiling” 等更贴语义。速度见上。

### E 进一步讨论

- **码插值**（Fig. A5）：在 $D_s$ 中线性混合两码再取最近码，逐步 morph 形状。
- **局限**（Fig. A6）：码本超参未穷举；解耦位置偶发框裁切笔画（「硬裁剪」），可加覆盖正则。
- **形状码本泛化**（Fig. A7）：未微调可在 QuickDraw 上重建。
- **点序列 vs 位图**（ContextSeg 设定延伸）：用 Sketchformer 替 CNN 训 VQ+生成，重建 Acc/Rec 远低于位图（0.44/0.32 vs 0.96/0.97），生成 FID 75.42 vs **15.78**，确认**光栅单笔**编码更适合本文生成场景。

### F 网络结构细节

Fig. A8：嵌入网络 10 层 2D CNN，块维 64→128→256→512，输出 $\mathbb{R}^{256}$，CoordConv 输入，双解码支路重建笔画与距离图。Token 化网络：输入 $N\times256$ 序列，Conv1d + 池化/反卷积压到 $N\times512$，最近邻替换码后解码重建潜码。形状嵌入与码本学习**分开**：前者逐笔抓结构，后者在整序列上压缩——合并训练难同时最优，故分阶段。

---

参考文献列表见 PDF 第 9 页起，共 45 条，此处从略。
