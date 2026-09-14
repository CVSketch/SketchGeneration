---
title: "Stroke-based Neural Painting and Stylization with Dynamically Predicted Painting Region"
short: Stroke-based-Neural-Painting
source: "[[Stroke-based-Neural-Painting]]"
pdf: "[[Stroke-based-Neural-Painting.pdf]]"
tags:
  - 译文
  - 全文
status: 全文
---

# Stroke-based-Neural-Painting · 全文译文

对应笔记：[[Stroke-based-Neural-Painting]]。要点：[[Stroke-based-Neural-Painting-要点]]。原文：[[Stroke-based-Neural-Painting.pdf]]。代码：https://github.com/sjtuplayer/Compositional_Neural_Painter

> [!abstract] 这篇怎么用
> 按原论文章节用中文写全。标题编号跟原文一致，方便从 [[Stroke-based-Neural-Painting-要点]] 点进来。
> 公式、表号、图号跟 PDF。不要整段粘英文。
>
> **和本课题的关系：** [[Inverse-Painting]]、[[ProcessPainter]] 在笔画渲染（SBR）与过程生成相关讨论里会引用本文。本文核心流程是 **先根据当前画布预测下一步涂哪一块矩形区域，再在该区域内预测多笔笔触并贴回画布**（Compositional Neural Painter）。$T$ 是合成器—画家交替的**作画步数**，不是扩散去噪步；总笔触数由每步笔数与步数共同决定。与均匀切 $k \times k$ 网格再分块画的方法不同，动态区域能减轻块边界处笔触断裂、缺失等伪影。

## 摘要

笔触式渲染（SBR）用一组笔触重建一张图。多数现有方法在测试时对画面做**均匀分块**：每块单独预测笔触，再拼成整图，块与块交界处容易出现**边界不一致**（笔触断线、缺笔等）。为此提出 **Compositional Neural Painter**：从空白画布出发，把作画拆成若干步；每一步里，先用带**分阶段奖励**的 [[强化学习]] 训练 **合成器网络（compositor）**，根据**当前画布**动态预测下一块**作画区域**；再用带 **WGAN** 判别器的 **画家网络（painter）** 预测该区域的笔触参数，并由**笔触渲染器**画到区域上后贴回画布。方法还扩展到笔触式风格迁移，提出可微**距离变换（distance transform）损失**，在笔触化过程中更好保留输入图结构。实验表明，在神经笔触绘画与笔触风格迁移上均优于现有方法。

## 1 引言

SBR 用笔触而非像素作为基本单元，按顺序在画布上落笔，更贴近人类作画方式。传统 SBR 多靠贪心搜索、启发式优化等，效率低。深度学习方法大致分 [[强化学习]] 路线 [14, 22, 23]、纯深度学习路线 [20]、以及逐图优化路线 [17, 30]。

![[图/Stroke-based-Neural-Painting/fig1.png]]

Fig. 1：(a) 测试时均匀 $k \times k$ 分块、分块出笔再拼整图；(b) 每步动态预测矩形区域、裁剪出笔、贴回画布并迭代。**$T$ 是合成器与画家交替的作画步，不是课堂四阶段，也不是扩散去噪步。**

![[图/Stroke-based-Neural-Painting/fig2.png]]

Fig. 2：3000 笔触下悉尼歌剧院；Paint Transformer、Learning To Paint、Stylized Neural Painting 在块边界出现断笔、缺笔；本文动态选区无均匀网格缝。

现有方法对**单个图像块**能预测的笔触数有限，更多笔触意味着更高训练或优化成本；而 ImageNet 等真实图往往过于复杂，有限笔触难以重建细节。因此 [14, 20, 30] 等在测试阶段把画面**均匀划成 $k \times k$ 块**，每块独立出笔（Fig. 1(a)）。这带来两点问题：（1）分块策略常只在测试用，与训练设定不一致；（2）各块互不可见，块边界两侧笔触不协调（Fig. 2），出现边界不一致伪影。

真实作画里，画家往往**先定这一笔要画哪一片**，再在该片里落笔。本文据此提出 Compositional Neural Painter：**先回答「涂哪里」再回答「涂什么」**。与 [[Intelli-Paint]] 用滑动注意力窗、且依赖目标检测不同，本文在**全局**视野下预测作画区域，不绑检测器，对多主体或无清晰前景的场景更稳。

结构上分两块：**合成器**预测涂哪里，**画家**决定涂什么。合成器用带分阶段奖励的 RL 训练；画家为 CNN，用 WGAN 判别器抑制「反复画同一类粗笔触」（纯像素损失时常见）。另将框架用于笔触风格迁移，并加入可微距离变换损失以保结构。

**贡献概括：**

- Compositional Neural Painter：依当前画布动态预测下一块作画区域，缓解均匀分块的边界伪影，重建质量更好。
- 合成器（分阶段 RL 奖励）+ 画家（WGAN 判别）+ 神经笔触渲染器的组合训练流程。
- 笔触风格迁移与可微距离变换损失，结构保持更好。

## 2 相关工作

### 2.1 笔触式渲染（SBR）

SBR 用笔触序列模拟绘画，相对 VAE [16]、GAN [6]、[[扩散模型]] [12] 等像素生成，局部纹理与笔触感更强。传统方法包括贪心 [11, 19, 26]、启发式 [27]、用户交互 [10, 25] 等；Im2Oil [26] 用自适应采样与概率图贪心，效果较好。

早期深度方法用 RNN [7, 9]，但依赖标注。Ganin 等 [4]、Zhou 等 [29] 引入 RL，多限于素描级简单图。Huang 等 [14]（[[Learning-to-Paint]]）用 DDPG [18] 与 WGAN [1] 奖励画复杂图；Singh 与 Zheng [23]、Singh 等 [22]（[[Intelli-Paint]]）加入语义引导，让 RL 更关注主体，但检测依赖与 RL 效率限制复杂图与细粒度笔触数。Liu 等 [20] 的 Paint Transformer 去掉 RL、训练更稳，但同精度往往需要更多笔触。优化类 [17, 30]（含 [[Stylized-Neural-Painting]]）逐图迭代，单图耗时长。上述方法或受边界伪影困扰，或难以在复杂图上兼顾细节与块间一致。本文用合成器动态选区 + 画家局部出笔，并相对 Intelli-Paint 不依赖检测。

### 2.2 笔触式风格迁移

风格迁移常先在像素域完成 [2, 5, 13, 21, 24]。笔触式风格迁移 [17, 30] 优化笔触参数而非像素，但结构往往保不住。本文在笔触风格化中加入可微距离变换损失，用边缘图约束结构。

## 3 方法

### 3.1 概述

SBR 用笔触重建目标图 $I$。既有方法 [14, 20, 22, 23, 30] 对单块只能预测有限笔触；为细节，测试时均匀分 $k \times k$ 块独立出笔，导致 Fig. 2 所示边界不一致（断笔、缺笔）。语义方法 [22, 23] 不用均匀分块、在前景多画几笔，但依赖检测且 RL 训练效率低，对多主体或无清晰前景图能力有限。

Compositional Neural Painter 含三模块：（1）**合成器**：输入目标图 $I$ 与画布 $C$，预测下一块作画区域 $r$；（2）**画家**：输入按 $r$ 裁剪的 $I^r$ 与 $C^r$，预测笔触参数 $s$；（3）**渲染器**：把 $s$ 画回当前画布。

![[图/Stroke-based-Neural-Painting/fig3.png]]

Fig. 3：合成器预测 $r_t$（Where）→ 画家预测 $s_t$（What）→ 渲染器贴回得 $C_{t+1}$；空画布起共 $T$ 步。整图为方法总览，不含页眉。

从空画布 $C^0$ 起，过程拆成 $T$ 步（Fig. 3）。第 $t$ 步：合成器得 $r_t$；画家预测 $N$ 笔参数 $s_t$；渲染器在区域 $r_t$ 上作画并更新画布。$T$ 步后得 $I_r = C^T$。当像素损失很小时，原文用分阶段奖励放大后期细粒度重建的信号（式 (2)）。

### 3.2 合成器网络：涂哪里？

人类通常**先看当前画布，再决定下一片要画哪里**，而不是固定从左上扫到右下。合成器在每一步输入 $I$ 与 $C_t$，输出矩形区域 $r_t = (x, y, w, h)$：$(x,y)$ 为左上角，$w,h$ 为宽高。得到 $r_t$ 后，画家与渲染器在该区域出笔，得到 $C_{t+1}$。

**训练。** 与 [14, 20, 22, 23] 的自监督像素距离不同，本文需按 $r_t$ 裁剪，$(x,y,w,h)$ 要取整，反传不可微，故用 DDPG [18] 训合成器。训练顺序：先训渲染器与画家，再**固定**二者，训合成器。

DDPG 类方法 [14, 23] 常用原始奖励：

$$R_{\mathrm{ori}} = \frac{\|I - C_t\|_2^2}{\|I\|_2^2} - \frac{\|I - C_{t+1}\|_2^2}{\|I\|_2^2} \tag{1}$$

意在缩小 $C_{t+1}$ 与 $I$ 的差距。但当 $C_t$ 已接近 $I$ 时，奖励过小，critic 难捕捉细微改进。定义 $d = \|I - C_{t+1}\|_2 / \|I\|_2$，**分阶段奖励**为：

$$R_{\mathrm{phasic}} = \beta R_{\mathrm{ori}}, \quad \beta = \begin{cases} 1, & d > 0.005 \\ f(1-d), & d \leq 0.005 \end{cases} \tag{2}$$

其中 $f(x) = \alpha \ln\frac{1-x}{1+x}$ 为反 sigmoid；$\alpha$ 为常数。实现里在 $f$ 分母加 $\epsilon = 10^{-6}$，避免 $d \to 0$ 时数值问题。相对式 (1)，后期小奖励被放大，有利于纹理与细节。

### 3.3 画家网络：涂什么？

画家在区域 $r_t$ 内用 $N$ 笔重建局部。许多方法 [4, 14, 23, 29] 用 RL 训画家；[[Learning-to-Paint]] 需多个辅助网络。Paint Transformer [20] 不用 RL，但同精度往往要更多笔。本文发现：**CNN 画家 + WGAN 判别器**即可，训练更简单，重建精度更好。

**结构。** 第 $t$ 步输入 $I$ 与 $C_t$（实现上配合裁剪区域），输出 $N$ 笔参数 $s_t = \{s_t^{(1)}, \ldots, s_t^{(N)}\}$。渲染器 $R$ 得笔触图 $I_s = R(s_t)$ 与二值蒙版 $M_s$，贴回画布：

$$C_{t+1} = I_s \odot M_s + C_t \odot (1 - M_s) \tag{3}$$

$T$ 步后 $I_r = C^T$。

**训练。** 像素损失：

$$L_{\mathrm{pixel}} = \|I - C_{t+1}\|_2^2 \tag{4}$$

仅优化 $L_{\mathrm{pixel}}$ 时，模型易重复相似粗笔触，细节不足（见 4.5 消融）。加入 WGAN 判别器 $D$：生成画布 $C_{t+1}$ 为假样本，$D$ 惩罚与历史迭代中见过的笔触过于相似的输出，促使画家探索多样笔触。训练采用 WGAN-GP [8]：

$$L_{\mathrm{adv}} = D(C_{t+1}) - D(x) - \lambda(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2 \tag{5}$$

$x$ 为真图，$\hat{x}$ 在 $C_{t+1}$ 与 $x$ 之间插值。与 [[Learning-to-Paint]] 把两张相同真图拼接当真样本不同，本文 $D$ 只盯「是否重复旧笔触」；真样本 $x$ 甚至可以是随机噪声（4.5 验证仍有效）。总损失：

$$L_{\mathrm{total}} = L_{\mathrm{pixel}} + \gamma L_{\mathrm{adv}} \tag{6}$$

$\gamma = \lambda \|L_{\mathrm{pixel}}\| / \|L_{\mathrm{adv}}\|$ 为自适应权重，$\lambda$ 为常数。

### 3.4 笔触渲染器

渲染器由笔触参数 $s = \{x, y, w, h, \theta, r, g, b\}$ 生成笔触图 $I_s$ 与蒙版 $M_s$：中心 $(x,y)$，宽高 $w,h$，旋转 $\theta$，RGB。基本笔形采用真实油画笔触 [30]，再按参数变换。仿射变换在图形学里常不可微，故与 [14, 20, 30] 一样用**神经网络**渲染：输入参数，输出 $M_s$ 与 $I_s = (M_s \cdot r,\, M_s \cdot g,\, M_s \cdot b)$。以仿射变换得到的 $\hat{M}_s$ 为监督：

$$L_{\mathrm{renderer}} = \|\hat{M}_s - M_s\|_2^2 \tag{7}$$

训练流程跟随 [[Learning-to-Paint]]。渲染器还可输出笔触**边缘图** $E_s$（与 $M_s$ 同法训练），供 3.5 风格迁移使用。

### 3.5 笔触式风格迁移

像素域风格迁移 [5, 13, 21] 与笔触式 [17, 30] 相比，后者难保结构。本文在笔触风格化中加入可微距离变换损失：渲染器额外输出 $E_s$，通过让 $E_s$ 与输入边缘图 $E_{\mathrm{gt}}$ 在距离变换意义下接近，得到**保边缘**的笔触化结果。

在像素 $(i,j)$ 的 $K \times K$ 邻域 $\mathcal{N}(i,j)$ 上，用核 $D$ 存到中心像素的欧氏距离，近似边缘图 $E$ 的距离变换：

$$DT(E)_{(i,j)} = \min_{(k,l) \in \mathcal{N}(i,j)} \left[ E(k,l) \cdot D(k-i, l-j) + (1 - E(k,l)) \cdot d_{\max} \right] \tag{8}$$

最小值用 $\lambda \to 0$ 的 soft-min 近似（实验取 $\lambda = 0.3$）。距离变换损失：

$$L_{DT}(E_s, E) = \mathbb{E}_{(i,j)} \left[ DT(E_s)_{(i,j)} \cdot E(i,j) \right] \tag{10}$$

给定内容图 $I_s$ 与风格图 $I_t$：先用 Compositional Neural Painter 从 $I_s$ 得到笔触图 $I_r$ 及笔触参数 $S = \{s_1, \ldots, s_n\}$，再渲染边缘 $E_s$。记输入二值边缘为 $E_{\mathrm{gt}}$，优化：

$$S^* = \min_S \left[ \lambda_{\mathrm{sty}} L_{\mathrm{style}} + \lambda_{\mathrm{con}} L_{\mathrm{con}} + \lambda_{DT} L_{DT}(E_s, E_{\mathrm{gt}}) \right] \tag{11}$$

$L_{\mathrm{style}}, L_{\mathrm{con}}$ 沿用 [5]。

## 4 实验

### 4.1 数据集与设置

**数据。** 主要在 CelebA-HQ [15] 与 ImageNet [3] 上训练与测试；各随机 1 000 张测试，其余训练。

**指标。**（1）**L2 距离**：渲染图与目标图像素 L2，越低越好。（2）**PSNR**，越高越好。（3）**LPIPS** [28]，越低表示感知上越接近。

### 4.2 训练细节

三步训练：

1. **渲染器**：随机采样 $s$，仿射得到 $\hat{M}_s$，最小化式 (7)，**100 万** iter，batch **32**。
2. **画家**：在 CelebA-HQ 或 ImageNet 上 **200 万** iter，batch **32**。
3. **合成器**：固定渲染器与画家，DDPG 训合成器，**200 万** iter，batch **32**。

对比实验除 Parameterized Brushstrokes [17] 用其自带 Bézier 渲染器外，其余方法统一使用相同**油画笔触**素材，并采用官方实现。

### 4.3 图像到绘画（重建）

与 RL 类（[[Learning-to-Paint]] [14]、Semantic Guidance+RL [23]）、Paint Transformer [20]、优化类（[[Stylized-Neural-Painting]] [30]、Parameterized Brushstrokes [17]）、传统 Im2Oil [26] 对比。笔触数取 **200、500、1 000、3 000、5 000**。

![[图/Stroke-based-Neural-Painting/fig4.png]]

Fig. 4：200 / 1000 / 3000 / 5000 笔触下两场景×八列对比（目标、Img2Oil、Parameterized Brushstrokes、Semantic+RL、Learning To Paint、Paint Transformer、Stylized Neural Painting、Ours）；整页对比网格一张图，勿拆小格。

**定性（Fig. 4）。** Parameterized Brushstrokes 与 Semantic+RL 难以重建复杂图。Learning To Paint、Paint Transformer、Stylized Neural Painting 均有明显**边界不一致**；Im2Oil 观感尚可但笔触少时缺细节。本文既减轻边界问题，细节与纹理也更丰富。

![[图/Stroke-based-Neural-Painting/table1.png]]

Table 1：ImageNet 与 CelebA-HQ 上 L2、PSNR、LPIPS；逐图优化三法仅抽 100 张评。

**定量（Table 1）。** 在 ImageNet 与 CelebA-HQ 上，多数笔触设定下，本文 L2 更低、PSNR 更高、LPIPS 更低。Stylized Neural Painting、Parameterized Brushstrokes、Im2Oil 因逐图优化慢，Table 1 中仅抽 **100** 张评估。

### 4.4 笔触式风格迁移

![[图/Stroke-based-Neural-Painting/fig5.png]]

Fig. 5：1000 / 2000 笔触、六组内容+风格；Ours 与 Stylized Neural Painting [30]、Parameterized Brushstrokes [17] 并列；右栏整图一张，无邻栏正文。

与 [[Stylized-Neural-Painting]] [30]、Parameterized Brushstrokes [17] 在 **1 000、2 000** 笔触及默认风格化设定下对比（Fig. 5）。Stylized Neural Painting 边界伪影重；Parameterized Brushstrokes 内容损失大、物体轮廓模糊（其论文中需 **1 万+** 笔触才较好看）。本文结构保留更好，风格化观感更佳。用生成图与风格图 **Gram 矩阵距离**衡量风格贴合度：本文平均 **0.6943**，Stylized Neural Painting **1.4048**，Parameterized Brushstrokes **4.0274**。

### 4.5 消融实验

![[图/Stroke-based-Neural-Painting/fig6.png]]

Fig. 6：消融八格 (a) 源图—(h) 完整 C+P；(c) 测试均匀分块再现边界伪影；(e) 无 $L_{\mathrm{adv}}$ 重复大块笔触。**分块是空间策略，不是课堂阶段。**

![[图/Stroke-based-Neural-Painting/table2.png]]

Table 2：ImageNet 上合成器/画家消融；1000 与 5000 笔触的 L2、PSNR、LPIPS。

**合成器（Table 2，Fig. 6）。** 设置包括：无合成器（仅画家）；无合成器但测试时**均匀分块**；无分阶段奖励；完整模型。无合成器或无阻奖励时细节不足；仅分块则再现边界伪影（Fig. 6(c)）。无分阶段奖励比完整模型细节少（Fig. 6(d) vs (h)）。

**画家。** 去掉 $L_{\mathrm{adv}}$ 会反复画大块相似笔触（Fig. 6(e)）；去掉自适应 $\gamma$ 训练不稳、细节差（Fig. 6(f)）；WGAN 真样本用随机噪声（Fig. 6(g)）与完整模型接近，说明 $D$ 的主要作用是**惩罚见过的假样本、避免重复笔触**。

## 5 结论

本文提出 Compositional Neural Painter：依当前画布**动态预测**下一块作画区域，而不是把画面均匀切成 $K \times K$ 块；并扩展到保边缘的笔触风格迁移（可微距离变换损失）。实验表明，在神经笔触渲染与笔触风格化上优于现有方法。

## 致谢

受国家自然科学基金、上海市扬帆计划、上海市科委重大项目、上海市科委项目、CCF-腾讯犀牛鸟基金、中国科协青年人才托举、北京市自然科学基金、中央高校基本科研业务费等资助（详见原文列项）。

## 参考文献

见 PDF 第 9 页起 [1]–[30] 条目列表；与正文引用编号一致。

## 附录

### A 附录总览（p10）

附录含：B 训练细节；C 距离变换损失消融；D 作画过程可视化；E 用户研究；F 与各画家网络对比；G 与 Intelli-Paint 对比；H 更多绘画对比；I 更多风格化（含像素法）；J 时间复杂度。

### B 训练细节（p10）

与正文 4.2 相同的三阶段流程与迭代数、batch 大小。

### C 距离变换损失消融（p10，Fig. 7）

去掉 $L_{DT}$ 时，模糊边缘区域丢结构、直线轮廓出现扰动；加入 $L_{DT}$ 后两种情形结构均保留更好。

### D 作画过程可视化（p11–12，Fig. 8）

展示 **5、100、500、2 000、5 000** 笔触时的中间结果：早期先抓整体轮廓，随后逐步补细节，笔触越多重建越细。

### E 用户研究（p11，Table 3）

30 名志愿者对本文、Learning To Paint、Paint Transformer、Stylized Neural Painting、Im2Oil 的结果排序。**65%** 的样本把本文排第一；平均名次 **1.56**（越低越好），优于其余方法。

### F 画家网络对比（p11，Table 4）

在 ImageNet 与 CelebA-HQ 上、**200 笔触**、**不做分块**时，仅比较各方法的画家：本文 L2 / LPIPS 最低、PSNR 最高。相对 Paint Transformer：本文用真实数据训练并加对抗学习，探索能力更强。相对 RL 画家：本文深度学习训练框架更简单、更稳，重建更好。

### G 与 Intelli-Paint 对比（p10–11，Fig. 9–10）

Intelli-Paint 用检测得全局窗，再在一个网络里同时预测局部注意力窗与笔触，RL 负担大，长序列（如超 2 000 笔）困难。本文**分阶段**：合成器选区、画家出笔，分开训练，笔触数可更大。动机上，Intelli-Paint 的窗是为了模拟「相邻两笔位置接近」；本文的作画区域是为让画家**只关注局部细节**、提高重建。Intelli-Paint 依赖清晰物体，风景等难画细；本文不依赖检测。代码未开源，作者基于 Learning To Paint 复现其 attention window（Fig. 9），细节重建仍不如本文（Fig. 10）。

### H 更多绘画对比（p14 起，Fig. 11–14）

**H.1 油画笔触（Fig. 11–14）。** 3 000 或 5 000 笔触下，Parameterized Brushstrokes、Semantic+RL 仍难重建；语义预测错时 Semantic+RL 信息丢失严重。均匀分块三方法（Learning To Paint、Paint Transformer、Stylized Neural Painting）边界断笔明显；本文无此问题且细节最多。

**H.2 透明笔触（Table 5）。** 用 Learning To Paint 式**透明圆点笔触**重训本文，与 [14]、[23] 默认透明笔对比。ImageNet 与 CelebA-HQ 各 1 000 张、笔触数 200–5 000，本文 L2 / PSNR / LPIPS 仍全面优于两者，说明对不同笔刷类型也鲁棒。CelebA-HQ 上 5 000 透明笔时，简单人脸结构与 Learning To Paint 接近；ImageNet 更复杂时本文优势更大。

### I 更多笔触风格化（p13、18，Fig. 15）

除笔触法 [30]、[17] 外，还与像素法 Gatys [5]、AdaIN [13]、AdaAttN [21] 对比。笔触数：本文与 Stylized Neural Painting **2 000**，Parameterized Brushstrokes **10 000+**。本文比其它笔触法更好保留语义内容；观感可与像素法相当。

### J 时间复杂度（p13，Table 6）

在单卡 **24G RTX 3090** 上，**1 000** 笔触、CelebA-HQ **100** 张图平均推理时间（不含加载）：Learning To Paint **0.2066 s** 最快，本文 **0.2162 s** 次之；Semantic+RL **2.9921 s**；Paint Transformer **0.3725 s**；Im2Oil **55.37 s**；Stylized Neural Painting **124.94 s**；Parameterized Brushstrokes **247.61 s**。本文在质量与速度之间较均衡。
