---
title: DoodleFormer 要点
short: DoodleFormer
source: "[[DoodleFormer]]"
translation: "[[DoodleFormer-译文]]"
tags:
  - 要点
  - 参考文献
status: 要点
---

# DoodleFormer · 要点

对应笔记：[[DoodleFormer]]。全文译文：[[DoodleFormer-译文]]。原文：[[DoodleFormer.pdf]]。对照任务：[[任务定义]]。

> [!tip] 怎么翻原文
> 每个要点都链到全文译文的某一节。点开就能对着中文读，不必再猜在 PDF 哪一页。附录节标题是「1 由粗到细的生成过程（Supp. Fig. 1）」这种，不要和正文「1 引言」搞混。

## 关键图

- Fig. 1：三任务示例——(a) 潜空间创意鸟、(b) 文本控形、(c) 起稿补全。图在 [[DoodleFormer-译文#1 引言]]。
- Fig. 2：同一初始笔画下 DoodlerGAN 与 DoodleFormer 在鸟/生物上的对比；后者拓扑更稳。图在 [[DoodleFormer-译文#1 引言]]。
- Fig. 3：**PL-Net + PS-Net** 方法总图：先 GMM 出部件框，再 PS-Net 栅格成图；空间上由粗到细，不是课堂阶段。图在 [[DoodleFormer-译文#整体框架]]。
- Fig. 4：从单阶段到 +GAT、+GMM 的消融视觉；姿态与尺寸更散。图在 [[DoodleFormer-译文#整体框架]]。
- Fig. 5：GAT 块——邻接图谱卷积与 MHSA 融合。图在 [[DoodleFormer-译文#图感知 Transformer（GAT）块]]。
- Fig. 6：(a) 随机笔画生成、(b) 补全、(c) 文本→简笔，对 DG/AG/SG。图在 [[DoodleFormer-译文#4.1 定量与定性对比]]、[[DoodleFormer-译文#4.4 相关应用]]。
- Fig. 7：用户 study 五题柱状图（Birds / Creatures）。图在 [[DoodleFormer-译文#4.2 用户 study]]。
- Fig. 8：气泡图→户型，对 House-GAN 等。图在 [[DoodleFormer-译文#4.4 相关应用]]。
- Tab. 1–4：主结果、两阶段/GAT/GMM 消融、GAT 设计对比、户型 FID/Compatibility。图在 [[DoodleFormer-译文#4.1 定量与定性对比]]、[[DoodleFormer-译文#4.3 消融实验]]、[[DoodleFormer-译文#4.4 相关应用]]。

## 这篇在干什么

- DoodleFormer 生成创意简笔图像：画日常物体想象化、以前少见的组合，不是贴近真实概念的典型简笔。详见 [[DoodleFormer-译文#摘要]]。
- 框架由粗到细两阶段：先定粗构图，再补细线条。编码器用图感知 [[Transformer]]（GAT）。粗阶段用高斯混合模型（GMM）建模各部件位置和尺度。详见 [[DoodleFormer-译文#摘要]]。
- 常规简笔生成追求可辨认、典型画法。创意简笔常以外部随机初始笔画为条件，要拼出更少见的组合。详见 [[DoodleFormer-译文#1 引言]]。
- DoodlerGAN 按部件各训一个 GAN，再顺序拼图。没有显式保证相对位置，容易多头、断连，多样性也不够。详见 [[DoodleFormer-译文#1 引言]]。
- 作者模仿自然涂鸦：先画整体粗结构，再填细节。粗结构先定各部件位置与大小。详见 [[DoodleFormer-译文#1 引言]]。
- 贡献是两阶段编码–解码、GAT 编码器、GMM 粗解码器，以及文本条件、简笔补全、气泡图转户型。详见 [[DoodleFormer-译文#1.1 贡献]]。
- 标准简笔生成里有 SketchRNN、可微渲染、[[强化学习]] 等。创意简笔较新。DoodlerGAN 每个部件单独训练，推理顺序生成，开销大。详见 [[DoodleFormer-译文#2 相关工作]]。

## 方法要点

- 两个设计目标：显式建模整体部件排布，避免拓扑和连通出错；在框架里做显式概率建模，提高多样性。详见 [[DoodleFormer-译文#动机：两个设计目标]]。
- 第一阶段是 Part Locator（PL-Net）。条件是外部随机初始笔画点列。输出是各身体部件的包围盒，也就是粗结构。详见 [[DoodleFormer-译文#整体框架]]。
- 第二阶段是 Part Sketcher（PS-Net）。输入是 PL-Net 的框和初始笔画。输出是栅格化高质量简笔。详见 [[DoodleFormer-译文#整体框架]]。
- PL-Net 先框部件，PS-Net 再在框里填线。这是空间上由粗到细，不是课堂里的构图、结构、排线、收细。详见 [[DoodleFormer-译文#整体框架]]。
- Fig. 4 从单阶段基线起，逐步加入两阶段、GAT、GMM，展示先框后细如何改善构图和多样性。详见 [[DoodleFormer-译文#整体框架]]。
- GAT 编码器有两路：$E_b$ 编码各部件框和身份嵌入；$E_c$ 编码条件笔画。序列首加 cls token。详见 [[DoodleFormer-译文#图感知 Transformer 编码器]]。
- GAT 块在多头自注意力里融入邻接图上的谱图卷积。邻接图是静态、稀疏的局部连接；自注意力是动态全局关系。详见 [[DoodleFormer-译文#图感知 Transformer（GAT）块]]。
- 粗结构上，框与框重叠则连边。初始笔画上，同笔相邻点连边。详见 [[DoodleFormer-译文#图感知 Transformer（GAT）块]]。
- $E_c$ 的 cls 送入 Prior-Net。$E_b$ 与 $E_c$ 的 cls 送入 Recog-Net。训练时从变分分布采样潜变量，再送入粗解码器。详见 [[DoodleFormer-译文#图感知 Transformer（GAT）块]]。
- 概率粗解码器用 GMM 预测框的中心和宽高，不用确定性回归。另有头预测部件是否存在。详见 [[DoodleFormer-译文#概率粗简笔解码器]]。
- PL-Net 损失是重建项加 KL。重建含框的负对数似然和是否存在的交叉熵。$\lambda_{KL}=1$。详见 [[DoodleFormer-译文#PL-Net 损失]]。
- PS-Net 把框特征和光栅化后的初始笔画送进卷积编解码。Mask regressor 为每个框预测辅助 mask，做结构感知仿射。详见 [[DoodleFormer-译文#3.2 Part Sketcher Network（PS-Net）]]。
- 解码前给空间特征加零均值单位方差噪声。训练是标准 [[对抗与判别器]]：三个判别器对应图像级、部件级、外观。$\lambda_p=\lambda_a=10$。详见 [[DoodleFormer-译文#3.2 Part Sketcher Network（PS-Net）]]。
- 训练分两阶段。先用部件标注得到真值框，归一化到 $[0,1]$，训 PL-Net。再用光栅图和真值框训 PS-Net。详见 [[DoodleFormer-译文#实现细节]]。

## 数字要点

- Creative Birds 有 8067 张鸟创意简笔。Creative Creatures 有 9097 张各类生物。都带部件标注和自然语言短语。详见 [[DoodleFormer-译文#数据集]]。
- 每个 GAT 编码器 6 层、8 头。粗解码器里位置头和尺寸头各 3 层 attention。嵌入维 $d=512$。光栅 $128 \times 128$。batch 32，学习率 1e-4。详见 [[DoodleFormer-译文#实现细节]]。
- Tab. 1 上 Birds：DoodleFormer 的 [[FID]] 是 16.45，GD 是 18.33；DoodlerGAN 是 39.95 和 16.33。特征网在 QuickDraw3.8M 上训练。详见 [[DoodleFormer-译文#4.1 定量与定性对比]]。
- Creatures：DoodleFormer 的 FID 是 18.71，GD 是 16.89，SDS 是 1.78；DoodlerGAN 是 43.94、14.57、1.45。详见 [[DoodleFormer-译文#4.1 定量与定性对比]]。
- 贡献里写 Creatures / Birds 上 [[FID]] 绝对增益约 +25、+23。详见 [[DoodleFormer-译文#1.1 贡献]]。
- CS 衡量生成图被判成鸟或生物的比例。只画很典型的图，CS 也会高。Birds 上 DoodleFormer 的 CS 是 0.55，DoodlerGAN 是 0.69。详见 [[DoodleFormer-译文#4.1 定量与定性对比]]。
- 用户 study 100 人。Birds 上相对 DoodlerGAN，创意、更像人画、融合初始笔画约 82%、86%、85% 胜出。详见 [[DoodleFormer-译文#4.2 用户 study]]。
- 消融（Birds）：单阶段 baseline* 的 FID 是 46.45。两阶段无 GAT、无 GMM 是 20.14。加上 GAT 是 16.89。完整模型是 16.45。详见 [[DoodleFormer-译文#4.3 消融实验]]。
- 两阶段相对单阶段，FID 约 +26.3。GMM 使 GD 约 +1.17。完整模型相对单阶段 / 两阶段 baseline 的 FID 增益约 30.0 / 3.7。详见 [[DoodleFormer-译文#4.3 消融实验]]。
- GAT 设计（Birds）：纯 GCN 的 FID 是 27.45。标准 Transformer 是 20.14。Mesh Graphormer 式串联是 20.34。本文 GAT 是 16.45。详见 [[DoodleFormer-译文#4.3 消融实验]]。
- 文本→创意简笔用 80/20 划分。Birds 上 StackGAN 的 FID / GD 是 53.1 / 16.7，AttnGAN 是 45.2 / 16.5，DoodleFormer 是 18.5 / 17.3。详见 [[DoodleFormer-译文#4.4 相关应用]]。
- 简笔补全（Birds）：DoodlerGAN 的 FID / GD 是 44.2 / 15.1。DoodleFormer 是 18.3 / 17.8。详见 [[DoodleFormer-译文#4.4 相关应用]]。
- 评测从未见过的初始笔画随机采样，生成 10 000 张简笔，缩到 $64 \times 64$ 过 Inception，算 FID、GD、CS、SDS。详见 [[DoodleFormer-译文#2 评价指标补充说明]]。

## 对本课题

- 本课题要有顺序、有阶段、可中断的素描过程。这篇输出 [[栅格图]] 简笔终稿，属于成品创意简笔。详见 [[DoodleFormer-译文#5 结论]]。
- PL-Net 先框部件，PS-Net 再填线。两阶段是空间上先布局后画线，不是构图、结构、排线、收细那种课堂四阶段。详见 [[DoodleFormer-译文#整体框架]]。
- [[StrokeFusion]] 把这篇当栅格创意简笔对照：两阶段、部件级，输出是像素简笔，不是可编辑矢量。对照时应看部件框、GAT 和 GMM，不要把 PL/PS 读成绘画阶段扩散。详见 [[DoodleFormer-译文#3.2 Part Sketcher Network（PS-Net）]]。
- 补充图并排展示粗框和最终简笔。强调先 holistic 粗结构，再填细节。不要读成课堂示范的先后阶段。详见 [[DoodleFormer-译文#1 由粗到细的生成过程（Supp. Fig. 1）]]。
- FID、GD、CS、SDS 都评成品分布。CS 高也可能只是画得很典型。过程任务还要另报阶段和人看像不像示范。详见 [[DoodleFormer-译文#4.1 定量与定性对比]]。
- 可借鉴：大形阶段先落部件框，再在框里填线。框管「放哪」，线管「画形」。详见 [[DoodleFormer-译文#3.1 Part Locator Network（PL-Net）]]。
- 条件是随机初始笔画，不是成品线稿或课堂中间稿。续画、中途改不能照搬这套输入。详见 [[DoodleFormer-译文#1 引言]]。
- 扩展有文本条件、简笔补全、气泡图转户型。户型只用 PL-Net 出房间框。这些仍是布局或终稿，不是素描过程。详见 [[DoodleFormer-译文#4.4 相关应用]]。
- 成图偏鸟和生物的创意涂鸦，不是调子素描。不要混成「又做了一张更好看的成品」。详见 [[DoodleFormer-译文#摘要]]。
- GMM 采样多组框布局，外观、尺寸、朝向、姿态更分散。这是空间多样性，不是作画顺序多样性。详见 [[DoodleFormer-译文#3 更多定性结果]]。
