---
title: Inverse Painting 要点
short: Inverse-Painting
source: "[[Inverse-Painting]]"
translation: "[[Inverse-Painting-译文]]"
tags:
  - 要点
  - 论文
status: 要点
---

# Inverse-Painting · 要点

对应笔记：[[Inverse-Painting]]。全文译文：[[Inverse-Painting-译文]]。原文：[[Inverse-Painting.pdf]]。

> [!tip] 怎么翻原文
> 每个要点都链到全文译文的某一节。点开就能对着中文读，不必再猜在 PDF 哪一页。

## 这篇在干什么

- 给定一张成品画，模型重建一段延时关键帧，用来展示它可能是怎样画出来的。详见 [[Inverse-Painting-译文#摘要]]。
- 结果只是说得通的过程，不能当成某张画的真实考古复原。详见 [[Inverse-Painting-译文#1 引言]]。
- 作者把任务写成[[自回归]]图像生成：从空白画布一步步更新，直到接近成品。详见 [[Inverse-Painting-译文#3 方法]]。
- 以往工作多靠手工规则；最接近的 Timecraft 只在 $50\times 50$ 的小块上做。详见 [[Inverse-Painting-译文#1 引言]]。
- 方法先出文字指令和区域[[掩码]]，再用[[扩散模型]]渲染器更新画布。详见 [[Inverse-Painting-译文#1 引言]]。
- 训练数据主要是丙烯风景，但仍能外推到多种画风。详见 [[Inverse-Painting-译文#摘要]]。

## 方法要点

- 输入目标画 $I_T$，从空白 $I_0$ 生成 $T$ 个关键帧；相邻帧对应固定时间间隔。详见 [[Inverse-Painting-译文#3 方法]]。
- 单阶段渲染器以 Stable Diffusion 的去噪 UNet 为主，并用 ReferenceNet 注入目标画。详见 [[Inverse-Painting-译文#3.1 单阶段画布渲染]]。
- 加上时间间隔后，每步加多少更可控，但山仍会分层错乱，所以还要语义指令。详见 [[Inverse-Painting-译文#3.1 单阶段画布渲染]]。
- 文本指令生成器采用 LLaVA，把目标画和当前画左右拼接，输出短指令。详见 [[Inverse-Painting-译文#3.2.1 文本指令生成器]]。
- [[掩码]]指令生成器同时看当前画、目标画、文本、未完成差异和时间；差异阈值 $\alpha=0.2$，损失是二值交叉熵。详见 [[Inverse-Painting-译文#3.2.2 掩码指令生成器]]。
- 画布渲染在单阶段模型上接入文本和[[掩码]]；训练时用真值指令。详见 [[Inverse-Painting-译文#3.3 训练：画布渲染]]。
- 下一帧 CLIP 预测能补上「mountain」这类词没说清的粗铺或收细。详见 [[Inverse-Painting-译文#3.3 训练：画布渲染]]。
- 测试时固定间隔[[自回归]]：先文本、再[[掩码]]、再去噪；更新极小则停。详见 [[Inverse-Painting-译文#3.4 测试时生成]]。
- 测试用 $S=25$ 步去噪；相邻两步感知距离都小于 $10^{-3}$ 就停止。详见 [[Inverse-Painting-译文#A.4 测试时生成]]。

## 数字要点

- 收集 294 段丙烯风景录像，265 幅训练、29 幅验证；训练 7261 对、验证 783 对。详见 [[Inverse-Painting-译文#数据集]]。
- 每幅平均 27 帧；训练相邻间隔约 23.6 秒，验证约 22.0 秒；测试默认 20 秒。详见 [[Inverse-Painting-译文#数据集]]。
- Table 1 完整模型：[[LPIPS]] 0.364、[[IoU]] 0.418、DDC 32.66、DTS 1.273、[[FID]] 150.6，优于三个基线。详见 [[Inverse-Painting-译文#与基线对比]]。
- 三基线 Table 1 的 [[LPIPS]] / [[IoU]]：Paint Transformer 为 0.643 / 0.104，Timecraft 为 0.602 / 0.251，SVD 为 0.500 / 0.197。详见 [[Inverse-Painting-译文#与基线对比]]。
- 去掉[[掩码]]生成器后 [[IoU]] 降到 0.175；去掉文本与[[掩码]]后只有 0.128 或 0.139。详见 [[Inverse-Painting-译文#消融实验]]。
- 测试间隔改成 10 秒、30 秒时，[[IoU]] 为 0.349、0.353，都低于默认 20 秒的 0.418。详见 [[Inverse-Painting-译文#消融实验]]。
- 用户研究未归一化均分：本文 4.21，SVD 2.96，Paint Transformer 2.11，Timecraft 1.52；归一化后约为 SVD 的 1.9 倍。详见 [[Inverse-Painting-译文#用户研究]]。
- 文本：不考虑顺序时，91% 生成词落在真值集合里；单步一致率 72%，预训练 LLaVA 只有 31%。详见 [[Inverse-Painting-译文#文本与掩码生成器分析]]。
- [[掩码]]：真值文本下 [[IoU]] 0.70，预测文本 0.64，无文本条件 0.58，预训练分割只有 0.42。详见 [[Inverse-Painting-译文#文本与掩码生成器分析]]。
- 用预测文本与[[掩码]]训练渲染器，[[LPIPS]] 从 0.364 升到 0.438，过程更不像真录像。详见 [[Inverse-Painting-译文#误差累积]]。
- 裁剪画上本文 [[LPIPS]] 0.452、[[IoU]] 0.296，Timecraft 为 0.647、0.165。详见 [[Inverse-Painting-译文#与基线对比]]。

## 关键图

- Fig. 1：上下两例，左列「Input」成品，右列各 10 张关键帧；上排丙烯山景，下排梵高风夜景，从大块铺色到细节。图在 [[Inverse-Painting-译文#摘要]]。
- Fig. 3：左「Training: Instruction Generation」（文本 $g_{\text{text}}$、[[掩码]] $g_{\text{mask}}$），中「Canvas Rendering」（ReferenceNet、去噪 UNet），右测试步 $t-1$ 的绿/橙/蓝闭环。图在 [[Inverse-Painting-译文#3 方法]]。
- Fig. 5：(a) 当前画布与目标 inset；(b) 仅 CLIP 预测改动过大；(c) 加时间仍分层乱；(d) 无[[掩码]]一次画整山；(e) 无文本时绿湖先出现（红箭头）；(f) 完整条件。图在 [[Inverse-Painting-译文#3.1 单阶段画布渲染]]。
- Fig. 8：五行七列过程对比（GT、Ours、SVD、Paint Transformer、Timecraft）；SVD 天空红块与湖心竖条伪影（红箭头），Paint Transformer 块状笔触，Timecraft 全程发糊。图在 [[Inverse-Painting-译文#与基线对比]]。
- Fig. 11：四格人像过程，从糊色块到无五官头肩，最后一格才有清晰五官，说明仅风景训练时人像不自然。图在 [[Inverse-Painting-译文#局限与未来工作]]。

## 对本课题

- 导演和画工分开是骨干：先定「画什么、画哪里」，再让[[扩散模型]]改画布。详见 [[Inverse-Painting-译文#1 引言]]。
- 本课题也是过程生成，不是只出最后一张图；这篇对应「反过程」，对象却是油画铺色。详见 [[Inverse-Painting-译文#摘要]]。
- 不要直接把油画权重搬到线稿：训练全是风景丙烯，人像过程已经不自然。详见 [[Inverse-Painting-译文#局限与未来工作]]。
- 评价不要只报 [[FID]]。至少还要看改动区域 [[IoU]]、和真过程比的 [[LPIPS]]，以及人评是否像在画。详见 [[Inverse-Painting-译文#评价指标]]。
- 中间结果是栅格关键帧，不能当可擦的线。本课题若对标这篇，贡献应写在线和阶段上。详见 [[Inverse-Painting-译文#3 方法]]。
- 指令生成不必上英文大模型。课堂四个阶段名就能顶这里的短指令。详见 [[Inverse-Painting-译文#3.2.1 文本指令生成器]]。
- 训练给真值指令、测试用预测指令，方便单独做[[消融]]。详见 [[Inverse-Painting-译文#误差累积]]。
- 模型会涌现从后往前的分层，这是风景铺色习惯，不能当成素描课堂顺序。详见 [[Inverse-Painting-译文#涌现性质]]。
