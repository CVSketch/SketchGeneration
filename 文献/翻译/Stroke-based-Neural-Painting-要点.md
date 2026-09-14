---
title: Stroke-based Neural Painting 要点
short: Stroke-based-Neural-Painting
source: "[[Stroke-based-Neural-Painting]]"
translation: "[[Stroke-based-Neural-Painting-译文]]"
tags:
  - 要点
  - 论文
status: 要点
---

# Stroke-based-Neural-Painting · 要点

对应笔记：[[Stroke-based-Neural-Painting]]。全文译文：[[Stroke-based-Neural-Painting-译文]]。原文：[[Stroke-based-Neural-Painting.pdf]]。

> [!tip] 怎么翻原文
> 每个要点都链到全文译文的某一节。点开就能对着中文读，不必再猜在 PDF 哪一页。

## 这篇在干什么

- 笔触式渲染用一组笔触重建目标图。旧方法测试时往往把画面均匀切块，块边界容易断笔、缺笔。本文提出 Compositional Neural Painter。详见 [[Stroke-based-Neural-Painting-译文#摘要]]。
- 从空白画布走若干步。每一步先由合成器按当前画布预测一块矩形区域，再由画家在该区内出笔，然后贴回画布。详见 [[Stroke-based-Neural-Painting-译文#摘要]]。
- 作者把作画拆成「涂哪里」和「涂什么」。合成器看整图选区，不绑检测器。这和 [[Intelli-Paint]] 的滑动注意力窗不同。详见 [[Stroke-based-Neural-Painting-译文#1 引言]]。
- 贡献三条：动态选区减轻边界伪影；合成器、画家、渲染器分开训；笔触风格迁移加上可微距离变换损失。详见 [[Stroke-based-Neural-Painting-译文#1 引言]]。
- [[强化学习]]、Paint Transformer、逐图优化都能出画。复杂图上仍难同时保住细节和块间一致。详见 [[Stroke-based-Neural-Painting-译文#2.1 笔触式渲染（SBR）]]。
- 笔触式风格迁移常保不住结构。本文用边缘图上的距离变换损失压结构。详见 [[Stroke-based-Neural-Painting-译文#2.2 笔触式风格迁移]]。

## 方法要点

- 三个模块：合成器预测矩形 $r$，画家在裁剪块上预测笔触 $s$，渲染器把 $s$ 画回画布。从空画布 $C^0$ 走 $T$ 步，得到 $I_r=C^T$。详见 [[Stroke-based-Neural-Painting-译文#3.1 概述]]。
- 第 $t$ 步顺序固定：先出区域 $r_t$，再出 $N$ 笔 $s_t$，再更新画布。这是空间分块，不是构图、结构、排线、收细。$T$ 是合成器与画家交替的作画步，不是扩散去噪步。详见 [[Stroke-based-Neural-Painting-译文#3.1 概述]]。
- 合成器输入目标图 $I$ 和当前画布 $C_t$，输出矩形 $r_t=(x,y,w,h)$。$(x,y)$ 是左上角，$w,h$ 是宽高。画家只在这块里落笔。详见 [[Stroke-based-Neural-Painting-译文#3.2 合成器网络：涂哪里？]]。
- 裁剪坐标要取整，反传不可微。先训渲染器和画家，再固定二者，用 DDPG 训合成器。详见 [[Stroke-based-Neural-Painting-译文#3.2 合成器网络：涂哪里？]]。
- 原始奖励看这一步像素差缩小多少。画布已经很接近目标时，奖励太小。分阶段奖励在 $d\le 0.005$ 时放大后期信号。这里的「分阶段」是强化学习的奖励调度，不是课堂四阶段。详见 [[Stroke-based-Neural-Painting-译文#3.2 合成器网络：涂哪里？]]。
- 画家用 CNN，不用强化学习。只靠像素损失会反复画同类粗笔。加上 WGAN 判别器，惩罚和见过的笔触太像。详见 [[Stroke-based-Neural-Painting-译文#3.3 画家网络：涂什么？]]。
- 画布用蒙版混合更新。总损失是像素项加对抗项。对抗权重 $\gamma$ 按两项范数自适应。真样本甚至可以是随机噪声。详见 [[Stroke-based-Neural-Painting-译文#3.3 画家网络：涂什么？]]。
- 渲染器把参数 $s=\{x,y,w,h,\theta,r,g,b\}$ 变成笔触图和蒙版。笔形用真实油画笔触，网络渲染可微。详见 [[Stroke-based-Neural-Painting-译文#3.4 笔触渲染器]]。
- 风格迁移时，渲染器再出边缘图 $E_s$。可微距离变换损失让 $E_s$ 贴近输入边缘。先得到笔触，再优化笔触参数。详见 [[Stroke-based-Neural-Painting-译文#3.5 笔触式风格迁移]]。
- 没有合成器就缺细节。测试时均匀分块会再现边界伪影。去掉对抗损失会反复画大块相似笔触。详见 [[Stroke-based-Neural-Painting-译文#4.5 消融实验]]。
- 去掉距离变换损失后，模糊边缘丢结构，直线轮廓会抖。加上后两种情形都更稳。详见 [[Stroke-based-Neural-Painting-译文#C 距离变换损失消融（p10，Fig. 7）]]。
- 和 [[Intelli-Paint]] 比：对方一个网络同时预测窗和笔，还依赖检测。本文合成器选区、画家出笔，分开训练，不绑检测器。详见 [[Stroke-based-Neural-Painting-译文#G 与 Intelli-Paint 对比（p10–11，Fig. 9–10）]]。

## 关键图

- Fig. 1：右栏 (a) 均匀 $k \times k$ 分块测试 vs (b) 动态矩形、裁剪出笔、贴回迭代。图在 [[Stroke-based-Neural-Painting-译文#1 引言]]。
- Fig. 2：3000 笔触悉尼歌剧院；三基线在块边界断笔/缺笔，Ours 无网格缝。图在 [[Stroke-based-Neural-Painting-译文#1 引言]]。
- Fig. 3：合成器 + 画家 + 渲染器总图；$I,C_t \to r_t \to s_t \to C_{t+1}$，$T$ 步。图在 [[Stroke-based-Neural-Painting-译文#3.1 概述]]。
- Fig. 4：笔数档是 $200$ / $1000$ / $3000$ / $5000$（这张图没有 $500$ 那一行），两场景 × 八方法定性大网格（整图一张）。图在 [[Stroke-based-Neural-Painting-译文#4.3 图像到绘画（重建）]]。
- Table 1：ImageNet / CelebA-HQ 定量对比。档是 $200$ / $500$ / $1000$ / $3000$ / $5000$ 笔，不是构图、结构、排线、收细。图在 [[Stroke-based-Neural-Painting-译文#4.3 图像到绘画（重建）]]。
- Fig. 5：1000 / 2000 笔触风格迁移六组；Ours vs [30]、[17]。图在 [[Stroke-based-Neural-Painting-译文#4.4 笔触式风格迁移]]。
- Fig. 6：合成器/画家消融八格；(c) 均匀分块边界伪影，(e) 无对抗损失重复粗笔。图在 [[Stroke-based-Neural-Painting-译文#4.5 消融实验]]。
- Table 2：ImageNet 消融数值（1000 / 5000 笔触）。图在 [[Stroke-based-Neural-Painting-译文#4.5 消融实验]]。

## 数字要点

- 主要在 CelebA-HQ 和 ImageNet 上训和测。各随机抽 1 000 张测试，其余训练。指标是 L2、PSNR、[[LPIPS]]。详见 [[Stroke-based-Neural-Painting-译文#4.1 数据集与设置]]。
- 三步训练：渲染器 100 万 iter，画家 200 万 iter，合成器 200 万 iter。batch 都是 32。详见 [[Stroke-based-Neural-Painting-译文#4.2 训练细节]]。
- 重建实验笔触数取 200、500、1 000、3 000、5 000。逐图优化的三篇只抽 100 张评 Table 1。详见 [[Stroke-based-Neural-Painting-译文#4.3 图像到绘画（重建）]]。
- 风格迁移对比用 1 000、2 000 笔。Gram 矩阵距离：本文平均 0.6943，Stylized Neural Painting 为 1.4048，Parameterized Brushstrokes 为 4.0274。详见 [[Stroke-based-Neural-Painting-译文#4.4 笔触式风格迁移]]。
- 分阶段奖励阈值 $d=0.005$。实现里给 $f$ 的分母加 $\epsilon=10^{-6}$。详见 [[Stroke-based-Neural-Painting-译文#3.2 合成器网络：涂哪里？]]。
- 距离变换的 soft-min 取 $\lambda=0.3$。详见 [[Stroke-based-Neural-Painting-译文#3.5 笔触式风格迁移]]。
- 附录可视化展示 5、100、500、2 000、5 000 笔时的中间结果。早期先抓轮廓，再补细节。详见 [[Stroke-based-Neural-Painting-译文#D 作画过程可视化（p11–12，Fig. 8）]]。
- 用户研究：30 名志愿者排序。65% 的样本把本文排第一。平均名次 1.56。详见 [[Stroke-based-Neural-Painting-译文#E 用户研究（p11，Table 3）]]。
- 只比画家、200 笔、不做分块时，本文 L2 / [[LPIPS]] 最低、PSNR 最高。详见 [[Stroke-based-Neural-Painting-译文#F 画家网络对比（p11，Table 4）]]。
- [[Intelli-Paint]] 长序列困难，例如超过 2 000 笔。本文把选区和出笔分开训，笔触数可以更大。详见 [[Stroke-based-Neural-Painting-译文#G 与 Intelli-Paint 对比（p10–11，Fig. 9–10）]]。
- 用透明圆点笔触重训后，ImageNet 与 CelebA-HQ 各 1 000 张，笔触数 200–5 000，三项指标仍优于 [[Learning-to-Paint]] 和 Semantic+RL。详见 [[Stroke-based-Neural-Painting-译文#H 更多绘画对比（p14 起，Fig. 11–14）]]。
- 风格化补充对比：本文与 Stylized Neural Painting 用 2 000 笔，Parameterized Brushstrokes 用 10 000+ 笔。详见 [[Stroke-based-Neural-Painting-译文#I 更多笔触风格化（p13、18，Fig. 15）]]。
- 单卡 24G RTX 3090，1 000 笔、CelebA-HQ 100 张平均推理：[[Learning-to-Paint]] 为 0.2066 s，本文为 0.2162 s，Paint Transformer 为 0.3725 s，Semantic+RL 为 2.9921 s，Im2Oil 为 55.37 s，Stylized Neural Painting 为 124.94 s，Parameterized Brushstrokes 为 247.61 s。详见 [[Stroke-based-Neural-Painting-译文#J 时间复杂度（p13，Table 6）]]。

## 对本课题

- [[Inverse-Painting]]、[[ProcessPainter]] 拿它当笔画渲染基线。对照的是成品重建和中间帧，不是素描课里的构图、结构、排线、收细。详见 [[Stroke-based-Neural-Painting-译文#摘要]]。
- 这篇每一步先预测矩形，再在矩形里出笔。这是空间分块，不是课堂四阶段。详见 [[Stroke-based-Neural-Painting-译文#3.1 概述]]。
- $T$ 是合成器与画家交替的次数。总笔触数由每步笔数和步数一起决定。不要把 $T$ 写成扩散去噪步。详见 [[Stroke-based-Neural-Painting-译文#3.1 概述]]。
- 奖励和损失都在缩小画布和目标图的差。这不能当本课题的过程主损失。详见 [[Stroke-based-Neural-Painting-译文#3.2 合成器网络：涂哪里？]]。
- 输出是油画笔触铺色，用来重建照片。本课题要的是有顺序、有阶段、可中断的素描过程。详见 [[Stroke-based-Neural-Painting-译文#5 结论]]。
- 可借鉴「先定这一片，再在片里落笔」。不要把动态矩形当成课堂阶段名。详见 [[Stroke-based-Neural-Painting-译文#1 引言]]。
- 均匀 $k\times k$ 分块会在边界断笔。动态区域能减轻伪影，目的仍是贴原图，不是给课堂示范「一次只改一块」。详见 [[Stroke-based-Neural-Painting-译文#4.5 消融实验]]。
- 5 笔到 5 000 笔的中间结果是由粗到细的重建轨迹，不是教学顺序。详见 [[Stroke-based-Neural-Painting-译文#D 作画过程可视化（p11–12，Fig. 8）]]。
