---
title: Learning to Paint 要点
short: Learning-to-Paint
source: "[[Learning-to-Paint]]"
translation: "[[Learning-to-Paint-译文]]"
tags:
  - 要点
  - 论文
status: 要点
---

# Learning-to-Paint · 要点

对应笔记：[[Learning-to-Paint]]。全文译文：[[Learning-to-Paint-译文]]。原文：[[Learning-to-Paint.pdf]]。

> [!tip] 怎么翻原文
> 每个要点都链到全文译文的某一节。点开就能对着中文读，不必再猜在 PDF 哪一页。

## 这篇在干什么

- 教智能体按顺序落笔，用很少几笔贴近目标图。训练不需要画家经验，也不需要笔画追踪。详见 [[Learning-to-Paint-译文#摘要]]。
- 要画人脸和自然场景这类纹理密、结构复杂的图。没有「这一步该画哪一笔」的标准答案，所以不用逐步监督。改用强化学习，看整段作画的累计奖励。详见 [[Learning-to-Paint-译文#1 引言]]。
- 作画写成序列决策：看当前画布和目标图，预测下一笔。目标是给定笔数全部画完后的累计奖励最大，而不是只贪当前这一笔。详见 [[Learning-to-Paint-译文#3.1 概述]]。
- 学成后把目标图拆成有序笔画，在画布上重建。作者说智能体可预测几百甚至几千笔。详见 [[Learning-to-Paint-译文#6 结论]]。

## 方法要点

- 状态写成 $s_t = (C_t, I, t)$：当前画布、目标图、步数。步数告诉智能体还剩几步。动作是连续参数，控制位置、形状、颜色、透明度。详见 [[Learning-to-Paint-译文#3.2 模型]]。
- 一步奖励是 $r(s_t, a_t) = L_t - L_{t+1}$。$L$ 量当前画布和目标差多远。整局最大化折扣未来奖励之和 $R_t$，不能只看眼前一笔。详见 [[Learning-to-Paint-译文#3.2 模型]]。
- 没有标准笔序，所以不用逐步监督。作者用基于模型的 DDPG：可微渲染器显式建模环境，演员能拿到环境梯度。详见 [[Learning-to-Paint-译文#3.3 学习]]。
- Action Bundle：智能体每步预测 $k$ 笔，渲染器按顺序画上。实验里 $k = 5$ 较好。折扣从 $\gamma$ 改成 $\gamma^k$。这是一次决策覆盖多笔，不是课堂里的作画阶段。详见 [[Learning-to-Paint-译文#3.3.2 动作束]]。
- 度量损失 $L$ 用 WGAN 判别器分数，不用死板的 $\ell_2$。假样本是作品与目标配对，真样本是两张相同目标图。作者说判别器奖励比直接用 $\ell_2$ 更好。详见 [[Learning-to-Paint-译文#3.3.3 WGAN 奖励]]。
- 默认笔画是带粗细的二次 Bézier 曲线。一步参数是 13 维：三个控制点、两端粗细与透明度、RGB。详见 [[Learning-to-Paint-译文#4.2 笔画设计]]。
- 神经渲染器把笔画参数映射成笔画图像，可微，能端到端训练。训练样本用图形学程序随机生成。详见 [[Learning-to-Paint-译文#4.1 神经渲染器]]。

## 数字要点

- 四个数据集的笔画数：MNIST 为 5 笔，SVHN 为 40 笔，CelebA 为 200 笔，ImageNet 为 400 笔。详见 [[Learning-to-Paint-译文#5.1 数据集]]。
- MNIST 共 7 万张，官方划分 6 万训练、1 万测试，每张 $28 \times 28$。SVHN 随机抽 20 万张，每张 $32 \times 32$。CelebA 约 20 万张。ImageNet 随机抽 20 万张，覆盖 1000 类。除 MNIST 外，其余各划 2000 张作测试。详见 [[Learning-to-Paint-译文#5.1 数据集]]。
- 喂给智能体前，图缩到 $128 \times 128$。每束 5 笔时，2.2 GHz Intel Core i7 画 200 笔约 2.1 秒；NVIDIA 2080Ti 约 9.5 倍加速。画一束时，演员约 554 MFLOPs，渲染器约 217 MFLOPs。详见 [[Learning-to-Paint-译文#5.2 训练]]。
- ImageNet 与 CelebA 训 $2 \times 10^5$ 个 mini-batch；SVHN 为 $10^5$；MNIST 为 $2 \times 10^4$。Adam，batch 大小 96，单 GPU。ImageNet 与 CelebA 约 40 小时，SVHN 约 20 小时，MNIST 约 2 小时。神经渲染器约 5 到 15 小时。回放池存最近 800 个 episode。详见 [[Learning-to-Paint-译文#5.2 训练]]。
- 和 SPIRAL 比：同样在 CelebA 上训 20 笔、用不透明笔画。这篇的 $\ell_2$ 距离约为 SPIRAL 的三分之一。详见 [[Learning-to-Paint-译文#5.3 结果]]。
- 基于模型的 DDPG，测试 $\ell_2$ 比 PatchQ 版约小 5 倍，比原版 DDPG 约小 20 倍。CelebA 上另训了 100、200、400、1000 笔的智能体。每步预测 5 笔时测试损失最好。详见 [[Learning-to-Paint-译文#5.4 消融研究]]。
- 附录默认超参数：每束 5 笔，每 episode 40 步，回放池 800 个 episode，折扣因子 0.955。演员输出 $5 \times 13$。渲染器输入 13 维，输出 $128 \times 128$ 笔画。详见 [[Learning-to-Paint-译文#7.1 架构]]。

## 关键图

- Fig. 1：四行过程——MNIST 数字、SVHN 门牌、CelebA 人脸、ImageNet 鸟；每行左列目标，右列若干步中间画布，色块由粗到细。图在 [[Learning-to-Paint-译文#摘要]]。
- Fig. 2：（a）推理：Policy → Action → Renderer，Canvas 从空到叠色块；（b）训练：回放池采样 → Actor / Renderer → 判别器给 Reward、评论家出 $Q(s,a)$。图在 [[Learning-to-Paint-译文#3.1 概述]]。
- Fig. 4：（a）原版 DDPG：Canvas、Target、Action 一起进 Critic，出 $Q(s,a)$；（b）基于模型：Renderer 先出 Rendered Image，再与 Target 比，出 $V(s)$ 与 Reward。图在 [[Learning-to-Paint-译文#3.3 学习]]。
- Fig. 3（PDF 第 3 页，库内未裁）：四数据集成对对比，笔画数 5 / 40 / 200 / 400。译文文字见 [[Learning-to-Paint-译文#5.3 结果]]。

## 对本课题

- 本课题是 Inverse Sketching：由成品线稿反推有顺序、有阶段、可中断的素描过程。这篇的奖励是「画布更接近目标图」，来自像素差或判别器分数。这种像素奖励不能当过程主损失。详见 [[Learning-to-Paint-译文#3.2 模型]]。
- WGAN 奖励只说明作品更像目标照片。消融里它连测试 $\ell_2$ 也比直接用 $\ell_2$ 奖励更低。评价仍停在贴原图，不能当成过程主损失。详见 [[Learning-to-Paint-译文#5.4 消融研究]]。
- Action Bundle 是强化学习里的 frame skip：一次预测 $k$ 笔再按序画上。$k = 5$ 是训练技巧，不是构图、结构、排线这类课堂阶段。详见 [[Learning-to-Paint-译文#3.3.2 动作束]]。
- 消融也写了：束里笔画增多，决策轮次变少，有助于长期规划。这仍是一次决策预测多笔，不是人类分阶段作画。详见 [[Learning-to-Paint-译文#5.4 消融研究]]。
- 作者写智能体倾向由粗到细。这是为贴紧目标照片规划出的落笔顺序，不是课堂里教的起稿阶段。详见 [[Learning-to-Paint-译文#1 引言]]。
- 训练不需要真人笔序，开题可以先做到「能出有序笔画」。这也是上限：学不到人从后往前、一次一块区域的画法。详见 [[Learning-to-Paint-译文#摘要]]。
- 可借鉴两处：状态里加入剩余步数；可微渲染让笔画参数能反传。动作应定义成当前阶段允许的那类线，不要定义成任意位置下一笔。详见 [[Learning-to-Paint-译文#3.2 模型]]。
- 笔画数 5 / 40 / 200 / 400 只说明多画几笔更能贴原图。素描若只用同样奖励，模型容易在五官上堆线，大关系反而乱。详见 [[Learning-to-Paint-译文#5.3 结果]]。
