---
title: Stylized Neural Painting 要点
short: Stylized-Neural-Painting
source: "[[Stylized-Neural-Painting]]"
translation: "[[Stylized-Neural-Painting-译文]]"
tags:
  - 要点
  - 论文
status: 要点
---

# Stylized-Neural-Painting · 要点

对应笔记：[[Stylized-Neural-Painting]]。全文译文：[[Stylized-Neural-Painting-译文]]。原文：[[Stylized-Neural-Painting.pdf]]。

> [!tip] 怎么翻原文
> 每个要点都链到全文译文的某一节。点开就能对着中文读，不必再猜在 PDF 哪一页。

## 这篇在干什么

- 输入一张照片，输出一串有物理含义的矢量笔触，再渲染成画。最后一张很像画。中间各步是在参数空间里搜索，不是逐像素填图。详见 [[Stylized-Neural-Painting-译文#摘要]]。
- 作者不直接改输出图的每个像素，而是按顺序生成矢量化笔触。这些矢量可在任意分辨率下重渲染。方法支持油画笔、水彩、马克笔、色块胶带，也可按参考图做风格迁移。详见 [[Stylized-Neural-Painting-译文#1 引言]]。
- 和逐步贪心、RNN、强化学习不同，本文把预测下一笔写成自监督的参数搜索。目标是让渲染画布尽量像输入照片。详见 [[Stylized-Neural-Painting-译文#1 引言]]。
- 贡献有三条：笔触预测等于参数搜索，并可和风格迁移联合优化；用可微传输损失缓解零梯度；用栅格化加着色的双通路渲染，分开形状和颜色。详见 [[Stylized-Neural-Painting-译文#1 引言]]。
- 作者在人像、动物、风景、日常物体、艺术摄影、卡通上试过。成图的写实感和艺术感都较强。详见 [[Stylized-Neural-Painting-译文#1 引言]]。

## 方法要点

- 方法分三块：神经渲染器由矢量参数生成单笔前景和 alpha；笔触混合器把多笔叠到画布上；相似度模块约束最终画布去重建输入图。详见 [[Stylized-Neural-Painting-译文#3 方法]]。
- 从空白画布 $h_0$ 出发，逐步叠加笔触。第 $t$ 步，渲染器 $G$ 读入参数 $x_t$，输出前景 $s_t$ 和 alpha $\alpha_t$，再用软混合保证可微。$T$ 是优化迭代步，不是画家教学里的阶段名。中间 $h_t$ 只要损失下降即可，不必像真人起稿。详见 [[Stylized-Neural-Painting-译文#3.1 概述]]。
- 渲染器拆成着色网络 $G_s$ 和栅格化网络 $G_r$。$G_s$ 同时读颜色和形状，出前景色。$G_r$ 忽略颜色，只出清晰轮廓。训练时用 $\ell_2$ 去回归矢量引擎的真值。详见 [[Stylized-Neural-Painting-译文#3.2 解耦的神经渲染]]。
- 笔触和目标区域没有重叠时，像素 $\ell_1$ / $\ell_2$ 对位置参数的梯度是 0。只靠像素损失，笔几乎推不动。详见 [[Stylized-Neural-Painting-译文#3.3 像素相似度与零梯度问题]]。
- 作者把画布和参考图的相似度写成最小搬运代价，用带熵正则的 Sinkhorn 距离。总损失是 $\ell_1$ 和传输项加权相加。详见 [[Stylized-Neural-Painting-译文#3.4 笔触搜索的最优传输]]。
- 同一套搜索里可以加上 Gatys 式风格损失。风格项用 VGG-19 特征 Gram 矩阵的平方误差，在第 2、4、7 个卷积层上计算。详见 [[Stylized-Neural-Painting-译文#3.5 与神经风格迁移联合优化]]。
- 为了补细节，先在整幅 $128 \times 128$ 上搜索，再把画布划成有重叠的 $m \times m$ 块，逐块搜索。每块同时优化活跃笔触集。详见 [[Stylized-Neural-Painting-译文#3.6 实现细节]]。
- 四类笔刷的参数个数不同。油画笔像带固定纹理的色块矩形。马克笔和水彩走二次 Bézier。色块胶带是可旋转的实心矩形。详见 [[Stylized-Neural-Painting-译文#6.2 笔触参数化（Table 4–7）]]。

## 数字要点

- 渲染器训练用 Adam，batch 大小 64，学习率 $2 \times 10^{-4}$，$\beta = (0.9, 0.999)$。一共 400 epoch，每 100 epoch 学习率乘 $1/10$。每个 epoch 随机生成 $50\,000 \times 64$ 条笔触真值，输出分辨率 $128 \times 128$。每种笔刷单独训练一份渲染器。详见 [[Stylized-Neural-Painting-译文#3.6 实现细节]]。
- 笔触搜索取 $\beta_{\ell_1} = 1.0$，$\beta_{\mathrm{ot}} = 0.1$。梯度下降用 RMSprop，学习率 $\mu = 0.01$。做风格迁移时 $\beta_{\mathrm{sty}} = 0.5$，更新 200 步。计算 Sinkhorn 前把画布缩到 $48 \times 48$，熵系数 $\epsilon = 0.01$，迭代 5 步。每块跑 $20 \times N$ 步，$N$ 是该块的笔触数。详见 [[Stylized-Neural-Painting-译文#3.6 实现细节]]。
- 和 [[Learning-to-Paint]] 对比时，两边都用 400 笔。SPIRAL 用 20 笔。本文笔刷纹理更清晰，另外两篇更容易糊。详见 [[Stylized-Neural-Painting-译文#4.2 与其他方法对比]]。
- Table 1 报验证集上前景加 alpha 的平均 PSNR。双通路：油画笔 26.982，水彩 31.389。只留栅格化：24.015 / 25.769。只留着色：26.048 / 29.045。详见 [[Stylized-Neural-Painting-译文#4.3 受控实验]]。
- 着色网络有六层转置卷积，通道依次为 512→512→256→128→64→3，空间尺寸从 $4 \times 4$ 上采样到 $128 \times 128 \times 3$。栅格化网络先经四层全连接 512→1024→2048→4096，再看成 $16 \times 16 \times 16$，最后得到 $128 \times 128 \times 1$ 轮廓。详见 [[Stylized-Neural-Painting-译文#6.1 神经渲染器结构（Table 2、Table 3）]]。
- 每笔参数个数：油画笔 $N = 11$，马克笔 $N = 10$，水彩 $N = 15$，色块胶带 $N = 8$。色块胶带的旋转角 $\theta \in [0, 180°]$。详见 [[Stylized-Neural-Painting-译文#6.2 笔触参数化（Table 4–7）]]。
- 矢量结果可以重渲染到 $1024 \times 1024$。放大不依赖训练时的分辨率。详见 [[Stylized-Neural-Painting-译文#6.3 高分辨率结果]]。

## 关键图

- Fig. 1：上排四种笔刷（油画、马克笔、水彩等）照片转绘画；下排输入→成画→风格迁移（只迁颜色 / 迁颜色与纹理）。图在 [[Stylized-Neural-Painting-译文#1 引言]]。
- Fig. 2：空白画布 $h_0$ 起逐步软混合叠加笔触，最终 $h_T$ 与参考 $\hat{h}$ 算损失并反传优化参数 $x_t$；$t$ 是优化步。图在 [[Stylized-Neural-Painting-译文#3.1 概述]]。
- Fig. 3：双通路渲染——着色 $G_s$ 读颜色+形状，栅格化 $G_r$ 只读形状，相乘得前景 $s$ 与 alpha $\alpha$。图在 [[Stylized-Neural-Painting-译文#3.2 解耦的神经渲染]]。
- Fig. 4：两方块不重叠时像素 $\ell_1$ 对位移 $s$ 梯度为 0；传输（W 距离）梯度仍非零。图在 [[Stylized-Neural-Painting-译文#3.3 像素相似度与零梯度问题]]。
- Fig. 5：同一笔从初态「推」向目标：仅像素 $\ell_1$ 时笔几乎消失；加传输损失则能到位。图在 [[Stylized-Neural-Painting-译文#3.3 像素相似度与零梯度问题]]。
- Fig. 6：(a) 像素距离逐格对齐；(b) 传输距离允许跨位置搬质量。图在 [[Stylized-Neural-Painting-译文#3.4 笔触搜索的最优传输]]。
- Fig. 7：马克笔/油画笔逐笔重建与损失曲线；$m$ 为笔触数（搜索进度），不是课堂阶段。图在 [[Stylized-Neural-Painting-译文#4.1 风格化绘画生成]]。
- Fig. 8：(a)–(c) 油画、马克笔、水彩成图；(d) 极少笔触色块胶带抽象角色。图在 [[Stylized-Neural-Painting-译文#4.1 风格化绘画生成]]。
- Fig. 9：风格迁移——上行迁颜色+纹理，下行只迁颜色。图在 [[Stylized-Neural-Painting-译文#4.1 风格化绘画生成]]。
- Fig. 10：与 Huang 等、SPIRAL、本文对比；右半与手工低多边形风（Adam Lister）对照。图在 [[Stylized-Neural-Painting-译文#4.2 与其他方法对比]]。
- Fig. 11：纽约艺术家 Adam Lister 手工块面画与本文自动块面结果并列。图在 [[Stylized-Neural-Painting-译文#4.2 与其他方法对比]]。
- Fig. 12：MNIST 数字与向日葵上，仅 $\ell_1$ 缺笔划/发糊，$\ell_1$+传输更完整。图在 [[Stylized-Neural-Painting-译文#4.3 受控实验]]。
- Fig. 13：DCGAN-G、UNet、PxlShuffleNet 与本文渲染器验证 PSNR 曲线。图在 [[Stylized-Neural-Painting-译文#4.3 受控实验]]。
- Fig. 14：单笔 GT 与 UNet / DCGAN-G / PxlShuffleNet / 本文可视化；本文最接近 GT。图在 [[Stylized-Neural-Painting-译文#4.3 受控实验]]。
- Table 1：栅格化 only、着色 only、双通路完整渲染在油画笔/水彩上的验证 PSNR 消融。图在 [[Stylized-Neural-Painting-译文#4.3 受控实验]]。

## 对本课题

- 本课题要输出有顺序、有阶段、可中断的素描过程，不是只出最后一张更好看的图。本文最后一张像画；中间 $h_t$ 是优化步，只要损失下降即可，不必像真人课堂上的起稿。详见 [[Stylized-Neural-Painting-译文#3.1 概述]]。
- [[Inverse-Painting]] 和 [[ProcessPainter]] 在笔画渲染基线里常拿本文当对照。ProcessPainter 造伪过程时，也会用本文生成过程帧。能当基线，是因为最后一张像画、输出是矢量笔触。中间步仍是优化轨迹。详见 [[Stylized-Neural-Painting-译文#摘要]]。
- Fig. 7 的逐笔结果是损失驱动的由粗到细：先抓住物体整体，再补细部。这不等于人类固定的教学顺序。详见 [[Stylized-Neural-Painting-译文#4.1 风格化绘画生成]]。
- 和 [[Learning-to-Paint]] 动机相近，都追求笔触式写实绘画。对方用强化学习出笔，本文用参数搜索。两边都没有真人笔序当监督。详见 [[Stylized-Neural-Painting-译文#2 相关工作]]。
- 像素损失和传输损失都在缩小画布和照片的差。这种损失不能当过程主损失。详见 [[Stylized-Neural-Painting-译文#3.4 笔触搜索的最优传输]]。
- 渐进分块 $m \times m$ 是为了补细节的搜索策略，不是构图、结构、排线这类课堂阶段。详见 [[Stylized-Neural-Painting-译文#3.6 实现细节]]。
- 输出是矢量笔触，可在任意分辨率重渲染。这是成品绘画，不是可擦的线、结构和排线。本课题若对标这篇，贡献应写在课堂阶段上，不要只报最后一张像不像画。详见 [[Stylized-Neural-Painting-译文#5 结论]]。
