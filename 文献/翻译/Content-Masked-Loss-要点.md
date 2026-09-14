---
title: Content Masked Loss 要点
short: Content-Masked-Loss
source: "[[Content-Masked-Loss]]"
translation: "[[Content-Masked-Loss-译文]]"
tags:
  - 要点
  - 参考文献
status: 要点
---

# Content-Masked-Loss · 要点

对应笔记：[[Content-Masked-Loss]]。全文译文：[[Content-Masked-Loss-译文]]。原文：[[Content-Masked-Loss.pdf]]。

> [!tip] 怎么翻原文
> 每个要点都链到全文译文的某一节。点开就能对着中文读，不必再猜在 PDF 哪一页。

## 这篇在干什么

- 多数[[强化学习]]画家用[[对抗与判别器|对抗]]损失或 $L_2$，成画可以，但笔序不像人。这篇提出 Content Masked Loss：用目标图上的 VGG 特征做成掩码，给像素损失加权，让智能体先画能认出主体的区域。详见 [[Content-Masked-Loss-译文#摘要]]。
- 不必采集昂贵的真人笔迹。332 名评估者表明：相对只用对抗或 $L_2$ 的基线，过程更早能看出画的是什么，最终画质没有明显变差。详见 [[Content-Masked-Loss-译文#摘要]]。
- 画家常用铺大面（blocking in）：过程还没结束，主体已经能认出来。作者把绘画看成 anytime 算法，也看成通向成画的笔序列规划。这里的「像人」只指主体更早可辨，不是课堂里的构图、结构、排线、收细。详见 [[Content-Masked-Loss-译文#1 引言]]。
- 机器绘画还要评整个过程，不能只评最后一张。现有笔画渲染加[[强化学习]]，奖励只看整体差异，对抽象内容不敏感。笔序可以任意，只要最后损失低。详见 [[Content-Masked-Loss-译文#1 引言]]。
- SPIRAL、Huang et al. 等靠试错学画，没有人类示范。笔够多几乎能复刻输入，过程里却常常看不出题材。这篇用 Content Masked Loss，在不加人类标注的前提下改笔序。详见 [[Content-Masked-Loss-译文#2 相关工作]]。
- 作者把笔画约束成真实笔刷能执行的指令，并在机械臂上画。Huang et al. 的笔形，真实画笔很难复现，也就难判断笔序像不像人。详见 [[Content-Masked-Loss-译文#2 相关工作]]。

## 方法要点

- 给定输入图和空白画布，生成笔序列。最终要接近输入，过程风格要像人。重点在奖励如何诱导类人规划，不是另写一套课堂阶段。详见 [[Content-Masked-Loss-译文#3 方法（Approach）]]。
- 框架改自 Huang et al. 2019 的 DDPG 画家。改了三处：笔宽固定，长度取画布宽度的 5%；画布从白底起画；最大笔长为半幅画布，颜料按不透明处理。动作仍是 Bézier 曲线。Neural Renderer 当转移函数。详见 [[Content-Masked-Loss-译文#3.1 强化学习画家模型（Reinforcement Learning Painter Model）]]。
- 奖励是 $t$ 到 $t+1$ 的损失差。基线损失是判别器分数或像素 $L_2$。判别器只学视觉差异，不考虑内容和抽象结构。详见 [[Content-Masked-Loss-译文#3.2 奖励函数（Reward Functions）]]。
- Content Masked Loss 不直接在 VGG 特征上比距离。目标图送进 VGG-16，取某层输出，通道维平均，再归一化到 0–1，当作位置权重。详见 [[Content-Masked-Loss-译文#3.2 奖励函数（Reward Functions）]]。
- 像素差乘上这个掩码。对人认出主体更重要的位置，损失更大。[[强化学习]]为了降加权损失，会更早在这些区域落笔，而不是均匀铺像素。详见 [[Content-Masked-Loss-译文#3.2 奖励函数（Reward Functions）]]。
- 白底上，$L_2$ 会让模型先画深色，快速拿奖励，把掩码效果盖住。作者改用裁剪 $L_1$，逐像素损失不超过 $\lambda$。主结果模型是 CM + $L_1^*$。详见 [[Content-Masked-Loss-译文#3.2 奖励函数（Reward Functions）]]。
- 若奖励直接用画布和目标的 VGG 特征 $L_2$（Content Loss），模型容易只输出空白画布。照片上训出的 VGG，对只有几笔的画布几乎随机。详见 [[Content-Masked-Loss-译文#4.2 Content Loss 奖励]]。
- VGG-16 浅层掩码偏边缘纹理，深层偏眼、嘴。最终成画相近，深层中间过程更像人。后续实验固定第 17 层。详见 [[Content-Masked-Loss-译文#4.3 Content Masked 奖励与 VGG 层选择]]。
- 固定笔宽、限笔长之后，基线还能复现目标，保真度低于 Huang et al.。把画布划成子块再画，最终损失可以接近对方。详见 [[Content-Masked-Loss-译文#4.1 笔约束与基线画质]]。

## 数字要点

- 机械臂实验里，目标图经 k-means 得到 10 色，再人工调色。单刷、可水洗，过程全自动。详见 [[Content-Masked-Loss-译文#1 引言]]。
- 笔长取画布宽度的 5%。最大笔长为半幅画布，经验来自 20 cm 纸面试画。详见 [[Content-Masked-Loss-译文#3.1 强化学习画家模型（Reinforcement Learning Painter Model）]]。
- Fig. 6：整图 125 笔，再加 25 个 $1/25$ 子块，每块各 125 笔。详见 [[Content-Masked-Loss-译文#4.1 笔约束与基线画质]]。
- VGG-16 有 13 个卷积层（各带激活）和 5 个池化层，一共 31 层都能做掩码。Fig. 8 画的是前 29 层。后续固定第 17 层。详见 [[Content-Masked-Loss-译文#4.3 Content Masked 奖励与 VGG 层选择]]。
- CelebA 留出 2000 张肖像，做 5 折交叉验证。详见 [[Content-Masked-Loss-译文#4.4 最终绘画质量]]。
- Table 1 的 [[FID]] / IS：GAN 为 $240.05 \pm 0.43$ / $3.05 \pm 0.08$；$L_2$ 为 $241.02 \pm 0.24$ / $3.05 \pm 0.14$；CM + $L_2$ 为 $242.67 \pm 0.24$ / $3.01 \pm 0.04$；$L_1^*$ 为 $241.58 \pm 0.14$ / $3.45 \pm 0.09$；CM + $L_1^*$ 为 $243.13 \pm 0.20$ / $3.47 \pm 0.05$。没有一种损失在两项上全面最好。详见 [[Content-Masked-Loss-译文#4.4 最终绘画质量]]。
- FaceNet 对 2000 张 hold-out 肖像逐笔检测。Content Masking 提高 50–200 笔区间的人脸检出率。CM + $L_1^*$ 最高。200 笔以内，检测和人评结论一致。详见 [[Content-Masked-Loss-译文#4.5 类人规划（Human-Like Planning）]]。
- 人类评估展示 10、30、100、200、750 笔时的画布。332 名 MTurk 工人，400 张图，每题 4 人评。30–200 笔时，CM + $L_1^*$ 被选为「最像脸」的比例明显更高。750 笔附近各模型差距缩小。详见 [[Content-Masked-Loss-译文#4.5 类人规划（Human-Like Planning）]]。

## 关键图

- Fig. 1：Baseline 与 CM + $L_1^*$ 对比。列是 10、30、100、300、750 笔和目标图。Ours 更早出现可辨五官。人评用的是 10、30、100、200、750 笔，和这张列不完全一样。图在 [[Content-Masked-Loss-译文#1 引言]]。
- Fig. 2：目标图 → Neural Renderer → 机械臂实画四联；TinkerKit 臂、k-means 十色。图在 [[Content-Masked-Loss-译文#1 引言]]。
- Fig. 3：(a) Huang et al. 无笔约束；(b) 本文白底固定笔宽/笔长基线，同笔数下过程更碎、终稿保真更低。图在 [[Content-Masked-Loss-译文#2 相关工作]]。
- Fig. 4：式 (1)–(7) 汇总即时奖励与各损失（GAN、$L_2$、Content Loss、CM + $L_2$、$L_1^*$、CM + $L_1^*$）。图在 [[Content-Masked-Loss-译文#3.2 奖励函数（Reward Functions）]]。
- Fig. 5：Predicted / Target / Mask 三联，掩码来自目标图 VGG 特征。图在 [[Content-Masked-Loss-译文#3.2 奖励函数（Reward Functions）]]。
- Fig. 6：目标与 (a) Huang 原模型、(b) 约束基线；整图 125 笔再加 25 子块各 125 笔。图在 [[Content-Masked-Loss-译文#4.1 笔约束与基线画质]]。
- Fig. 7：上行左三为模型成画，右为目标照片；下行是各图用 VGG 前 17 层抽出的特征。成画已经很像目标时，特征和照片目标仍差很远。原文没有给这三张成画的笔数。图在 [[Content-Masked-Loss-译文#4.2 Content Loss 奖励]]。
- Fig. 8：VGG-16 第 0–28 层特征可视化；浅层偏边缘，深层偏眼嘴。图在 [[Content-Masked-Loss-译文#4.3 Content Masked 奖励与 VGG 层选择]]。
- Fig. 9：不同 VGG 层做掩码时，5、20、50、100、200 笔的中间过程。行是第 2、8、12、17、22、31 层。深层更盯五官，不是课堂四阶段。图在 [[Content-Masked-Loss-译文#4.3 Content Masked 奖励与 VGG 层选择]]。
- Fig. 10：五组损失在 10–750 笔的过程格；CM 两行更早出现脸形，不是课堂四阶段。图在 [[Content-Masked-Loss-译文#4.4 最终绘画质量]]。
- Table 1：五组损失的 FID / IS；无一项两项全胜，说明掩码未明显伤终稿。图在 [[Content-Masked-Loss-译文#4.4 最终绘画质量]]。
- Fig. 11：FaceNet 检出比例随笔数；50–200 笔 CM + $L_1^*$ 最高。图在 [[Content-Masked-Loss-译文#4.5 类人规划（Human-Like Planning）]]。
- Fig. 12：MTurk 五选一界面示例（白底粗块 vs 黑底成画）。图在 [[Content-Masked-Loss-译文#4.5 类人规划（Human-Like Planning）]]。
- Fig. 13：30–200 笔 CM + $L_1^*$ 被选「最像脸」比例最高。图在 [[Content-Masked-Loss-译文#4.5 类人规划（Human-Like Planning）]]。

## 对本课题

- 本课题要输出有顺序、有阶段、可中断的素描过程。这篇的「类人规划」只是：用 VGG 掩码加权像素损失，让[[强化学习]]先画能认出主体的区域。不要把它写成构图、结构、排线、收细这四个课堂阶段。详见 [[Content-Masked-Loss-译文#摘要]]。
- 铺大面在这里的意思是：过程中途主体已经能认出来。anytime 指随时停下还有一张还能看的画。这是油画笔序上的可辨性，不是素描课的阶段划分。详见 [[Content-Masked-Loss-译文#1 引言]]。
- 可借鉴的是损失设计，不是课堂教案。Content Masked Loss 把「哪里对认出主体重要」写成位置权重，乘在像素差上。本课题若只学「先画五官」，仍然只是贴原图，学不到线、结构和排线。详见 [[Content-Masked-Loss-译文#3.2 奖励函数（Reward Functions）]]。
- Table 1 的 [[FID]] 和 IS 只说明最后一张味道差不多。过程好不好，作者另用中间笔数的 FaceNet 检出和人排序来评。本课题也不能只报 [[FID]]。详见 [[Content-Masked-Loss-译文#4.4 最终绘画质量]]。
- 人评问的是「哪张最像脸」，看的是 10 到 750 笔的中间画布。这是主体可辨，不是阶段分类准不准，也不是一次只改一块。详见 [[Content-Masked-Loss-译文#4.5 类人规划（Human-Like Planning）]]。
- 谁在引：[[ProcessPainter]]。ProcessPainter 在相关工作里把这篇写成强化学习笔画渲染的一条，训练用的是扩散和 8 帧，没有用 Content Masked Loss。两边都不是线稿课堂上的构图到收细。详见 [[Content-Masked-Loss-译文#5 结论与未来工作（Conclusion and Future Work）]]。
- 相对采集人类笔迹，这是更省数据的类人规划手段。开题可以先做到「主体更早能认出来」。这也是上限：学不到人按课堂阶段改线、一次一块区域的画法。详见 [[Content-Masked-Loss-译文#5 结论与未来工作（Conclusion and Future Work）]]。
