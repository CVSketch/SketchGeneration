---
title: "DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models"
short: DiffSketcher
source: "[[DiffSketcher]]"
tags:
  - 译文
  - 论文
status: 第一轮
---

# DiffSketcher · 故事与方法译文

> [!warning] 旧意译，只留底
> 请改读 [[DiffSketcher-译文]]。不要再改本页。


对应笔记：[[DiffSketcher]]。原文：[[DiffSketcher.pdf]]。

讲解页是 NeurIPS 2023 Poster，不是 Oral：https://neurips.cc/virtual/2023/poster/72425

> [!abstract] 版权与写法
> 这里是**意译**，不是逐句复述全文。故事按作者口吻用中文重讲。方法按节写清。只对几句关键原文做对照精翻。图用库内裁图，不整页贴 PDF。

## 作者想讲的故事

一句话和一张简笔都能抓住要点，不必铺满细节。设计师听客户说话，会先画一张线稿对一下。机器这边，以前多半是「给你一张照片，再描成线」。CLIPasso 一类还得有图，而且不能凭空长出新东西。

近几年，文生图 [[扩散模型]] 已经能听一句话出照片。作者问：这些模型主要在普通图片上训练，能不能当素描老师？

答案是能。但老师不会自己下笔。它只负责看：你现在这组线，画成普通图以后，像不像那句话该有的样子。学生是一组 [[贝塞尔曲线]]。[[可微渲染器]] 把线画成图，再用改过的 [[分数蒸馏]] 拧控制点。没有「这句话对应这张素描」的成对数据。

Fig. 4 图注说得更直：扩散模型冻住，不往它里面反传。它像一个只会给改图方向的鉴定师。

## 方法按节

### 总体在干什么

进一句话。出一组带透明度的矢量线。大模型参数不动。动的是这张纸上的控制点和深浅。换一句话，就要重新拧。

三条办法：

1. **ASDS**（增强版分数蒸馏）：给 SDS 加上透视、裁切、锐度扰动。让潜空间里的扩散老师盯「像不像那句话」。
2. **注意力初始化**：把交叉注意力和自注意力合成一张热力图。第一批锚点放在主体附近。
3. **透明度**：线可以浓可以淡，更像手绘的轻重。

先用 JVSP（感知差 + 图意差）贴住扩散自己采样出的图。再上 ASDS 对着句子微调。作者说这样质量和速度最好。

### 数据怎么来

没有自建素描过程库。也没有文–素描对。老师是现成的潜空间文生图模型。笔画数 $n$ 由人定。抽象程度主要靠线的多少，不靠句子自动决定。作者后来说这是局限。

### 模型怎么走

1. 从扩散模型的注意力图采样 $n$ 个点。附近再撒 3 个控制点，组成初始贝塞尔。
2. [[可微渲染器]] 把 $\theta$（控制点 + 透明度）画成线稿 $S=R(\theta)$。
3. 线稿做几种小扰动，送进冻住的扩散模型。
4. ASDS 给出「该往哪拧」的方向。JVSP 再拿线稿和老师自己出的图比一比。
5. 梯度只回到控制点和透明度。重复到几百步。

交叉注意力管「哪个词占哪一块」。自注意力管外形和前后关系。融合写成：

$$\mathrm{FinalAttn}=\lambda\cdot\mathrm{CrossAttn}_i+(1-\lambda)\cdot\mathrm{Mean}(\mathrm{SelfAttn})$$

再做 softmax，抽每条线的第一个点。另外三点落在半径约为图宽 $0.05$ 的圈里。

随机初始化要更久。只用 CLIP 显著图，容易只画主体、丢掉背景。

### 怎么知道自己错了

三项加在一起：

1. **ASDS**。线稿加噪后，问冻住的扩散模型：噪声预测和真噪声差多少。差就是拧曲线的方向。时间步抽 $t\sim U(0.05,0.95)$。无分类器引导取 $\omega=100$。
2. **JVSP 里的感知差**。线稿和扩散采样图，用 LPIPS 比。
3. **JVSP 里的图意差**。两者在 CLIP 视觉特征上的距离。权重：LPIPS $0.2$，CLIP 视觉 $1$。

ASDS 直接给梯度，不另外调这项的损失权重。

## 图在讲什么

### Fig. 1：终稿，以及优化步

![[图/DiffSketcher/fig1.png]]

上排：不同句子的矢量线。右边箭头从抽象到具体。下排：同一句金刚鹦鹉。蓝数字是迭代次数。

![[图/DiffSketcher/fig1-iters.png]]

**0 / 20 / 100 不能当画家阶段。** 图注写明：blue number indicates the number of iterations。第 0 步是初始化后的点，线还没拧拢。第 20 步是同一组线一起拧了 20 次。第 100 步还是这组线，只是损失又降了一截。画家阶段有顺序：先大形，再局部。这里没有笔序。线从一开始就全部在纸上。附录 Fig. 15 也把这一串叫 optimization process。

### Fig. 5：注意力初始化

![[图/DiffSketcher/fig5-attn.png]]

用「Eiffel Tower」里 Tower 那一张交叉注意力，加上自注意力平均，得到落笔热力图。再抽样成初始曲线。

### Fig. 6：和描边、CLIPasso 比

![[图/DiffSketcher/fig6.png]]

Canny 边太多、线脏。CLIPasso 做场景时容易只画前景。作者强调场景级也能画。

### Fig. 7：和 VectorFusion 比

![[图/DiffSketcher/fig7.png]]

两边都是扩散当老师、拧矢量。这篇盯素描。图里为了公平，只用 ASDS、随机初始化。完整模型还加注意力初始化和透明度。

### Fig. 8：换损失、换初始化，步数含义不变

![[图/DiffSketcher/fig8.png]]

同一句宇航员。换 CLIP 显著图或随机点，收敛快慢不同。步数仍是优化进度。只靠 JVSP 会贴住扩散出的那张图。只靠 ASDS 语义对，但布局可以漂。两个一起用，细节更多。

## 关键句对照

### 1. 为什么扩散能当老师（摘要）

> Even though trained mainly on images, we discover that pretrained diffusion models show impressive power in guiding sketch synthesis.

**精翻：** 这些扩散模型虽然主要在普通图片上训练，我们却发现它们指导素描合成的能力很强。

**为什么重要：** 没有素描大数据，也能借照片模型盯语义。借来的是「像那个东西」，不是「像人那样一步步画」。

### 2. 优化的是曲线，损失是改过的 SDS（摘要）

> It performs the task by directly optimizing a set of Bézier curves with an extended version of the score distillation sampling (SDS) loss

**精翻：** 它直接优化一组贝塞尔曲线。损失是改过的分数蒸馏。

**为什么重要：** 优化的是这张图的线，不是一个以后能秒出图的网络。[[CoProSketch]] 后来批评它慢、不能中途改布局。

### 3. 扩散模型只当冻住的鉴定师（Fig. 4 图注）

> Since the diffusion model directly predicts the update direction, we do not need to backpropagate through the diffusion model; the model simply acts like an efficient, frozen critic that predicts image-space edits.

**精翻：** 扩散模型直接给出更新方向，不必把梯度传进它里面。它像一个冻住的鉴定师，只预测图上该怎么改。

**为什么重要：** 这就是「扩散当老师」的具体做法。老师不会排笔序。

### 4. 注意力初始化为什么存在（4.2 节）

> The highly non-convex nature of the ASDS loss function makes the optimization process susceptible to initialization

**精翻：** ASDS 损失很不凸，优化结果对初始化敏感。

**为什么重要：** 线不能在整张纸上乱撒。注意力图只解决「第一下落在哪」，不解决「先画什么」。

## 这篇实现了什么（给组会）

- 输入 → 输出：一句话 → 一组带透明度的贝塞尔线。没有文–素描对。
- 新模块：ASDS（给 SDS 加输入扰动）；交叉 / 自注意力融合初始化；透明度可优化。
- 公开数字（原文有的）：CLIP 余弦 $0.3494$，高于 Canny 的 $0.328$、CLIPasso 的 $0.3075$。美感均分 $4.8206$，高于 Canny 的 $4.3682$、CLIPasso 的 $4.0821$。用户混淆分 $0.65$，真人素描 $0.67$，CLIPasso $0.39$，VectorFusion $0.33$。
- 缺什么：没有笔序，没有画家阶段。一张图要拧到一两百步以上。句子和抽象程度没有挂钩。风格偏能认出物体的简笔。
