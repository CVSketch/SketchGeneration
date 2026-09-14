---
title: "ChiroDiff: Modelling chirographic data with Diffusion Models"
short: ChiroDiff
source: "[[ChiroDiff]]"
pdf: "[[ChiroDiff.pdf]]"
tags:
  - 译文
  - 全文
status: 全文
---

# ChiroDiff · 全文译文

对应笔记：[[ChiroDiff]]。要点：[[ChiroDiff-要点]]。原文：[[ChiroDiff.pdf]]。项目页：https://ayandas.me/chirodiff

> [!abstract] 这篇怎么用
> 按原论文章节用中文写全。标题编号跟 PDF 一致，方便从 [[ChiroDiff-要点]] 点进来。
> 公式、图号跟原文。不要整段粘英文。
>

> **和本课题的界线：** 扩散时间步 $t$ 是[[扩散模型]]去噪步，**不是**人类作画的阶段顺序。Fig. 3、Fig. 4 里「前向 / 反向扩散」展示的是噪声如何加在整条折线序列上、又如何整段去噪，不能读成「先画轮廓再填细节」的过程生成。无条件 [[FID]] 量的是**成品**矢量草图分布。

## 摘要

对手写、草图、绘画等**连续时间几何结构**（手绘数据）做生成建模，长期依赖[[自回归]]分布。严格按顺序的离散分解有两个硬伤：因果方向只能看过去，建不起对时间概念的整体理解；结果往往把连续轨迹压成**固定采样率**的离散 token 序列，而不是学到底层概念。

本文把**去噪扩散概率模型**（DDPM）用到手绘数据上，提出 **ChiroDiff**。它是**非自回归**的：一次看见整段序列，能学整体概念，对更高的时间采样率也相对稳。条件采样、创意混合等下游用法可以灵活接在同一框架上；随机矢量化、去噪修复、可控抽象等也是这一模型类的特色能力。在相关数据集上做定量和定性评估，整体优于或持平于对比方法。更多结果见项目页。

## 1 引言

触摸屏和 AR/VR 等设备普及后，手写、草图、绘画等手绘数据在数字内容里很常见。草图检索、语义分割、分类等监督任务需求大，发展很快；无监督生成建模相对少，但大规模数据集出现后，对手绘数据的生成也开始变热。已有工作覆盖通用涂鸦（Ha and Eck 2018）、字体、图表、SVG 图标等。无条件神经生成模型既能刻画手绘数据分布，也能为分割、翻译等任务提供条件生成基础。

![[图/ChiroDiff/fig1.png]]

Fig. 1 展示了在 VMNIST、KanjiVG、Quick, Draw! 上无条件采样的例子。

### 1.1 表示与自回归的局限

![[图/ChiroDiff/fig2.png]]

连续时间手绘结构常见两种表示：**栅格图像**和**矢量图形**。栅格是自然图像的默认格式，也常用在手绘上（Yu et al. 2015; 2017），但静态像素难以刻画「画是怎么一步步画出来的」。**创意模型**多用带拓扑的矢量格式（笔画方向、笔序等），学分布 $p_\theta(X)$。其中多数仍是[[自回归]]（Ha and Eck 2018; Aksan et al. 2020; Ribeiro et al. 2020）：变长序列好建模，但模型看不到全局，对时间概念只有局部理解。Fig. 2 的潜空间插值对比说明：自回归模型的插值在复杂组合结构上明显不如 ChiroDiff（上：DDIM 采样；下：自回归）。

另一条路是把点序扔掉，当成二维点集，借用三维点云方法（Luo and Hu 2021a,b; Cai et al. 2020）。点集对手绘的笔画拓扑又不合适。ChiroDiff 取中间路线：**非自回归**密度 + **保留序列顺序**。

### 1.2 时间分辨率

手绘结构本质是连续时间的（Das et al. 2022）。SketchRNN 等把连续轨迹离散成 token 或「运动程序」；因果可见性有限，很难适配多种采样率，往往只记住训练时的那一档分辨率，丢失数字内容需要的时空可缩放性。CoSE、BézierSketch 等虽用几何参数表示连续实体，生成端仍常带自回归成分。SketchODE 用神经 ODE 学时间导数，但 NODE 训练代价高，复杂结构难扩展。ChiroDiff 能看见**整段序列**，可从数据里隐式学采样率，对离散运动程序背后的连续时间概念更鲁棒；在相近数据上，表示能力与训练成本都优于 SketchODE 一类方案。

### 1.3 为何选扩散模型

作者选用 DDPM，是因为它在多样性与保真度上表现突出（Ramesh et al. 2021; Nichol et al. 2022），训练稳定，在视觉合成里几乎取代 GAN。现有扩散工作大多面向栅格图像，序列模态关注少；把序列当定长实体处理的零星工作（Tashiro et al. 2021）也不适合变长手绘。ChiroDiff 是较早把扩散用到**连续时间、变长**手绘实体上的尝试之一：生成过程可理解为把**单位步长的离散布朗运动**逐步变成有结构的手绘样本。

### 1.4 条件生成与应用

作者同时学无条件与条件生成。与自回归不同，ChiroDiff 在同质数据上甚至可以在**没有显式编码器**的情况下做条件采样（见 [[#5.4.2 隐式条件]]）。还考虑**随机矢量化**：从感知输入（栅格或点云）采样 plausible 的拓扑重建。DDIM 变体支持潜空间插值（类似 Ha and Eck 2018; Das et al. 2022）。**创意混合**（Creative Mixing）允许插值端点不必都来自训练分布。去噪 / 修复、可控抽象等也在文中展示。

代价是：自回归擅长的随机补全等能力会弱一些。

**贡献小结：**

- 提出面向连续时间手绘数据的扩散框架 ChiroDiff（[[#4 手绘数据的扩散模型]]），生成端**非自回归**，更利于整体时间概念，重建与生成指标更好（[[#5.3 定量评估]]）。
- 首个能处理**变长、时间连续**手绘模态的扩散生成模型（在作者表述范围内）。
- 展示多种手绘下游应用（[[#5.4 下游应用]]）。

## 2 相关工作

因果[[自回归]] RNN（LSTM、GRU）长期是序列建模默认工具，用于 NLP、视频、音频等。NLP 突破后，其他模态也开始探索非自回归（Girdhar et al. 2019; Huang et al. 2019）。手绘表示学习从 LSTM 转向 Transformer（Ribeiro et al. 2020; Aksan et al. 2020），但生成端仍多为因果 Transformer。也有把手绘当成笔画集合的 set 结构（Carlier et al. 2020），生成 set 需要对齐难题（Zaheer et al. 2017）。ChiroDiff 在生成端**非自回归**，同时保留顺序信息。Luhman and Luhman（2020）曾把扩散直接用于手写，但缺少针对手绘的设计、解释与系统实验。

[[扩散模型]]早年即有（Sohl-Dickstein et al. 2015），Ho et al.（2020）、Dhariwal and Nichol（2021）等推动其在图像生成成为主流。高效采样、潜空间扩散、分类器（无）引导等不断改进。理论形式通用，但非图像模态关注仍少（Lam et al. 2022; Hoogeboom et al. 2022; Xu et al. 2022）。基础概念见库内 [[扩散模型]]，此处不重复 DDPM 通用背景的全部推导。

## 3 去噪扩散概率模型（DDPM）

DDPM（Ho et al. 2020; Sohl-Dickstein et al. 2015）用随机**反向扩散**把各向同性高斯先验 $p(X_T)=\mathcal{N}(X_T;0,I)$ 经 $T$ 步去噪变成 $p_\theta(X_0)$。马尔可夫反向链 $p_\theta(X_{t-1}|X_{t:T})=p_\theta(X_{t-1}|X_t)$ 可取高斯（$T$ 足够大时）：

$$p_\theta(X_{t-1}|X_t) := \mathcal{N}(X_{t-1}; \mu_\theta(X_t,t), \Sigma_\theta(X_t,t)). \tag{1}$$

![[图/ChiroDiff/fig3.png]]

Fig. 3：手绘数据上的前向与反向扩散；「断线」来自笔状态与坐标一起扩散，颜色只标拓扑。横轴是扩散步 $t$，不是人类作画的阶段顺序。

从先验 $X_T\sim p(X_T)$ 出发，用训练好的 $p_{\theta^*}(X_{t-1}|X_t)$ 做祖先采样直到 $t=0$。

直接优化 $\log p_\theta(X_0)$ 困难。先通过**前向扩散**采样潜变量：

$$q(X_t|X_0)=\mathcal{N}(X_t;\sqrt{\alpha_t}\,X_0,(1-\alpha_t)I), \tag{2}$$

其中 $\alpha_t\in(0,1)$ 随 $t$ 单调减，完全指定加噪过程。令 $\Sigma_\theta(X_t,t):=\sigma_t^2 I$，可得到实用的简化变分目标 $L_{\mathrm{simple}}(\theta)$：

$$L_{\mathrm{simple}}(\theta)=\mathbb{E}_{X_0\sim q(X_0),\,t\sim U[1,T],\,\epsilon\sim\mathcal{N}(0,I)}\left[\|\epsilon-\epsilon_\theta(X_t(X_0,\epsilon),t)\|^2\right],$$

其中 $X_t(X_0,\epsilon)=\sqrt{\alpha_t}X_0+\sqrt{1-\alpha_t}\,\epsilon$，网络改为预测噪声 $\epsilon_\theta$。$\mu_\theta$ 与 $\epsilon_\theta$ 的关系见 Ho et al.（2020）原文；$\beta_t=1-\alpha_t/\alpha_{t-1}$。

## 4 手绘数据的扩散模型

### 4.1 表示：三点格式与速度编码

沿用 Ha and Eck（2018）的**三点格式**折线序列 $X=\cdots,(x^{(j)},p^{(j)}),\cdots$，$x^{(j)}\in\mathbb{R}^2$，$p^{(j)}\in\{-1,1\}$ 表示笔状态（提笔 / 落笔）。预处理（等距重采样、空间缩放等）与 SketchRNN 一致。序列长度 $|X|$ 随样本变化。

ChiroDiff 把序列 $X$ 按固定拓扑排成向量做 DDPM，但实验发现**直接扩散绝对坐标**不如扩散**速度** $V=\cdots,(v^{(j)},p^{(j)}),\cdots$，$v^{(j)}=x^{(j+1)}-x^{(j)}$（简单差分）。生成后可积分还原 $x^{(j)}=\sum_{j'\le j}v^{(j')}$。对速度（更高阶导数）建模，更关注高层概念而非局部时间细节（Ha and Eck 2018; Das et al. 2022）。$X$ 与 $V$ 可廉价互转。

下标 $t$ 表示**扩散步**，上标 $(j)$ 表示序列元素。

### 4.2 序列扩散（笔画级前向加噪）

主模型 $p_\theta(V)$ 仍是 DDPM。前向过程称为 **sequence-diffusion**：对每个元素 $(v_0^{(j)},p_0^{(j)})$ **独立**加噪，类比式 (2)：

$$q(V_t|V_0)=\prod_{j=1}^{|V|} q(v_t^{(j)}|v_0^{(j)})\prod_{j=1}^{|V|} q(p_t^{(j)}|p_0^{(j)}),$$

$$q(v_t^{(j)}|v_0^{(j)})=\mathcal{N}(v_t^{(j)};\sqrt{\alpha_t}\,v_0^{(j)},(1-\alpha_t)I),\quad q(p_t^{(j)}|p_0^{(j)})=\mathcal{N}(p_t^{(j)};\sqrt{\alpha_t}\,p_0^{(j)},(1-\alpha_t)I).$$

即：**整条笔画轨迹在序列维度上同步扩散**——每个时间索引上的 $(v,p)$ 对共享同一扩散时间表 $\alpha_t$，但各元素独立高斯扰动。$t=T$ 时先验为各分量标准正态。

二值笔状态当作连续变量扩散，生成时用 $p=0$ 阈值还原 $\{-1,1\}$（Chen et al. 2022 的 analog bits 思路；作者实验有效）。

### 4.3 反向过程、损失与非自回归

反向链 $p_\theta(V_{t-1}|V_t):=\mathcal{N}(V_{t-1};\mu_\theta(V_t,t),\sigma_t^2 I)$，同样用 $\epsilon_\theta$ 参数化，最小化

$$L_{\mathrm{simple}}(\theta)=\mathbb{E}_{V_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(V_t(V_0,\epsilon),t)\|^2\right]. \tag{3}$$

训练好后用 DDPM 从 $t=T$ 迭代采样；也可用确定性 **DDIM**（Song et al. 2021a）：

$$V_{t-1}=\sqrt{\alpha_{t-1}}\left(\frac{V_t-\sqrt{1-\alpha_t}\,\epsilon_{\theta^*}(V_t,t)}{\sqrt{\alpha_t}}\right)+\sqrt{1-\alpha_{t-1}}\,\epsilon_{\theta^*}(V_t,t). \tag{4}$$

像素扩散常用 U-Net；这里需要**序列编码器** $\epsilon_\theta(v_t^{(j)},V_t,t)$，在整段 $V_t$ 上下文中编码每个元素。作者比较双向 RNN 与带位置编码的 Transformer 编码器：Bi-RNN 收敛更快、效果更好。有益的设计是把绝对位置 $X_t$ 与 $V_t$ 拼接输入：$\epsilon_\theta(\cdot,[V_t;X_t],t)$，让网络看见带噪样本的绝对状态，而不只速度动力学（后文记号省略 $X_t$）。

**生成是非因果的**：扩散每一步都能看见**整段序列**，因此是**非自回归**模型，强调整体概念而非低层运动程序；反向过程可以修正任意位置，自回归模型做不到。

### 4.4 从布朗运动到有结构的轨迹

反向过程从 $V_T$ 开始，各 $v_T^{(j)}\sim\mathcal{N}(0,I)$。在速度–位置编码下，对应位置 $x_T^{(j)}=\sum_{j'} v_T^{(j')}$ 是**单位步长离散布朗运动**。随反向扩散展开，随机游走逐渐变成有结构的手绘样本（Fig. 3）。

### 4.5 按长度重采样生成

$\epsilon_\theta(\cdot,t)$ 不硬约束 $|V|$。反向过程可从**任意长度** $L$ 的先验 $p(V_T)=\prod_{j=1}^L q(v_T^{(j)})q(p_T^{(j)})$ 启动，$L$ 可以大于训练时常见长度。作者假设：训练充分时，模型学到几何高层概念，能在更高采样率下生成相似数据。整段 $V_t$（及 $X_t$）可见，有利于隐式全局表示，因此对提高时间分辨率更稳（[[#5.3 定量评估]] 用 CD 验证）。

![[图/ChiroDiff/fig4.png]]

Fig. 4：前向较低采样、反向以更高折线点数 $|X|$ 启动；这是时间采样率差异，不是课堂上的分阶段作画。

## 5 实验与结果

### 5.1 数据集

- **VMNIST**（Das et al. 2022）：MNIST 的矢量版，1 万样本、10 类数字，折线序列；划分 80-10-10。
- **KanjiVG**：汉字矢量数据，用预处理 SVG→折线版本，笔画多、结构复杂。
- **Quick, Draw!**（Ha and Eck 2018）：大规模众包简笔涂鸦。本文类别：cat、crab、bus、mosquito、fish、yoga、flower。

### 5.2 实现细节

前向噪声 schedule：线性 $\beta_{\min}=10^{-4}\cdot 1000/T$，$\beta_{\max}=2\times 10^{-2}\cdot 1000/T$（Nichol and Dhariwal 2021; Dhariwal and Nichol 2021）。扩散步数 $T=1000$。噪声网络 $\epsilon_\theta$ 为**双向 GRU**（Cho et al. 2014），VMNIST 2 层、隐藏维 $D=48$；QuickDraw 与 KanjiVG 为 3 层 GRU，$D=128$ / $96$。带位置编码的 Transformer 效果不佳，作者认为位置编码不适合表达连续时间。

AdamW 优化式 (3)；学习率每 epoch 乘 $0.9997$，初值 $\gamma_0=6\times 10^{-3}$。扩散步 $t$ 用正弦嵌入拼到每层每个元素上。反向方差 $\sigma_t^2=0.8\tilde{\beta}_t$ 较稳，$\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$（Ho et al. 2020 后验方差）。代码见项目页。

### 5.3 定量评估

对比 SketchRNN（Ha and Eck 2018）、CoSE（Aksan et al. 2020）、SketchODE（Das et al. 2022）等，参数量大致对齐。

#### 重建（条件模型）

构造条件 ChiroDiff：编码器 $E_V$ 为 Bi-GRU，把样本 $V$ 编成 $z$；解码器为第 4 节扩散模型，学 $p_\theta(V_0|z=E_V(V))$，即 $\epsilon_{\theta^*}(V_t,t,z)$，$z$ 拼到每个元素与各扩散步。还测试**解码时提高时间采样率**（[[#4.5 按长度重采样生成]]）。

![[图/ChiroDiff/fig5.png]]

Fig. 5：（A–C）重建 Chamfer Distance 随采样率因子变化；（D）相对训练 / 采样耗时；（E）无条件 FID。采样率因子和扩散训练时间都不是课堂阶段。图在 PDF 第 6 页。

![[图/ChiroDiff/fig6.png]]

Fig. 6：条件重建。每组左列采样率 1、右列采样率 2。行依次是条件、SketchRNN、CoSE、SketchODE、ChiroDiff。五组是数字 5、数字 9、猫、蟹、瑜伽小人。图在 PDF 第 7 页。

自回归的 SketchRNN 无法直接提采样率，只能为不同重采样数据各训一版，已处于劣势。指标：条件重建的 **Chamfer Distance（CD）**（忽略 pen-up 位）。Fig. 5(A–C) 为 CD 随**采样率因子**（相对原始点数的倍数）变化：ChiroDiff 在高采样率下更稳；SketchRNN 在长序列上明显变差。CoSE、SketchODE 曲线较平；KanjiVG 上 SketchODE 因训练 / 收敛问题未列入 Fig. 5。Fig. 6 为采样率 1 与 2 的重建定性对比。

#### 无条件生成

用 **DDIM、50 步**无条件采样，对真实样本算 **FID**。Inception 未在手绘上预训练，作者在 Quick, Draw! 上按 Ge et al.（2021）自训特征网络。三数据集上与 SketchRNN、CoSE 对比。Fig. 5(E) 柱状图显示 ChiroDiff 在三套数据上无条件 [[FID]] 均为最低（QuickDraw 为文内各类别平均）。图中可读的大致量级（纵轴 0–60，以 PDF Fig. 5(E) 为准）：

| 数据集 | Ours（ChiroDiff） | CoSE | S.RNN（SketchRNN） |
| :--- | ---: | ---: | ---: |
| VMNIST | 约 10 | 约 11 | 约 13 |
| KanjiVG | 约 15 | 约 23 | 约 32 |
| Quick, Draw!（文内 7 类平均） | 约 25 | 约 35 | 约 40 |

原文未在正文给出 FID 小数；上表依 Fig. 5(E) 柱高估读。定性样本见 Fig. 1。

> [!tip] 与 [[StrokeFusion]] 的分组 [[FID]]（跨论文，非本篇 Table）
> 本文实验是**全类或固定子集上的平均 FID**，没有按笔画数分组。后续 [[StrokeFusion]] 在 QuickDraw 测试集上按平均笔画数三分（低 &lt; 4、中 &lt; 8、高 ≥ 8），用 RDP 简化后同一套 [[FID]] 协议复现对比（见其 PDF Table 1）。**少于 4 笔**一组：ChiroDiff **17.17**，StrokeFusion **19.53**——简单类上 ChiroDiff 的非自回归整段扩散仍略优；笔画变多后 StrokeFusion 在 ≥ 8 笔组报告 **17.76**，高于 ChiroDiff 的 **27.78**。数字出自 StrokeFusion 原文表，便于和库内成品草图基准对照；**均不评价绘画过程**。

#### 计算效率

Fig. 5(D) 相对 ChiroDiff 的收敛时间与采样时间：扩散类训练动态较稳、收敛更快；SketchODE 明显更慢。

### 5.4 下游应用

#### 5.4.1 随机矢量化

![[图/ChiroDiff/fig7.png]]

从感知输入（如点集化的折线，预处理含密采样）恢复 plausible 的矢量拓扑。用 Set Transformer 编码器 $E_R$（max pooling，Lee et al. 2019）得 $z$，条件生成 $p_\theta(V|z=E_R(X))$，$X=\{x^{(j)}\}$ 为去掉笔状态的点集。测试集 CD 与 Das et al.（2021）对比，Fig. 7 显示 ChiroDiff 更优；采样拓扑（颜色）可多样。

#### 5.4.2 隐式条件

与 [[#5.3 定量评估]] 里带 $E_V$ 的**显式条件**不同，**隐式条件**不需要编码器。给定条件 $V_0^{\mathrm{cond}}$，在前向过程取 $t=T_c<T$，得

$$V_{T_c}^{\mathrm{cond}}=\sqrt{\bar{\alpha}_{T_c}}\,V_0^{\mathrm{cond}}+\sqrt{1-\bar{\alpha}_{T_c}}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I),$$

再用 $p_{\theta^*}$ 从 $t=T_c$ 去噪到 $0$：

$$V_{t-1}\sim p_{\theta^*}(V_{t-1}|V_t),\quad T_c>t>0,\quad V_{T_c}:=V_{T_c}^{\mathrm{cond}}.$$

![[图/ChiroDiff/fig8.png]]

$T_c$ 控制与条件的相似度：越大越像条件（Fig. 8 左）。VMNIST 与 Quick, Draw! 上，生成样本与条件**同类**的比例平均约 **93%**。

#### 5.4.3 修复（Healing）

自回归「随机补全」难以双向修复劣质草图；点云领域常谈 healing（Luo and Hu 2021a）。对劣质 $\tilde{V}_0$，把隐式条件里的条件换成 $\tilde{V}_0$，从 $t=T_h$ 启动反向，$V_{T_h}:=\tilde{V}_{T_h}$，在语义附近采样「 healed」分布。$T_h$ 权衡修复强度与概念漂移；示例取 $T_h=T/5$（Fig. 8 右）。

#### 5.4.4 创意混合

![[图/ChiroDiff/fig9.png]]

传统做法是在自编码潜空间插值（Ha and Eck 2018; Das et al. 2022）。DDIM 条件模型可 decode 插值 latent：$z_{\mathrm{interp}}=(1-\delta)E_V(V_0^1)+\delta E_V(V_0^2)$，从 $V_T=0$ 固定点跑式 (4)。KanjiVG、VMNIST 上效果见 Fig. 9 左。

Quick, Draw! 上采用受 ILVR（Choi et al. 2021）启发的 **DDPM 混合**：给定 $V_0$ 与参考 $V_0^{\mathrm{ref}}$，修改反向更新

$$V_{t-1}^{0}=V_{t-1}^{0}-\Phi_\omega(V_{t-1}^{0})+\Phi_\omega(V_{t-1}^{\mathrm{ref}}),$$

其中 $V_{t-1}^{0}\sim p_{\theta^*}(V_{t-1}|V_t,z=E_V(V_0))$，$V_{t-1}^{\mathrm{ref}}\sim q(V_{t-1}^{\mathrm{ref}}|V_0^{\mathrm{ref}})$。$\Phi_\omega$ 为时间轴上一维低通（卷积窗 $\omega=7$）。两序列需等长，条件序列重采样对齐。参考样本**不必**在训练分布内。Fig. 9 右为 Quick, Draw! 跨类混合三元组。

#### 5.4.5 可控抽象

![[图/ChiroDiff/fig10.png]]

**视觉抽象**：学一个更「概括」、细节更少但分布上仍像原数据的分布（Muhammad et al. 2019; Das et al. 2022）。ChiroDiff 可在**同一模型**上通过采样控制实现：令反向方差 $\sigma_t^2=k\cdot\tilde{\beta}_t$，$k\in[0,1]$。$k$ 接近 0 时，反向过程少探索，收敛到主导模式，得到更 canonical、更抽象的样本（Fig. 10，Quick, Draw!）。Das et al.（2022）需为不同控制重训；这里只改 $k$。

## 6 结论、局限与未来工作

![[图/ChiroDiff/fig11.png]]

Fig. 11：上排同一 $\alpha$ 下，矢量折线比栅格数字更怕噪；下排反向方差过大时的去噪失败样。图在 PDF 第 9 页。

ChiroDiff 是基于 DDPM 的**非自回归**手绘生成模型，整体概念建模更好，支持多种自回归难以实现的下游任务。局限包括：（1）矢量表示比栅格更怕噪（Fig. 11 上：同一 $\alpha$ 下矢量扰动更伤结构）；（2）反向方差 $\sigma_t^2$ 为经验设定，噪声有时压过预测均值（Fig. 11 下）；（3）速度积分使绝对位置噪声随序列长度累积。未来可让加噪过程随生成长度或数据基数自适应。

---

## 参考文献

正文引用以 PDF 第 10–12 页 References 列表为准；库内笔记 [[ChiroDiff]] 文首 YAML 含 arXiv 与项目链接。
