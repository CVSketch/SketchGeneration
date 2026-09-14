---
title: "Synthesizing Programs for Images using Reinforced Adversarial Learning"
short: SPIRAL
source: "[[SPIRAL]]"
pdf: "[[SPIRAL.pdf]]"
tags:
  - 译文
  - 全文
status: 全文
---

# SPIRAL · 全文译文

对应笔记：[[SPIRAL]]。要点：[[SPIRAL-要点]]。原文：[[SPIRAL.pdf]]。代码：https://github.com/google-deepmind/spiral。演示视频：https://youtu.be/iSyvwAwa7vk

> [!abstract] 这篇怎么用
> 按原论文章节用中文写全。标题编号跟原文一致，方便日后写 [[SPIRAL-要点]] 并链到各节。
> 公式、表号、图号跟 PDF。正文 Fig. 已裁入 `附件/图/SPIRAL/`。不要整段粘英文。
>
> **本课题对照：** [[Learning-to-Paint]] 在相关工作与实验中把 SPIRAL 当基线。SPIRAL 用**不可微**的 libmypaint / MuJoCo 当绘画环境，整局结束才用 **WGAN 判别器分数**当稀疏奖励，走 model-free 的 A2C；Learning to Paint 用**可微神经渲染器**、逐步 WGAN 奖励和基于模型的 DDPG，在 CelebA 上报告约 3× 更小的 $\ell_2$ 误差（见其译文 §4.3）。

## 摘要

深度生成网络近年进展很大，但解码器归纳偏置弱时，容量常被数据集上的细枝末节占满。图形引擎把图像抽象成高层「程序」，可减轻这一问题。现有把深度学习与渲染器结合的工作，往往依赖手工似然或距离、需要大量对齐监督，或难以扩展到更复杂数据。为此我们提出 **SPIRAL**：对抗训练的[[强化学习]]智能体，合成一段由图形引擎执行的程序，用于解释或采样图像。智能体的目标是骗过判别器——判别器区分真实图与渲染图，训练在**无额外监督**的分布式[[强化学习]]下进行。一个关键发现是：**把判别器输出当奖励**，智能体才能在匹配目标渲染上取得实质进展。据我们所知，这是首次在 MNIST、Omniglot、CelebA 等真实数据与合成 3D 数据上，端到端、无监督、对抗式地训练逆图形智能体。

## 1 引言

人类能从原始感官里恢复结构化表示，并频繁使用这一能力。看到手写字符的照片，拆成笔画有助于分类或重画；知道房间布局有助于规划、导航与交互；这种结构还能帮助泛化、快速学习，甚至与其他智能体通信。Lake 等认为，人类会借助内部模拟来学习（Lake et al., 2017）：在纸笔上试动作与字符的对应，在想象中把建筑布局与现实对应起来。

在视觉里，为理解场景而**反演渲染器**，通常叫逆图形（inverse graphics）。用逆图形训练视觉系统一直很难：渲染器输入的程序往往序列语义明确、由离散符号组成（如 CAD  keystroke）、长度可达几十到上百；把渲染图与真实数据对齐时，黑盒模拟器一般**不可微**，优化成了黑盒问题。

我们提出 **Deep Reinforced Adversarial Learning**：对抗训练的智能体生成视觉程序，由图形引擎执行以生成图像，可无条件也可条件于输入。智能体靠骗过判别器获得奖励，用分布式[[强化学习]]训练，**不需要**程序—图像对之类的强监督。判别器本身学区分渲染结果与真实图。

贡献概括：

- 对抗训练的[[强化学习]]智能体，在**视觉程序空间**里解释并生成图像；架构对程序语义与任务域都尽量通用。

![[图/SPIRAL/fig1.png]]

Fig. 1：四行任务——无条件噪声生成曲线；条件重建 Omniglot 字符、蒙娜丽莎、MuJoCo 3D 场景。每行左为输入（或噪声），中为 SPIRAL，右为同一轨迹上若干中间渲染，不是课堂意义上的「作画阶段」。

- 把逆图形扩展到真实世界与程序生成数据，**无需标签**：在 MNIST、Omniglot 上发现笔画；在 CelebA 上发现刷笔触；还能合成 3D 场景描述，渲染后重建图像。
- 证据表明：用判别器输出当[[强化学习]]奖励，比直接优化像素 $\ell_2$ **显著更好**地压低渲染与数据间的像素误差。
- 展示可与逆图形、程序合成衔接的规模化深度[[强化学习]]训练管线（基于 IMPALA 扩展）。

## 2 相关工作

**反演模拟器以解释图像**已有大量工作（Nair et al., 2008; Paysan et al., 2009; Mansinghka et al., 2013; Loper & Black, 2014; Kulkarni et al., 2015a; Jampani et al., 2015）。Wu 等提出对象—属性结构的「去渲染」模型用于图像与视频。同期工作用构造实体几何（CSG）基元解释二值图（Sharma et al., 2017）。Loper & Black (2014) 提出可微逆图形，适合连续变量优化，难处理离散变量。Xie 等 (2013) 用[[强化学习]]自动生成单道水墨笔画；扩展到更大真实数据集、尤其在测试时推理，仍困难。

**MNIST 上的运动程序**最早见 Nair & Hinton (2006)：生成模型用两对「弹簧」刚度由运动程序控制；训练从原型程序与观测出发，加噪扩展分布直到覆盖训练数字流形。SPIRAL 则通过判别器与同一智能体**自动形成训练课程**，且同一套智能体可覆盖 3D 等场景理解问题。

**视觉程序归纳：** Lake et al. (2015) 在 Omniglot 上表现很好，但需手工解析器初始化，且未展示超出手写字符。Ellis et al. (2017) 推断 LATEX 程序以理解图表。Sketch-RNN（Ha & Eck, 2017）用 seq2seq 生成素描，但需要笔画序列监督。与这些不同，SPIRAL **不需要**对齐的程序—图像对。

神经网络社区也有推断前馈或循环生成过程的工作（LeCun et al., 2015; Goodfellow et al., 2014; Kingma & Welling, 2013 等），生成能力强，但少结构化逆推断。

SPIRAL 沿用对抗训练（Goodfellow et al., 2014; Ganin & Lempitsky, 2015）。GAN 已用于音频、文本、运动（Ho & Ermon, 2016; Merel et al., 2017）；在域迁移里，pix2pix、CycleGAN、AIGN 等把分割图映射到像素。SPIRAL 在此基础上：**手工设计少、无程序—图像对监督、跨域可用**（见第 4 节）。

## 3 SPIRAL 智能体

### 3.1 概述

目标：构造生成模型 $G$，从目标数据分布 $p_d$ 采样。使用外部**黑盒**渲染模拟器 $R$：接受命令序列 $a=(a_1,\ldots,a_N)$，映射到目标域（如位图）。$R$ 可以是 CAD 程序，把基元描述渲染成 3D 场景。

任务等价于恢复分布 $p_a$，使 $p_d \approx R(p_a)$。用循环神经网络 $\pi$（策略网络 / 智能体）建模 $p_a$。生成过程 $G=(\pi,R)$ 见 Fig. 2a（PDF 第 2 页）。

### 3.2 目标函数

实验发现原始 minimax GAN 目标难优化，改用 **Wasserstein 距离**变体（Gulrajani et al., 2017），对 $p_g$ 与 $p_d$ 差异很大时更稳；其他 GAN 目标也可替换。

**判别器** $D$：

$$L_D = -\mathbb{E}_{x\sim p_d}[D(x)] + \mathbb{E}_{x\sim p_g}[D(x)] + R \tag{1}$$

$R$ 为梯度惩罚，软约束 $D$ 为 Lipschitz 连续。式 (1) 的解只差一个加性常数；训练 $\pi$ 时，鼓励 $x\sim\frac{1}{2}p_g+\frac{1}{2}p_d$ 时 $D(x)$ 均值接近 0，以固定这一歧义。

**生成器 / 策略：** $\pi$ 在每一步 $t$ 预测命令分布 $\pi_t=\pi(a_t|s_t;\theta)$，$s_t$ 为 RNN 状态。采样 $(a_t\sim\pi_t)$ 后，$p_g$ 的样本为 $R(a_1,\ldots,a_N)$。因 $R$ 任意**不可微**，不能直接对

$$L_G = -\mathbb{E}_{x\sim p_g}[D(x)] \tag{2}$$

做朴素梯度下降。改为最大化期望回报，用 **advantage actor-critic（A2C）**（REINFORCE 变体）：

$$L_G = -\sum_t \log \pi(a_t|s_t;\theta)\,[R_t - V^\pi(s_t)] \tag{3}$$

$V^\pi$ 为值函数近似（对 $\theta$ 视为常数），$R_t=\sum_{t'} r_{t'}$ 为回报的 1-sample 蒙特卡洛估计。若设奖励为：

$$r_t = \begin{cases} 0, & t < N \\ D(R(a_1,\ldots,a_N)), & t = N \end{cases} \tag{4}$$

则优化 (3) 等价于解 (2)。**只有最后一步**有非零奖励：中间渲染 $R$ 每步都会更新画布，但信用分配完全依赖终局判别器分数——这是与 [[Learning-to-Paint]]（每步判别器差分奖励 + 可微环境）的核心差别之一。

新 formulation 还允许**中间奖励**（依赖 $R$ 输出或命令），第 4 节 MNIST 无条件实验用了辅助惩罚。

> [!tip] 不可微环境 + 判别器奖励（读法）
> - **环境 $R$：** libmypaint、MuJoCo 等真实渲染器；对 $\pi$ 的参数**没有**反传梯度，智能体只能把 $R$ 当黑盒，用策略梯度学「下什么命令」。
> - **奖励：** 终局 $D(\text{最终渲染})$（条件任务里 $D(x|x_{\mathrm{target}})$）；不是像素 MSE，也不是逐步 $\ell_2$ 差分。
> - **对照 [[Learning-to-Paint]]：** 后者用神经渲染器把 $s_{t+1}=\mathrm{trans}(s_t,a_t)$ 可微化，WGAN 分数做成 $r_t=L_t-L_{t+1}$，并走基于模型的 DDPG。

![[图/SPIRAL/fig2.png]]

Fig. 2：（a）MuJoCo 示例轨迹——策略 $\pi$ 逐步输出放置/修改物体的程序片段，渲染器 $R$ 每步更新场景；中间画面可反馈给策略，但默认只在终局拿判别器奖励。（b）IMPALA 式异步训练：多路 actor 写 replay；policy learner 用轨迹与终局 $D$ 分数更新 $\pi$；discriminator learner 区分真实图与生成终局图。

### 3.3 条件生成

无条件生成之外，常需条件于辅助输入（Mirza & Osindero, 2014）。例如给定目标图 $x_{\mathrm{target}}$，找能生成它的程序：把 $x_{\mathrm{target}}$ 同时喂给策略与判别器，

$$p_g = R(p_a(a|x_{\mathrm{target}})) \tag{5}$$

$p_d$ 变为以 $x_{\mathrm{target}}$ 为中心的 Dirac。式 (1) 前两项化为

$$-D(x_{\mathrm{target}}|x_{\mathrm{target}}) + \mathbb{E}_{x\sim p_g}[D(x|x_{\mathrm{target}})] \tag{6}$$

此设定下，$\ell_2$ 距离是**最优**判别器之一（附录 A），但一般不是 (1) 的唯一解，作生成器奖励时与学习的 $D$ **不等价**（Fig. 8a，PDF 第 8 页）。第 4 节对比固定 $\ell_2$ 与判别器分数，结论：实践中必须用学习的 $D$ 才训得动。

### 3.4 分布式学习

训练管线是 IMPALA（Espeholt et al., 2018）的扩展（Fig. 2b）。三类 worker：

- **Actors：** 策略与 $R$ 交互生成轨迹，含 $(\pi_t,a_t)$ 序列及 $R$ 产生的**全部中间渲染**。
- **Policy learner：** 收轨迹成 batch，对 (2)/(3) 做 SGD；并加**熵正则**鼓励探索（Mnih et al., 2016）。
- **Discriminator learner（相对 IMPALA 新增）：** 消费 $p_d$ 随机样本与 actor 终局渲染，优化 $L_D$ (1)。

WGAN-GP 原文建议 $D$ 更新比 $G$ 更频繁；这里每条样本生成都很贵（多次调用外部模拟器），policy learner **不丢弃**轨迹。改为 **replay buffer** 在 actor 与 discriminator learner 之间解耦，让 $D$ 可以比 $\pi$（多步 RNN）更新更快。replay 会平滑 $p_g$，实践中仍有效。

## 4 实验

### 4.1 数据集

在三个真实与一个合成数据集上验证。图像统一缩放到 **64×64**，网络结构可复用。

| 数据集 | 要点 | PDF 页 |
| :--- | :--- | :--- |
| MNIST | 70000 手写数字，10000 测试；28×28 灰度，常被认为 GAN/VAE「已解」，但不强调可解释笔画结构 | 第 4 页 |
| Omniglot | 1623 字符、50 字母表；变异性更高、符号更复杂、每类仅 20 样本 | 第 4–5 页 |
| CelebA | 20 万+ 彩色名人头像，姿态背景光照变化大 | 第 5 页 |
| MuJoCo Scenes | 程序生成的简单 3D 场景（最多 5 个基元），50000 张 RGB 训练图 | 第 5、9 页 |

MNIST / Omniglot 是线稿域；CelebA 与 MuJoCo 用来测彩色与 3D 逆图形。

### 4.2 环境（不可微渲染）

引入两个渲染环境；**对 $\pi$ 均不可微**。

**libmypaint（MNIST、Omniglot、CelebA）：** 开源绘画库。智能体控制画笔，在画布 $C$ 上产生（可不相连的）笔画序列。环境状态 = 画布内容 + 当前笔位置 $l_t$。每步动作 $a_t$ 为 **8 个离散决策** $(a_t^1,\ldots,a_t^8)$（Fig. 3，PDF 第 5 页）：

- 控制点 $p_c$ 与终点 $l_{t+1}$，定义二次[[贝塞尔曲线]]：
  $$p(\tau)=(1-\tau)^2 l_t + 2(1-\tau)\tau p_c + \tau^2 l_{t+1},\quad \tau\in[0,1] \tag{7}$$
- 有效位置在 $C$ 上的 **32×32** 网格；$l_0$ 为画布左上角。
- 笔压（10 档）、笔刷大小、RGB（各 20 bin）；灰度数据省略颜色。
- 最后一维为二元标志：**画一笔**或**跳到** $l_{t+1}$ 不画。

![[图/SPIRAL/fig3.png]]

Fig. 3：libmypaint 单步动作示意——从当前笔位 $l_t$ 选控制点与终点，得到红/绿/蓝等不同二次贝塞尔笔画；右侧列出笔压、笔刷大小与 RGB 等离散维度。

**MuJoCo Scenes：** 每步决定物体类型（4 种）、16×16 网格位置、大小（3 档）、颜色（3 通道各 4 bin）；环境按规格加物体；还可跳过、或修改最近放置的物体（Fig. 2a 示例命令）。

### 4.3 MNIST

![[图/SPIRAL/fig4.png]]

Fig. 4：（a）MNIST 无条件——训练帧数从 0 增至 $25\times 10^6$ 时，乱涂逐渐变成可辨数字；终局网格展示多风格样本。（b）条件重建——左为 SPIRAL 输出，右为真值，成对对齐。

**无条件：** 除判别器奖励外，加体现归纳偏置的辅助项：鼓励**一笔连续运动**画完数字（每段连续笔画开始给小负奖励）；并对「完全没画出可见笔画」加惩罚。训练约 **$25\times 10^6$** 帧后，生成样本可辨为手写数字，风格与笔刷大小有多模态（Fig. 4a，PDF 第 6 页）。

**条件重建：** 给定目标数字，比较 §3.3 两种奖励：**固定 $\ell_2$** vs **判别器分数**（Fig. 8a 蓝线，PDF 第 8 页）。判别器路线收敛更快、终局 $\ell_2$ 更低；无辅助奖励时，纯 $\ell_2$ 训不出合理重建。Fig. 4b 为条件生成样例。

**盲智能体（blind）：** 不向 $\pi$ 喂中间画布状态（Sharma et al., 2017 同类设定），无法依赖逐步反馈。Fig. 8a 虚线：性能低于完整模型，仍能产生合理重建，说明在**无中间状态**的程序合成设定下也有潜力。

### 4.4 Omniglot

![[图/SPIRAL/fig5.png]]

Fig. 5：（a）Omniglot 无条件生成随训练变清晰，但终局仍不如 MNIST 整齐。（b）条件重建——左重建、右真值，字符轮廓大体对齐。

无条件生成质量低于 MNIST，对其他神经网络生成方法也偏难（如 Rezende et al., 2016 的样本不像真实笔画顺序）。**条件**智能体重建质量令人信服。此数据集上 **$\ell_2$ 奖励训不好**（Fig. 8a 红虚线）：说明判别器奖励不仅加速学习，还在 $\ell_2$ 无法探索成功时**允许训练成功**。

![[图/SPIRAL/fig6.png]]

Fig. 6：Omniglot 训练后的**域外线稿解析**——输入 64×64、64×64 与 256×256 重建，以及 256×256 彩色笔画轨迹（按时间着色，表示命令顺序而非语义阶段）。

训练后的智能体对**域外**线稿输入做解析；重建尚可，略逊于 Omniglot 测试集。利用底层笔画可高分辨率或换风格重渲染。

### 4.5 CelebA

libmypaint 也能画复杂彩色图。条件智能体在 CelebA 上训练：**20 步** episode，**无中间奖励**。除重建奖励（$\ell_2$ 或判别器）外，可选加输出与 $x_{\mathrm{target}}$ **颜色直方图**的 earth mover 距离惩罚，略提性能但**非必需**。

![[图/SPIRAL/fig7.png]]

Fig. 7：CelebA 条件重建网格——每对左为 20 步 libmypaint 输出、右为真值；块面笔触导致偏糊，但背景、脸位与发色等大块结构仍能对上。

未针对人脸改动作空间，策略发现较慢。**$\ell_2$ 智能体**几乎学不会（Fig. 8b）。

现象：episode 前半画的笔常被后半完全盖住，作者归因于**信用分配难**；中途用模糊版目标的中间奖励留作未来工作。[[Learning-to-Paint]] 在相同 20 笔、不透明笔设定下与之对比，报告更小 $\ell_2$（见其译文），并归因于可微渲染与逐步奖励等差异。

### 4.6 MuJoCo Scenes

仅**条件**生成：从输入图推断简单 CAD 式程序。奖励为 $\ell_2$ 或判别器输出，无辅助奖励。**20 步**、最多 20 个物体；训练集最多 5 个物体，智能体**事先不知道**应放几个。动作空间 cardinality 极大：单物体设置数 $M=4\cdot 16^2\cdot 3\cdot 4^3\cdot 3$，轨迹长 $N=20$，穷举 $M^N$ 不可行（脚注：真实场景配置数更小但仍难解）。

![[图/SPIRAL/fig8.png]]

Fig. 8：（a）MNIST / Omniglot、（b）CelebA / MuJoCo 上，训练过程中重建与真值的 $\ell_2$ 距离；实线为判别器奖励 $D$，虚线为直接 $\ell_2$ 奖励，前者终局误差更低（Omniglot 上 $D$ 与 blind 等见图例）。

![[图/SPIRAL/fig9.png]]

Fig. 9：MuJoCo holdout 上 Blocked MCMC 与 SPIRAL 的 $\ell_2$ 随迭代——SPIRAL 一次前向即近零误差，MCMC 数千步仍远高于 SPIRAL。

**Blocked Metropolis-Hastings** 基线：100 张 holdout，每步随机翻转一个物体的属性块，高斯似然（对角方差 0.25）做 MH 接受拒绝。数千次评估仍失败；SPIRAL **单次前向**即可处理每张图。判别器变体达近完美 holdout 重建；纯 $\ell_2$ 像素误差显著更高。

![[图/SPIRAL/fig10.png]]

Fig. 10：MuJoCo 条件重建样例——左为 SPIRAL 渲染、右为输入场景，物体数量、位置、大小与颜色大体一致。

## 5 讨论

把视觉程序合成扩展到真实、组合爆炸数据一直很难。本文表明：用**黑盒渲染模拟器** + **Wasserstein 判别器输出作奖励** + **异步[[强化学习]]**，可以训练对抗式生成智能体。当前探索主要靠熵；未来可用 MCTS（类比 AlphaGo Zero）或通用推理算法。动作空间参数化也可改——双控制点[[贝塞尔曲线]]对直线不友好等。

奖励侧，可试 **BiGAN/ALI 式联合图像—动作判别器**（策略当编码器、渲染器当解码器），让 $D$ 更关注语义。作者希望本文为逆模拟、程序合成在视觉、图形、语音、音乐、科学模拟等方向提供一条路径。

## 致谢

感谢 DeepMind 同事在稿件与讨论上的帮助（名单见 PDF 第 10 页）。

## 附录

### A 条件生成下的最优 $D$（PDF 第 11 页）

条件设定下 $p_d$ 为 Dirac，可写出最优（非参数）判别器形式。式 (1) 对应 Wasserstein-1 对偶；原问题 (8) 在 $p_d$ 为点质量时化为 (9)：

$$W_1(p_g,p_d)=\mathbb{E}_{x\sim p_g}\|x-x_{\mathrm{target}}\|_2 \tag{9}$$

故 $D(x)=\|x-x_{\mathrm{target}}\|_2$ 满足 (1) 之一解 (10)。**但 $\ell_2$ 不是唯一解：** 当 $p_g$ 也是 Dirac 于 $x_g$ 时，任意满足 (11) 且 Lipschitz $\le 1$ 的 $D$ 均可。在 $x_g$ 附近、到 $x_{\mathrm{target}}$ 等距的点集 $V$ 上，$\ell_2$ 不区分语义更近的点；而某些超平面解在 $V$ 上梯度非零，可能更快把搜索推向好区域。

另一差异：实践里 WGAN 目标与 SGD **未收敛到** Wasserstein 对偶精确解。

![[图/SPIRAL/fig11.png]]

Fig. 11：玩具圆盘数据上，$\ell_2$ 相对目标的 3D 曲面在圆不相交处几乎平坦；学习的 $D(x,x_{\mathrm{target}})$ 仍有谷但梯度更易引导搜索。

### B 网络结构（PDF 第 11–12 页）

![[图/SPIRAL/fig12.png]]

Fig. 12：策略网络——画布 $C_t$ 与上一步动作 $a_t$ 经卷积、ResNet、FC 与 LSTM 得隐状态，再送入 Fig. 13 解码器输出 $a_{t+1}$。

![[图/SPIRAL/fig13.png]]

Fig. 13：动作自回归解码——标量维走 FC 分类；空间位置走 ResNet + 反卷积；每采一维嵌入 16 维并与 $z_i$ 拼接得 $z_{i+1}$，直至 8 维动作 tuple 完成。

**策略 $\pi$：** 输入观测（画布 $C_t$）与上一步动作 $a_t$；步长卷积降采样 + ResNet 栈 + 全连接 → 嵌入 → **LSTM** → 隐向量 $z_0$ 驱动自回归解码。

**动作解码：** 动作各维 $a_t^i$ 自回归采样。标量（笔刷大小等）用 FC 出类别分布；空间位置（控制点等）用 ResNet + 转置卷积 + 卷积。每采一维，嵌入 16 维与 $z_i$ 拼接得 $z_{i+1}$，直至整 tuple 完成。

**判别器：** 类似 DCGAN 的常规 CNN。条件任务中 $D$ 接收图像对。

### C 训练细节（PDF 第 12 页）

- 判别器：**Adam**，$10^{-4}$ 学习率，$\beta_1=0.5$。
- 策略：**PBT**（Jaderberg et al., 2017）搜熵系数与学习率；种群 12 个实例，每实例 **64 CPU actor** + **2 GPU**（policy learner + discriminator learner）。假设不同实例的判别器分数可比，用作 PBT 适应度。
- Batch size **64**（policy 与 $D$ 相同）；生成数据从容量 **20 batch** 的 replay buffer 均匀采样。

**参考文献：** 正文 PDF 第 10–11 页起，不逐条译文。

---

## 标题清单与 PDF 页码

| 原文 | 译文标题 | PDF 页 |
| :--- | :--- | :--- |
| Abstract | ## 摘要 | 1 |
| 1. Introduction | ## 1 引言 | 1 |
| 2. Related Work | ## 2 相关工作 | 2–3 |
| 3. The SPIRAL Agent | ## 3 SPIRAL 智能体 | 3 |
| 3.1 Overview | ### 3.1 概述 | 3 |
| 3.2 Objectives | ### 3.2 目标函数 | 3–4 |
| 3.3 Conditional Generation | ### 3.3 条件生成 | 4 |
| 3.4 Distributed Learning | ### 3.4 分布式学习 | 4 |
| 4. Experiments | ## 4 实验 | 4 |
| 4.1 Datasets | ### 4.1 数据集 | 4–5 |
| 4.2 Environments | ### 4.2 环境（不可微渲染） | 5 |
| 4.3 MNIST | ### 4.3 MNIST | 5–6 |
| 4.4 O MNIGLOT | ### 4.4 Omniglot | 6–7 |
| 4.5 C ELEBA | ### 4.5 CelebA | 7–8 |
| 4.6 M U J O C O S CENES | ### 4.6 MuJoCo Scenes | 8–9 |
| 5. Discussion | ## 5 讨论 | 9 |
| Acknowledgements | ## 致谢 | 10 |
| References | （不逐条译） | 10–11 |
| A. Optimal D… | ### A 条件生成下的最优 $D$ | 11 |
| B. Network Architectures | ### B 网络结构 | 11–12 |
| C. Training Details | ### C 训练细节 | 12 |

**主要图号与页码：** Fig. 1（1）；Fig. 2（2）；Fig. 3（5）；Fig. 4–5 MNIST/Omniglot（6）；Fig. 6 Omniglot 解析（7）；Fig. 7 CelebA（7）；Fig. 8 $\ell_2$ 曲线（8）；Fig. 9 MCMC 对比（8）；Fig. 10 3D 重建（8）；Fig. 11–13 附录（12）。

**PDF 总页数：** 12。
