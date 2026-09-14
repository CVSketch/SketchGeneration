---
title: "Content Masked Loss: Human-like Brush Stroke Planning in a Reinforcement Learning Painting Agent"
short: Content-Masked-Loss
source: "[[Content-Masked-Loss]]"
pdf: "[[Content-Masked-Loss.pdf]]"
tags:
  - 译文
  - 全文
status: 全文
---

# Content-Masked-Loss · 全文译文

对应笔记：[[Content-Masked-Loss]]。要点：[[Content-Masked-Loss-要点]]。原文：[[Content-Masked-Loss.pdf]]。代码：https://github.com/pschaldenbrand/ContentMaskedLoss

> [!abstract] 这篇怎么用
> 按原论文章节用中文写全。标题跟 PDF 一致，方便以后从 [[Content-Masked-Loss-要点]] 点进来。
> 核心想法：**[[强化学习]]作画时，奖励里的像素损失先盯「人能认出来」的区域**（用目标图上的 VGG 特征做掩码加权），让笔序更像人类「铺大面、早认主体」，又不明显牺牲成画质量。
> 公式、表号、图号跟原文；正文 Fig. 旁已嵌裁图。

## 摘要

多数[[强化学习]]绘画智能体的目标，是缩小目标图与画布之间的损失。人类画家更强调目标里的重要结构，而不是像素级照搬（DiPaola 2007）。在 RL 绘画模型里用[[对抗与判别器|对抗]]损失或 $L_2$ 损失，成画往往不错，但笔序和人类差很远——模型不知道图像里哪些是抽象意义上的「内容」。

为在不采集昂贵人类笔迹数据的前提下，让规划更像人，我们提出用于奖励函数的新损失：**Content Masked Loss（内容掩码损失）**。在机器人绘画场景里，它借助目标检测式特征提取（VGG-16），给画布上**对人识别主体更重要**的区域更高权重。332 名人类评估者表明：相对仅用对抗或 $L_2$ 的基线，本方法在笔序更早阶段就能让人看出画的是什么，而最终画质没有明显变差。

## 1 引言

熟练画家和机器打印机的做法差别很大，成品有时却很像。画写实肖像时，画家希望用颜料尽量贴近目标图；打印机目标类似，但过程完全不同——例如从左上到右、自上而下扫像素。画家则更抽象：常用「blocking in（铺大面、起形）」（Scott 2017），**过程还没结束，主体往往已经能认出来**，随后细节慢慢加上。绘画因此像一种 **anytime 算法**：随时停下都有一张「还能看」的画；画得越久通常越好。

现有笔画渲染（Stroke-Based Rendering, SBR）里的[[强化学习]]方法（Ganin et al. 2018；Huang, Heng, and Zhou 2019；Nakano 2019；Xie, Hachiya, and Sugiyama 2012）多在奖励里用[[对抗与判别器|对抗]]或 $L_2$ 损失，只关心画布与目标的整体差异，**对抽象内容不敏感**，笔序可以任意，只要最后损失低就行。

![[图/Content-Masked-Loss/fig1.png]]

Fig. 1：本文模型用带 Content Masking 的裁剪 $L_1$ 损失训练；基线用对抗损失。列是 10、30、100、300、750 笔和目标图。这是中间成画对比，不是课堂四阶段。

和现有 SBR+RL 类似，很多模型**要到最后一批笔才看得清画的是什么**（Fig. 3）。在 GAN 等生成艺术里，往往只评最终图；**机器绘画还要评整个过程**。本文把绘画看成通向成画的**动作（笔）序列规划**，希望笔序更接近人类：主体随笔数增加而越来越可辨。

一种做法是收集真人逐笔数据，但样本要多、每条耗时长、题材还要多样，成本高。本文目标是：**让 RL 画家的笔序规划更像人**，聚焦「blocking in」——主体在过程早期就要能认出来。为此引入 Content Masked Loss：在奖励里**更奖励画在「对认出主体重要」的区域**。用目标识别模型标出这些区域。Amazon Mechanical Turk 上，相对对抗或 $L_2$ 基线，用 Content Masked Loss 的模型**更早**就能让人看出题材。

![[图/Content-Masked-Loss/fig2.png]]

Fig. 2：RL 画家生成笔参数，经 Neural Renderer 或机器人执行。实验用 TinkerKit Braccio 机械臂；目标图经 k-means 得 10 色，人工调色；单刷、可水洗。绘画过程全自动。

## 2 相关工作

**笔画渲染（SBR）** 把形状、笔刷等元素叠到数字画布上，复现给定图像；目标可以是完美复刻，也可以带风格化抽象。

深度学习 SBR（Ganin et al. 2018；Huang, Heng, and Zhou 2019；Nakano 2019；Xie, Hachiya, and Sugiyama 2012）之前，已有大量确定性方法（Hertzmann 2003）：每步放一笔，使下一步画布与目标的损失下降。「Artist Agent」（Xie, Hachiya, and Sugiyama 2012）用 RL 设计单道笔画，并用反馈判断像不像人画的；单笔画很像人画的，但整幅画笔序仍靠人工。

**整幅画的 RL SBR**：SPIRAL（Ganin et al. 2018）用 RL 生成笔，使成画接近目标。后续工作（Nakano 2019；Zheng, Jiang, and Huang 2019；Huang, Heng, and Zhou 2019）常用 world model 加速训练；绘画里 world model 即 **Neural Renderer**，可微以便反传。Huang et al. 2019 在把图像拆成笔序列上效果突出；RL 目标仍是缩小成画与目标的差异，作者发现 WGAN 的 Wasserstein-1 距离优于 $L_2$。笔够多时几乎能复刻输入，但**过程里看不出题材**。

**配对笔迹数据** 对肖像质量不够：QuickDraw（Ha and Eck 2017）有笔与图，但太简；SketchRNN 能学人类素描规划，但只能画训练集里标过的类，且是素描不是油画式绘画。

Huang et al. 与 SPIRAL 等靠试错学画，奖励只驱动「哪一笔能降损失」，**没有人类示范**。本文用 Content Masked Loss 在不加人类标注的前提下提高笔序的人味。

Huang et al. 的笔形**真实画笔难以复现**（Fig. 3），难以判断笔序是否像人。本文把 SBR+RL **约束为真实笔刷可执行的指令**（Fig. 2），便于评估并在机器人上执行。

![[图/Content-Masked-Loss/fig3.png]]

Fig. 3：行 (a) 为 Huang et al. 2019 模型；行 (b) 为本文基线（约束 Neural Renderer）。约束：固定笔宽、不透明、有最大笔长；基线从**白**画布起，Huang et al. 从黑画布起。

## 3 方法（Approach）

**问题（非形式）**：给定输入图与空白画布，生成笔序列，使画布最终接近输入，且过程风格像人。下文是 RL 形式化，重点在**奖励如何诱导类人规划**。

### 3.1 强化学习画家模型（Reinforcement Learning Painter Model）

在 Huang, Heng, and Zhou 2019 的 deep RL 框架上改动，去掉若干不真实现实绘画的假设：

1. **笔的尺寸**：原文允许超大笔（可超过画布半宽，Fig. 3）。本文按机器人单刷能力约束：固定笔宽、长度为画布宽度的 5%，与实刷一致，便于把指令交给 Fig. 2 的臂。
2. **初始画布**：改为**白色**（常见纸/画布），而非黑色。
3. **笔长与不透明**：刷上颜料有限，最大笔长设为画布半宽（按 20 cm 纸面试画经验）；丙烯不透明，渲染里按不透明处理。

**MDP 设定**（同 Huang et al.）：状态是画布从空到完成的过程；动作为 Bézier 曲线笔，含三个位置控制点、两端粗细、颜色等。Neural Renderer 为转移函数，把动作画到当前画布。

动作连续，采用 **DDPG**（Lillicrap et al. 2016）。奖励为 $t$ 与 $t+1$ 时刻损失之差（式 (1)）。Huang et al. 用判别器算损失（式 (2)）；DDPG 的 critic 预测期望回报：

$$V(c_t) = r(c_t, a_t) + \gamma V(c_{t+1})$$

$r(c_t, a_t)$ 为在画布 $c_t$ 执行 $a_t$ 的即时奖励，$\gamma$ 控制未来权重。$c_{t+1} = \mathrm{trans}(c_t, a_t)$ 由 Neural Renderer 渲染得到。Actor $\pi(c_t)$ 最大化 $r(c_t, \pi(c_t)) + \gamma V(\mathrm{trans}(c_t, \pi(c_t)))$。

### 3.2 奖励函数（Reward Functions）

![[图/Content-Masked-Loss/fig4.png]]

Fig. 4 汇总：奖励为**逐像素**损失从 $t$ 到 $t+1$ 的均值变化；$c$ 为画布，$y$ 为目标图。

**基线（Baseline）**

$$\mathrm{Reward} = \mathrm{mean}\big[L(c_t, y) - L(c_{t+1}, y)\big] \tag{1}$$

$$L_{\mathrm{GAN}}(c, y) = 1 - \mathrm{Discriminator}(c, y) \tag{2}$$

$$L_{L_2}(c, y) = \|c - y\|_2^2 \tag{3}$$

判别器输出 1 表示 $c$ 与 $y$ 相同、0 表示完全不同；只学视觉差异，**不考虑内容与抽象结构**。

**Content Loss（内容损失，式 (4)）**

$$L_{\mathrm{CL}}(c, y) = \big(\mathrm{VGG}_l(c) - \mathrm{VGG}_l(y)\big)^2 \tag{4}$$

在 VGG-16（Simonyan and Zisserman 2015）特征上做 $L_2$，而非 RGB 像素；VGG 参数固定。直觉：若成画与目标在「识别用特征」上接近，应更像同一对象。VGG 在照片上训练，对绘画泛化存疑；更鲁棒的特征提取器可能是未来方向。

**Content Masked Loss（内容掩码损失，本文核心，式 (5)(7)）**

画家会抽象地考虑题材；blocking in 让主体在过程早期可辨。除 Content Loss 外，本文主要用 **Content Masked** 思路：**在「重要」区域加大逐像素损失权重**。重要性由 VGG-16 对**目标图**提取的特征决定。

![[图/Content-Masked-Loss/fig5.png]]

Fig. 5：在 $L_2$ 差分上，乘以从目标图得到的特征掩码。

流程：目标图喂入 VGG-16，取某中间层输出，**通道维平均**，再 **0–1 归一化**，得到 $\mathrm{norm}(\mathrm{VGG}_l(y))$，与逐像素差相乘：

$$L_{\mathrm{CM}+L_2}(c, y) = (c - y)^2 * \mathrm{norm}(\mathrm{VGG}_l(y)) \tag{5}$$

即：**损失先盯 VGG 认为与识别相关的像素位置**（通常对应脸、眼、嘴等高层结构），RL 为降加权损失会更早在这些区域落笔，而不是均匀铺像素。

**$L_1$ 与损失裁剪（Loss Clipping）**

除 Content Loss 外，其余损失都是画布与目标的像素距离。RGB 下白–黑距离最大；画布初始为白，模型会偏向先画深色以快速拿奖励，**掩盖 Content Masking 的效果**。作者试验 $L_1$ 并对逐像素损失设上限 $\lambda$：

$$L_{L_1^*}(c, y) = \min(|c - y|, \lambda) \tag{6}$$

$$L_{\mathrm{CM}+L_1^*}(c, y) = \min(|c - y|, \lambda) * \mathrm{norm}(\mathrm{VGG}_l(y)) \tag{7}$$

后文主结果模型为 **CM + $L_1^*$**（Fig. 1）。

## 4 实验（Results）

### 4.1 笔约束与基线画质

![[图/Content-Masked-Loss/fig6.png]]

Fig. 6：左为目标图。(a) Huang et al. 原模型；(b) 加笔能力约束的基线。各生成 125 笔（整图）再加 25 块各 125 笔。

Fig. 3 已展示固定笔宽、最大笔长、不透明、白底的影响：基线能复现目标，但保真度低于 Huang et al.；笔更细、更短，需要更多笔；固定宽度难以画嘴、眼等细节。把画布划成子块、分块跑模型，等价于用更细的刷。子块策略下，基线最终损失可接近 Huang et al.

### 4.2 Content Loss 奖励

![[图/Content-Masked-Loss/fig7.png]]

Fig. 7：上行左三为模型成画，右为目标照片；下行是各图用 VGG 前 17 层抽出的特征。原文没有写这三张成画各用了多少笔。

用式 (4) 作奖励时，模型易陷局部最优，**只输出空白画布**（参数怎么调都这样）。原因：VGG 在照片上学到的特征，对只有少量笔的画布几乎随机；即使成画已很像目标，抽出的特征仍与照片目标差很远。未来可试纹理偏置更小的分类器（Geirhos et al. 2019）。

### 4.3 Content Masked 奖励与 VGG 层选择

![[图/Content-Masked-Loss/fig8.png]]

Fig. 8：VGG-16 第 0–28 层特征。浅层偏边缘和纹理，深层偏眼、嘴。

VGG-16 有 13 个卷积层（各带激活）和 5 个池化层，一共 31 层都能做掩码。Fig. 8 画的是前 29 层。用式 (5) 在不同层训练多组模型，中间笔序不同：浅层更铺整脸，深层更盯抽象部位；**最终成画相近**，但深层掩码的中间过程更像人。后续实验固定 **第 17 层**，在高低层特征间折中。

![[图/Content-Masked-Loss/fig9.png]]

Fig. 9：不同 VGG 层做掩码时的中间笔序。列是 5、20、50、100、200 笔；行是第 2、8、12、17、22、31 层。深层更盯五官，不是课堂四阶段。

### 4.4 最终绘画质量

![[图/Content-Masked-Loss/table1.png]]

用 **Fréchet Inception Distance（FID）** 与 **Inception Score（IS）** 比较不同奖励（Barratt and Sharma 2018 指出 IS 等有局限）。在 CelebA **2000** 张 hold-out 肖像上，5 折交叉验证，结果见 Table 1。

| 损失函数 | FID ↓ | IS ↑ |
| :--- | :--- | :--- |
| GAN | 240.05 ± 0.43 | 3.05 ± 0.08 |
| $L_2$ | 241.02 ± 0.24 | 3.05 ± 0.14 |
| CM + $L_2$ | 242.67 ± 0.24 | 3.01 ± 0.04 |
| $L_1^*$ | 241.58 ± 0.14 | 3.45 ± 0.09 |
| CM + $L_1^*$ | 243.13 ± 0.20 | 3.47 ± 0.05 |

Table 1：FID 越小表示成画与照片越「像」；IS 越大表示分布越接近。本文成画是绘画风格，FID/IS 绝对值常与 photorealistic GAN 文献不可比；**相对比较**即可。没有一种损失在 FID 与 IS 上全面碾压其余，说明 **Content Masking 没有显著牺牲最终画质**。

![[图/Content-Masked-Loss/fig10.png]]

Fig. 10：不同损失训练模型的绘画过程。GAN、$L_1^*$、$L_2$ 的中间结果几乎不受「画的是谁」影响；**CM + $L_2$、CM + $L_1^*$ 会更鼓励先画对识别重要的区域**，观者更早看出题材。

### 4.5 类人规划（Human-Like Planning）

人类 blocking in 下，**主体应随过程变清晰**。验证分两条线：FaceNet 人脸检测随笔数变化；Amazon MTurk 人类排序。

**说明**：人类偏形状，CNN（含 FaceNet）有纹理偏置（Geirhos et al. 2019）；两实验形式不同（检测 vs 排序），但在 **200 笔以内** 结论一致——**Content Masking 让主体更早可辨**。

对比五组模型，**仅奖励里的损失不同**：

- **GAN**：对抗损失（式 (2)），等同 Huang et al. 损失 + 本文笔约束。
- **$L_2$**：式 (3)。
- **$L_1^*$**：式 (6)。
- **CM + $L_2$**：式 (5)。
- **CM + $L_1^*$**：式 (7)。

![[图/Content-Masked-Loss/fig11.png]]

Fig. 11：FaceNet 在指定笔数内检出人脸的比例。横轴是笔数。

**AI 评估**：各模型对 hold-out 中 **2000** 张肖像逐笔作画；每加一笔用 FaceNet（Schroff, Kalenichenko, and Philbin 2015）检测是否有人脸。Content Masking 总体提升 **50–200 笔** 区间的检出率；对 $L_2$ 提升较弱（深色偏好可能掩盖掩码）。**CM + $L_1^*$** 检出最高。

![[图/Content-Masked-Loss/fig12.png]]

Fig. 12：MTurk 题目示例。问哪张最像人脸。上行是早段白底粗笔，下行是晚段接近成画。

![[图/Content-Masked-Loss/fig13.png]]

Fig. 13：工人把哪一组选成「最像脸」。横轴是 10、30、100、200、750 笔。

**人类评估**：五模型逐笔作画；向工人展示 **10、30、100、200、750** 笔时的画布。五张画来自五模型，问「哪张最像脸」。算法顺序与笔数顺序随机。**332** 名独立 MTurk 工人；**400** 张图；每题 **4** 人评。在 **30–200 笔**，**CM + $L_1^*$** 被选为「最像脸」的比例明显更高；CM + $L_2$ 提升不显著，作者归因于 $L_2$ 的深色偏置——**CM 配 $L_1^*$ 更成功**。750 笔附近各模型差距缩小。

## 5 结论与未来工作（Conclusion and Future Work）

本文在[[强化学习]] SBR 中强调**绘画过程**与最终质量，提出 Content Masked Loss：用 VGG-16 特征给目标图上「对识别重要」的像素更高损失权重，使笔序更早呈现可辨主体，**332 人实验与 FaceNet 检测均支持这一点**，且 Table 1 表明最终画质大致持平。相对采集人类笔迹，这是**数据高效**的类人规划手段。

VGG 特征与面部结构相关（Fig. 8），故可用于掩码；未来计划换**纹理偏置更小、更接近人类感知**的分类器来生成权重矩阵。

## 致谢（Acknowledgements）

感谢 Huang、Heng、Zhou 的开源代码（https://github.com/hzwer/ICCV2019-LearningToPaint）作为 RL 基础；感谢 Nicholas Eisley 提供 Fig. 1、3、6、7、10 等照片。

## 参考文献（References）

正文引用含 DiPaola 2007；Ganin et al. 2018 SPIRAL；Huang, Heng, and Zhou 2019；Hertzmann 2003；Ha and Eck 2017 QuickDraw / SketchRNN；Geirhos et al. 2019；Simonyan and Zisserman 2015 VGG；Lillicrap et al. 2016 DDPG；Heusel et al. 2017 FID；Salimans et al. 2016 IS；Schroff et al. 2015 FaceNet；Scott 2017 blocking in 等。完整条目见 PDF 第 7–8 页。
