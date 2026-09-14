---
title: "ProcessPainter: Learn Painting Process from Sequence Data"
short: ProcessPainter
source: "[[ProcessPainter]]"
tags:
  - 译文
  - 论文
status: 第一轮
---

# ProcessPainter · 故事与方法译文

> [!warning] 旧意译，只留底
> 请改读 [[ProcessPainter-译文]]。不要再改本页。


对应笔记：[[ProcessPainter]]。原文：[[ProcessPainter.pdf]]。

> [!abstract] 版权与写法
> 这里是**意译**，不是逐句复述全文。故事（为什么要做）按作者口吻用中文重讲。方法按节用自己的话写清。只对 2～4 段关键句做对照精翻。图用库内裁图，不整页贴 PDF。

## 作者想讲的故事

画家不是一次出成品。他们先铺大关系，再往上加细节。人跟人、画种跟画种，步骤差得很远。

美术课很需要看见「怎么画」。临摹名作、老师改半成品，都靠中间稿。机器以前两条路都不对。

一条是笔画渲染。把静图拆成一串笔触。目标是缩小当前画布和原图的差。步骤可以显得由粗到细，但仍是在贴像素。它没有去学某个画家自己的习惯。

另一条是文生图 [[扩散模型]]。从噪声一步步变清楚。中间帧只是糊图。把潜变量用 VAE 解出来，也还是糊的，不是能临摹的中间稿。

网上的作画录像也不好用。手挡住画面，镜头推拉，角度乱变。真过程太少，直接当视频模型的训练集不够。

作者换了一个问法：把作画过程当成一段很短的视频。先用假过程，让网络学会「图会随时间变」。再用很少的真人过程做 [[LoRA]] [[微调]]，把步骤纠成人。换一小截边路，就能换一种画法。再加一个能卡住任意帧的参考图网络。同一套模型就能做三件事：看文字出过程、看成品反推过程、看半成品继续画。

这篇不是 [[自回归]] 逐步出帧。它一次生成整段 8 帧。和一张推一张的写法不同。

## 方法按节

### 总体在干什么

主干是带时间注意力的潜扩散。思路接近 AnimateDiff。原来的 VAE 和 UNet 冻住。每一层后面加帧间自注意力，让 8 帧在内容上对得上。

时间注意力把各帧特征排成一条，再做自注意力：

$$
\mathrm{Temporal\text{-}Attention}(Q,K,V)=\mathrm{Softmax}\left(\frac{QK^{T}}{\sqrt{c}}\right)V
$$

另外一个旁路叫参考图网络（Artwork Replication Network）。做法像 ControlNet。指定第 $\tau$ 帧必须像参考图。输出加到去噪 UNet 上：

$$
\hat{S}_{n}^{u}=S_{n}^{u}+\lambda\,ARN_{n}
$$

$\tau$ 放最后一帧，就是成品反推过程。$\tau$ 放开头，就是半成品续画。$\tau$ 越靠近开头，后面画得越满。

三个任务装在一篇里：

1. **文生过程（Text2Painting）**：只给提示词，关掉参考图网络，出 8 帧。
2. **成品反推（Image2Painting）**：参考图卡住最后一帧。文字可有可无。
3. **半成品续画（Semi2Complete）**：半成品卡住第一帧，再用文字把后面画完。同一张半成品，换一句话，可以走出不同结果。

换 UNet 或换 [[LoRA]]，就能换画法。厚涂油画、半透明笔触、水墨，都可以插拔。

### 数据怎么来

真过程太少。预训练靠假过程。

正文第 3.3 节写：预训练 **3 万** 段合成序列；每段 **8** 帧、**512×512**。[[LoRA]] 再用 **95** 段画家过程。图 2 图说写成 4 万，和正文不一致。以第 3.3 节、第 4.1 节为准。

假过程这样造：

1. 从 DiffusionDB 按美感抽出 **1 万** 张静图。
2. 每张用三种笔画渲染做成过程：Learn-to-Paint、Paint Transformer、Stylized Neural Painting。这些方法后几百步几乎只修细节。所以不均匀抽 8 帧，不要把几乎不变的尾巴都留下来。
3. 再加第四种造法。用 Segment Anything 切物体，用 Depth Anything 估远近，从近到远往白画布上贴。这是在模拟「先画前面，再画后面」。

假过程本身不像人。它只让网络先知道：画布会从空变满。

真人过程：3 位画家，共 95 段。画种是厚涂肖像、风景色块、线稿上色。每位画家各训一份 Painting LoRA。说明文字里加触发词 `sks`。

第 4.2 节还写：预训练过之后，十来个样本也能复现一位画家的步骤和味道。图 4 图说写成 10 到 50 段。

### 模型怎么走

训练分几段。

1. 在 3 万段假过程上预训练 Painting Model，**5 万** 步。时间模块先学会「从空到满」。
2. 冻住 Painting Model，再训参考图网络。正式训练也是 **5 万** 步。每次随机抽 0 到 3 帧当条件。首帧、尾帧各三分之一概率。中间帧按中间高的正态分布。作者说这样更像真实使用。
3. Painting LoRA 拆成两步。目的是把「最后一张长什么样」和「中间怎么走」分开。
    - 先只用最后一帧训空间注意力 LoRA。没画完的中间稿如果也拿来改画质，会把成品带糊。
    - 再冻住这一截，用整段 8 帧训时间注意力 LoRA。
4. 优化器是 Prodigy。学习率 $2\times 10^{-5}$，batch 大小是 1，分辨率 512×512，显卡是 NVIDIA A100。

LoRA 本身只在注意力层上加一小截：

$$
W'=W+\Delta W=W+AB^{T}
$$

推理时：

- 纯文生：不用参考图网络。
- 反推、续画：参考图先过 VAE。第 $\tau$ 帧的噪声换成这张图的潜变量。再用 DDIM inversion 做噪声替换，让指定帧贴住。采样是 DDIM **50** 步。

### 怎么知道自己错了

作者说：文生过程和半成品续画，当时没有现成对手。所以定量比较只做「成品反推」。对手是四种笔画渲染：LearnToPaint、Paint Transformer、Intelli-Paint、Stylized Neural Painting。

表 1 只量最后一帧和参考图有多像。

| 方法 | MSE↓ | LPIPS↓ | L1↓ |
| :--- | :--- | :--- | :--- |
| LearnToPaint | 0.016181 | 0.033240 | 0.087082 |
| Paint Transformer | 0.087695 | 0.153372 | 0.187685 |
| Intelli-Paint | 0.247486 | 0.397536 | 0.350746 |
| Stylized Neural Painting | 0.084447 | 0.141126 | 0.185832 |
| 本文 | 0.014820 | 0.024517 | 0.082165 |

过程像不像人，靠用户研究。44 人，每人看 28 对过程，每段是 5 秒 GIF。问两件事：哪段更像人在画，哪段更喜欢。

| 对比 | 更像人 (%) | 更喜欢 (%) |
| :--- | :--- | :--- |
| vs LearnToPaint | 78.2 | 68.2 |
| vs Paint Transformer | 80.4 | 65.9 |
| vs Intelli-Paint | 84.5 | 78.4 |
| vs Stylized Neural Painting | 78.6 | 71.6 |

消融在原文图 7。参考图网络和 DDIM 噪声替换都要留下。少一个，最后一帧就对不齐。

局限也写在原文第 5 节。序列只有 8 帧、512 边长，显存不够就加不了帧。真过程仍然难收。输出是关键帧栅格，不是每一笔矢量轨迹。

## 图在讲什么

每张裁图：先嵌入，再写两句人话。

![[图/ProcessPainter/p01-01.png]]

图 1 是全文任务图。上行：一句话走出从空到满的风景。中行：红框是成品，前面几帧是反推出来的过程。下行：红框是半成品第一帧，后面按文字画完。

![[图/ProcessPainter/p04-02.png]]

图 2 是总流程。左：假过程预训练，空间注意力和时间注意力一起学。中：真人过程上训两截 LoRA。右：推理时用 DDIM inversion 换噪声，参考图网络卡住指定帧。

![[图/ProcessPainter/p06-03.png]]

图 3 是假过程预训练之后，纯文生能走出的四种机器画法。半透明贝塞尔由糊到清楚；小方块笔触慢慢聚成人；大色块收到五官；物体按分区依次出现。这些步骤来自合成数据，还不是某位画家的习惯。

![[图/ProcessPainter/p06-04.png]]

图 4 是 10 到 50 段真过程微调之后。上排是画家过程样本，下排是生成结果。左：风景色块。中：线稿上色。右：厚涂肖像。换一份 LoRA，步骤和最终味道一起换。

![[图/ProcessPainter/p07-06.png]]

图 6 是参考图网络的两个用法。(a) 成品卡在最后一帧，向日葵和马都从大色块收到原图。(b) 半成品卡在第一帧。同一座山的剪影，一句话走出雪景，另一句话走出火山夕照。

## 关键句对照

> Because painting is generally a gradual instantiation process, moving from abstract to specific, from macro to detail, which is far removed from how diffusion models generate images through a denoising process. Decoding the latent from the denoising process through a VAE decoder only yields blurry images, not meaningful painting processes.

**精翻：** 作画通常是由抽象到具体、由大关系到细节，一步步把东西画实。这和扩散模型靠去噪出图差得很远。把去噪中的潜变量用 VAE 解出来，得到的只是糊图，不是有意义的作画过程。

**为什么重要：** 这篇的对手不是「最后一张漂不漂亮」，而是「中间帧算不算在画画」。素描若只报成品像不像石膏像，别人会当成普通生成。

> [B]aseline methods … are fundamentally designed to minimize the difference between the real image and the current canvas, resulting in a painting process that does not conform to human painting habits.

**精翻：** 基线本质上是在缩小原图和当前画布的差，所以过程不符合人的作画习惯。

**为什么重要：** 假过程可以用来预训练。它不能代替「像人」。像人这件事，要靠真过程和用户研究。

> [W]e initially fine-tuned the Spatial-Attention LoRA using only the final frame to prevent unfinished painting works from damaging the image quality of the model. After this step, we froze the parameters of the Spatial-Attention LoRA and fine-tuned the Temporal-Attention LoRA using the complete painting sequence.

**精翻：** 先只用最后一帧微调空间注意力 LoRA，避免没画完的图把画质带坏。然后冻住它，再用整段过程微调时间注意力 LoRA。

**为什么重要：** 真过程里大量是半成品。两步拆开，是这篇能用 10 到 50 段就换画法的关键。素描真过程同样少，可以抄这个拆法。

> A Painting LoRA can be fine-tuned only on 10-50 sequences of artists' painting process, which can effectively capture the characteristics of the artists' painting process and the style of the final results.

**精翻：** 只用 10 到 50 段画家过程微调 Painting LoRA，就能抓住这位画家的步骤习惯和最终风格。

**为什么重要：** 不要等几百段课堂延时才开工。先造假过程，再收十几段真人过程。

## 这篇实现了什么

- 输入 → 输出：文字，或成品 / 半成品参考图 → 8 帧、512×512 的作画关键帧。不是一笔一笔的矢量轨迹。
- 新模块：带时间注意力的 Painting Model；空间、时间拆开的 Painting LoRA；能卡住任意帧的参考图网络。
- 公开数字（只写原文有的）：合成 3 万段（图 2 图说写成 4 万）。真人 95 段、3 位画家。LoRA 常用 10 到 50 段。预训练和正式训练各 5 万步。学习率 $2\times 10^{-5}$。DDIM 50 步。表 1 最后一帧 LPIPS 是 0.024517。表 2 有 44 人，更像人大约 78% 到 85%。
- 缺什么：关键帧栅格，不是每笔矢量。真过程覆盖面窄。8 帧是显存上限。表 1 不衡量中间帧顺序。直接搬到素描会丢掉线序和疏密。
