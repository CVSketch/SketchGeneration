---
title: "VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation"
short: VQ-SGen
authors: Jiawei Wang, Zhiming Cui, Changjian Li
year: 2025
venue: ICCV 2025
ccf: A
arxiv: "2411.16446"
pdf: "[[VQ-SGen.pdf]]"
tags:
  - 参考文献
  - 素描生成
status: 已译
---

# VQ-SGen

谁在引：[[StrokeFusion]]。FaceX 上 StrokeFusion 拿它当对照。

全文译文：[[VQ-SGen-译文]]。要点：[[VQ-SGen-要点]]。原文：[[VQ-SGen.pdf]]。

## 一句话

先把每一笔收成词典里的词，再用 Transformer 生成整张创意简笔。

对着 [[任务定义]]：输出是拼好的创意简笔终稿。[[StrokeFusion]] 在引，把它当「栅格 + 离散 token + 自回归」对照。Stage 1 / Stage 2 是码本加抽码，不是课堂四阶段。解码一步只是下一个 token，不是排线。
