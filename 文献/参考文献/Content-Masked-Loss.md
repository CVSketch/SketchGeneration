---
title: "Content Masked Loss: Human-like Brush Stroke Planning in a Reinforcement Learning Painting Agent"
short: Content-Masked-Loss
authors: Peter Schaldenbrand, Jean Oh
year: 2021
venue: AAAI 2021
ccf: A
arxiv: "2012.10043"
pdf: "[[Content-Masked-Loss.pdf]]"
code: https://github.com/pschaldenbrand/ContentMaskedLoss
tags:
  - 参考文献
  - 油画过程
status: 已译
---

# Content Masked Loss

谁在引：[[ProcessPainter]]。

全文译文：[[Content-Masked-Loss-译文]]。要点：[[Content-Masked-Loss-要点]]。原文：[[Content-Masked-Loss.pdf]]。

## 一句话

强化学习作画时，损失先盯人能认出来的地方，再补其余。

这是给过程中间帧加权的损失。[[ProcessPainter]] 只在相关工作里点过这篇，训练没有用这个损失，也没有课堂阶段监督。
