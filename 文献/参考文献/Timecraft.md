---
title: "Painting Many Pasts: Synthesizing Time Lapse Videos of Paintings"
short: Timecraft
authors: Amy Zhao, Guha Balakrishnan, Kathleen M. Lewis, Frédo Durand, John V. Guttag, Adrian V. Dalca
year: 2020
venue: CVPR 2020
ccf: A
arxiv: "2001.01026"
pdf: "[[Timecraft.pdf]]"
code: https://github.com/xamyzhao/timecraft
tags:
  - 参考文献
  - 油画过程
status: 已译
---

# Timecraft

谁在引：[[Inverse-Painting]]。项目常叫 Timecraft。

全文译文：[[Timecraft-译文]]。要点：[[Timecraft-要点]]。原文：[[Timecraft.pdf]]。

## 一句话

从一张画完的画，反推它可能怎么一步步画出来，做成延时视频。

训练先把帧缩到 $126 \times 168$，再裁 $50 \times 50$ 小块。铺的是数字画和水彩，不是铅笔排线。[[Inverse-Painting]] 批评的就是这块：只看小块，看不见整幅在画什么。
