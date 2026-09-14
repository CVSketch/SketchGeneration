---
title: "CLIPasso: Semantically-Aware Object Sketching"
short: CLIPasso
authors: Yael Vinker, Eitan Pajouheshgar, Jessica Y. Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, Ariel Shamir
year: 2022
venue: SIGGRAPH 2022
ccf: A
arxiv: "2202.05822"
pdf: "[[CLIPasso.pdf]]"
code: https://clipasso.github.io/clipasso/
tags:
  - 论文
  - 素描生成
  - 矢量图
  - CCF-A
status: 对照
---

# CLIPasso

先看 [[入门-读论文前先看]]。这篇用到：[[分数蒸馏]]、[[可微渲染器]]、[[CLIP]]、[[贝塞尔曲线]]。

原文：[[CLIPasso.pdf]]。这轮不写全文译文，方法要点写在 [[分数蒸馏]]。

## 一句话

用 CLIP 的语义距离当损失，优化一组贝塞尔曲线，把一张图收成能认出来的简笔。

## 和本课题的关系

成品线质量的老基线。优化轨迹不是作画阶段。[[DiffSketcher]] 拿它当对照。
