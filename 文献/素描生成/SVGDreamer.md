---
title: "SVGDreamer: Text Guided SVG Generation with Diffusion Model"
short: SVGDreamer
authors: Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, Qian Yu
year: 2024
venue: CVPR 2024
ccf: A
arxiv: "2312.16476"
pdf: "[[SVGDreamer.pdf]]"
code: https://github.com/ximinng/SVGDreamer
project: https://ximinng.github.io/SVGDreamer-project/
tags:
  - 论文
  - 素描生成
  - SVG
  - CCF-A
status: 精读
---

# SVGDreamer

先看 [[入门-读论文前先看]]。这篇用到：[[扩散模型]]、[[分数蒸馏]]、[[矢量图]]、[[注意力]]。

原文：[[SVGDreamer.pdf]]。全文译文还没写。

项目页：https://ximinng.github.io/SVGDreamer-project/

## 一句话

当场优化一组 SVG 路径，让冻住的文生图模型觉得「像那句话」，并尽量拆开前后景好改。

## 要解决什么问题

[[VectorFusion]] 一类方法路径糊、颜色过饱和、不好单独改一块。

## 原理

语义拆图（SIVE）+ 粒子化分数蒸馏（VPSD）。一张大约十几分钟，显存十几到三十 GB。

## 局限

可编辑取决于文生图注意力。控制点数量不能自适应。中间步是优化步，不是画家阶段。

## 和本课题的关系

成品 SVG 质量基线。不要把优化轨迹写成作画过程。
