---
title: "VectorFusion: Text-to-SVG by Abstracting Pixel-Based Diffusion Models"
short: VectorFusion
authors: Ajay Jain, Amber Xie, Pieter Abbeel
year: 2023
venue: CVPR 2023
ccf: A
arxiv: "2211.11319"
pdf: "[[VectorFusion.pdf]]"
project: https://ajayj.com/vectorfusion
tags:
  - 论文
  - 素描生成
  - SVG
  - CCF-A
status: 精读
---

# VectorFusion

先看 [[入门-读论文前先看]]。这篇用到：[[扩散模型]]、[[分数蒸馏]]、[[可微渲染器]]、[[矢量图]]。

原文：[[VectorFusion.pdf]]。全文译文还没写。

## 一句话

把像素扩散模型当成老师，拧 SVG 控制点，让渲染出来的图像那句话。

## 局限

每张都要优化。笔画无序。是 [[DiffSketcher]]、[[SVGDreamer]] 的前身。
