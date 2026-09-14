---
title: "Synthesizing Programs for Images using Reinforced Adversarial Learning"
short: SPIRAL
authors: Yaroslav Ganin, Tejas Kulkarni, Igor Babuschkin, S. M. Ali Eslami, Oriol Vinyals
year: 2018
venue: ICML 2018
ccf: A（机器学习）
arxiv: "1804.01118"
pdf: "[[SPIRAL.pdf]]"
code: https://github.com/google-deepmind/spiral
tags:
  - 参考文献
  - 油画过程
status: 已译
---

# SPIRAL

谁在引：[[Learning-to-Paint]]。

全文译文：[[SPIRAL-译文]]。要点：[[SPIRAL-要点]]。原文：[[SPIRAL.pdf]]。

## 一句话

强化学习智能体往不可微的绘画程序里下指令，用判别器当奖励。

命令序列有先后，奖励只看终局图像不像真图。中间渲染是强化学习轨迹，不是构图、结构、排线、收细。
