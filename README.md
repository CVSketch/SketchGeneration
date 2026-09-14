# SketchGeneration

素描过程生成知识库。用 [Obsidian](https://obsidian.md/) 打开本文件夹即可。

课题暂定 **Inverse Sketching**：从成品线稿反推课堂式四阶段（构图 / 大形 → 结构线 → 排线调子 → 细节收束）。目标不是再画一张更好看的成品草图。

油画已经用分阶段扩散做到过程生成，并连着发到 CCF-A。素描还缺同等的过程定义、过程数据和过程模型。

这是第一版：文献笔记、按章节写完的中文译文、从 PDF 裁下的图，以及研究路线。仓库里暂时没有我们自己的训练代码。

## 怎么读

1. 安装 Obsidian，选「打开文件夹作为库」，选中本目录。
2. 打开 [00-首页.md](00-首页.md)。
3. 再按 [研究/使用说明.md](研究/使用说明.md) 往下读。

建议顺序：

1. [知识基础/00-目录.md](知识基础/00-目录.md)：一词一页
2. [研究/入门-读论文前先看.md](研究/入门-读论文前先看.md)
3. [研究/任务定义.md](研究/任务定义.md)、[研究/10-研究路线.md](研究/10-研究路线.md)
4. [研究/00-本轮目录.md](研究/00-本轮目录.md)：这轮精读和参考文献

每篇论文一般有四页：笔记、全文译文、要点、原文 PDF。要点里的链接点进译文某一节。

## 目录

| 目录 | 里面是什么 |
| :--- | :--- |
| `文献/油画过程/`、`文献/素描生成/`、`文献/深大相关/` | 论文笔记 |
| `文献/翻译/` | 按原文章节写完的译文，以及链到各节的要点 |
| `文献/参考文献/` | 从精读论文参考文献里拆出来的笔记 |
| `文献/不读/` | 本期明确不读 |
| `研究/` | 路线、索引、任务定义、复现说明 |
| `知识基础/` | 一词一页 |
| `附件/pdf/` | 顶会公开原文；参考文献 PDF 在 `附件/pdf/参考文献/` |
| `附件/图/` | 译文和笔记里用到的裁图 |
| `模板/` | 新笔记、译文、要点 |

写法约定见 [AGENTS.md](AGENTS.md)。缺的原文记在 [附件/待补下载.md](附件/待补下载.md)。

## 复现代码不在这个仓库里

`复现/` 下的两个官方克隆没有推进来（它们自己有 git 历史）。需要时按说明自行克隆：

- Inverse Painting：<https://github.com/ArmastusChen/inverse_painting>  
  步骤见 [研究/30-复现Inverse-Painting.md](研究/30-复现Inverse-Painting.md)
- Primitive Operation Painter：<https://github.com/wonderfulearth/primitive-operation-painter>  
  步骤见 [研究/40-复现Primitive-Operation-Painter.md](研究/40-复现Primitive-Operation-Painter.md)

## 论文版权

笔记和译文是本组写的。PDF 和裁图来自各论文公开版本，版权仍归原作者和出版社。引用请回原文。

## 自己怎么更新这个仓库

本机已经有 git 和 [GitHub CLI](https://cli.github.com/)（`gh`）时：

```bash
cd /你的路径/SketchGeneration
git status
git add 你改过的文件
git commit -m "用一两句话写为什么改"
git push
```

第一次在本机拉下来：

```bash
gh auth login
git clone https://github.com/CVSketch/SketchGeneration.git
```

新建一个空仓库再推（下次换题目时可以照做）：

```bash
cd /你的项目目录
git init -b main
git add .
git commit -m "第一版说明"
gh repo create CVSketch/仓库名 --public --source=. --remote=origin --push
```

`--public` 表示公开。只要自己看得见，改成 `--private`。
