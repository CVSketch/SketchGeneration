---
tags:
  - 研究
  - 复现
---

# 复现 Inverse Painting

官方仓库已克隆到 `复现/inverse_painting`。论文见 [[Inverse-Painting]]。

本机是 Apple M3 Max，没有 NVIDIA。官方 `demo.py` 写死了 `cuda`，还依赖 `flash-attn`、`xformers`。要出视频，必须到 Linux + NVIDIA 上跑。

作者在单卡 A40（48 GB 显存）上测过。Hugging Face 整包约 50 GB，里面有重复文件。demo 实际大约：

| 占用 | 大约 | 说明 |
|---|---|---|
| 磁盘（整包） | 50 GB | 含第二份 LLaVA、CLIP 的 Flax/TF、多余的 safety checker |
| 磁盘（demo 精简） | 28 GB | 只要 `TP_llava`、渲染器、掩码网络、CLIP、RealisticVision 必要部分 |
| 显存 | 20～28 GB | `demo.py` 把 LLaVA-7B 和扩散模型同时留在 GPU 上 |
| 内存 | 再加几 GB | 权重先读进 CPU 再上卡 |

学方法不必下载。看仓库里的示例数据和 `demo.py` 循环就够。

## 今天先做哪一档

1. **官方 demo（先做这个）**：下载预训练权重，对 `data/demo` 里的成品画跑 `python demo.py`，得到延时关键帧。
2. **完整训练（以后再做）**：自己准备作画视频，生成真值文本和掩码，再分别训文本、掩码、渲染器。

## 官方 demo 在 GPU 机器上的命令

```bash
git clone https://github.com/ArmastusChen/inverse_painting.git
cd inverse_painting

conda create --name inverse_painting python=3.10 -y
conda activate inverse_painting

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA && pip install -e . && cd ..

pip3 install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

# 权重放到仓库根目录，目录要对齐 README
git lfs install
git clone https://huggingface.co/boweiche/inverse_painting hf_tmp
# 把 hf_tmp 里的 checkpoints/ 和 base_ckpt/ 挪到当前目录
# demo 不需要 checkpoints/TP_llava_annotator

python demo.py
```

结果在 `results/`。每张输入会得到 `sample_0.jpg`（白画布）到 `sample_N.jpg`（逐步接近成品），以及带文字指令的 `vis_sample_*.jpg`。

## 推理时每一步在干什么

对应 `demo.py` 主循环，最多 50 步：

1. **文本指令（TP）**：把「当前画布 | 成品」拼成左右图，问微调过的 LLaVA：下一步该画什么，答案不超过 2 个词。
2. **区域掩码（RP）**：根据当前画布、成品和这句话，预测这一步该动哪一块。太小会降低阈值再试。
3. **扩散渲染**：用文本、掩码、当前画布、成品，画出下一张画布。
4. **何时停**：相邻两帧几乎不变，或已经很接近成品。

示例数据里第一步的真值文本是 `Mountain, sky`，见 `data/sample_data_processed/train/text/example/white_10_3:21.json`。

## 还没做

- [ ] 选定 GPU 机器（实验室或云卡）
- [ ] 装环境并下载权重
- [ ] 跑通 `python demo.py`
- [ ] 把 `results/` 拷回本库，对照论文图看
