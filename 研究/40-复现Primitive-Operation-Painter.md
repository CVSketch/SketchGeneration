---
tags:
  - 复现
  - 项目
---

# 复现 Primitive Operation Painter

对应笔记 [[Primitive-Operation-Painter]]。代码已克隆到 `复现/primitive-operation-painter`。

## 你要准备什么

- Python 3.10 以上，能装对应系统的 PyTorch
- 推理：一份 Hugging Face 权重（`config.json` + `model.safetensors`）
- 自己造数据：Rust 工具链 + 能跑 WGPU 的 GPU（macOS 用 Metal）

权重页：https://huggingface.co/doubixz/primitive-operation-painter-weight  
代码页：https://github.com/wonderfulearth/primitive-operation-painter

## 跑自带的 6 条续画

1. 把权重放到 `复现/primitive-operation-painter/model/`，里面只要 `config.json` 和 `model.safetensors`。
2. 在该目录安装 `requirements.txt`。
3. 运行：

```bash
python example.py
```

默认读 `example/sequences/v1/data_part_1.csv`（每条 $11$ 步），补到 $144$ 步，写出 `example/example_inference.png`。

权重在别处时：

```bash
python example.py --model-dir /path/to/primitive-operation-painter-weight
```

## 自己把图收成序列

转换器在 `fast_shape_render/`。先设输入图文件夹，再：

```bash
cd fast_shape_render
cargo run --release
```

默认写出 `data/output_256`，也就是 Python 训练默认读的位置。可用环境变量 `SHAPE_RENDERER_INPUT_DIR`、`SHAPE_RENDERER_OUTPUT_DIR`。

## 接着训

```bash
export ANIME_PAINTER_DATA_DIR=/path/to/output_256
python train_gpt_pretrain.py --initial-model-dir /path/to/primitive-operation-painter-weight
```

新断点写到 `checkpoints_gpt_fullseq_144ctx_256reso/`。下次去掉 `--initial-model-dir` 就会从最新断点接着训。

## 这轮先不用做的

- 不要指望它直接出石膏像素描。基元是色块。
- 不要把训练损失 $3.81$ 写进论文当质量指标。
- 没有官方和 [[StrokeFusion]]、[[SketchAgent]] 的同一张表。要当基线，得自己在同一数据上跑。
