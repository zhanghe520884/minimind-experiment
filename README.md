# MiniMind 本地训练与华中科技大学研究生手册文档 LoRA 微调实验

_**说明：训练好的权重约130MB，无法上传至仓库，故out文件夹下没有训练好的.pth文件，只有LoRA训练好的小权重**_

基于开源项目 [MiniMind](https://github.com/jingyaogong/minimind),完整复现了 minimind-3 (64M) 的预训练、SFT 与 LoRA 微调全流程。

## 📌 实验亮点

- 🔥 完整复现 minimind-3 主线版本 (hidden_size=768, n_layers=8, ~64M 参数)
- 📊 自训模型 vs 官方 minimind-3 三方对比测试
- 🎯 12 万字本地文档（华中科技大学研究生手册） LoRA 微调,产出 779 KB 权重
- 📈 详尽的 loss 曲线、训练日志与对比报告

## 🛠️ 实验环境

- **GPU**: 3 × 24 GB,DDP 多卡并行
- **PyTorch**: 2.3.0 + CUDA 12.1
- **Python**: 3.10

## 📂 项目结构
```
.
├── model/                    # 模型架构(沿用官方)
├── trainer/                  # 训练脚本(添加了 loss 落盘代码)
├── dataset/                  # 训练数据(已 gitignore,需自行下载)
├── exp_lora/                 # LoRA 微调实验工作区
│   ├── scripts/              # 数据处理与对比脚本
│   ├── data/                 # LoRA 训练数据(已 gitignore)
│   └── models/               # LoRA 权重产物
│       └── from_mine_768.pth # 文档 LoRA 权重 (779 KB)
├── eval_compare.py           # 三方对比测试脚本(基于官方 eval_llm 改造)
├── plot_loss.py              # loss 曲线绘制脚本
└── mydoc.docx                # 华中科技大学研究生手册.docx文档
```
## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch==2.3.0 transformers==4.57 huggingface_hub==0.34
pip install "setuptools<81" "pyarrow==15.0.2"
pip install python-docx modelscope
```

### 2. 下载训练数据

```bash
# 从 ModelScope 下载
modelscope download --dataset jingyaogong/minimind_dataset \
    pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl \
    --local_dir ./dataset
```

### 3. 训练

```bash
cd trainer
torchrun --nproc_per_node=3 train_pretrain.py \
    --batch_size 32 --hidden_size 768 --num_hidden_layers 8 \
    --epochs 2 --data_path ../dataset/pretrain_t2t_mini.jsonl
```

完整命令请参见实验报告。

### 4. 对比测试

```bash
python eval_compare.py
```

## 📊 主要结果

| 指标 | 预训练 | SFT |
|------|-------|-----|
| 起始 loss | 7.21 | 2.05 |
| 最终 loss | 1.86 | 1.63 |
| 单 epoch 步数 | 13,232 | 18,870 |
| epochs | 2 | 2 |

## 📄 详细文档

完整的实验过程、对比分析、问题排查见 [实验报告](./实验报告.docx)。

## 🙏 致谢

本项目基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 开源项目。
