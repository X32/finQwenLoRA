# FinQwen LoRA训练项目

## 项目简介

本项目是基于Qwen系列大模型的LoRA微调训练框架，专注于金融领域的NER和SQL生成任务。

## 项目结构

```
myFinQwen/
├── train_lora/          # 训练脚本和配置
│   ├── finetune_qwen.py # 主训练脚本
│   ├── run_ner_lora_*.sh # NER训练启动脚本
│   ├── A10配置说明.md   # A10显卡配置指南
│   └── Python3.11兼容性指南.md
├── DOC/                 # 文档和教程
├── data/                # 数据目录
└── README.md           # 项目说明
```

## 功能特性

- ✅ 支持Qwen1.5B、7B等多种模型
- ✅ 支持LoRA、QLoRA、AdaLoRA微调方法
- ✅ 支持多种GPU配置（1080Ti、A10、V100等）
- ✅ Python 3.7-3.11兼容支持
- ✅ 详细的训练配置和优化指南

## 快速开始

### 环境要求

- Python 3.7-3.11（推荐3.8-3.10）
- PyTorch 2.0+
- CUDA 11.8+

### 安装依赖

```bash
pip install torch transformers peft accelerate bitsandbytes modelscope
```

### 训练示例

```bash
# NER任务训练（A10显卡）
cd train_lora
bash run_ner_lora_a10_1_5b.sh
```

## 硬件配置

| GPU型号 | 显存 | 推荐模型 | 训练时间 | 准确率 |
|---------|------|----------|----------|--------|
| GTX 1080 Ti | 11GB | Qwen-1.5B | 3-4小时 | 85-88% |
| NVIDIA A10 | 24GB | Qwen-7B | 1.5-2小时 | 92-94% |
| Tesla V100 | 16GB | Qwen-7B | 8-10小时 | 90-92% |

## 支持的任务

- **NER（命名实体识别）**: 从文本中提取公司名称、关键词等重要实体
- **SQL生成**: 根据自然语言生成SQL查询语句
- **文本分类**: 情感分析、主题分类等任务

## 配置说明

详细配置请参考：
- [A10显卡配置指南](train_lora/A10配置说明.md)
- [Python 3.11兼容性指南](train_lora/Python3.11兼容性指南.md)

## 技术栈

- **框架**: PyTorch, Transformers, PEFT
- **优化**: 混合精度训练、梯度检查点、量化
- **硬件**: NVIDIA GPU (CUDA 11.8+)

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

- GitHub: X32/finQwenLoRA
- 项目基于不会ML团队的开源工作

---

**注意**: 本项目主要用于教育和研究目的，请确保模型和数据的使用符合相关法律法规。