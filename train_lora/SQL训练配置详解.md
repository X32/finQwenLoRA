# SQL生成任务训练配置详解

## 🎯 SQL vs NER 任务对比分析

### 任务特性差异

```python
task_comparison = {
    "NER任务": {
        "输入": "文本段落",
        "输出": "实体标签（公司名、关键词等）",
        "复杂度": "较低，直接信息提取",
        "序列长度": "通常较短（<384 tokens）",
        "训练难度": "较低，收敛快"
    },
    "SQL任务": {
        "输入": "自然语言问题",
        "输出": "SQL查询语句",
        "复杂度": "较高，需要逻辑推理",
        "序列长度": "通常较长（<512 tokens）",
        "训练难度": "较高，需要更多训练"
    }
}
```

## 📊 配置参数调整说明

### 关键参数对比

| 参数 | NER配置 | SQL配置 | 调整原因 |
|------|---------|---------|----------|
| **MAX_LENGTH** | 384 | 512 | SQL语句更长 |
| **BATCH_SIZE** | 16 (A10) | 8 | SQL序列长，显存需求大 |
| **NUM_EPOCHS** | 3 | 4 | SQL任务更复杂 |
| **LEARNING_RATE** | 8e-5 | 5e-5 | SQL需要更稳定训练 |
| **LORA_RANK** | 128 | 96 | SQL逻辑需要适度rank |
| **SYSTEM_MESSAGE** | NER专家 | SQL专家 | 任务特定提示 |

### 系统消息设计

```python
system_messages = {
    "NER系统消息": {
        "role": "命名实体识别专家",
        "focus": "提取公司名称、关键词等重要实体信息",
        "skills": "实体识别、关系抽取、信息过滤"
    },
    "SQL系统消息": {
        "role": "SQL查询专家", 
        "focus": "精通金融数据分析，准确生成SQL查询语句",
        "skills": "数据库理解、SQL语法、复杂查询、聚合分析"
    }
}
```

## 🔧 不同GPU配置方案

### A10 24GB配置

```bash
#!/bin/bash
# SQL任务A10专用配置

# 环境设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 训练参数
NUM_TRAIN_EPOCHS=4              # SQL任务需要更多轮次
BATCH_SIZE=8                    # 适度batch size
GRADIENT_ACCUMULATION_STEPS=2  # 补偿有效batch
LEARNING_RATE=5e-5              # 保守学习率
MAX_LENGTH=512                  # SQL序列较长

# LoRA参数  
LORA_RANK=96                   # 处理复杂SQL逻辑
LORA_ALPHA=192                  # 标准alpha配置
LORA_DROPOUT=0.1                # 防止过拟合

# SQL专用系统消息
SYSTEM_MESSAGE='你是一个SQL查询专家，精通金融数据分析，能够根据自然语言问题准确生成SQL查询语句，特别是涉及多表连接、复杂查询和聚合分析的场景'

# 预期性能
显存使用: 14-17GB / 24GB
训练时间: 2-3小时
准确率: 90%+ (SQL生成正确性)
```

### RTX 3090 24GB配置

```bash
#!/bin/bash
# SQL任务3090优化配置

# 训练参数 - 3090性能更强
BATCH_SIZE=12                   # 更大batch
GRADIENT_ACCUMULATION_STEPS=2  # 保持累积
LEARNING_RATE=5e-5              # 相同学习率
MAX_LENGTH=512                  # SQL序列较长

# LoRA参数
LORA_RANK=96                   # 保持相同rank
LORA_ALPHA=192

# 3090优势
显存使用: 14-17GB / 24GB
训练时间: 1.5-2.5小时 (比A10快20-30%)
性能提升: 计算能力更强，复杂查询生成更好
```

### GTX 1080 Ti 11GB配置

```bash
#!/bin/bash  
# SQL任务1080Ti配置（需要调整）

# 模型：Qwen-1.5B, FP16
# 训练参数：需要大幅降低batch和rank

NUM_TRAIN_EPOCHS=4
BATCH_SIZE=2                    # 小batch避免OOM
GRADIENT_ACCUMULATION_STEPS=8  # 大梯度累积
LEARNING_RATE=5e-5
MAX_LENGTH=512

# LoRA参数 - 降低rank节省显存
LORA_RANK=32                    # 降低rank
LORA_ALPHA=64

# 预期性能
显存使用: 10-11GB / 11GB (接近极限)
训练时间: 4-6小时 (较长但可行)
准确率: 88-90% (略低于大rank配置)
```

## 📈 SQL任务训练特点

### 数据格式分析

```python
sql_data_structure = {
    "输入问题": "在20241231，按照申万行业分类的行业划分标准，哪个一级行业的A股公司数量最多？",
    "预期输出": "SELECT \"一级行业名称\", COUNT(DISTINCT \"股票代码\") AS \"公司数量\" FROM \"A股公司行业划分表\" WHERE \"交易日期\" = '20241231' AND \"行业划分标准\" = '申万行业分类' GROUP BY \"一级行业名称\" ORDER BY \"公司数量\" DESC LIMIT 1",
    "复杂度因素": [
        "多表连接",
        "复杂WHERE条件", 
        "聚合函数(COUNT, SUM等)",
        "GROUP BY和ORDER BY",
        "子查询和LIMIT"
    ]
}
```

### 训练策略差异

```python
training_strategy_comparison = {
    "NER训练策略": {
        "重点": "模式识别和信息提取",
        "loss函数": "实体标签分类",
        "收敛速度": "快（1-2 epochs）",
        "难点": "实体边界识别"
    },
    "SQL训练策略": {
        "重点": "逻辑推理和语法生成",
        "loss函数": "序列生成",
        "收敛速度": "较慢（2-4 epochs）",
        "难点": [
            "复杂SQL语法",
            "表关系理解",
            "查询逻辑推理",
            "SQL语句完整性"
        ]
    }
}
```

## 🚀 性能优化建议

### 针对SQL任务的优化

#### 1. 数据预处理优化
```bash
# SQL数据特点
- 问题长度差异大
- SQL语句格式严格
- 需要理解表结构

# 优化建议
--lazy_preprocess True    # 大数据集懒加载
--max_length 512          # 足够长度覆盖复杂SQL
--gradient_checkpointing  # 处理长序列
```

#### 2. 学习率调优
```python
sql_learning_rate_strategy = {
    "初始学习率": {
        "推荐": "5e-5",
        "原因": "SQL任务需要稳定训练",
        "调整": "如不收敛可降低到3e-5"
    },
    "warmup_ratio": {
        "推荐": "0.05",
        "作用": "前5%步数线性增加学习率",
        "好处": "提高训练稳定性"
    },
    "lr_scheduler": {
        "推荐": "cosine",
        "原因": "平滑下降，避免震荡"
    }
}
```

#### 3. Batch Size调整
```python
batch_size_recommendations = {
    "24GB GPU (A10/3090)": {
        "推荐": "batch=8-12",
        "有效batch": "16-24 (with grad accum)",
        "显存使用": "14-17GB"
    },
    "16GB GPU (V100)": {
        "推荐": "batch=4-6",
        "有效batch": "8-12 (with grad accum)",
        "显存使用": "15-16GB"
    },
    "11GB GPU (1080Ti)": {
        "推荐": "batch=2-3",
        "有效batch": "8-16 (with grad accum)",
        "显存使用": "10-11GB (接近极限)"
    }
}
```

## 💡 实际使用指南

### A10快速开始

```bash
# 1. 确认环境
source ../venv_lora/bin/activate
python -c "import torch; print(torch.cuda.is_available())"

# 2. 检查数据
ls ../data/lora_data/sql-lora-train-end.json

# 3. 启动训练
cd train_lora
bash run_sql_lora_a10_1_5b.sh

# 4. 监控训练
watch -n 1 nvidia-smi
tail -f output/sql_lora_fp16_qwen2.5_1_5b_a10/trainer_log.jsonl
```

### RTX 3090快速开始

```bash
# 直接使用3090优化版本
cd train_lora  
bash run_sql_lora_3090_1_5b.sh

# 3090优势：
# - 训练快20-30%
# - 性能更强
# - 适合复杂SQL训练
```

## 📊 预期结果分析

### SQL生成质量评估

```python
sql_quality_metrics = {
    "语法正确性": {
        "良好": "SQL语法完全正确，可执行",
        "问题": "缺少引号、括号不匹配等"
    },
    "逻辑准确性": {
        "良好": "查询逻辑正确，答案准确", 
        "问题": "表连接错误、条件遗漏"
    },
    "完整性": {
        "良好": "涵盖所有查询要素",
        "问题": "缺少关键字段或条件"
    }
}
```

### 典型SQL查询类型

```python
complexity_levels = {
    "简单查询": {
        "特征": "单表、简单条件",
        "示例": "SELECT * FROM table WHERE id = 1",
        "训练效果": "95%+ 准确率"
    },
    "中等查询": {
        "特征": "多表连接、聚合函数",
        "示例": "SELECT category, COUNT(*) FROM products JOIN orders ON ...",
        "训练效果": "90-95% 准确率"
    },
    "复杂查询": {
        "特征": "子查询、多表、复杂条件",
        "示例": "SELECT * FROM (SELECT ...) WHERE ... GROUP BY ... HAVING ...",
        "训练效果": "85-90% 准确率"
    }
}
```

## 🎯 故障排除

### 常见SQL训练问题

```python
sql_training_issues = {
    "问题1: SQL语法错误": {
        "原因": "模型未学会正确的SQL语法",
        "解决": "增加训练epochs，检查数据质量",
        "预防": "确保训练数据SQL语句格式正确"
    },
    "问题2: 逻辑错误": {
        "原因": "模型理解错误表关系",
        "解决": "增加相关训练样本",
        "预防": "提供更多表关系示例"
    },
    "问题3: 生成不完整": {
        "原因": "max_length设置太小",
        "解决": "增加到512或768",
        "预防": "分析训练数据SQL长度分布"
    },
    "问题4: 收敛慢": {
        "原因": "学习率过高或数据质量差",
        "解决": "降低学习率，增加warmup",
        "预防": "使用更保守的学习率"
    }
}
```

## 📈 性能预期

### A10 24GB SQL训练预期

```python
a10_sql_expectations = {
    "训练时间": {
        "预期": "2-3小时",
        "因素": "SQL序列较长，计算量大",
        "优化": "可适当降低batch_size加速"
    },
    "显存使用": {
        "预期": "14-17GB / 24GB",
        "峰值": "复杂的SQL生成时",
        "安全": "仍有7GB余量"
    },
    "准确率预期": {
        "简单查询": "95%+",
        "中等查询": "90-95%",
        "复杂查询": "85-90%",
        "平均准确率": "90%+"
    }
}
```

### RTX 3090 SQL训练预期

```python
rtx3090_sql_expectations = {
    "训练时间": {
        "预期": "1.5-2.5小时",
        "优势": "比A10快20-30%",
        "原因": "更强的计算性能和显存带宽"
    },
    "显存使用": {
        "预期": "14-17GB / 24GB", 
        "相同": "配置相同，显存需求相同",
        "优势": "GDDR6X更快"
    },
    "准确率预期": {
        "简单查询": "95%+",
        "中等查询": "90-95%",
        "复杂查询": "85-90%",
        "平均准确率": "90%+ (与A10相同)"
    }
}
```

## 🎓 最佳实践总结

### SQL生成任务最佳配置

```python
best_practices = {
    "硬件选择": {
        "推荐": "RTX 3090 或 A10 (24GB)",
        "原因": "SQL任务需要大显存和强计算",
        "备选": "V100 16GB (INT4量化)"
    },
    "训练配置": {
        "Epochs": "4轮 (SQL任务更复杂)",
        "Batch Size": "8-12 (平衡速度和显存)",
        "Learning Rate": "5e-5 (保守稳定)",
        "Max Length": "512 (覆盖长SQL)"
    },
    "LoRA配置": {
        "Rank": "96 (处理复杂逻辑)",
        "Alpha": "192 (标准比例)",
        "Target Modules": "全部线性层",
        "Dropout": "0.1 (标准防过拟合)"
    },
    "系统消息": {
        "关键": "强调SQL专家身份",
        "技能": "金融数据分析、复杂查询",
        "语气": "专业、准确"
    }
}
```

## 🚀 总结

**SQL生成任务相比NER任务：**
- ⏱️ **训练时间更长**: 2-3小时 vs 1.5-2小时
- 🧠 **任务复杂度更高**: 需要逻辑推理而非模式识别
- 📏️ **序列长度更长**: 需要max_length=512
- 🎯 **准确率略低**: 90%+ vs 92%+ (更难的任务)
- 💡 **配置更保守**: 需要更稳定的学习率和更多训练轮次

**选择建议：**
- **追求性能**: RTX 3090配置 (训练快，性能强)
- **平衡选择**: A10配置 (稳定可靠)
- **预算有限**: 1080Ti配置 (可行但需优化)

**SQL生成任务是NL2SQL的核心挑战，需要精心配置和充分训练！** 🚀