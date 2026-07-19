#!/bin/bash

# 激活虚拟环境
source ../venv_lora/bin/activate

# 修复 CUDA_HOME 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 减少显存碎片（OOM 时官方推荐）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== NVIDIA A10 24GB NER训练配置 (FP16 LoRA - 高性能) ==="
echo "GPU: NVIDIA A10 (24GB显存, Ampere架构)"
echo "模型: Qwen2.5-7B-Instruct (fp16, 不量化)"
echo "任务: 命名实体识别 (NER)"
echo "优化: FP16 + 大Rank LoRA + 梯度检查点"
echo ""
echo "💡 为何A10用FP16: 24GB显存充足，无需量化性能更好；"
echo "   Ampere架构支持BF16，混合精度训练效率高。"
echo ""

# 模型配置 - 针对7B模型
MODEL="/home/x32/Desktop/apfs/workplace/FinQwen/models/Qwen2.5-7B-Instruct"
MODEL_SAVE_DIR="./output/ner_lora_fp16_qwen2.5_7b_a10"
DATA='../data/lora_data/NER_lora_train.json'

# 训练参数 - A10优化配置
NUM_TRAIN_EPOCHS=2
BATCH_SIZE=4                      # A10 24GB可以用更大batch
SAVE_STEPS=200                    # 降低保存频率，训练中途也有检查点
GRADIENT_ACCUMULATION_STEPS=4     # 有效batch=16 (A10性能强)
LEARNING_RATE=5e-5                # 稳定学习率
MAX_LENGTH=384
SYSTEM_MESSAGE='你是一个命名实体识别专家，请从文本中提取公司名称、关键词等重要实体信息'

# LoRA参数 - 针对7B模型和A10显卡优化
LORA_RANK=64                      # A10显存充足，用大rank提升性能
LORA_ALPHA=128                    # alpha = 2*rank，标准配置
LORA_DROPOUT=0.1                  # 标准dropout
LORA_BIAS='none'
SAVE_TOTAL_LIMIT=2                # 只保留最近2个检查点

# A10专属优化
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # 完整注意力模块
QUANTIZATION="fp16"               # 不量化，保持FP16精度
USE_BF16="False"                  # 如果数据支持可改为True

echo "✅ 使用模型: Qwen2.5-7B-Instruct (FP16)"
echo "显存预估: ~18-20GB / 24GB"
echo "训练参数: ~50M (0.7%)"
echo "预估时长: 6-8小时"
echo "预估性能: 92-94% 准确率"

python finetune_qwen.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --use_lora \
  --fp16 True \
  --output_dir $MODEL_SAVE_DIR \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps $SAVE_STEPS \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --learning_rate $LEARNING_RATE \
  --weight_decay 0.01 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --report_to "none" \
  --model_max_length $MAX_LENGTH \
  --lazy_preprocess True \
  --system_message "$SYSTEM_MESSAGE" \
  --lora_r $LORA_RANK \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --lora_bias $LORA_BIAS \
  --gradient_checkpointing True \
  --optim "adamw_torch"

echo ""
echo "🎉 训练完成！"
echo "模型保存位置: $MODEL_SAVE_DIR"
echo ""
echo "💡 A10性能优化建议："
echo "   - 如需更高性能，可增加rank到128"
echo "   - 如需更快训练，可增加batch_size到8"
echo "   - 如显存不足，降低batch_size到2或启用量化"
