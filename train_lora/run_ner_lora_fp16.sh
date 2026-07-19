#!/bin/bash

# 激活虚拟环境
source ../venv_lora/bin/activate

# 修复 CUDA_HOME 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 减少显存碎片（OOM 时官方推荐）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GTX 1080 Ti NER训练配置 (fp16 LoRA - 推荐) ==="
echo "GPU: NVIDIA GTX 1080 Ti (11GB显存, Pascal sm_61)"
echo "模型: Qwen2.5-1.5B-Instruct (fp16, 不量化)"
echo "任务: 命名实体识别 (NER)"
echo "优化: fp16 + LoRA + 梯度检查点"
echo ""
echo "💡 为何不用QLoRA: 1.5B模型fp16仅~3GB，11GB够用；"
echo "   且1080 Ti(Pascal)对bitsandbytes 4-bit支持差。"
echo ""

# 模型配置
MODEL="/home/x32/Desktop/apfs/workplace/FinQwen/models/Qwen2.5-1.5B-Instruct"
MODEL_SAVE_DIR="./output/ner_lora_fp16_qwen2.5"
DATA='../data/lora_data/NER_lora_train.json'

# 训练参数
NUM_TRAIN_EPOCHS=2
BATCH_SIZE=2                  # 1080 Ti 11GB，先用1确保不OOM
SAVE_STEPS=200                # 降低保存频率，训练中途也有检查点（容错，防中断丢进度）
GRADIENT_ACCUMULATION_STEPS=8 # 有效batch=8
LEARNING_RATE=5e-5
MAX_LENGTH=384
SYSTEM_MESSAGE='你是一个命名实体识别专家，请从文本中提取公司名称、关键词等重要实体信息'

# LoRA参数
LORA_RANK=16
LORA_ALPHA=8
LORA_DROPOUT=0.1
LORA_BIAS='none'
SAVE_TOTAL_LIMIT=2

echo "✅ 使用模型: Qwen2.5-1.5B-Instruct (fp16)"
echo "显存预估: ~6-8GB"

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
echo "💡 如果显存不足(OOM)，把 BATCH_SIZE 改成 1，GRADIENT_ACCUMULATION_STEPS 改成 8"
