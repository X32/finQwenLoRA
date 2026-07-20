#!/bin/bash

# 激活虚拟环境
source ../venv_lora/bin/activate

# 修复 CUDA_HOME 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 减少显存碎片（OOM 时官方推荐）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== RTX 3090 24GB SQL生成训练配置 (Qwen2.5-1.5B - 超高性能) ==="
echo "GPU: NVIDIA RTX 3090 (24GB显存, Ampere架构)"
echo "模型: Qwen2.5-1.5B-Instruct (fp16, 不量化)"
echo "任务: NL2SQL (自然语言转SQL查询)"
echo "优化: FP16 + 大Rank LoRA + 大Batch + 梯度检查点"
echo ""
echo "💡 RTX 3090训练优势: 24GB显存充足，计算性能比A10更强；"
echo "   SQL任务需要复杂推理，3090性能优势明显，训练时间更短。"
echo ""

# 模型配置 - 保持1.5B模型
MODEL="/home/x32/Desktop/apfs/workplace/FinQwen/models/Qwen2.5-1.5B-Instruct"
MODEL_SAVE_DIR="./output/sql_lora_fp16_qwen2.5_1_5b_3090"
DATA='../data/lora_data/sql-lora-train-end.json'

# 训练参数 - RTX 3090针对SQL任务的优化
NUM_TRAIN_EPOCHS=4              # SQL任务复杂，增加epoch数
BATCH_SIZE=12                   # RTX 3090性能更强，可以使用更大batch
SAVE_STEPS=100                  # 更频繁保存 (训练快了)
GRADIENT_ACCUMULATION_STEPS=2  # 适度累积
LEARNING_RATE=5e-5              # SQL任务使用保守学习率
MAX_LENGTH=512                  # SQL生成任务需要更长序列
SYSTEM_MESSAGE='你是一个SQL查询专家，精通金融数据分析，能够根据自然语言问题准确生成SQL查询语句，特别是涉及多表连接、复杂查询和聚合分析的场景'

# LoRA参数 - RTX 3090显存和性能优势
LORA_RANK=96                   # 大rank处理复杂SQL逻辑
LORA_ALPHA=192                  # alpha = 2*rank
LORA_DROPOUT=0.1                # 标准dropout
LORA_BIAS='none'
SAVE_TOTAL_LIMIT=3              # 保留更多检查点

# RTX 3090专属优化 - SQL任务优化
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"  # 全部线性层
QUANTIZATION="fp16"             # 不量化，保持FP16精度
USE_BF16="False"               # 如果数据支持可改为True

echo "✅ 使用模型: Qwen2.5-1.5B-Instruct (FP16)"
echo "显存预估: ~14-17GB / 24GB"
echo "训练参数: ~25M (1.7%)"
echo "预估时长: 1.5-2.5小时 (比A10快20-30%)"
echo "预估性能: SQL生成准确率90%+"
echo ""

# 启动训练
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
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
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
echo "💡 RTX 3090 SQL训练优势："
echo "   - 计算性能: 比A10快15-25%"
echo "   - 显存带宽: GDDR6X提供更快数据传输"
echo "   - 适合复杂SQL逻辑训练"
echo ""
echo "🎯 与A10配置差异："
echo "   - Batch Size: 12 vs 8 (3090性能更强)"
echo "   - 训练时间: 快20-30%"
echo "   - 准确率: 相同（数据质量决定）"
echo ""
echo "🔧 优化建议："
echo "   - 如需更高性能：增加rank到128，batch到16"
echo "   - 如需更快训练：增加batch到16，降低epoch到3"
echo "   - 如显存不足：降低batch到8，或max_length到384"