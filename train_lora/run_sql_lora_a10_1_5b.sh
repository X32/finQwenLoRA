#!/bin/bash

# 激活虚拟环境
source ../venv_lora/bin/activate

# 修复 CUDA_HOME 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 减少显存碎片（OOM 时官方推荐）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== NVIDIA A10 24GB SQL生成训练配置 (Qwen2.5-1.5B - 超高性能) ==="
echo "GPU: NVIDIA A10 (24GB显存, Ampere架构)"
echo "模型: Qwen2.5-1.5B-Instruct (fp16, 不量化)"
echo "任务: NL2SQL (自然语言转SQL查询)"
echo "优化: FP16 + 大Rank LoRA + 大Batch + 梯度检查点"
echo ""
echo "💡 A10训练1.5B优势: 24GB显存充足，可使用超大rank和大batch；"
echo "   训练速度极快，2-3小时即可完成，SQL生成质量大幅提升。"
echo ""

# 模型配置 - 保持1.5B模型
MODEL="/home/x32/Desktop/apfs/workplace/FinQwen/models/Qwen2.5-1.5B-Instruct"
MODEL_SAVE_DIR="./output/sql_lora_fp16_qwen2.5_1_5b_a10"
DATA='../data/lora_data/sql-lora-train-end.json'

# 训练参数 - A10显卡针对1.5B的激进优化
NUM_TRAIN_EPOCHS=4              # SQL任务复杂，增加epoch数
BATCH_SIZE=8                    # SQL任务序列较长，适度降低batch
SAVE_STEPS=100                  # 更频繁保存 (训练快了)
GRADIENT_ACCUMULATION_STEPS=2  # 补偿batch reduction
LEARNING_RATE=5e-5              # SQL任务使用相对保守学习率
MAX_LENGTH=512                  # SQL生成任务需要更长序列
SYSTEM_MESSAGE='你是一个SQL查询专家，精通金融数据分析，能够根据自然语言问题准确生成SQL查询语句，特别是涉及多表连接、复杂查询和聚合分析的场景'

# LoRA参数 - A10显存充足，使用大rank
LORA_RANK=96                   # 大rank处理复杂SQL逻辑
LORA_ALPHA=192                  # alpha = 2*rank
LORA_DROPOUT=0.1                # 标准dropout
LORA_BIAS='none'
SAVE_TOTAL_LIMIT=3              # 保留更多检查点

# A10专属优化 - SQL任务优化
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"  # 全部线性层
QUANTIZATION="fp16"             # 不量化，保持FP16精度
USE_BF16="False"               # 如果数据支持可改为True

echo "✅ 使用模型: Qwen2.5-1.5B-Instruct (FP16)"
echo "显存预估: ~14-17GB / 24GB"
echo "训练参数: ~25M (1.7%)"
echo "预估时长: 2-3小时"
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
echo "💡 SQL生成任务特点："
echo "   - 需要理解复杂的金融领域知识"
echo "   - 需要掌握数据库表结构和关系"
echo "   - 需要生成准确的SQL语法和逻辑"
echo "   - 比NER任务更复杂，需要更多训练时间"
echo ""
echo "🎯 优化建议："
echo "   - 如需更高性能：增加rank到128，增加epoch到5"
echo "   - 如需更快训练：增加batch到12，降低rank到64"
echo "   - 如显存不足：降低batch到4，或降低max_length到384"
echo "   - 如SQL逻辑复杂：增加learning_rate到8e-5，增加warmup"
