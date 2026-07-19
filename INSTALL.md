# FinQwen LoRA训练环境安装指南

## 📋 依赖包说明

本项目提供了4个不同场景的requirements文件：

| 文件 | 适用场景 | 功能完整性 |
|------|----------|------------|
| **requirements.txt** | 完整功能训练 | ✅ 100% |
| **requirements_minimal.txt** | 快速体验 | ✅ 70% |
| **requirements_python311.txt** | Python 3.11环境 | ✅ 100% |
| **requirements_cloud_training.txt** | 云端训练 | ✅ 100% |

## 🚀 快速安装

### 方式1: 标准安装（推荐）
```bash
# 创建虚拟环境
python3.9 -m venv venv_lora
source venv_lora/bin/activate  # Linux/Mac
# venv_lora\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 方式2: Python 3.11专用安装
```bash
# Python 3.11环境
python3.11 -m venv venv_lora_311
source venv_lora_311/bin/activate

# 安装Python 3.11优化版本
pip install -r requirements_python311.txt
```

### 方式3: 最小化安装（快速体验）
```bash
# 创建环境
python3.9 -m venv venv_lora
source venv_lora/bin/activate

# 最小依赖安装
pip install -r requirements_minimal.txt
```

### 方式4: 云端训练安装
```bash
# 云端虚拟环境
python3.9 -m venv venv_lora
source venv_lora/bin/activate

# 云端训练依赖
pip install -r requirements_cloud_training.txt
```

## 🎯 按需安装说明

### 核心依赖（必需）
```bash
# 最基础的5个包
pip install torch>=2.0.0 transformers>=4.35.0 peft>=0.7.0 accelerate>=0.25.0 modelscope>=1.0.0
```

### QLoRA训练（4-bit量化）
```bash
# 需要额外安装
pip install bitsandbytes>=0.41.0
```

### 分布式训练
```bash
# 需要额外安装
pip install deepspeed>=0.12.0
```

### 中文文本处理
```bash
# 中文分词支持
pip install jieba>=0.42.0 sentencepiece>=0.1.99
```

## 🔧 PyTorch安装注意事项

### CUDA版本匹配
```bash
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121
```

### RTX 3090/4090专用
```bash
# 最新PyTorch版本（推荐）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### A10/V100专用
```bash
# 稳定版本
pip install torch==2.0.1 transformers==4.35.0
```

## 🌐 国内镜像加速

### 使用清华镜像
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 使用阿里云镜像
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 永久配置镜像源
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🐛 常见问题解决

### 问题1: PyTorch安装失败
```bash
# 解决方案：先安装PyTorch，再安装其他依赖
pip install torch==2.0.1 torchvision==0.15.2
pip install -r requirements.txt --no-deps
pip install -r requirements.txt
```

### 问题2: BitsAndBytes安装失败
```bash
# 编译安装
pip install bitsandbytes>=0.41.0
# 如果失败，尝试预编译版本
pip install bitsandbytes==0.41.0
```

### 问题3: ModelScope下载慢
```bash
# 配置国内镜像
export HF_ENDPOINT=https://hf-mirror.com
# 或者使用gitee镜像
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题4: Python 3.11兼容性问题
```bash
# 使用Python 3.11专用requirements
pip install -r requirements_python311.txt
```

## 💻 不同硬件环境配置

### RTX 3090/4090 (24GB)
```bash
# 高性能配置
pip install torch==2.1.0 transformers==4.36.0 peft==0.8.0 accelerate==0.25.0
```

### A10/V100 (16-24GB)
```bash
# 标准配置
pip install torch==2.0.1 transformers==4.35.0 peft==0.7.0 accelerate==0.25.0
```

### GTX 1080 Ti (11GB)
```bash
# 兼容配置
pip install torch==2.0.0 transformers==4.30.0 peft==0.5.0 accelerate==0.20.0
```

## 📦 安装验证

### 快速验证脚本
```bash
# 验证关键库
python -c "
import torch
import transformers
import peft
import accelerate
import modelscope

print('✅ PyTorch:', torch.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ PEFT:', peft.__version__)
print('✅ Accelerate:', accelerate.__version__)
print('✅ ModelScope:', modelscope.__version__)

# CUDA检查
if torch.cuda.is_available():
    print('✅ CUDA:', torch.version.cuda)
    print('✅ GPU:', torch.cuda.get_device_name(0))
else:
    print('⚠️  CUDA不可用')
"
```

### 完整验证
```bash
# 运行训练脚本帮助信息
python train_lora/finetune_qwen.py --help
```

## 🎓 推荐安装顺序

### 新手安装（零基础）
```bash
# 1. 安装Python 3.9
# 2. 创建虚拟环境
python3.9 -m venv venv_lora
source venv_lora/bin/activate

# 3. 升级pip
pip install --upgrade pip

# 4. 安装PyTorch (根据CUDA版本)
pip install torch==2.0.1 torchvision==0.15.2

# 5. 安装其他依赖
pip install -r requirements.txt
```

### 快速安装（有经验）
```bash
# 一键安装
pip install torch==2.0.1 transformers==4.35.0 peft==0.7.0 accelerate==0.25.0 modelscope==1.0.0
```

## 🔄 环境迁移

### 导出当前环境
```bash
pip freeze > requirements_export.txt
```

### 在新环境恢复
```bash
pip install -r requirements_export.txt
```

## 📞 获取帮助

如果遇到安装问题：
1. 检查Python版本：`python --version`
2. 检查CUDA版本：`nvidia-smi`
3. 尝试不同的requirements文件
4. 查看错误信息并搜索解决方案

**安装完成后，你就可以开始LoRA训练了！** 🚀