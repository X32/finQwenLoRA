"""
=============================================================================
Qwen模型LoRA微调训练脚本
=============================================================================

功能：支持Qwen系列模型的参数高效微调(PEFT)
支持方法：LoRA、QLoRA、AdaLoRA
优化技术：DeepSpeed ZeRO、混合精度训练、梯度检查点

作者：不会ML团队
适用：金融领域的NL2SQL、NER等下游任务
Python支持：3.7-3.11 (推荐3.8-3.10)
=============================================================================
"""

# ============================================================================
# Python版本兼容性检查
# ============================================================================
import sys
if sys.version_info < (3, 7):
    raise RuntimeError("此脚本需要Python 3.7或更高版本")
if sys.version_info >= (3, 12):
    import warnings
    warnings.warn("Python 3.12+兼容性未完全测试，推荐使用3.8-3.10")

# ============================================================================
# 1. 基础库导入 - Python标准库和数据处理
# ============================================================================
from dataclasses import dataclass, field  # 数据类装饰器，用于定义配置参数类
import json                              # JSON数据处理，用于读取训练数据
import math                             # 数学运算，用于学习率调度等
import logging                          # 日志记录，用于训练过程监控
import os                               # 操作系统接口，用于文件路径和环境变量
from typing import Dict, Optional, List, Any, Union  # 类型提示，Python 3.11兼容
import warnings                         # Python 3.11兼容性警告处理

# ============================================================================
# 2. 深度学习框架 - PyTorch生态
# ============================================================================
import torch                            # PyTorch深度学习框架
from torch.utils.data import Dataset     # PyTorch数据集基类，用于自定义数据加载

# Python 3.11兼容性：检查PyTorch版本
if sys.version_info >= (3, 11):
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (2, 0):
        warnings.warn(f"Python 3.11+推荐PyTorch 2.0+,当前版本: {torch.__version__}")

# ============================================================================
# 3. 大模型框架 - Transformers、ModelScope、PEFT
# ============================================================================
import transformers                     # HuggingFace Transformers库
import modelscope                       # 阿里云ModelScope模型库（国内访问更快）

# Transformers核心组件
from transformers import (
    Trainer,                           # 训练器类，封装训练循环和优化逻辑
    GPTQConfig,                        # GPTQ量化配置（4-bit模型量化）
    AutoTokenizer,                     # 自动分词器加载（适配不同模型）
    AutoModelForCausalLM,              # 自动因果语言模型加载
    BitsAndBytesConfig                 # bitsandbytes量化配置（QLoRA专用）
)

# 可选的transformers.deepspeed导入（兼容不同版本）
try:
    from transformers import deepspeed as transformers_deepspeed
except (ImportError, Exception):
    transformers_deepspeed = None

from transformers.trainer_pt_utils import LabelSmoother  # 标签平滑处理

# PEFT参数高效微调库
try:
    from peft import (
        LoraConfig,                        # LoRA配置类
        get_peft_model,                    # 应用PEFT到模型
        PeftModel,                         # PEFT模型类
        AdaLoraConfig,                     # AdaLoRA配置类
        AdaLoraModel                       # AdaLoRA模型类
    )
except ImportError as e:
    raise ImportError(
        f"PEFT库导入失败，请确保安装了正确版本的peft库: pip install peft>=0.7.0\n"
        f"Python 3.11可能需要更新版本的peft库\n"
        f"详细错误: {e}"
    )

# 兼容不同版本的peft导入（prepare_model_for_kbit_training位置可能变化）
# Python 3.11兼容性：更详细的错误处理
prepare_model_for_kbit_training = None
prepare_model_error = None

try:
    from peft import prepare_model_for_kbit_training  # 量化模型训练准备
except ImportError as e:
    prepare_model_error = e
    try:
        from peft.utils.other import prepare_model_for_kbit_training
        prepare_model_error = None
    except ImportError as e2:
        try:
            from peft.utils import prepare_model_for_kbit_training
            prepare_model_error = None
        except ImportError as e3:
            # 如果完全找不到，记录警告（兼容性处理）
            if sys.version_info >= (3, 11):
                warnings.warn(
                    f"Python 3.11环境无法导入prepare_model_for_kbit_training，"
                    f"请确保安装最新版本的peft库: pip install --upgrade peft"
                )
            else:
                prepare_model_for_kbit_training = None

from accelerate.utils import DistributedType  # 分布式训练类型枚举
import numpy as np                        # NumPy数值计算库
import random                            # 随机数生成

# ============================================================================
# 4. DeepSpeed分布式训练框架（可选）
# ============================================================================
try:
    from deepspeed import zero                                      # DeepSpeed核心模块
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus  # ZeRO参数状态
    HAS_DEEPSPEED = True                                            # 标记DeepSpeed可用
except (ImportError, Exception):
    HAS_DEEPSPEED = False                                           # 标记DeepSpeed不可用
    ZeroParamStatus = None                                          # 占位符

# ============================================================================
# 工具函数定义
# ============================================================================

def is_deepspeed_zero3_enabled():
    """
    检查是否启用DeepSpeed ZeRO-3优化阶段

    ZeRO-3是DeepSpeed的最激进优化策略，会将模型参数、梯度、优化器状态都分片到不同GPU
    返回值：True表示启用ZeRO-3，False表示未启用或不可用

    为什么需要检查：
    - ZeRO-3与某些PEFT方法（如QLoRA）不兼容
    - 不同版本的DeepSpeed/Transformers可能有不同的检查方式
    - 需要提前检查避免训练中断
    """
    if not HAS_DEEPSPEED:
        return False
    try:
        if transformers_deepspeed is not None:
            return transformers_deepspeed.is_deepspeed_zero3_enabled()
        return False
    except:
        return False

def seed_it(seed):
    """
    设置全局随机种子，确保训练可复现

    参数说明：
    seed：随机种子值（如2024）

    为什么需要随机种子：
    - 确保实验可重复：相同种子+相同数据=相同结果
    - 便于调试问题：固定随机性便于定位问题
    - 公平对比实验：消除随机因素影响

    设置范围：
    - Python内置random模块
    - NumPy随机数生成器
    - PyTorch CPU和GPU随机数生成器
    - CUDA后端确定性行为
    """
    random.seed(seed)                      # Python随机数
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python哈希种子
    np.random.seed(seed)                   # NumPy随机数
    torch.manual_seed(seed)                # PyTorch CPU随机数

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)       # 当前GPU随机数
        torch.cuda.manual_seed_all(seed)   # 所有GPU随机数
        torch.backends.cudnn.deterministic = True   # 确定性算法（可复现）
        torch.backends.cudnn.benchmark = True      # 性能优化（自动选择最快算法）
        torch.backends.cudnn.enabled = True        # 启用cuDNN加速

def torch_gc():
    """
    清理GPU显存的垃圾回收函数

    作用：
    1. 清空PyTorch缓存区中未使用的显存
    2. 清理进程间通信的显存碎片
    3. 避免显存泄漏和OOM错误

    使用场景：
    - 推理完成后：清理显存供下次使用
    - 模型加载失败后：释放已分配的显存
    - 训练异常中断后：清理状态重新开始
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # 清空缓存
        torch.cuda.ipc_collect()      # 清理进程间通信显存

# 特殊标记ID，用于在计算loss时忽略某些token（如padding、用户输入）
# LabelSmoother是Transformers提供的标签平滑工具，ignore_index通常设为-100
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# ============================================================================
# 配置参数类定义 - 使用dataclass组织训练参数
# ============================================================================

@dataclass
class ModelArguments:
    """
    模型相关参数配置类

    作用：定义模型加载相关的所有参数
    优势：使用dataclass自动生成__init__、__repr__等方法，减少样板代码
    """
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen-7B",
        metadata={"help": "模型路径或ModelScope ID，支持本地路径和远程模型"}
    )


@dataclass
class DataArguments:
    """
    数据相关参数配置类

    作用：定义数据加载和预处理的参数
    """
    data_path: str = field(
        default="/root/main/bojin_LLM/boshi-sample-solution/peft_model/sft_data/NER_lora.json",
        metadata={"help": "训练数据文件路径（JSON格式）"}
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "验证数据文件路径（可选，用于训练过程中评估）"}
    )
    lazy_preprocess: bool = field(
        default=False,
        metadata={"help": "是否使用懒加载模式（大数据集推荐启用，节省启动内存）"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    训练相关参数配置类

    继承自transformers.TrainingArguments，扩展了PEFT特有的参数
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "模型缓存目录，用于存储下载的模型文件"}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "优化器类型（adamw_torch/adamw_anyprecision等）"}
    )
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "模型最大序列长度（tokens数），影响显存占用和上下文理解能力"
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用LoRA微调方法"}
    )
    use_adalora: bool = field(
        default=False,
        metadata={"help": "是否使用AdaLoRA微调方法（自适应rank分配）"}
    )
    system_message: str = field(
        default='You are a helpful assistant',
        metadata={"help": "系统提示词，用于引导模型行为（如SQL专家、NER助手等）"}
    )
    


@dataclass
class LoraArguments:
    """
    LoRA相关参数配置类

    作用：详细定义LoRA/QLoRA/AdaLoRA的各种超参数
    """
    # ===== LoRA基础参数 =====
    lora_r: int = field(
        default=96,  # 7B模型推荐80，1.5B模型推荐16-32
        metadata={"help": "LoRA矩阵的秩，决定适配器容量（rank越大，参数越多，效果越好但显存占用越大）"}
    )
    lora_alpha: int = field(
        default=24,  # 通常设为lora_r的一半
        metadata={"help": "LoRA缩放因子，控制适配器权重的影响程度"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA层的dropout比率，用于防止过拟合（0.05-0.15之间）"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",  # 注意力模块
                                 "gate_proj", "up_proj", "down_proj"],   # 前馈网络模块
        metadata={"help": "应用LoRA的模块列表（全量覆盖或仅注意力层）"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "bias训练策略：none/所有/lora_only"}
    )

    # ===== QLoRA量化参数 =====
    q_lora: bool = field(
        default=False,
        metadata={"help": "是否使用QLoRA（LoRA + 4bit量化，大幅节省显存）"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "是否使用4-bit量化加载（需要bitsandbytes库）"}
    )

    # ===== 高级参数 =====
    lora_modules_to_save: bool = field(
        default=False,
        metadata={"help": "是否额外训练embedding和lm_head（省显存建议关闭）"}
    )
    lora_weight_path: str = field(
        default="",
        metadata={"help": "预训练LoRA权重路径（用于继续训练）"}
    )

    # ===== AdaLoRA参数 =====
    r: int = field(
        default=20,
        metadata={"help": "AdaLoRA初始rank（会根据重要性动态调整）"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "AdaLoRA目标模块列表"}
    )
    



# ============================================================================
# 全局变量和工具函数
# ============================================================================

local_rank = None  # 当前进程的本地rank（用于分布式训练）

def rank0_print(*args):
    """
    只在rank 0进程打印信息（分布式训练中使用）

    作用：在多GPU训练时，避免所有GPU都重复打印相同信息
    原理：只有rank 0的进程会执行print，其他进程跳过

    使用场景：
    - 打印训练进度信息
    - 输出日志和警告
    - 显示模型信息
    """
    if local_rank == 0:
        print(*args)




def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str
) -> Dict:
    """
    数据预处理函数：将对话数据转换为模型输入格式

    参数说明：
        sources: 对话数据列表，格式为 [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]
        tokenizer: 分词器，用于将文本转换为token IDs
        max_len: 最大序列长度，超过此长度的序列会被截断
        system_message: 系统提示词，用于引导模型行为

    返回值：
        包含input_ids、labels、attention_mask的字典

    核心功能：
        1. 将对话转换为Qwen聊天模板格式
        2. 创建训练标签（只训练assistant回复部分）
        3. 处理序列截断和padding
        4. 生成attention_mask

    训练策略：
        - 用户输入部分：label设为IGNORE_TOKEN_ID，不参与loss计算
        - 助手回复部分：正常计算loss
        - 系统提示词：label设为IGNORE_TOKEN_ID，不参与loss计算

    为什么需要标签掩码：
        - 只让模型学习如何生成正确的回复
        - 避免模型学习用户输入的模式
        - 提高训练效率和效果
    """
    # 角色到特殊token的映射（Qwen聊天模板格式）
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    # 兼容Qwen1.5和Qwen2.5的不同tokenizer实现
    if hasattr(tokenizer, 'im_start_id'):
        # Qwen1.5使用专用的im_start_id属性
        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
    else:
        # Qwen2.5需要手动转换特殊tokens
        im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')

    # 预先计算常用的token序列，避免重复计算
    nl_tokens = tokenizer('\n').input_ids                              # 换行符的token IDs
    _system = tokenizer('system').input_ids + nl_tokens                # "system" + 换行符
    _user = tokenizer('user').input_ids + nl_tokens                    # "user" + 换行符
    _assistant = tokenizer('assistant').input_ids + nl_tokens          # "assistant" + 换行符

    # 应用prompt模板，将对话转换为模型输入格式
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        # 确保第一个对话来自用户（如果不是则跳过）
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        # 构造系统提示词部分
        # 格式：<|im_start|>system{system_message}<|im_end|>\n
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        # 系统提示词部分：全部忽略，不参与loss计算
        target += [IGNORE_TOKEN_ID] * len(system)
        assert len(input_id) == len(target)  # 确保长度一致

        # 处理对话轮次
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]  # 获取当前对话角色
            # 构造输入：<|im_start|>role\n{content}<|im_end|>\n
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id

            if role == '<|im_start|>user':
                # 用户输入：全部忽略，不参与loss计算
                # 原因：我们只教模型如何生成回复，不教模型如何生成用户输入
                _target = [IGNORE_TOKEN_ID] * len(_input_id)

            elif role == '<|im_start|>assistant':
                # 助手回复：只训练实际回复内容，忽略角色标记
                role_len = len(tokenizer(role).input_ids) + len(nl_tokens)
                # 复制input_id确保长度一致
                _target = list(_input_id)
                # 忽略角色标记部分（<|im_start|>assistant\n）
                for i in range(role_len):
                    _target[i] = IGNORE_TOKEN_ID
                # 忽略结束标记（<|im_end|>\n）
                _target[-2] = IGNORE_TOKEN_ID
                _target[-1] = IGNORE_TOKEN_ID
            else:
                raise NotImplementedError(f"Unknown role: {role}")

            target += _target

        assert len(input_id) == len(target)  # 确保长度一致

        # 序列截断和padding
        # 如果序列过短，用pad_token_id填充到max_len
        # 如果序列过长，在后面截断到max_len
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])  # 截断到最大长度
        targets.append(target[:max_len])

    # 转换为PyTorch张量
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,                                   # 模型输入的token IDs
        labels=targets,                                       # 训练标签（只有assistant回复部分有值）
        attention_mask=input_ids.ne(tokenizer.pad_token_id), # 注意力掩码（1表示有效token，0表示padding）
    )


class SupervisedDataset(Dataset):
    """
    监督学习数据集类（预处理模式）

    作用：在初始化时预处理所有数据，训练时直接读取
    优势：训练时速度快，数据一致性好
    劣势：启动慢，占用内存较多
    适用：中小型数据集（<10万样本）
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int,system_message:str):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len,system_message = system_message)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        """返回数据集大小"""
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        获取单个样本数据

        参数：
            i: 样本索引

        返回：
            包含input_ids、labels、attention_mask的字典
        """
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """
    懒加载监督学习数据集类

    作用：在首次访问时才处理数据，并缓存结果
    优势：启动速度快，内存占用小，适合大数据集
    劣势：首次访问时有轻微延迟
    适用：大型数据集（>10万样本）或内存受限场景

    工作原理：
        1. 初始化时只存储原始数据和配置
        2. 首次访问某个样本时才进行预处理
        3. 将预处理结果缓存，避免重复计算
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, system_message):
        """
        初始化懒加载数据集

        参数：
            raw_data: 原始JSON数据
            tokenizer: 分词器
            max_len: 最大序列长度
            system_message: 系统提示词
        """
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data                       # 存储原始数据
        self.cached_data_dict = {}                      # 缓存已处理的数据
        self.system_message = system_message           # 系统提示词

    def __len__(self):
        """返回数据集大小"""
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        获取单个样本数据（懒加载 + 缓存）

        参数：
            i: 样本索引

        返回：
            包含input_ids、labels、attention_mask的字典
        """
        # 检查缓存，如果已处理则直接返回
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        # 首次访问时进行预处理
        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len, system_message=self.system_message)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )

        # 缓存结果，避免重复处理
        self.cached_data_dict[i] = ret
        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
    system_message
) -> Dict:
    """
    创建监督学习数据模块

    参数：
        tokenizer: 分词器
        data_args: 数据参数配置
        max_len: 最大序列长度
        system_message: 系统提示词

    返回：
        包含train_dataset和eval_dataset的字典

    功能：
        1. 根据lazy_preprocess参数选择合适的Dataset类
        2. 加载训练数据
        3. 可选加载验证数据
        4. 返回数据集字典
    """
    # 根据配置选择数据集类
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # 加载训练数据
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len, system_message=system_message)

    # 可选加载验证数据
    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len, system_message=system_message)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# ============================================================================
# 核心训练函数
# ============================================================================

def train():
    """
    主训练函数：执行完整的模型微调流程

    功能流程：
        1. 参数解析和验证
        2. 分布式训练初始化
        3. 模型和分词器加载
        4. PEFT配置和应用
        5. 数据加载
        6. 训练器初始化和启动

    这是整个脚本的核心函数，协调整个训练流程
    """
    global local_rank

    # 创建参数解析器，支持四类参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    # 解析命令行参数到各个数据类
    # 支持从命令行、配置文件或默认值获取参数
    (
        model_args,      # 模型相关参数（模型路径、缓存目录等）
        data_args,       # 数据相关参数（数据路径、懒加载等）
        training_args,   # 训练相关参数（学习率、batch size、序列长度等）
        lora_args,       # LoRA相关参数（rank、alpha、dropout等）
    ) = parser.parse_args_into_dataclasses()

    # ===== 梯度检查点配置（显存优化关键技术）=====
    # 强制使用非重入式梯度检查点，避免显存不降或报错
    # 这是QLoRA/LoRA + gradient_checkpointing的必需设置
    # 直接在代码里设置，绕过transformers 4.38命令行解析dict参数的bug
    if training_args.gradient_checkpointing:
        gck = training_args.gradient_checkpointing_kwargs or {}
        gck.setdefault("use_reentrant", False)  # 使用新的非重入式实现
        training_args.gradient_checkpointing_kwargs = gck

    # ===== 分布式训练配置 =====
    # 单GPU使用DeepSpeed时的特殊处理
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank  # 当前进程的本地rank

    # ===== 设备映射配置 =====
    device_map = None  # 设备映射策略（自动分布到可用GPU）
    world_size = int(os.environ.get("WORLD_SIZE", 2))  # 总进程数
    ddp = world_size != 1  # 是否为分布式数据并行

    # QLoRA的设备映射配置
    if lora_args.q_lora:
        # 多GPU：手动指定每个进程使用的GPU
        # 单GPU：自动分配到可用GPU
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"

        # 检查兼容性：QLoRA与FSDP/ZeRO-3不兼容
        if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are incompatible with QLoRA.")

    # ===== 模型兼容性检查 =====
    # 检查是否为chat模型（有特殊处理逻辑）
    is_chat_model = 'chat' in model_args.model_name_or_path.lower()

    # 兼容性检查：ZeRO-3 + LoRA + base model 组合不兼容
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    # ===== 模型加载参数配置 =====
    model_load_kwargs = {
        'low_cpu_mem_usage': not is_deepspeed_zero3_enabled(),  # ZeRO-3时禁用以避免问题
    }

    # ===== 模型配置加载 =====
    config = modelscope.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,        # 模型缓存目录
        trust_remote_code=True,                  # 信任远程代码（Qwen模型必需）
    )
    config.use_cache = False  # 禁用KV缓存以节省显存（训练时不使用）

    # ===== 量化配置（QLoRA专用）=====
    # 根据配置选择量化方法
    quantization_config = None
    if training_args.use_lora and lora_args.q_lora:
        if lora_args.load_in_4bit:
            # 使用bitsandbytes进行4-bit量化（推荐用于QLoRA）
            # NF4量化：4-bit NormalFloat量化，专为LLM设计
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,                          # 启用4-bit加载
                bnb_4bit_use_double_quant=True,              # 双重量化（进一步压缩）
                bnb_4bit_quant_type="nf4",                   # NF4量化类型
                bnb_4bit_compute_dtype=torch.float16,        # 计算时使用FP16
            )
        else:
            # 使用GPTQ量化（需要optimum库）
            # GPTQ是一种更精确的4-bit量化方法
            quantization_config = GPTQConfig(
                bits=4,                 # 4-bit量化
                disable_exllama=True   # 禁用exllama加速（兼容性考虑）
            )

    # ===== 模型加载 =====
    model = modelscope.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,                   # 自动分配到可用GPU
        trust_remote_code=True,                 # 信任远程代码（Qwen必需）
        quantization_config=quantization_config, # 量化配置（QLoRA使用）
        **model_load_kwargs,
    )

    # ===== Tokenizer加载 =====
    tokenizer = modelscope.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,  # 模型最大长度
        padding_side="right",                         # 右侧填充（生成任务标准）
        use_fast=False,                               # 使用慢tokenizer（兼容性更好）
        trust_remote_code=True,
    )
    # ===== Tokenizer padding配置 =====
    # Qwen2.5 uses eos_token_id instead of eod_id
    if hasattr(tokenizer, 'eod_id'):
        tokenizer.pad_token_id = tokenizer.eod_id      # Qwen1.5使用eod_id
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Qwen2.5+使用eos_token_id

    # ===== 模型embedding调整（可选）=====
    # Resize model embeddings to match tokenizer vocabulary
    # This is important for Qwen2.5 which adds new special tokens
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        print(f"Resizing model embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # ===== LoRA配置和应用 =====
    if training_args.use_lora:
        print('========================当前正在使用LORA微调===================')

        # modules_to_save控制：决定是否训练embedding和lm_head
        # 默认不训练（省显存），仅在需要时训练
        need_resize = model.get_input_embeddings().weight.shape[0] != len(tokenizer)
        if lora_args.lora_modules_to_save or need_resize:
            modules_to_save = ["wte", "lm_head"]  # 训练embedding和输出层
            print('  -> 训练 embedding 和 lm_head (modules_to_save)')
        else:
            modules_to_save = None
            print('  -> 仅训练 LoRA 层，不训练 embedding/lm_head (省显存)')

        # 创建LoRA配置对象
        lora_config = LoraConfig(
            r=lora_args.lora_r,                        # LoRA秩（决定适配器容量）
            lora_alpha=lora_args.lora_alpha,            # 缩放因子
            target_modules=lora_args.lora_target_modules,  # 目标模块
            lora_dropout=lora_args.lora_dropout,       # Dropout比率
            bias=lora_args.lora_bias,                   # Bias训练策略
            task_type="CAUSAL_LM",                    # 任务类型（因果语言建模）
            modules_to_save=modules_to_save,           # 需要额外训练的模块
        )

        # ===== QLoRA特殊处理 =====
        if lora_args.q_lora:
            # 准备量化模型进行训练
            if prepare_model_for_kbit_training is not None:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=training_args.gradient_checkpointing
                )
            else:
                # 没有 prepare_model_for_kbit_training 时的替代处理
                # 注意：不要对量化模型调用.half()，会破坏4-bit量化状态
                if training_args.gradient_checkpointing:
                    model.gradient_checkpointing_enable()
                model.enable_input_require_grads()

        # 应用LoRA到模型
        model = get_peft_model(model, lora_config)

        # 打印可训练参数统计信息
        model.print_trainable_parameters()

        # 启用输入梯度（梯度检查点需要）
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    # ===== AdaLoRA配置（可选）=====
    elif training_args.use_adalora:
        print('========================当前正在使用adaLORA微调===================')

        # AdaLoRA的modules_to_save配置
        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]

        # 创建AdaLoRA配置（自适应rank分配）
        ada_lora_config = AdaLoraConfig(
            r=lora_args.r,                            # 初始rank
            target_modules=lora_args.target_modules,  # 目标模块
            lora_dropout=lora_args.lora_dropout,      # Dropout比率
            task_type="CAUSAL_LM",                   # 任务类型
            modules_to_save=modules_to_save          # 需要额外训练的模块
        )
        model = get_peft_model(model, ada_lora_config)

        # 打印可训练参数统计信息
        model.print_trainable_parameters()
        model.is_parallelizable = True          # 支持并行化
        model.model_parallel = True              # 启用模型并行

        # 启用输入梯度（梯度检查点需要）
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # ===== 打印训练配置信息 =====
    print(training_args)

    # ===== 数据加载 =====
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
        system_message=training_args.system_message
    )

    # ===== 训练器初始化和启动 =====
    trainer = Trainer(
        model=model,                    # 模型
        tokenizer=tokenizer,            # 分词器
        args=training_args,             # 训练参数
        **data_module                  # 数据集（train_dataset, eval_dataset）
    )

    # 开始训练
    trainer.train()

    # 保存最终模型（LoRA adapter 权重 + 配置）
    # 对 PEFT 模型，save_model 会生成 adapter_config.json 和 adapter_model.safetensors
    # 这是后续 PeftModel.from_pretrained / merge_and_unload 加载所必需的
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"\n✅ 训练完成，LoRA adapter 已保存到: {training_args.output_dir}")


def merge_model():
    """
    模型合并函数：将LoRA权重合并到基础模型

    功能：
        1. 加载基础模型
        2. 加载训练好的LoRA适配器
        3. 将LoRA权重合并到基础模型
        4. 返回合并后的完整模型

    使用场景：
        - 训练完成后部署模型
        - 推理时需要完整模型
        - 避免每次推理都要加载LoRA适配器

    合并后的优势：
        - 推理速度更快（不需要动态计算LoRA）
        - 模型部署更简单
        - 可以直接用于生产环境
    """
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",           # 自动分配到可用GPU
        trust_remote_code=True
    )

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, training_args.output_dir)

    print("Loaded PEFT model. Merging...")

    # 合并LoRA权重到基础模型并卸载LoRA适配器
    model.merge_and_unload()

    print("Merge complete.")
    return model, model_args.model_name_or_path


def test_lora_model():
    """
    测试训练好的LoRA模型

    功能：
        1. 加载合并后的模型
        2. 进行推理测试
        3. 打印生成结果

    使用场景：
        - 快速验证训练效果
        - 调试模型问题
        - 生成示例输出
    """
    # 合并模型
    model, old_model_path = merge_model()

    # 清理显存
    torch_gc()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        old_model_path,
        trust_remote_code=True,
    )

    # 测试问题（可根据具体任务修改）
    prompt = "大连派思燃气系统股份有限公司在何时获得美国GE公司的合格供应商认证？"

    # 进行推理
    response, history = model.chat(tokenizer, prompt, history=None)

    # 打印结果
    print(response)

    # 清理显存
    torch_gc()


# ============================================================================
# Python 3.11兼容性检查函数
# ============================================================================

def check_python_311_compatibility():
    """
    检查Python 3.11环境的兼容性

    主要检查：
    1. PyTorch版本兼容性
    2. Transformers版本兼容性
    3. PEFT版本兼容性
    4. CUDA可用性
    """
    if sys.version_info < (3, 11):
        return  # Python 3.11以下版本无需特殊检查

    print("🔍 检测到Python 3.11+环境，进行兼容性检查...")

    # 检查关键库版本
    issues = []
    recommendations = []

    # PyTorch检查
    try:
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 0):
            issues.append(f"PyTorch版本过低: {torch.__version__}，推荐2.0+")
            recommendations.append("pip install --upgrade torch torchvision")
    except Exception as e:
        issues.append(f"PyTorch检查失败: {e}")

    # Transformers检查
    try:
        transformers_version = tuple(map(int, transformers.__version__.split('.')[:2]))
        if transformers_version < (4, 35):
            issues.append(f"Transformers版本过低: {transformers.__version__}，推荐4.35+")
            recommendations.append("pip install --upgrade transformers")
    except Exception as e:
        issues.append(f"Transformers检查失败: {e}")

    # PEFT检查
    try:
        import peft
        peft_version = tuple(map(int, peft.__version__.split('.')[:2])) if hasattr(peft, '__version__') else (0, 0)
        if peft_version < (0, 7):
            issues.append(f"PEFT版本过低: {peft.__version__}，推荐0.7+")
            recommendations.append("pip install --upgrade peft")
    except Exception as e:
        issues.append(f"PEFT检查失败: {e}")

    # 检查结果输出
    if issues:
        print("⚠️  发现兼容性问题:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 建议:")
        for rec in recommendations:
            print(f"   - {rec}")
        print("\n继续训练可能会遇到兼容性问题，建议先升级相关库。")
        response = input("是否继续训练? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        print("✅ Python 3.11兼容性检查通过")

    # CUDA检查
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.version.cuda}")
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        warnings.warn("⚠️  CUDA不可用，训练会使用CPU，速度会很慢")


# ============================================================================
# 主函数入口
# ============================================================================

if __name__ == "__main__":
    """
    主函数入口：执行训练流程

    功能：
        1. Python 3.11兼容性检查
        2. 设置随机种子（确保可复现）
        3. 调用训练函数
        4. 可选：测试训练好的模型

    使用方式：
        python finetune_qwen.py \\
            --model_name_or_path ./models/Qwen2.5-1.5B-Instruct \\
            --data_path ../data/lora_data/sql-lora-train-end.json \\
            --use_lora --lora_r 16 \\
            --num_train_epochs 4 \\
            --per_device_train_batch_size 1 \\
            --gradient_accumulation_steps 8
    """
    # Python 3.11兼容性检查
    check_python_311_compatibility()

    seed_it(2024)        # 设置随机种子
    train()              # 开始训练并保存 LoRA adapter 到 output_dir
    # 训练完成。如需把 LoRA 权重合并进基础模型，单独运行 merge_model（取消下行注释）：
    # model, old_model_path = merge_model()
    torch_gc()