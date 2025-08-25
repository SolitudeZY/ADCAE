from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """ADCAE模型配置"""
    # 基本参数
    input_channels: int = 1
    encoding_dim: int = 128
    
    # 编码器配置
    encoder_channels: List[int] = None
    encoder_kernels: List[int] = None
    encoder_strides: List[int] = None
    encoder_paddings: List[int] = None
    
    # 解码器配置
    decoder_channels: List[int] = None
    decoder_kernels: List[int] = None
    decoder_strides: List[int] = None
    decoder_paddings: List[int] = None
    decoder_output_paddings: List[int] = None
    
    # 注意力机制配置
    use_attention: bool = True
    attention_type: str = 'cbam'  # 'cbam', 'channel', 'spatial', 'none'
    attention_reduction: int = 16
    attention_kernel_size: int = 7
    
    # 激活函数配置
    activation: str = 'elu'  # 'elu', 'relu', 'sigmoid'
    
    # 正则化配置
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    
    def __post_init__(self):
        # 设置默认的编码器配置（5层）
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128, 256, 512, 1024]
        if self.encoder_kernels is None:
            self.encoder_kernels = [7, 5, 3, 3, 2]
        if self.encoder_strides is None:
            self.encoder_strides = [2, 2, 2, 2, 2]
        if self.encoder_paddings is None:
            self.encoder_paddings = [3, 2, 1, 1, 0]
            
        # 设置默认的解码器配置（5层，与编码器对称）
        if self.decoder_channels is None:
            self.decoder_channels = [512, 256, 128, 64, 32]
        if self.decoder_kernels is None:
            self.decoder_kernels = [2, 3, 3, 5, 7]
        if self.decoder_strides is None:
            self.decoder_strides = [2, 2, 2, 2, 2]
        if self.decoder_paddings is None:
            self.decoder_paddings = [0, 1, 1, 2, 3]
        if self.decoder_output_paddings is None:
            self.decoder_output_paddings = [0, 1, 1, 1, 1]

@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 25
    
    # 学习率调度
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    
    # 数据配置
    validation_split: float = 0.2
    max_files_per_category: Optional[int] = None
    
@dataclass
class DataConfig:
    """数据配置"""
    file_length: int = 1024
    image_size: int = 32

# 实验配置生成器
class ExperimentConfigs:
    """实验配置生成器"""
    
    # 在 ExperimentConfigs 类中修复层数配置
    @staticmethod
    def generate_layer_configs():
        """生成不同层数的配置"""
        configs = {}
        
        # 2层配置 - 保持不变
        configs['2_layers'] = ModelConfig(
            encoder_channels=[64, 128],
            encoder_kernels=[7, 5],
            encoder_strides=[2, 2],
            encoder_paddings=[3, 2],
            decoder_channels=[64, 32],
            decoder_kernels=[5, 7],
            decoder_strides=[2, 2],
            decoder_paddings=[2, 3],
            decoder_output_paddings=[1, 1]
        )
        
        # 4层配置 - 保持不变
        configs['4_layers'] = ModelConfig(
            encoder_channels=[32, 64, 128, 256],
            encoder_kernels=[7, 5, 3, 3],
            encoder_strides=[2, 2, 2, 2],
            encoder_paddings=[3, 2, 1, 1],
            decoder_channels=[128, 64, 32, 16],
            decoder_kernels=[3, 3, 5, 7],
            decoder_strides=[2, 2, 2, 2],
            decoder_paddings=[1, 1, 2, 3],
            decoder_output_paddings=[1, 1, 1, 1]
        )
        
        # 6层配置 - 适度增加深度
        configs['6_layers'] = ModelConfig(
            encoder_channels=[32, 64, 128, 256, 512, 1024],
            encoder_kernels=[3, 3, 3, 3, 3, 2],  # 最后一层用小核
            encoder_strides=[2, 2, 2, 2, 2, 1],  # 5次下采样 + 1次特征提取
            encoder_paddings=[1, 1, 1, 1, 1, 0],
            decoder_channels=[512, 256, 128, 64, 32, 16],
            decoder_kernels=[2, 3, 3, 3, 3, 3],
            decoder_strides=[1, 2, 2, 2, 2, 2],  # 对应的上采样
            decoder_paddings=[0, 1, 1, 1, 1, 1],
            decoder_output_paddings=[0, 1, 1, 1, 1, 1]
        )
        
        # 8层配置 - 增加特征学习层
        configs['8_layers'] = ModelConfig(
            encoder_channels=[16, 32, 64, 128, 256, 512, 1024, 1024],
            encoder_kernels=[3, 3, 3, 3, 3, 3, 3, 1],  # 最后一层使用1x1卷积
            encoder_strides=[2, 2, 2, 2, 2, 1, 1, 1],  # 5次下采样 + 3次特征学习
            encoder_paddings=[1, 1, 1, 1, 1, 1, 1, 0],
            decoder_channels=[1024, 512, 256, 128, 64, 32, 16, 16],
            decoder_kernels=[1, 3, 3, 3, 3, 3, 3, 3],  # 第一层使用1x1卷积
            decoder_strides=[1, 1, 1, 2, 2, 2, 2, 2],
            decoder_paddings=[0, 1, 1, 1, 1, 1, 1, 1],
            decoder_output_paddings=[0, 0, 0, 1, 1, 1, 1, 1]
        )
        
        # 10层配置 - 折中方案：5次下采样 + 5次深度特征学习
        configs['10_layers'] = ModelConfig(
            encoder_channels=[8, 16, 32, 64, 128, 256, 512, 512, 512, 512],
            encoder_kernels=[3, 3, 3, 3, 3, 3, 3, 3, 3, 1],  # 最后一层使用1x1卷积
            encoder_strides=[2, 2, 2, 2, 2, 1, 1, 1, 1, 1],  # 5次下采样 + 5次特征学习
            encoder_paddings=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            decoder_channels=[512, 512, 512, 256, 128, 64, 32, 16, 16, 16],
            decoder_kernels=[1, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # 第一层使用1x1卷积
            decoder_strides=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2],  # 对应的上采样
            decoder_paddings=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            decoder_output_paddings=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        )
        
        # 12层配置 - 折中方案：5次下采样 + 7次深度特征学习
        configs['12_layers'] = ModelConfig(
            encoder_channels=[4, 8, 16, 32, 64, 128, 256, 256, 256, 256, 256, 256],
            encoder_kernels=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1],  # 最后一层使用1x1卷积
            encoder_strides=[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],  # 5次下采样 + 7次特征学习
            encoder_paddings=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            decoder_channels=[256, 256, 256, 256, 256, 128, 64, 32, 16, 16, 16, 16],
            decoder_kernels=[1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # 第一层使用1x1卷积
            decoder_strides=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],  # 对应的上采样
            decoder_paddings=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            decoder_output_paddings=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        )
        
        return configs
    
    @staticmethod
    def generate_activation_configs(base_config=None):
        """生成不同激活函数的配置"""
        if base_config is None:
            base_config = ModelConfig()  # 使用默认配置
        
        configs = {
            'elu': ModelConfig(
                activation='elu',
                encoder_channels=base_config.encoder_channels,
                encoder_kernels=base_config.encoder_kernels,
                encoder_strides=base_config.encoder_strides,
                encoder_paddings=base_config.encoder_paddings,
                decoder_channels=base_config.decoder_channels,
                decoder_kernels=base_config.decoder_kernels,
                decoder_strides=base_config.decoder_strides,
                decoder_paddings=base_config.decoder_paddings,
                decoder_output_paddings=base_config.decoder_output_paddings
            ),
            'relu': ModelConfig(
                activation='relu',
                encoder_channels=base_config.encoder_channels,
                encoder_kernels=base_config.encoder_kernels,
                encoder_strides=base_config.encoder_strides,
                encoder_paddings=base_config.encoder_paddings,
                decoder_channels=base_config.decoder_channels,
                decoder_kernels=base_config.decoder_kernels,
                decoder_strides=base_config.decoder_strides,
                decoder_paddings=base_config.decoder_paddings,
                decoder_output_paddings=base_config.decoder_output_paddings
            ),
            'sigmoid': ModelConfig(
                activation='sigmoid',
                encoder_channels=base_config.encoder_channels,
                encoder_kernels=base_config.encoder_kernels,
                encoder_strides=base_config.encoder_strides,
                encoder_paddings=base_config.encoder_paddings,
                decoder_channels=base_config.decoder_channels,
                decoder_kernels=base_config.decoder_kernels,
                decoder_strides=base_config.decoder_strides,
                decoder_paddings=base_config.decoder_paddings,
                decoder_output_paddings=base_config.decoder_output_paddings
            )
        }
        
        return configs
    
    @staticmethod
    def generate_attention_configs(base_config=None):
        """生成注意力机制对比配置"""
        if base_config is None:
            base_config = ModelConfig()  # 使用默认配置
        
        configs = {
            'with_cbam': ModelConfig(
                use_attention=True,
                attention_type='cbam',
                encoder_channels=base_config.encoder_channels,
                encoder_kernels=base_config.encoder_kernels,
                encoder_strides=base_config.encoder_strides,
                encoder_paddings=base_config.encoder_paddings,
                decoder_channels=base_config.decoder_channels,
                decoder_kernels=base_config.decoder_kernels,
                decoder_strides=base_config.decoder_strides,
                decoder_paddings=base_config.decoder_paddings,
                decoder_output_paddings=base_config.decoder_output_paddings
            ),
            'without_attention': ModelConfig(
                use_attention=False,
                encoder_channels=base_config.encoder_channels,
                encoder_kernels=base_config.encoder_kernels,
                encoder_strides=base_config.encoder_strides,
                encoder_paddings=base_config.encoder_paddings,
                decoder_channels=base_config.decoder_channels,
                decoder_kernels=base_config.decoder_kernels,
                decoder_strides=base_config.decoder_strides,
                decoder_paddings=base_config.decoder_paddings,
                decoder_output_paddings=base_config.decoder_output_paddings
            )
        }
        
        return configs