import torch
import torch.nn as nn
from .blocks import AsymmetricConvBlock, AsymmetricDeconvBlock, get_activation
from .config import ModelConfig

class ADCAE(nn.Module):
    """可配置的非对称深度卷积自编码器"""
    
    def __init__(self, config: ModelConfig):
        super(ADCAE, self).__init__()
        
        self.config = config
        self.input_channels = config.input_channels
        self.encoding_dim = config.encoding_dim
        
        # 计算编码器输出的特征图尺寸
        self.encoded_h, self.encoded_w = self._calculate_encoded_size()
        
        # 构建编码器
        self.encoder = self._build_encoder()
        
        # 瓶颈层
        self.bottleneck = self._build_bottleneck()
        
        # 解码器恢复层
        self.decoder_fc = self._build_decoder_fc()
        
        # 构建解码器
        self.decoder = self._build_decoder()
        
        # 最终输出层
        self.final_layer = nn.Sequential(
            nn.Conv2d(config.decoder_channels[-1], config.input_channels, 
                     kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _calculate_encoded_size(self):
        """计算编码器输出的特征图尺寸"""
        h, w = 32, 32  # 输入图像尺寸
        
        for i in range(len(self.config.encoder_channels)):
            stride = self.config.encoder_strides[i]
            padding = self.config.encoder_paddings[i]
            kernel_size = self.config.encoder_kernels[i]
            
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1
        
        return h, w
    
    def _calculate_flattened_size(self):
        """计算编码器输出展平后的尺寸"""
        return self.config.encoder_channels[-1] * self.encoded_h * self.encoded_w
    
    def _build_encoder(self):
        """构建编码器"""
        layers = nn.ModuleList()
        
        in_channels = self.config.input_channels
        for i, out_channels in enumerate(self.config.encoder_channels):
            layer = AsymmetricConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.config.encoder_kernels[i],
                stride=self.config.encoder_strides[i],
                padding=self.config.encoder_paddings[i],
                use_attention=self.config.use_attention,
                attention_type=self.config.attention_type,
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate,
                use_batch_norm=self.config.use_batch_norm,
                reduction=self.config.attention_reduction,
                attention_kernel_size=self.config.attention_kernel_size
            )
            layers.append(layer)
            in_channels = out_channels
        
        return layers
    
    def _build_decoder(self):
        """构建解码器"""
        layers = nn.ModuleList()
        
        in_channels = self.config.encoder_channels[-1]
        for i, out_channels in enumerate(self.config.decoder_channels):
            layer = AsymmetricDeconvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.config.decoder_kernels[i],
                stride=self.config.decoder_strides[i],
                padding=self.config.decoder_paddings[i],
                output_padding=self.config.decoder_output_paddings[i],
                use_attention=self.config.use_attention,
                attention_type=self.config.attention_type,
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate,
                use_batch_norm=self.config.use_batch_norm,
                reduction=self.config.attention_reduction,
                attention_kernel_size=self.config.attention_kernel_size
            )
            layers.append(layer)
            in_channels = out_channels
        
        return layers
    
    def _build_bottleneck(self):
        """构建瓶颈层"""
        activation_fn = get_activation(self.config.activation)
        flattened_size = self._calculate_flattened_size()
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            activation_fn,
            nn.Dropout(0.2),
            nn.Linear(512, self.config.encoding_dim),
            activation_fn
        )
    
    def _build_decoder_fc(self):
        """构建解码器恢复层"""
        activation_fn = get_activation(self.config.activation)
        flattened_size = self._calculate_flattened_size()
        
        return nn.Sequential(
            nn.Linear(self.config.encoding_dim, 512),
            activation_fn,
            nn.Dropout(0.2),
            nn.Linear(512, flattened_size),
            activation_fn
        )
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """编码过程"""
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        encoded = self.bottleneck(x)
        return encoded
    
    def decode(self, encoded):
        """解码过程"""
        x = self.decoder_fc(encoded)
        x = x.view(-1, self.config.encoder_channels[-1], self.encoded_h, self.encoded_w)
        
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        
        decoded = self.final_layer(x)
        return decoded
    
    def forward(self, x):
        """前向传播"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_layers': len(self.config.encoder_channels),
            'decoder_layers': len(self.config.decoder_channels),
            'encoding_dim': self.config.encoding_dim,
            'attention_type': self.config.attention_type,
            'activation': self.config.activation,
            'encoded_size': f'{self.encoded_h}x{self.encoded_w}'
        }