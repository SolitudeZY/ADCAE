import torch.nn as nn
from .attention import AttentionBlock

def get_activation(activation_name):
    """获取激活函数"""
    activations = {
        'elu': nn.ELU(inplace=True),
        'relu': nn.ReLU(inplace=True),
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.2, inplace=True)
    }
    return activations.get(activation_name, nn.ELU(inplace=True))

class AsymmetricConvBlock(nn.Module):
    """可配置的非对称卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_attention=True, attention_type='cbam', activation='elu',
                 dropout_rate=0.1, use_batch_norm=True, **attention_kwargs):
        super(AsymmetricConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # 可选的批归一化
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        
        # 可配置的激活函数
        self.activation = get_activation(activation)
        
        # 可配置的dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # 可配置的注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(out_channels, attention_type, **attention_kwargs)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        return x

class AsymmetricDeconvBlock(nn.Module):
    """可配置的非对称反卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                 output_padding=1, use_attention=True, attention_type='cbam',
                 activation='elu', dropout_rate=0.1, use_batch_norm=True, **attention_kwargs):
        super(AsymmetricDeconvBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                       stride, padding, output_padding, bias=False)
        
        # 可选的批归一化
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        
        # 可配置的激活函数
        self.activation = get_activation(activation)
        
        # 可配置的dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # 可配置的注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(out_channels, attention_type, **attention_kwargs)
        
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        return x