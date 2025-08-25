import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction=16, activation='elu'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 确保reduction后的通道数至少为1
        reduced_channels = max(1, in_channels // reduction)
        
        # 可配置的激活函数
        if activation == 'elu':
            act_fn = nn.ELU(inplace=True)
        elif activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ELU(inplace=True)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            act_fn,
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        return self.sigmoid(x_cat)

class CBAM(nn.Module):
    """卷积块注意力模块"""
    def __init__(self, in_channels, reduction=16, kernel_size=7, activation='elu'):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction, activation)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class AttentionBlock(nn.Module):
    """可配置的注意力块"""
    def __init__(self, in_channels, attention_type='cbam', **kwargs):
        super(AttentionBlock, self).__init__()
        self.attention_type = attention_type
        
        # 处理参数名映射
        if 'attention_kernel_size' in kwargs:
            kwargs['kernel_size'] = kwargs.pop('attention_kernel_size')
        
        if attention_type == 'cbam':
            self.attention = CBAM(in_channels, **kwargs)
        elif attention_type == 'channel':
            self.attention = ChannelAttention(in_channels, **kwargs)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention(**kwargs)
        elif attention_type == 'none':
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
    def forward(self, x):
        return self.attention(x)