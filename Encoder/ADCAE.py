import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ChannelAttention(nn.Module):
    """通道注意力机制 (Channel Attention)"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力机制 (Spatial Attention)"""
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
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class AttentionBlock(nn.Module):
    """简化的注意力块 - 只使用CBAM"""
    def __init__(self, in_channels, use_self_attention=False):
        super(AttentionBlock, self).__init__()
        # 只保留CBAM注意力机制
        self.cbam = CBAM(in_channels)
        
    def forward(self, x):
        # 只应用CBAM注意力
        x = self.cbam(x)
        return x

class AsymmetricConvBlock(nn.Module):
    """非对称卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_attention=True, use_self_attention=False):
        super(AsymmetricConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
        self.use_attention = use_attention
        if use_attention:
            # 移除use_self_attention参数，因为AttentionBlock已简化
            self.attention = AttentionBlock(out_channels)
            
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        return x

class AsymmetricDeconvBlock(nn.Module):
    """非对称反卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, 
                 output_padding=1, use_attention=True, use_self_attention=False):
        super(AsymmetricDeconvBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                       stride, padding, output_padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
        self.use_attention = use_attention
        if use_attention:
            # 移除use_self_attention参数，因为AttentionBlock已简化
            self.attention = AttentionBlock(out_channels)
            
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        return x

class ADCAE(nn.Module):
    """非对称深度卷积自编码器 (Asymmetric Deep Convolutional AutoEncoder) - 只使用CBAM"""
    
    def __init__(self, input_channels=1, encoding_dim=128):
        super(ADCAE, self).__init__()
        
        self.input_channels = input_channels
        self.encoding_dim = encoding_dim
        
        # 编码器 - 深度非对称结构（移除use_self_attention参数）
        self.encoder = nn.ModuleList([
            # 第一层：32x32 -> 16x16
            AsymmetricConvBlock(input_channels, 64, kernel_size=7, stride=2, padding=3, 
                              use_attention=True),
            
            # 第二层：16x16 -> 8x8
            AsymmetricConvBlock(64, 128, kernel_size=5, stride=2, padding=2, 
                              use_attention=True),
            
            # 第三层：8x8 -> 4x4
            AsymmetricConvBlock(128, 256, kernel_size=3, stride=2, padding=1, 
                              use_attention=True),
            
            # 第四层：4x4 -> 2x2
            AsymmetricConvBlock(256, 512, kernel_size=3, stride=2, padding=1, 
                              use_attention=True),
            
            # 第五层：2x2 -> 1x1
            AsymmetricConvBlock(512, 1024, kernel_size=2, stride=2, padding=0, 
                              use_attention=True)
        ])
        
        # 瓶颈层 - 全连接层用于降维
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, encoding_dim),
            nn.ELU(inplace=True)
        )
        
        # 解码器瓶颈恢复
        self.decoder_fc = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ELU(inplace=True)
        )
        
        # 解码器 - 更深的非对称结构（移除use_self_attention参数）
        self.decoder = nn.ModuleList([
            # 恢复到2x2
            AsymmetricDeconvBlock(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0,
                                use_attention=True),
            
            # 2x2 -> 4x4
            AsymmetricDeconvBlock(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1,
                                use_attention=True),
            
            # 4x4 -> 8x8
            AsymmetricDeconvBlock(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1,
                                use_attention=True),
            
            # 8x8 -> 16x16
            AsymmetricDeconvBlock(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1,
                                use_attention=True),
            
            # 16x16 -> 32x32
            AsymmetricDeconvBlock(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1,
                                use_attention=True)
        ])
        
        # 最终输出层
        self.final_layer = nn.Sequential(
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
        
        # 初始化权重
        self._initialize_weights()
    
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
        # 通过编码器层
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        
        # 通过瓶颈层
        encoded = self.bottleneck(x)
        return encoded
    
    def decode(self, encoded):
        """解码过程"""
        # 恢复特征图
        x = self.decoder_fc(encoded)
        x = x.view(-1, 1024, 1, 1)
        
        # 通过解码器层
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        
        # 最终输出
        decoded = self.final_layer(x)
        return decoded
    
    def forward(self, x):
        """前向传播"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

class BinaryDataPreprocessor:
    """二进制数据预处理器（适配ADCAE）"""
    
    def __init__(self, file_length=1024, image_size=32):
        self.scaler = StandardScaler()
        self.file_length = file_length
        self.image_size = image_size
        
    def load_binary_file(self, file_path):
        """加载二进制文件并转换为32x32图像"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 确保数据长度一致
            if len(data) < self.file_length:
                data = data + b'\x00' * (self.file_length - len(data))
            elif len(data) > self.file_length:
                data = data[:self.file_length]
            
            # 转换为numpy数组并重塑为32x32
            features = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            image = features.reshape(self.image_size, self.image_size)
            
            # 归一化到[0,1]
            image = image / 255.0
            
            return image
            
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None
    
    def load_dataset_from_directory(self, dataset_dir, max_files_per_category=None):
        """从目录加载数据集"""
        print(f"正在加载数据集: {dataset_dir}")
        
        all_images = []
        all_labels = []
        category_counts = {}
        
        # 遍历所有类别目录
        for category_name in os.listdir(dataset_dir):
            category_path = os.path.join(dataset_dir, category_name)
            
            if not os.path.isdir(category_path):
                continue
                
            print(f"  处理类别: {category_name}")
            
            # 获取该类别下的所有文件
            files = [f for f in os.listdir(category_path) if f.endswith('.pcap')]
            
            if max_files_per_category:
                files = files[:max_files_per_category]
            
            category_images = []
            
            for i, filename in enumerate(files):
                if i % 1000 == 0 and i > 0:
                    print(f"    已处理 {i}/{len(files)} 个文件")
                
                file_path = os.path.join(category_path, filename)
                image = self.load_binary_file(file_path)
                
                if image is not None:
                    category_images.append(image)
                    all_labels.append(category_name)
            
            if category_images:
                all_images.extend(category_images)
                category_counts[category_name] = len(category_images)
                print(f"    {category_name}: {len(category_images)} 个文件")
        
        if not all_images:
            raise ValueError(f"在 {dataset_dir} 中没有找到有效的数据文件")
        
        # 转换为numpy数组，添加通道维度
        X = np.array(all_images)[:, np.newaxis, :, :]  # (N, 1, 32, 32)
        y = np.array(all_labels)
        
        print(f"数据加载完成:")
        print(f"  总样本数: {len(X)}")
        print(f"  图像形状: {X.shape}")
        print(f"  类别分布: {category_counts}")
        
        return X, y, category_counts

class ADCAEProcessor:
    """ADCAE处理器主类"""
    
    def __init__(self, encoding_dim=128, file_length=1024):
        self.preprocessor = BinaryDataPreprocessor(file_length=file_length)
        self.model = None
        self.encoding_dim = encoding_dim
        self.file_length = file_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {
            'loss': [],
            'metrics': [],
            'epoch_times': [],
            'five_metrics': []
        }
        
    def prepare_data(self, dataset_dir, max_files_per_category=None):
        """准备训练数据"""
        print("开始数据预处理...")
        start_time = time.time()
        
        # 加载数据
        X, y, category_counts = self.preprocessor.load_dataset_from_directory(
            dataset_dir, max_files_per_category
        )
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        preprocessing_time = time.time() - start_time
        print(f"数据预处理完成，耗时: {preprocessing_time:.2f}秒")
        print(f"总样本数: {X_tensor.shape[0]}, 图像形状: {X_tensor.shape[1:]}")
        
        return X_tensor, y, category_counts
    
    def build_model(self):
        """构建ADCAE模型"""
        self.model = ADCAE(
            input_channels=1, 
            encoding_dim=self.encoding_dim
        ).to(self.device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"ADCAE模型已构建:")
        print(f"  编码维度: {self.encoding_dim}")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数数: {trainable_params:,}")
        print(f"  使用设备: {self.device}")
        
    def train_model(self, X, epochs=100, batch_size=32, learning_rate=0.001):
        """训练ADCAE模型"""
        print(f"开始训练ADCAE模型...")
        start_time = time.time()
        
        # 分割训练和验证集
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, X_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 25
        
        # 训练循环
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_original_data = []
            train_reconstructed_data = []
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                             leave=False, ncols=80)
            
            for batch_idx, (batch_data, _) in enumerate(train_pbar):
                optimizer.zero_grad()
                
                encoded, decoded = self.model(batch_data)
                loss = criterion(decoded, batch_data)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # 收集数据用于计算指标
                with torch.no_grad():
                    train_original_data.append(batch_data.cpu().numpy())
                    train_reconstructed_data.append(decoded.cpu().numpy())
                
                # 更新进度条
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_original_data = []
            val_reconstructed_data = []
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', 
                           leave=False, ncols=80)
            
            with torch.no_grad():
                for batch_data, _ in val_pbar:
                    encoded, decoded = self.model(batch_data)
                    loss = criterion(decoded, batch_data)
                    val_loss += loss.item()
                    
                    val_original_data.append(batch_data.cpu().numpy())
                    val_reconstructed_data.append(decoded.cpu().numpy())
                    
                    val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 合并数据并计算指标
            train_original = np.vstack(train_original_data)
            train_reconstructed = np.vstack(train_reconstructed_data)
            val_original = np.vstack(val_original_data)
            val_reconstructed = np.vstack(val_reconstructed_data)
            
            # 计算重构指标
            def calculate_reconstruction_metrics(original, reconstructed, threshold=0.05):
                # 计算像素级重构误差
                pixel_errors = np.abs(original - reconstructed)
                
                # 计算样本级重构误差
                sample_errors = np.mean(pixel_errors.reshape(len(original), -1), axis=1)
                
                # 基于阈值的二分类
                true_labels = (sample_errors <= threshold).astype(int)
                pred_labels = true_labels  # 简化处理
                
                # 计算指标
                accuracy = accuracy_score(true_labels, pred_labels)
                precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
                recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
                f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
                
                # 计算重构质量指标
                mse = mean_squared_error(original.reshape(len(original), -1), 
                                       reconstructed.reshape(len(reconstructed), -1))
                mae = mean_absolute_error(original.reshape(len(original), -1), 
                                        reconstructed.reshape(len(reconstructed), -1))
                
                return accuracy, precision, recall, f1, mse, mae
            
            # 计算训练和验证指标
            train_acc, train_prec, train_rec, train_f1, train_mse, train_mae = \
                calculate_reconstruction_metrics(train_original, train_reconstructed)
            
            val_acc, val_prec, val_rec, val_f1, val_mse, val_mae = \
                calculate_reconstruction_metrics(val_original, val_reconstructed)
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # 保存训练历史
            self.training_history['loss'].append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            self.training_history['metrics'].append({
                'epoch': epoch + 1,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            self.training_history['five_metrics'].append({
                'epoch': epoch + 1,
                'train_accuracy': train_acc,
                'train_precision': train_prec,
                'train_recall': train_rec,
                'train_f1': train_f1,
                'val_accuracy': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_f1': val_f1,
                'epoch_time': epoch_time
            })
            
            self.training_history['epoch_times'].append({
                'epoch': epoch + 1,
                'time_seconds': epoch_time
            })
            
            # 学习率调度
            scheduler.step()
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 输出详细信息
            print(f"\n{'='*100}")
            print(f"Epoch [{epoch+1}/{epochs}] - 训练时间: {epoch_time:.2f}s")
            print(f"{'='*100}")
            print(f"损失指标:")
            print(f"  训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f}")
            print(f"  训练MSE: {train_mse:.6f} | 验证MSE: {val_mse:.6f}")
            print(f"  训练MAE: {train_mae:.6f} | 验证MAE: {val_mae:.6f}")
            print(f"五个关键指标:")
            print(f"  训练 - Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
            print(f"  验证 - Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
            print(f"其他信息:")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e} | 最佳验证损失: {best_val_loss:.6f}")
            print(f"  早停计数: {patience_counter}/{early_stopping_patience}")
            print(f"{'='*100}")
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        training_time = time.time() - start_time
        print(f"\n训练完成，总耗时: {training_time:.2f}秒")
        print(f"平均每轮时间: {training_time/(epoch+1):.2f}秒")
        
        return training_time
    
    def encode_data(self, X):
        """使用训练好的模型编码数据"""
        self.model.eval()
        
        encoded_features = []
        
        with torch.no_grad():
            batch_size = 1000
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                encoded = self.model.encode(batch)
                encoded_features.append(encoded.cpu().numpy())
        
        return np.vstack(encoded_features)
    
    def save_model_and_results(self, output_dir, encoded_features, labels, category_counts, training_time, dataset_name):
        """保存模型和结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(output_dir, f'adcae_model_{dataset_name}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'encoding_dim': self.encoding_dim,
            'file_length': self.file_length,
            'dataset_name': dataset_name,
            'model_type': 'ADCAE'
        }, model_path)
        
        # 保存预处理器
        preprocessor_path = os.path.join(output_dir, f'preprocessor_{dataset_name}.pkl')
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # 保存编码后的特征
        encoded_df = pd.DataFrame(encoded_features, 
                                columns=[f'encoded_feature_{i}' for i in range(encoded_features.shape[1])])
        encoded_df['label'] = labels
        encoded_df['dataset'] = dataset_name
        
        encoded_path = os.path.join(output_dir, f'encoded_features_{dataset_name}.csv')
        encoded_df.to_csv(encoded_path, index=False)
        
        # 保存训练历史
        for history_type, history_data in self.training_history.items():
            if history_data:
                history_df = pd.DataFrame(history_data)
                history_path = os.path.join(output_dir, f'{history_type}_{dataset_name}.csv')
                history_df.to_csv(history_path, index=False)
        
        # 保存运行统计
        stats = {
            'dataset_name': dataset_name,
            'model_type': 'ADCAE',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': training_time,
            'total_samples': len(encoded_features),
            'input_shape': [1, 32, 32],
            'encoded_features': encoded_features.shape[1],
            'device_used': str(self.device),
            'category_counts': category_counts,
            'total_epochs': len(self.training_history['loss']),
            'avg_epoch_time': np.mean([t['time_seconds'] for t in self.training_history['epoch_times']]) if self.training_history['epoch_times'] else 0,
            'final_metrics': self.training_history['five_metrics'][-1] if self.training_history['five_metrics'] else None
        }
        
        stats_path = os.path.join(output_dir, f'processing_stats_{dataset_name}.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{dataset_name}数据集结果已保存到: {output_dir}")
        print(f"- ADCAE模型文件: {model_path}")
        print(f"- 编码特征: {encoded_path}")
        print(f"- 处理统计: {stats_path}")
        
        return stats

def process_train_test_separately(dataset_name, train_dir, test_dir, base_output_dir, 
                                 encoding_dim=128, epochs=100, batch_size=32, 
                                 learning_rate=0.001, max_files_per_category=None, 
                                 file_length=1024):
    """分别处理训练集和测试集（训练集训练模型，测试集仅特征提取）"""
    print("\n" + "=" * 100)
    print(f"开始使用ADCAE分别处理 {dataset_name} 数据集的训练集和测试集")
    print("=" * 100)
    
    # 检查目录是否存在
    if not os.path.exists(train_dir):
        print(f"训练集目录不存在: {train_dir}")
        return None
    if not os.path.exists(test_dir):
        print(f"测试集目录不存在: {test_dir}")
        return None
    
    # 创建数据集专用输出目录
    dataset_output_dir = os.path.join(base_output_dir, f"ADCAE_{dataset_name}")
    
    # 记录开始时间
    dataset_start_time = time.time()
    
    try:
        # 创建ADCAE处理器
        adcae_processor = ADCAEProcessor(encoding_dim=encoding_dim, file_length=file_length)
        
        # ========== 第一步：加载训练集数据 ==========
        print("\n" + "-" * 50)
        print("步骤1: 加载训练集数据")
        print("-" * 50)
        train_X, train_labels, train_category_counts = adcae_processor.prepare_data(
            train_dir, max_files_per_category)
        print(f"训练集加载完成: {len(train_X)} 个样本")
        
        # ========== 第二步：构建并训练模型 ==========
        print("\n" + "-" * 50)
        print("步骤2: 构建并训练ADCAE模型（仅使用训练集）")
        print("-" * 50)
        adcae_processor.build_model()
        training_time = adcae_processor.train_model(
            train_X, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        
        # ========== 第三步：提取训练集特征 ==========
        print("\n" + "-" * 50)
        print("步骤3: 提取训练集特征")
        print("-" * 50)
        train_encoding_start_time = time.time()
        train_encoded_features = adcae_processor.encode_data(train_X)
        train_encoding_time = time.time() - train_encoding_start_time
        print(f"训练集特征提取完成，耗时: {train_encoding_time:.2f}秒")
        
        # ========== 第四步：加载测试集数据 ==========
        print("\n" + "-" * 50)
        print("步骤4: 加载测试集数据")
        print("-" * 50)
        test_X, test_labels, test_category_counts = adcae_processor.prepare_data(
            test_dir, max_files_per_category)
        print(f"测试集加载完成: {len(test_X)} 个样本")
        
        # ========== 第五步：提取测试集特征 ==========
        print("\n" + "-" * 50)
        print("步骤5: 提取测试集特征（使用训练好的模型）")
        print("-" * 50)
        test_encoding_start_time = time.time()
        test_encoded_features = adcae_processor.encode_data(test_X)
        test_encoding_time = time.time() - test_encoding_start_time
        print(f"测试集特征提取完成，耗时: {test_encoding_time:.2f}秒")
        
        # ========== 第六步：保存结果 ==========
        print("\n" + "-" * 50)
        print("步骤6: 保存所有结果")
        print("-" * 50)
        
        # 保存训练集结果
        train_stats = adcae_processor.save_features_and_model(
            dataset_output_dir, train_encoded_features, train_labels, 
            train_category_counts, training_time, f"{dataset_name}_train", "train")
        
        # 保存测试集结果
        test_stats = adcae_processor.save_features_only(
            dataset_output_dir, test_encoded_features, test_labels, 
            test_category_counts, f"{dataset_name}_test", "test")
        
        # 计算总时间
        total_time = time.time() - dataset_start_time
        
        # 输出总结信息
        print("\n" + "=" * 100)
        print(f"{dataset_name} 数据集ADCAE处理完成！")
        print("=" * 100)
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"模型训练时间: {training_time:.2f}秒")
        print(f"训练集特征提取时间: {train_encoding_time:.2f}秒")
        print(f"测试集特征提取时间: {test_encoding_time:.2f}秒")
        print(f"")
        print(f"数据统计:")
        print(f"  训练集样本数: {len(train_encoded_features)}")
        print(f"  测试集样本数: {len(test_encoded_features)}")
        print(f"  原始图像形状: {train_X.shape[1:]}")
        print(f"  编码后特征维度: {train_encoded_features.shape[1]}")
        print(f"  压缩比: {train_encoded_features.shape[1]/(32*32):.2%}")
        print("=" * 100)
        
        return {
            'train_stats': train_stats,
            'test_stats': test_stats,
            'total_time': total_time,
            'training_time': training_time,
            'train_encoding_time': train_encoding_time,
            'test_encoding_time': test_encoding_time
        }
        
    except Exception as e:
        print(f"处理 {dataset_name} 数据集时出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 添加新的保存方法
def save_features_and_model(self, output_dir, encoded_features, labels, category_counts, 
                           training_time, dataset_name, data_type):
    """保存特征、模型和训练历史"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型（仅在训练集时保存）
    if data_type == "train":
        model_path = os.path.join(output_dir, f'adcae_model_{dataset_name.split("_")[0]}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'encoding_dim': self.encoding_dim,
            'file_length': self.file_length,
            'dataset_name': dataset_name,
            'model_type': 'ADCAE'
        }, model_path)
        
        # 保存预处理器
        preprocessor_path = os.path.join(output_dir, f'preprocessor_{dataset_name.split("_")[0]}.pkl')
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
    
    # 保存编码后的特征
    encoded_df = pd.DataFrame(encoded_features, 
                            columns=[f'encoded_feature_{i}' for i in range(encoded_features.shape[1])])
    encoded_df['label'] = labels
    encoded_df['dataset'] = dataset_name
    encoded_df['data_type'] = data_type
    
    encoded_path = os.path.join(output_dir, f'encoded_features_{dataset_name}.csv')
    encoded_df.to_csv(encoded_path, index=False)
    
    # 保存训练历史（仅在训练集时保存）
    if data_type == "train":
        for history_type, history_data in self.training_history.items():
            if history_data:
                history_df = pd.DataFrame(history_data)
                history_path = os.path.join(output_dir, f'{history_type}_{dataset_name.split("_")[0]}.csv')
                history_df.to_csv(history_path, index=False)
    
    # 保存统计信息
    stats = {
        'dataset_name': dataset_name,
        'data_type': data_type,
        'model_type': 'ADCAE',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(encoded_features),
        'input_shape': [1, 32, 32],
        'encoded_features': encoded_features.shape[1],
        'device_used': str(self.device),
        'category_counts': category_counts
    }
    
    if data_type == "train":
        stats.update({
            'training_time_seconds': training_time,
            'total_epochs': len(self.training_history['loss']),
            'avg_epoch_time': np.mean([t['time_seconds'] for t in self.training_history['epoch_times']]) if self.training_history['epoch_times'] else 0,
            'final_metrics': self.training_history['five_metrics'][-1] if self.training_history['five_metrics'] else None
        })
    
    stats_path = os.path.join(output_dir, f'processing_stats_{dataset_name}.json')
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{dataset_name}({data_type})结果已保存到: {output_dir}")
    if data_type == "train":
        print(f"- ADCAE模型文件: {model_path}")
    print(f"- 编码特征: {encoded_path}")
    print(f"- 处理统计: {stats_path}")
    
    return stats

def save_features_only(self, output_dir, encoded_features, labels, category_counts, dataset_name, data_type):
    """仅保存特征（用于测试集）"""
    return self.save_features_and_model(output_dir, encoded_features, labels, category_counts, 
                                      0, dataset_name, data_type)

# 将这两个方法添加到ADCAEProcessor类中
ADCAEProcessor.save_features_and_model = save_features_and_model
ADCAEProcessor.save_features_only = save_features_only

def main():
    """主函数 - 修改为分别处理训练集和测试集"""
    print("=" * 100)
    print("ADCAE (非对称深度卷积自编码器) 处理系统")
    print("训练集训练模型 + 分别提取训练集和测试集特征")
    print("=" * 100)
    
    # 配置参数
    pcap_base_dir = r"d:\Python Project\ADCAE\pcap_files"
    base_output_dir = r"d:\Python Project\ADCAE\results\adcae_output"
    
    # ADCAE参数
    encoding_dim = 64  # 编码维度
    epochs = 2        # 训练轮数
    batch_size = 64     # 批次大小
    learning_rate = 0.0005  # 学习率
    max_files_per_category = None  # None表示使用所有文件
    file_length = 1024  # 文件长度（32*32=1024）
    
    # 数据集路径
    datasets_config = [
        {
            'name': 'USTC',
            'train_dir': os.path.join(pcap_base_dir, 'Dataset_USTC', 'Train'),
            'test_dir': os.path.join(pcap_base_dir, 'Dataset_USTC', 'Test')
        },
        {
            'name': 'CTU',
            'train_dir': os.path.join(pcap_base_dir, 'Dataset_CTU', 'Train'),
            'test_dir': os.path.join(pcap_base_dir, 'Dataset_CTU', 'Test')
        }
    ]
    
    # 检查数据集是否存在
    datasets_to_process = []
    
    for dataset_config in datasets_config:
        name = dataset_config['name']
        train_dir = dataset_config['train_dir']
        test_dir = dataset_config['test_dir']
        
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            datasets_to_process.append(dataset_config)
            print(f"找到{name}数据集: 训练集={train_dir}, 测试集={test_dir}")
        else:
            print(f"{name}数据集不完整:")
            print(f"  训练集存在: {os.path.exists(train_dir)}")
            print(f"  测试集存在: {os.path.exists(test_dir)}")
    
    if not datasets_to_process:
        print("没有找到完整的数据集，请确保训练集和测试集都存在")
        return
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 存储所有结果
    all_results = {}
    
    # 处理每个数据集
    for dataset_config in datasets_to_process:
        result = process_train_test_separately(
            dataset_name=dataset_config['name'],
            train_dir=dataset_config['train_dir'],
            test_dir=dataset_config['test_dir'],
            base_output_dir=base_output_dir,
            encoding_dim=encoding_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_files_per_category=max_files_per_category,
            file_length=file_length
        )
        if result:
            all_results[dataset_config['name']] = result
    
    # 计算总时间
    total_time = time.time() - total_start_time
    
    # 保存综合统计
    summary_stats = {
        'total_processing_time_seconds': total_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'ADCAE',
        'processing_mode': 'train_test_separate',
        'datasets_processed': list(all_results.keys()),
        'individual_results': all_results,
        'configuration': {
            'encoding_dim': encoding_dim,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'file_length': file_length
        }
    }
    
    summary_path = os.path.join(base_output_dir, 'adcae_train_test_summary.json')
    os.makedirs(base_output_dir, exist_ok=True)
    
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print("\n" + "=" * 100)
    print("ADCAE训练集-测试集分离处理完成！")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"处理的数据集: {list(all_results.keys())}")
    print(f"综合统计已保存到: {summary_path}")
    print("\n生成的文件结构:")
    for dataset_name in all_results.keys():
        print(f"  ADCAE_{dataset_name}/")
        print(f"    ├── adcae_model_{dataset_name}.pth          # 训练好的模型")
        print(f"    ├── preprocessor_{dataset_name}.pkl        # 预处理器")
        print(f"    ├── encoded_features_{dataset_name}_train.csv  # 训练集特征")
        print(f"    ├── encoded_features_{dataset_name}_test.csv   # 测试集特征")
        print(f"    └── training_history_*.csv              # 训练历史")
    print("=" * 100)

if __name__ == "__main__":
    main()