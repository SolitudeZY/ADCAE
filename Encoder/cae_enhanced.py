import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import pickle
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 超参数配置区域 ====================
class CAEConfig:
    """增强版CAE超参数配置类"""
    
    # 模型架构参数
    ENCODING_DIM = 64          # 编码维度
    FILE_LENGTH = 1024          # 输入文件长度
    
    # 训练参数
    EPOCHS = 5                 # 训练轮数
    BATCH_SIZE = 64             # 批次大小
    LEARNING_RATE = 0.001       # 学习率
    WEIGHT_DECAY = 1e-4         # 权重衰减
    
    # 学习率调度参数
    SCHEDULER_T0 = 10           # 余弦退火初始周期
    SCHEDULER_T_MULT = 2        # 周期倍增因子
    
    # 早停参数
    EARLY_STOPPING_PATIENCE = 15  # 早停耐心值
    
    # 损失函数参数
    CONTRASTIVE_TEMPERATURE = 0.5  # 对比学习温度
    CONTRASTIVE_WEIGHT = 0.1       # 对比损失权重
    
    # 正则化参数
    DROPOUT_RATE_1 = 0.3        # 第一个Dropout率
    DROPOUT_RATE_2 = 0.2        # 第二个Dropout率
    GRAD_CLIP_NORM = 1.0        # 梯度裁剪阈值
    
    # 数据处理参数
    VAL_SPLIT = 0.2             # 验证集比例
    MAX_FILES_PER_CATEGORY = None  # 每类别最大文件数（None表示全部）
    
    # 输出参数
    SAVE_MODEL_INTERVAL = 5     # 模型保存间隔
    PRINT_INTERVAL = 5          # 打印间隔

# ==================== 数据预处理类 ====================
class BinaryDataPreprocessor:
    """二进制数据预处理器"""
    
    def __init__(self, file_length=1024):
        self.scaler = StandardScaler()
        self.file_length = file_length
        
    def load_binary_file(self, file_path):
        """加载二进制文件并转换为特征向量"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 确保数据长度一致
            if len(data) < self.file_length:
                # 填充零
                data = data + b'\x00' * (self.file_length - len(data))
            elif len(data) > self.file_length:
                # 截断
                data = data[:self.file_length]
            
            # 转换为numpy数组
            features = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            
            return features
            
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None
    
    def load_dataset_from_directory(self, dataset_dir, max_files_per_category=None):
        """从目录加载数据集"""
        print(f"正在加载数据集: {dataset_dir}")
        
        all_features = []
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
            
            category_features = []
            
            for i, filename in enumerate(files):
                # 删除这两行代码让输出更简洁
                # if i > 0 and i % 100 == 0:
                #     print(f"    已处理 {i}/{len(files)} 个文件")
                
                file_path = os.path.join(category_path, filename)
                features = self.load_binary_file(file_path)
                
                if features is not None:
                    category_features.append(features)
                    all_labels.append(category_name)
            
            if category_features:
                all_features.extend(category_features)
                category_counts[category_name] = len(category_features)
                print(f"    {category_name}: {len(category_features)} 个文件")
        
        if not all_features:
            raise ValueError(f"在 {dataset_dir} 中没有找到有效的数据文件")
        
        # 转换为numpy数组
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"数据加载完成:")
        print(f"  总样本数: {len(X)}")
        print(f"  特征维度: {X.shape[1]}")
        print(f"  类别分布: {category_counts}")
        
        return X, y, category_counts
    
    def normalize_features(self, X, fit_scaler=True):
        """标准化特征"""
        if fit_scaler:
            normalized_data = self.scaler.fit_transform(X)
        else:
            normalized_data = self.scaler.transform(X)
        
        return normalized_data

# ==================== 网络模块 ====================
class AttentionBlock(nn.Module):
    """注意力机制模块"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EnhancedConvolutionalAutoEncoder(nn.Module):
    """增强版卷积自编码器"""
    
    def __init__(self, config):
        super(EnhancedConvolutionalAutoEncoder, self).__init__()
        
        self.input_dim = config.FILE_LENGTH
        self.encoding_dim = config.ENCODING_DIM
        self.reshape_dim = 32  # 32*32 = 1024
        
        # 深层编码器
        self.encoder = nn.Sequential(
            # 第一层：32x32 -> 16x16
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            AttentionBlock(64),
            nn.MaxPool2d(2, 2),
            
            # 第二层：16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            AttentionBlock(128),
            nn.MaxPool2d(2, 2),
            
            # 第三层：8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            AttentionBlock(256),
            nn.MaxPool2d(2, 2),
            
            # 第四层：4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2d(2),
            
            # 全连接层
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE_1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE_2),
            nn.Linear(512, self.encoding_dim)
        )
        
        # 深层解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE_2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE_1),
            nn.Linear(1024, 512 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (512, 2, 2)),
            
            # 反卷积层：2x2 -> 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 1, self.reshape_dim, self.reshape_dim)
        
        encoded = self.encoder(x_reshaped)
        decoded = self.decoder(encoded)
        
        decoded = decoded.view(batch_size, self.input_dim)
        return encoded, decoded
    
    def encode(self, x):
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 1, self.reshape_dim, self.reshape_dim)
        return self.encoder(x_reshaped)

class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features):
        batch_size = features.size(0)
        
        # 计算相似度矩阵
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # 创建正样本掩码（对角线）
        mask = torch.eye(batch_size, device=features.device).bool()
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        exp_sim_masked = exp_sim.masked_fill(mask, 0)
        
        positive_sim = torch.diag(exp_sim)
        negative_sim = exp_sim_masked.sum(dim=1)
        
        loss = -torch.log(positive_sim / (positive_sim + negative_sim + 1e-8))
        return loss.mean()

# ==================== 增强版CAE处理器 ====================
class EnhancedCAEProcessor:
    """增强版CAE处理器"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = BinaryDataPreprocessor(file_length=config.FILE_LENGTH)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {
            'loss': [],
            'metrics': [],
            'epoch_times': [],
            'five_metrics': [],
            'detailed_metrics': []
        }
    
    def build_model(self):
        """构建增强版CAE模型"""
        self.model = EnhancedConvolutionalAutoEncoder(self.config).to(self.device)
        print(f"增强版CAE模型已构建，输入维度: {self.config.FILE_LENGTH}, 编码维度: {self.config.ENCODING_DIM}")
        print(f"使用设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, dataset_dir, max_files_per_category=None):
        """准备数据"""
        print("开始数据预处理...")
        start_time = time.time()
        
        # 加载数据
        X, y, category_counts = self.preprocessor.load_dataset_from_directory(
            dataset_dir, max_files_per_category
        )
        
        # 标准化
        print("正在标准化数据...")
        X_normalized = self.preprocessor.normalize_features(X, fit_scaler=True)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        preprocessing_time = time.time() - start_time
        print(f"数据预处理完成，耗时: {preprocessing_time:.2f}秒")
        print(f"总样本数: {X_tensor.shape[0]}, 特征维度: {X_tensor.shape[1]}")
        
        return X_tensor, y, category_counts
    
    def prepare_test_data(self, dataset_dir, max_files_per_category=None):
        """准备测试数据（使用已训练的预处理器）"""
        print("开始测试数据预处理...")
        start_time = time.time()
        
        # 加载数据
        X, y, category_counts = self.preprocessor.load_dataset_from_directory(
            dataset_dir, max_files_per_category
        )
        
        # 使用已训练的预处理器进行标准化（不重新拟合）
        print("正在标准化测试数据...")
        X_normalized = self.preprocessor.normalize_features(X, fit_scaler=False)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        preprocessing_time = time.time() - start_time
        print(f"测试数据预处理完成，耗时: {preprocessing_time:.2f}秒")
        print(f"总样本数: {X_tensor.shape[0]}, 特征维度: {X_tensor.shape[1]}")
        
        return X_tensor, y, category_counts
    
    def train_model(self, X):
        """训练增强版CAE模型"""
        print(f"开始训练增强版CAE模型...")
        start_time = time.time()
        
        # 分割训练和验证集
        X_train, X_val = train_test_split(X, test_size=self.config.VAL_SPLIT, random_state=42)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, X_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=0
        )
        
        # 优化器和损失函数
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.config.SCHEDULER_T0, 
            T_mult=self.config.SCHEDULER_T_MULT
        )
        
        reconstruction_criterion = nn.MSELoss()
        contrastive_criterion = ContrastiveLoss(temperature=self.config.CONTRASTIVE_TEMPERATURE)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 训练循环
        for epoch in range(self.config.EPOCHS):
            epoch_start_time = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_contrast_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS} [Train]', leave=False)
            
            for batch_data, _ in train_pbar:
                optimizer.zero_grad()
                
                encoded, decoded = self.model(batch_data)
                
                # 重构损失
                recon_loss = reconstruction_criterion(decoded, batch_data)
                
                # 对比学习损失
                contrast_loss = contrastive_criterion(encoded)
                
                # 总损失
                total_loss = recon_loss + self.config.CONTRASTIVE_WEIGHT * contrast_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRAD_CLIP_NORM)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_contrast_loss += contrast_loss.item()
                
                train_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.6f}',
                    'Recon': f'{recon_loss.item():.6f}',
                    'Contrast': f'{contrast_loss.item():.6f}'
                })
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            
            with torch.no_grad():
                for batch_data, _ in val_loader:
                    encoded, decoded = self.model(batch_data)
                    recon_loss = reconstruction_criterion(decoded, batch_data)
                    contrast_loss = contrastive_criterion(encoded)
                    total_loss = recon_loss + self.config.CONTRASTIVE_WEIGHT * contrast_loss
                    
                    val_loss += total_loss.item()
                    val_recon_loss += recon_loss.item()
            
            # 学习率调度
            scheduler.step()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_recon = train_recon_loss / len(train_loader)
            avg_val_recon = val_recon_loss / len(val_loader)
            avg_train_contrast = train_contrast_loss / len(train_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录训练历史
            self.training_history['loss'].append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_recon_loss': avg_train_recon,
                'val_recon_loss': avg_val_recon,
                'train_contrast_loss': avg_train_contrast
            })
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 输出训练信息
            if (epoch + 1) % self.config.PRINT_INTERVAL == 0 or epoch == 0:
                print(f"\nEpoch [{epoch+1}/{self.config.EPOCHS}] - 时间: {epoch_time:.2f}s")
                print(f"训练损失: {avg_train_loss:.6f} (重构: {avg_train_recon:.6f}, 对比: {avg_train_contrast:.6f})")
                print(f"验证损失: {avg_val_loss:.6f} (重构: {avg_val_recon:.6f})")
                print(f"学习率: {scheduler.get_last_lr()[0]:.2e}")
                print(f"早停计数: {patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
            
            # 早停
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
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
    
    def save_model_and_results(self, output_dir, train_features, test_features, 
                              train_labels, test_labels, category_counts, 
                              training_time, dataset_name):
        """保存模型和结果"""
        print("保存模型和结果...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存特征数据
        features_dir = os.path.join(output_dir, 'cae_features')
        os.makedirs(features_dir, exist_ok=True)
        
        # 保存训练集特征
        train_df = pd.DataFrame(train_features)
        train_df['label'] = train_labels
        train_df['dataset'] = 'train'
        train_path = os.path.join(features_dir, 'train_features.csv')
        train_df.to_csv(train_path, index=False)
        
        # 保存测试集特征
        test_df = pd.DataFrame(test_features)
        test_df['label'] = test_labels
        test_df['dataset'] = 'test'
        test_path = os.path.join(features_dir, 'test_features.csv')
        test_df.to_csv(test_path, index=False)
        
        # 保存模型
        model_path = os.path.join(output_dir, f'enhanced_cae_model_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'preprocessor_scaler': self.preprocessor.scaler
        }, model_path)
        
        # 保存训练历史
        history_path = os.path.join(output_dir, f'training_history_{dataset_name}.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 创建统计信息
        stats = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(train_features) + len(test_features),
            'train_samples': len(train_features),
            'test_samples': len(test_features),
            'input_features': self.config.FILE_LENGTH,
            'encoded_features': self.config.ENCODING_DIM,
            'compression_ratio': self.config.ENCODING_DIM / self.config.FILE_LENGTH,
            'category_counts': category_counts,
            'training_time_seconds': training_time,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'config': self.config.__dict__
        }
        
        print(f"- 训练集特征: {train_path}")
        print(f"- 测试集特征: {test_path}")
        print(f"- 模型文件: {model_path}")
        print(f"- 训练历史: {history_path}")
        
        return stats

# ==================== 数据集处理函数 ====================
def process_dataset(dataset_name, train_dataset_dir, test_dataset_dir, 
                   base_output_dir, config):
    """处理单个数据集"""
    try:
        print(f"\n{'='*80}")
        print(f"开始处理 {dataset_name} 数据集")
        print(f"{'='*80}")
        
        dataset_start_time = time.time()
        
        # 创建输出目录
        dataset_output_dir = os.path.join(base_output_dir, f"{dataset_name}_result", "cae_output")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 1. 初始化处理器
        cae_processor = EnhancedCAEProcessor(config)
        
        # 2. 构建模型
        cae_processor.build_model()
        
        # 3. 准备训练数据
        print(f"\n加载 {dataset_name} 训练数据...")
        X_train, labels_train, category_counts_train = cae_processor.prepare_data(
            train_dataset_dir, config.MAX_FILES_PER_CATEGORY
        )
        
        # 4. 训练模型
        print(f"\n开始训练 {dataset_name} 模型...")
        training_time = cae_processor.train_model(X_train)
        
        # 5. 准备测试数据
        print(f"\n加载 {dataset_name} 测试数据...")
        X_test, labels_test, category_counts_test = cae_processor.prepare_test_data(
            test_dataset_dir, config.MAX_FILES_PER_CATEGORY
        )
        
        # 6. 编码数据
        print(f"\n编码 {dataset_name} 数据...")
        encoding_start_time = time.time()
        encoded_train_features = cae_processor.encode_data(X_train)
        encoded_test_features = cae_processor.encode_data(X_test)
        encoding_time = time.time() - encoding_start_time
        print(f"数据编码完成，耗时: {encoding_time:.2f}秒")
        
        # 7. 合并标签和类别统计
        all_labels = np.concatenate([labels_train, labels_test])
        combined_category_counts = {}
        for key in set(list(category_counts_train.keys()) + list(category_counts_test.keys())):
            combined_category_counts[key] = category_counts_train.get(key, 0) + category_counts_test.get(key, 0)
        
        # 8. 保存结果
        print("保存结果...")
        stats = cae_processor.save_model_and_results(
            dataset_output_dir, 
            encoded_train_features, 
            encoded_test_features,
            labels_train,
            labels_test,
            combined_category_counts, 
            training_time, 
            dataset_name
        )
        
        # 计算总时间
        total_time = time.time() - dataset_start_time
        
        print(f"\n{dataset_name} 数据集处理完成！")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"训练时间: {training_time:.2f}秒")
        print(f"编码时间: {encoding_time:.2f}秒")
        print(f"原始特征维度: {X_train.shape[1]}")
        print(f"编码后特征维度: {encoded_train_features.shape[1]}")
        print(f"压缩比: {encoded_train_features.shape[1]/X_train.shape[1]:.2%}")
        print(f"训练集样本数: {len(encoded_train_features)}")
        print(f"测试集样本数: {len(encoded_test_features)}")
        print(f"总样本数: {len(encoded_train_features) + len(encoded_test_features)}")
        
        # 添加总时间到统计信息
        stats['total_processing_time_seconds'] = total_time
        stats['encoding_time_seconds'] = encoding_time
        
        return stats
        
    except Exception as e:
        print(f"处理 {dataset_name} 数据集时出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 80)
    print("增强版CAE (Enhanced Convolutional AutoEncoder) 二进制数据处理")
    print("=" * 80)
    
    # 初始化配置
    config = CAEConfig()
    
    # 打印配置信息
    print("\n当前配置:")
    print(f"- 编码维度: {config.ENCODING_DIM}")
    print(f"- 训练轮数: {config.EPOCHS}")
    print(f"- 批次大小: {config.BATCH_SIZE}")
    print(f"- 学习率: {config.LEARNING_RATE}")
    print(f"- 早停耐心值: {config.EARLY_STOPPING_PATIENCE}")
    
    # 配置路径
    pcap_base_dir = r"d:\Python Project\ADCAE\pcap_files"
    
    # 数据集路径
    ustc_train_dir = os.path.join(pcap_base_dir, 'Dataset_USTC', 'Train')
    ustc_test_dir = os.path.join(pcap_base_dir, 'Dataset_USTC', 'Test')
    ctu_train_dir = os.path.join(pcap_base_dir, 'Dataset_CTU', 'Train')
    ctu_test_dir = os.path.join(pcap_base_dir, 'Dataset_CTU', 'Test')
    
    # 检查数据集是否存在
    datasets_to_process = []
    
    if os.path.exists(ustc_train_dir) and os.path.exists(ustc_test_dir):
        datasets_to_process.append(('USTC', ustc_train_dir, ustc_test_dir, 
                                   r"D:\Python Project\ADCAE"))  # 修改这里，去掉USTC_result
        print(f"\n找到USTC数据集: 训练集 {ustc_train_dir}, 测试集 {ustc_test_dir}")
    
    if os.path.exists(ctu_train_dir) and os.path.exists(ctu_test_dir):
        datasets_to_process.append(('CTU', ctu_train_dir, ctu_test_dir, 
                                   r"D:\Python Project\ADCAE"))  # 修改这里，去掉CTU_result
        print(f"找到CTU数据集: 训练集 {ctu_train_dir}, 测试集 {ctu_test_dir}")
    else:
        print(f"CTU数据集不完整: 训练集 {ctu_train_dir}, 测试集 {ctu_test_dir}")
    
    if not datasets_to_process:
        print("没有找到任何完整的数据集，请确保数据集路径正确")
        return
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 存储所有结果
    all_results = {}
    
    # 处理每个数据集
    for dataset_name, train_dir, test_dir, output_dir in datasets_to_process:
        stats = process_dataset(
            dataset_name=dataset_name,
            train_dataset_dir=train_dir,
            test_dataset_dir=test_dir,
            base_output_dir=output_dir,
            config=config
        )
        if stats:
            all_results[dataset_name] = stats
    
    # 计算总时间
    total_time = time.time() - total_start_time
    
    # 保存综合统计
    summary_stats = {
        'total_processing_time_seconds': total_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'datasets_processed': list(all_results.keys()),
        'individual_results': all_results,
        'config': config.__dict__
    }
    
    # 输出最终结果
    print("\n" + "=" * 80)
    print("所有数据集处理完成！")
    print("=" * 80)
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"处理的数据集: {', '.join(all_results.keys())}")
    
    for dataset_name, stats in all_results.items():
        print(f"\n{dataset_name} 数据集:")
        print(f"  - 处理样本数: {stats['total_samples']}")
        print(f"  - 原始特征维度: {stats['input_features']}")
        print(f"  - 编码后特征维度: {stats['encoded_features']}")
        print(f"  - 压缩比: {stats['compression_ratio']:.2%}")
        print(f"  - 训练时间: {stats['training_time_seconds']:.2f}秒")
        print(f"  - 总处理时间: {stats['total_processing_time_seconds']:.2f}秒")
        print(f"  - 类别分布: {stats['category_counts']}")
        print(f"  - 输出目录: {output_dir}/cae_output")

if __name__ == "__main__":
    main()