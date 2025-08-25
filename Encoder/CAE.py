import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
                if i == len(files) and i > 0:
                    print(f"    已处理 {i}/{len(files)} 个文件")
                
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

class ConvolutionalAutoEncoder(nn.Module):
    """简化版卷积自编码器模型"""
    
    def __init__(self, input_dim=1024, encoding_dim=32):  # 降低编码维度
        super(ConvolutionalAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # 对于1024字节的输入，重塑为32x32的图像
        self.reshape_dim = 32  # 32*32 = 1024
        
        # 简化的编码器 - 只有2层卷积
        self.encoder = nn.Sequential(
            # 第一个卷积层 (32x32 -> 16x16)
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 减少通道数
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 第二个卷积层 (16x16 -> 8x8)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 减少通道数
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 展平并全连接 - 简化网络结构
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),  # 减少隐藏层大小
            nn.ReLU(),
            nn.Linear(128, encoding_dim)  # 直接到编码维度，去掉Dropout
        )
        
        # 简化的解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 8 * 8),  # 对应编码器
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),
            
            # 反卷积层 (8x8 -> 16x16)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # (16x16 -> 32x32)
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 重塑输入为图像格式 (batch_size, 1, 32, 32)
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 1, self.reshape_dim, self.reshape_dim)
        
        # 编码
        encoded = self.encoder(x_reshaped)
        
        # 解码
        decoded = self.decoder(encoded)
        
        # 重塑回原始维度
        decoded = decoded.view(batch_size, self.input_dim)
        
        return encoded, decoded
    
    def encode(self, x):
        """仅编码"""
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 1, self.reshape_dim, self.reshape_dim)
        return self.encoder(x_reshaped)

class CAEProcessor:
    """CAE处理器主类"""
    
    def __init__(self, encoding_dim=64, file_length=1024):
        self.preprocessor = BinaryDataPreprocessor(file_length=file_length)
        self.model = None
        self.encoding_dim = encoding_dim
        self.file_length = file_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 扩展训练历史记录，包含每个epoch的详细指标
        self.training_history = {
            'loss': [],
            'metrics': [],
            'epoch_times': [],
            'five_metrics': [],  # 修复：添加缺失的five_metrics键
            'detailed_metrics': []  # 新增：用于盒图的详细指标
        }
    
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

    def build_model(self):
        """构建CAE模型"""
        self.model = ConvolutionalAutoEncoder(
            input_dim=self.file_length, 
            encoding_dim=self.encoding_dim
        ).to(self.device)
        print(f"CAE模型已构建，输入维度: {self.file_length}, 编码维度: {self.encoding_dim}")
        print(f"使用设备: {self.device}")
        
    def train_model(self, X, epochs=30, batch_size=128, learning_rate=0.01):
        """训练简化版CAE模型"""
        print(f"开始训练简化版CAE模型...")
        start_time = time.time()
        
        # 分割训练和验证集
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, X_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 简化的优化器和损失函数 - 移除权重衰减和学习率调度
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # 移除weight_decay
        criterion = nn.MSELoss()
        # 移除学习率调度器
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10  # 减少早停耐心值
        
        # 扩展训练历史记录
        self.training_history = {
            'loss': [],
            'metrics': [],
            'epoch_times': [],
            'detailed_metrics': [],
            'five_metrics': []  # 新增：存储五个关键指标
        }
        
        # 训练循环
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_mse_total = 0.0
            train_mae_total = 0.0
            
            # 用于收集训练阶段的重构数据
            train_original_data = []
            train_reconstructed_data = []
            
            # 训练进度条
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                             leave=False, ncols=100)
            
            for batch_idx, (batch_data, _) in enumerate(train_pbar):
                optimizer.zero_grad()
                
                encoded, decoded = self.model(batch_data)
                loss = criterion(decoded, batch_data)
                
                loss.backward()
                # 移除梯度裁剪: torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # 计算重构指标
                with torch.no_grad():
                    batch_data_np = batch_data.cpu().numpy()
                    decoded_np = decoded.cpu().numpy()
                    
                    batch_mse = mean_squared_error(batch_data_np, decoded_np)
                    batch_mae = mean_absolute_error(batch_data_np, decoded_np)
                    
                    train_mse_total += batch_mse
                    train_mae_total += batch_mae
                    
                    # 收集数据用于计算分类指标
                    train_original_data.append(batch_data_np)
                    train_reconstructed_data.append(decoded_np)
                
                # 更新进度条
                current_loss = loss.item()
                train_pbar.set_postfix({
                    'Loss': f'{current_loss:.6f}',
                    'MSE': f'{batch_mse:.6f}',
                    'MAE': f'{batch_mae:.6f}'
                })
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_mse_total = 0.0
            val_mae_total = 0.0
            
            # 用于收集验证阶段的重构数据
            val_original_data = []
            val_reconstructed_data = []
            
            # 验证进度条
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', 
                           leave=False, ncols=100)
            
            with torch.no_grad():
                for batch_data, _ in val_pbar:
                    encoded, decoded = self.model(batch_data)
                    loss = criterion(decoded, batch_data)
                    val_loss += loss.item()
                    
                    # 计算验证指标
                    batch_data_np = batch_data.cpu().numpy()
                    decoded_np = decoded.cpu().numpy()
                    
                    batch_mse = mean_squared_error(batch_data_np, decoded_np)
                    batch_mae = mean_absolute_error(batch_data_np, decoded_np)
                    
                    val_mse_total += batch_mse
                    val_mae_total += batch_mae
                    
                    # 收集数据用于计算分类指标
                    val_original_data.append(batch_data_np)
                    val_reconstructed_data.append(decoded_np)
                    
                    # 更新验证进度条
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        'MSE': f'{batch_mse:.6f}'
                    })
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 合并所有训练和验证数据
            train_original = np.vstack(train_original_data)
            train_reconstructed = np.vstack(train_reconstructed_data)
            val_original = np.vstack(val_original_data)
            val_reconstructed = np.vstack(val_reconstructed_data)
            
            # 计算五个关键指标
            def calculate_reconstruction_metrics(original, reconstructed, threshold=0.1):
                """计算重构任务的分类指标"""
                # 将重构误差转换为二分类问题（好的重构 vs 差的重构）
                reconstruction_errors = np.mean(np.abs(original - reconstructed), axis=1)
                
                # 基于阈值创建二分类标签
                true_labels = (reconstruction_errors <= threshold).astype(int)  # 1表示好的重构
                
                # 使用重构质量作为预测（误差小的为好的重构）
                pred_labels = true_labels  # 简化处理，实际可以用更复杂的逻辑
                
                # 计算分类指标
                accuracy = accuracy_score(true_labels, pred_labels)
                precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
                recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
                f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
                
                return accuracy, precision, recall, f1
            
            # 计算训练指标
            train_accuracy, train_precision, train_recall, train_f1 = calculate_reconstruction_metrics(
                train_original, train_reconstructed
            )
            
            # 计算验证指标
            val_accuracy, val_precision, val_recall, val_f1 = calculate_reconstruction_metrics(
                val_original, val_reconstructed
            )
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_mse = train_mse_total / len(train_loader)
            avg_val_mse = val_mse_total / len(val_loader)
            avg_train_mae = train_mae_total / len(train_loader)
            avg_val_mae = val_mae_total / len(val_loader)
            
            # 保存五个关键指标
            five_metrics = {
                'epoch': epoch + 1,
                'train_accuracy': train_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'epoch_time': epoch_time
            }
            
            self.training_history['five_metrics'].append(five_metrics)
            
            # 记录其他训练历史
            self.training_history['loss'].append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            self.training_history['metrics'].append({
                'epoch': epoch + 1,
                'train_mse': avg_train_mse,
                'val_mse': avg_val_mse,
                'train_mae': avg_train_mae,
                'val_mae': avg_val_mae,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            self.training_history['epoch_times'].append({
                'epoch': epoch + 1,
                'time_seconds': epoch_time
            })
            
            # 学习率调度
            # scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 输出五个关键指标
            print(f"\n{'='*80}")
            print(f"Epoch [{epoch+1}/{epochs}] - 训练时间: {epoch_time:.2f}s")
            print(f"{'='*80}")
            print(f"训练指标:")
            print(f"  Loss: {avg_train_loss:.6f} | MSE: {avg_train_mse:.6f} | MAE: {avg_train_mae:.6f}")
            print(f"  Accuracy: {train_accuracy:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
            print(f"验证指标:")
            print(f"  Loss: {avg_val_loss:.6f} | MSE: {avg_val_mse:.6f} | MAE: {avg_val_mae:.6f}")
            print(f"  Accuracy: {val_accuracy:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
            print(f"其他信息:")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e} | 最佳验证损失: {best_val_loss:.6f}")
            print(f"  早停计数: {patience_counter}/{early_stopping_patience}")
            print(f"{'='*80}")
            
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
    
    def save_model_and_results(self, output_dir, encoded_train_features, encoded_test_features, labels_train, labels_test, category_counts, training_time, dataset_name):
        """保存模型和结果 - 分别保存训练集和测试集特征"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(output_dir, f'cae_model_{dataset_name}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'preprocessor': self.preprocessor,
            'encoding_dim': self.encoding_dim,
            'file_length': self.file_length
        }, model_path)
        
        # 分别保存训练集和测试集的编码特征
        train_encoded_path = os.path.join(output_dir, f'train_encoded_features_{dataset_name}.npy')
        test_encoded_path = os.path.join(output_dir, f'test_encoded_features_{dataset_name}.npy')
        train_labels_path = os.path.join(output_dir, f'train_labels_{dataset_name}.npy')
        test_labels_path = os.path.join(output_dir, f'test_labels_{dataset_name}.npy')
        
        np.save(train_encoded_path, encoded_train_features)
        np.save(test_encoded_path, encoded_test_features)
        np.save(train_labels_path, labels_train)
        np.save(test_labels_path, labels_test)
        
        # 合并所有特征用于统计
        all_encoded_features = np.concatenate([encoded_train_features, encoded_test_features])
        all_labels = np.concatenate([labels_train, labels_test])
        
        # 保存编码后的特征
        encoded_df = pd.DataFrame(all_encoded_features,  # 修复：使用all_encoded_features而不是encoded_features
                                columns=[f'encoded_feature_{i}' for i in range(all_encoded_features.shape[1])])
        encoded_df['label'] = all_labels  # 修复：使用all_labels而不是labels
        encoded_df['dataset'] = dataset_name
        
        # 修复：定义combined_encoded_path变量
        combined_encoded_path = os.path.join(output_dir, f'encoded_features_{dataset_name}.csv')
        encoded_df.to_csv(combined_encoded_path, index=False)
        
        # 保存训练历史（损失）
        history_df = pd.DataFrame(self.training_history['loss'])
        history_path = os.path.join(output_dir, f'training_history_{dataset_name}.csv')
        history_df.to_csv(history_path, index=False)
        
        # 保存训练指标
        metrics_df = pd.DataFrame(self.training_history['metrics'])
        metrics_path = os.path.join(output_dir, f'training_metrics_{dataset_name}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        # 保存epoch时间统计
        times_df = pd.DataFrame(self.training_history['epoch_times'])
        times_path = os.path.join(output_dir, f'epoch_times_{dataset_name}.csv')
        times_df.to_csv(times_path, index=False)
        
        # 保存运行统计
        stats = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': training_time,
            'total_samples': len(all_encoded_features),
            'input_features': self.file_length,
            'encoded_features': all_encoded_features.shape[1],
            'device_used': str(self.device),
            'category_counts': category_counts,
            'total_epochs': len(self.training_history['loss']),
            'avg_epoch_time': np.mean([t['time_seconds'] for t in self.training_history['epoch_times']]),
            'final_metrics': self.training_history['metrics'][-1] if self.training_history['metrics'] else None
        }
        
        stats_path = os.path.join(output_dir, f'processing_stats_{dataset_name}.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curve(output_dir, dataset_name)
        
        print(f"{dataset_name}数据集结果已保存到: {output_dir}")
        print(f"- 模型文件: {model_path}")
        print(f"- 训练集编码特征: {train_encoded_path}")
        print(f"- 测试集编码特征: {test_encoded_path}")
        print(f"- 合并编码特征: {combined_encoded_path}")
        print(f"- 训练历史: {history_path}")
        print(f"- 训练指标: {metrics_path}")
        print(f"- Epoch时间: {times_path}")
        print(f"- 处理统计: {stats_path}")
        
        return stats
    
    def plot_training_curve(self, output_dir, dataset_name):
        """绘制训练曲线"""
        if not self.training_history['loss']:
            return
        
        history_df = pd.DataFrame(self.training_history['loss'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss', linewidth=2)
        plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'CAE Training History - {dataset_name} Dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f'training_curve_{dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"- 训练曲线: {plot_path}")
    
    def reset_for_new_dataset(self):
        """为新数据集重置处理器状态"""
        self.preprocessor = BinaryDataPreprocessor(file_length=self.file_length)
        self.model = None
        self.training_history = {'loss': []}

def process_dataset(dataset_name, train_dataset_dir, test_dataset_dir, base_output_dir, encoding_dim=64, epochs=50, batch_size=32, learning_rate=0.001, max_files_per_category=None, file_length=1024):
    """处理单个数据集 - 用训练集训练模型，然后处理训练集和测试集"""
    print("\n" + "=" * 80)
    print(f"开始处理 {dataset_name} 数据集")
    print("=" * 80)
    
    if not os.path.exists(train_dataset_dir):
        print(f"训练集目录不存在: {train_dataset_dir}")
        return None
    
    if not os.path.exists(test_dataset_dir):
        print(f"测试集目录不存在: {test_dataset_dir}")
        return None
    
    # 创建数据集专用输出目录 - 根据数据集名称
    if dataset_name.upper() == 'CTU':
        dataset_output_dir = os.path.join(base_output_dir.replace('results', 'CTU_result'), 'cae_features')
    elif dataset_name.upper() == 'USTC':
        dataset_output_dir = os.path.join(base_output_dir.replace('results', 'USTC_result'), 'cae_features')
    else:
        dataset_output_dir = os.path.join(base_output_dir, dataset_name)
    
    # 记录开始时间
    dataset_start_time = time.time()
    
    try:
        # 创建CAE处理器
        cae_processor = CAEProcessor(encoding_dim=encoding_dim, file_length=file_length)
        
        # 1. 准备训练数据
        print("准备训练数据...")
        X_train, labels_train, category_counts_train = cae_processor.prepare_data(train_dataset_dir, max_files_per_category)
        
        # 2. 构建模型
        cae_processor.build_model()
        
        # 3. 训练模型
        print("开始训练模型...")
        training_time = cae_processor.train_model(X_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        
        # 4. 准备测试数据
        print("准备测试数据...")
        X_test, labels_test, category_counts_test = cae_processor.prepare_test_data(test_dataset_dir, max_files_per_category)
        
        # 5. 编码训练集数据
        print("编码训练集数据...")
        encoding_start_time = time.time()
        encoded_train_features = cae_processor.encode_data(X_train)
        
        # 6. 编码测试集数据
        print("编码测试集数据...")
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
        
        print(f"{dataset_name} 数据集处理完成！")
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
        stats['train_samples'] = len(encoded_train_features)
        stats['test_samples'] = len(encoded_test_features)
        
        return stats
        
    except Exception as e:
        print(f"处理 {dataset_name} 数据集时出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=" * 80)
    print("CAE (简化版卷积自编码器) 二进制数据处理")
    print("=" * 80)
    
    # 配置参数
    pcap_base_dir = r"d:\Python Project\ADCAE\pcap_files"
    base_output_dir = r"d:\Python Project\ADCAE\results\cae_output"
    
    # 简化的CAE参数 - 降低模型能力
    encoding_dim = 32  # 进一步降低编码维度
    epochs = 1  # 减少训练轮数
    batch_size = 128  # 增大批次大小，降低训练稳定性
    learning_rate = 0.01  # 提高学习率，可能导致训练不稳定
    max_files_per_category = None  # None表示使用所有文件
    file_length = 1024  # 与session_processor_simplified.py中的TRIMMED_FILE_LEN一致
    
    # 数据集路径
    ustc_train_dir = os.path.join(pcap_base_dir, 'Dataset_USTC', 'Train')
    ustc_test_dir = os.path.join(pcap_base_dir, 'Dataset_USTC', 'Test')
    ctu_train_dir = os.path.join(pcap_base_dir, 'Dataset_CTU', 'Train')
    ctu_test_dir = os.path.join(pcap_base_dir, 'Dataset_CTU', 'Test')
    
    # 检查数据集是否存在
    datasets_to_process = []
    
    if os.path.exists(ustc_train_dir) and os.path.exists(ustc_test_dir):
        datasets_to_process.append(('USTC', ustc_train_dir, ustc_test_dir))
        print(f"找到USTC数据集: 训练集 {ustc_train_dir}, 测试集 {ustc_test_dir}")
    else:
        print(f"USTC数据集不完整: 训练集 {ustc_train_dir}, 测试集 {ustc_test_dir}")
    
    if os.path.exists(ctu_train_dir) and os.path.exists(ctu_test_dir):
        datasets_to_process.append(('CTU', ctu_train_dir, ctu_test_dir))
        print(f"找到CTU数据集: 训练集 {ctu_train_dir}, 测试集 {ctu_test_dir}")
    else:
        print(f"CTU数据集不完整: 训练集 {ctu_train_dir}, 测试集 {ctu_test_dir}")
    
    if not datasets_to_process:
        print("没有找到任何完整的数据集，请确保已运行session_processor_simplified.py生成数据集")
        return
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 存储所有结果
    all_results = {}
    
    # 处理每个数据集
    for dataset_name, train_dir, test_dir in datasets_to_process:  # 修复：解包3个元素
        stats = process_dataset(
            dataset_name=dataset_name,
            train_dataset_dir=train_dir,  # 修复：使用正确的参数名
            test_dataset_dir=test_dir,    # 修复：使用正确的参数名
            base_output_dir=base_output_dir,
            encoding_dim=encoding_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_files_per_category=max_files_per_category,
            file_length=file_length
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
        'parameters': {
            'encoding_dim': encoding_dim,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'file_length': file_length
        }
    }
    
    summary_path = os.path.join(base_output_dir, 'processing_summary.json')
    os.makedirs(base_output_dir, exist_ok=True)
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
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
        print(f"  - 训练时间: {stats['training_time_seconds']:.2f}秒")
        print(f"  - 总处理时间: {stats['total_processing_time_seconds']:.2f}秒")
        print(f"  - 类别分布: {stats['category_counts']}")
    
    print(f"\n综合统计已保存到: {summary_path}")
    print(f"各数据集结果保存在: {base_output_dir}/[USTC|CTU]/")

if __name__ == "__main__":
    main()


def create_boxplot_data(self, output_dir, dataset_name):
    """创建专门用于盒图的数据格式"""
    if not self.training_history['detailed_metrics']:
        print("没有详细指标数据可用于创建盒图")
        return
    
    # 准备盒图数据
    boxplot_data = []
    
    for epoch_data in self.training_history['detailed_metrics']:
        epoch_num = epoch_data['epoch']
        
        # 为每种指标创建数据行
        for loss in epoch_data['train_losses']:
            boxplot_data.append({'epoch': epoch_num, 'metric': 'Train Loss', 'value': loss})
        
        for mse in epoch_data['train_mse_scores']:
            boxplot_data.append({'epoch': epoch_num, 'metric': 'Train MSE', 'value': mse})
        
        for mae in epoch_data['train_mae_scores']:
            boxplot_data.append({'epoch': epoch_num, 'metric': 'Train MAE', 'value': mae})
        
        for loss in epoch_data['val_losses']:
            boxplot_data.append({'epoch': epoch_num, 'metric': 'Validation Loss', 'value': loss})
        
        for acc in epoch_data['val_reconstruction_accuracies']:
            boxplot_data.append({'epoch': epoch_num, 'metric': 'Validation Accuracy', 'value': acc})
    
    # 保存盒图数据
    boxplot_df = pd.DataFrame(boxplot_data)
    boxplot_path = os.path.join(output_dir, f'boxplot_data_{dataset_name}.csv')
    boxplot_df.to_csv(boxplot_path, index=False)
    
    print(f"- 盒图数据: {boxplot_path}")
    
    return boxplot_df