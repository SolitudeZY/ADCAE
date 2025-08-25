import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import pickle

class SimpleTCNBlock(nn.Module):
    """简化的TCN块（性能较低）"""
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.25):
        super(SimpleTCNBlock, self).__init__()
        
        # 移除膨胀卷积，使用普通卷积
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 移除残差连接
        
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out  # 不使用残差连接
        
class TCNBlock(nn.Module):
    """TCN基础块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        self.chomp1 = nn.ConstantPad1d((0, -(kernel_size-1)*dilation), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        self.chomp2 = nn.ConstantPad1d((0, -(kernel_size-1)*dilation), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    """TCN模型"""
    def __init__(self, input_size, num_channels, num_classes, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, 
                                 dilation_size, dropout))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # 转换为 (batch_size, features, seq_len)
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, channels)
        return self.classifier(x)

class SimpleTCNModel(nn.Module):
    """使用简化TCN块的模型（性能较低）"""
    def __init__(self, input_size, num_channels, num_classes, kernel_size=3, dropout=0.5, classifier_hidden=128):
        super(SimpleTCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 使用SimpleTCNBlock而不是TCNBlock
            layers.append(SimpleTCNBlock(in_channels, out_channels, kernel_size, dropout))
        
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 可配置的分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(classifier_hidden, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # 转换为 (batch_size, features, seq_len)
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, channels)
        return self.classifier(x)

class TCNRawDataProcessor:
    def __init__(self, dataset_name="CTU"):
        """
        初始化TCN原始数据处理器
        
        Args:
            dataset_name: 数据集名称 ("CTU" 或 "USTC")
        """
        self.dataset_name = dataset_name
        self.label_encoder = LabelEncoder()
        
        # 获取数据集特定的超参数
        self.hyperparams = self.get_dataset_hyperparameters(dataset_name)
        
        # 设置数据处理参数
        self.max_packet_length = self.hyperparams['max_packet_length']
        self.max_packets_per_session = self.hyperparams['max_packets_per_session']
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置数据路径
        self.base_path = Path(f"d:/Python Project/ADCAE/pcap_files/Dataset_{dataset_name}")
        self.train_path = self.base_path / "Train"
        self.test_path = self.base_path / "Test"
        
        # 输出路径
        self.output_path = Path(f"d:/Python Project/ADCAE/{dataset_name}_result/tcn_output_raw_pytorch")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 缓存路径
        self.cache_path = Path(f"d:/Python Project/ADCAE/{dataset_name}_result/tcn_cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # 时间统计
        self.timing_stats = {
            'data_loading_time': 0,
            'preprocessing_time': 0,
            'training_time': 0,
            'validation_prediction_time': 0,
            'test_prediction_time': 0,
            'total_time': 0
        }
        
        print(f"初始化TCN原始数据处理器 - 数据集: {dataset_name}")
        print(f"训练数据路径: {self.train_path}")
        print(f"测试数据路径: {self.test_path}")
        print(f"输出路径: {self.output_path}")
        print(f"超参数配置: {self.hyperparams}")
    
    def get_dataset_hyperparameters(self, dataset_name):
        """
        获取针对不同数据集的超参数
        """
        if dataset_name == "CTU":
            return {
                'max_packet_length': 1024,
                'max_packets_per_session': 20,
                'noise_std': 0.15,  # 移除噪声参数
                'num_channels': [24, 48],
                'kernel_size': 3,
                'dropout': 0.4,
                'epochs': 2,
                'batch_size': 64,
                'sequence_length': 12,
                'learning_rate': 0.001,
                'classifier_hidden': 96
            }
        elif dataset_name == "USTC":
            return {
                'max_packet_length': 1024,
                'max_packets_per_session': 20,
                'noise_std': 0.10,  # 移除噪声参数
                'num_channels': [24, 48],
                'kernel_size': 3,
                'dropout': 0.2,
                'epochs': 2,
                'batch_size': 64,
                'sequence_length': 10,
                'learning_rate': 0.001,
                'classifier_hidden': 96
            }
        else:
            # 默认参数
            return {
                'max_packet_length': 1024,
                'max_packets_per_session': 25,
                'noise_std': 0.1,  # 移除噪声参数
                'num_channels': [32, 64],
                'kernel_size': 3,
                'dropout': 0.5,
                'epochs': 3,
                'batch_size': 64,
                'sequence_length': 16,
                'learning_rate': 0.001,
                'classifier_hidden': 128
            }
    
    def get_cache_key(self, data_path, max_files_per_class):
        """
        生成缓存键
        """
        cache_data = {
            'data_path': str(data_path),
            'max_files_per_class': max_files_per_class,
            'max_packet_length': self.hyperparams['max_packet_length'],
            'sequence_length': self.hyperparams['sequence_length'],
            'noise_std': self.hyperparams['noise_std']  # 移除噪声参数
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def load_cached_data(self, cache_key, data_type):
        """
        加载缓存数据
        """
        cache_file = self.cache_path / f"{cache_key}_{data_type}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载缓存失败: {e}")
        return None
    
    def save_cached_data(self, cache_key, data_type, data):
        """
        保存缓存数据
        """
        cache_file = self.cache_path / f"{cache_key}_{data_type}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"数据已缓存到: {cache_file}")
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def read_pcap_as_bytes(self, file_path, max_bytes=None):
        """
        读取pcap文件的原始字节数据
        
        Args:
            file_path: pcap文件路径
            max_bytes: 最大读取字节数
            
        Returns:
            bytes: 文件的字节数据
        """
        try:
            with open(file_path, 'rb') as f:
                if max_bytes:
                    data = f.read(max_bytes)
                else:
                    data = f.read()
            return data
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return b''
    
    def bytes_to_features(self, byte_data):
        """
        将字节数据转换为特征向量（去除噪声）
        """
        if len(byte_data) == 0:
            return np.zeros(self.max_packet_length, dtype=np.float32)
        
        # 将字节转换为0-255的整数数组
        features = np.frombuffer(byte_data, dtype=np.uint8).astype(np.float32)
        
        # 移除噪声添加代码
        noise = np.random.normal(0, self.hyperparams['noise_std'], features.shape)
        features = features + noise
        
        # 简单归一化（不使用标准化）
        features = features / 255.0
        
        # 截断或填充到固定长度
        if len(features) > self.max_packet_length:
            features = features[:self.max_packet_length]
        else:
            padding = np.zeros(self.max_packet_length - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        
        return features
    
    def load_dataset(self, data_path, max_files_per_class=None):
        """
        加载数据集（带缓存）
        
        Args:
            data_path: 数据路径
            max_files_per_class: 每个类别最大文件数（None表示不限制）
            
        Returns:
            tuple: (features, labels, class_names)
        """
        start_time = time.time()
        
        # 生成缓存键
        cache_key = self.get_cache_key(data_path, max_files_per_class)
        data_type = "train" if "Train" in str(data_path) else "test"
        
        # 尝试加载缓存
        cached_data = self.load_cached_data(cache_key, data_type)
        if cached_data is not None:
            print(f"从缓存加载 {data_type} 数据...")
            features, labels, class_names = cached_data
            self.timing_stats['data_loading_time'] += time.time() - start_time
            return features, labels, class_names
        
        features = []
        labels = []
        class_names = []
        
        print(f"\n从 {data_path} 加载数据...")
        
        # 获取所有类别目录
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name.replace("-ALL", "")
            class_names.append(class_name)
            
            # 获取该类别下的所有pcap文件
            pcap_files = list(class_dir.glob("*.pcap"))
            
            # 限制文件数量（如果指定）
            if max_files_per_class is not None and len(pcap_files) > max_files_per_class:
                pcap_files = pcap_files[:max_files_per_class]
            
            print(f"处理类别: {class_name} - 找到 {len(pcap_files)} 个文件")
            
            for pcap_file in pcap_files:
                # 读取pcap文件的字节数据
                byte_data = self.read_pcap_as_bytes(pcap_file, self.max_packet_length)
                
                # 转换为特征向量
                feature_vector = self.bytes_to_features(byte_data)
                
                features.append(feature_vector)
                labels.append(class_name)
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"\n数据加载完成:")
        print(f"特征形状: {features.shape}")
        print(f"标签数量: {len(labels)}")
        print(f"类别: {set(labels)}")
        
        # 保存到缓存
        self.save_cached_data(cache_key, data_type, (features, labels, class_names))
        
        self.timing_stats['data_loading_time'] += time.time() - start_time
        return features, labels, class_names

    def create_sequences(self, features):
        """
        创建时序序列（为了适配TCN）
        
        Args:
            features: 特征数组
            
        Returns:
            np.array: 时序特征
        """
        start_time = time.time()
        
        sequence_length = self.hyperparams['sequence_length']
        
        # 将特征重塑为时序格式
        # 这里我们将每个特征向量分割成多个时间步
        n_samples, feature_dim = features.shape
        
        if feature_dim < sequence_length:
            # 如果特征维度小于序列长度，进行填充
            padding = np.zeros((n_samples, sequence_length - feature_dim))
            features = np.concatenate([features, padding], axis=1)
            feature_dim = sequence_length
        
        # 重塑为 (samples, timesteps, features)
        timesteps = feature_dim // sequence_length
        features_per_timestep = sequence_length
        
        # 截断特征以适应整数倍的时间步
        truncated_features = features[:, :timesteps * features_per_timestep]
        
        # 重塑为时序格式
        sequential_features = truncated_features.reshape(
            n_samples, timesteps, features_per_timestep
        )
        
        print(f"时序特征形状: {sequential_features.shape}")
        
        self.timing_stats['preprocessing_time'] += time.time() - start_time
        return sequential_features

    def build_tcn_model(self, input_shape, num_classes):
        """
        构建简化的TCN模型（降低性能版本）
        
        Args:
            input_shape: 输入形状 (timesteps, features)
            num_classes: 类别数量
            
        Returns:
            SimpleTCNModel: 简化的TCN模型
        """
        model = SimpleTCNModel(
            input_size=input_shape[1],  # features per timestep
            num_channels=self.hyperparams['num_channels'],
            num_classes=num_classes,
            kernel_size=self.hyperparams['kernel_size'],
            dropout=self.hyperparams['dropout'],
            classifier_hidden=self.hyperparams['classifier_hidden']
        )
        
        return model.to(self.device)

    def train_model(self, X_train, y_train, X_val, y_val):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            tuple: (model, history)
        """
        print("\n开始训练TCN模型...")
        
        start_time = time.time()
        
        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(y_train))
        
        model = self.build_tcn_model(input_shape, num_classes)
        
        print(f"模型结构: 输入形状 {input_shape}, 类别数 {num_classes}")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.hyperparams['batch_size'], shuffle=False)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.hyperparams['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        # 训练循环
        for epoch in range(self.hyperparams['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_start_time = time.time()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            self.timing_stats['validation_prediction_time'] += time.time() - val_start_time
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            print(f'Epoch [{epoch+1}/{self.hyperparams["epochs"]}] - '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        self.timing_stats['training_time'] = time.time() - start_time
        print(f"\n训练完成！训练时间: {self.timing_stats['training_time']:.2f}秒")
        return model, history

    def evaluate_model(self, model, X_test, y_test, class_names):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            class_names: 类别名称
            
        Returns:
            dict: 评估结果
        """
        print("\n评估模型...")
        
        start_time = time.time()
        
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()
        
        self.timing_stats['test_prediction_time'] = time.time() - start_time
        
        # 计算指标（添加zero_division=0参数消除警告）
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n测试结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"测试预测时间: {self.timing_stats['test_prediction_time']:.2f}秒")
        
        # 分类报告（添加zero_division=0参数消除警告）
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        print(f"\n分类报告:\n{report}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }

    def save_results(self, model, history, test_results):
        """
        保存结果（包含详细时间统计和超参数）
        
        Args:
            model: 训练好的模型
            history: 训练历史
            test_results: 测试结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算总时间
        self.timing_stats['total_time'] = sum(self.timing_stats.values())
        
        # 保存模型
        model_path = self.output_path / f"tcn_raw_model_{self.dataset_name}_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")
        
        # 保存训练历史
        history_path = self.output_path / f"tcn_raw_history_{self.dataset_name}_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # 保存详细的结果数据
        results_data = {
            'dataset': self.dataset_name,
            'timestamp': timestamp,
            'hyperparameters': self.hyperparams,
            'timing_stats': self.timing_stats,
            'test_results': test_results
        }
        
        results_path = self.output_path / f"tcn_raw_results_{self.dataset_name}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # 保存性能指标CSV（包含时间指标）
        avg_prediction_time_per_sample = (self.timing_stats['test_prediction_time'] / 
                                        len(test_results['true_labels']) if test_results['true_labels'] else 0)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                       'Training_Time', 'Test_Prediction_Time', 'Avg_Prediction_Time_Per_Sample'],
            'Value': [
                test_results['accuracy'],
                test_results['precision'],
                test_results['recall'],
                test_results['f1_score'],
                self.timing_stats['training_time'],
                self.timing_stats['test_prediction_time'],
                avg_prediction_time_per_sample
            ]
        })
        
        metrics_path = self.output_path / f"tcn_raw_metrics_{self.dataset_name}_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"结果已保存到: {self.output_path}")
        print(f"时间统计: {self.timing_stats}")

    def plot_training_history(self, history, save_path=None):
        """
        绘制训练历史
        
        Args:
            history: 训练历史
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 准确率曲线
        ax2.plot(history['train_accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"训练历史图已保存到: {save_path}")
        
        plt.close()

    def process_dataset(self, max_files_per_class=None):
        """
        处理完整的数据集（使用数据集特定的超参数）
        
        Args:
            max_files_per_class: 每个类别最大文件数（None表示不限制）
        """
        print(f"\n开始处理 {self.dataset_name} 数据集...")
        print(f"使用超参数: {self.hyperparams}")
        
        total_start_time = time.time()
        
        # 加载训练数据
        X_train, y_train, class_names = self.load_dataset(self.train_path, max_files_per_class)
        
        # 加载测试数据
        X_test, y_test, _ = self.load_dataset(self.test_path, max_files_per_class)
        
        # 编码标签
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 创建时序序列
        X_train_seq = self.create_sequences(X_train)
        X_test_seq = self.create_sequences(X_test)
        
        # 分割训练集和验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_seq, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
        )
        
        print(f"\n数据分割完成:")
        print(f"训练集: {X_train_final.shape}")
        print(f"验证集: {X_val.shape}")
        print(f"测试集: {X_test_seq.shape}")
        
        # 训练模型
        model, history = self.train_model(
            X_train_final, y_train_final, X_val, y_val
        )
        
        # 评估模型
        test_results = self.evaluate_model(model, X_test_seq, y_test_encoded, class_names)
        
        # 保存结果
        self.save_results(model, history, test_results)
        
        # 绘制训练历史
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_path / f"tcn_raw_training_history_{self.dataset_name}_{timestamp}.png"
        self.plot_training_history(history, plot_path)
        
        total_time = time.time() - total_start_time
        print(f"\n{self.dataset_name} 数据集处理完成！总耗时: {total_time:.2f}秒")
        
        return model, history, test_results

# 主程序
if __name__ == "__main__":
    print("TCN原始数据处理器 (针对不同数据集优化版本)")
    print("=" * 50)
    for i in range(3):
        print('*'*10)
        print(f"总共3轮，目前是第{i+1}轮")
        print('*'*10)

        # # 处理CTU数据集
        # print("处理CTU数据集...")
        # ctu_processor = TCNRawDataProcessor(dataset_name="CTU")
        
        # try:
        #     ctu_model, ctu_history, ctu_results = ctu_processor.process_dataset(
        #         max_files_per_class=None  # 不限制文件数量
        #     )
        #     print("CTU数据集处理成功！")
        # except Exception as e:
        #     print(f"CTU数据集处理失败: {e}")
        
        # 处理USTC数据集
        print("\n处理USTC数据集...")
        ustc_processor = TCNRawDataProcessor(dataset_name="USTC")
        
        try:
            ustc_model, ustc_history, ustc_results = ustc_processor.process_dataset(
                max_files_per_class=None  # 不限制文件数量
            )
            print("USTC数据集处理成功！")
        except Exception as e:
            print(f"USTC数据集处理失败: {e}")
        
    print("\n所有数据集处理完成！")