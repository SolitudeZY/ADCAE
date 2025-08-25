import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

class TemporalBlock(nn.Module):
    """TCN的基本时序块"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """移除卷积后多余的填充"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    """时序卷积网络"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNClassifier(nn.Module):
    """TCN分类器"""
    def __init__(self, input_size, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, input_size, seq_len)
        y1 = self.tcn(x)
        # 全局平均池化
        y1 = torch.mean(y1, dim=2)
        y1 = self.dropout(y1)
        return self.linear(y1)

class TCNProcessor:
    """TCN处理器 - 模块化的TCN训练和评估系统"""
    
    def __init__(self, num_channels=[25, 25, 25, 25], kernel_size=7, dropout=0.2, 
                 seq_len=10, output_dir='results/tcn_output', dataset_name='unknown'):
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_source_name = feature_source_name  # 新增：特征来源标识
        
        self.model = None
        self.label_encoder = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'five_metrics': [],
            'epoch_times': []
        }
        
        os.makedirs(os.path.join(self.output_dir, 'tcn_features'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        
        print(f"TCN处理器初始化完成，使用设备: {self.device}")
    
    def load_features_from_files(self, train_file, test_file):
        """从特征文件加载数据"""
        print(f"\n加载特征文件...")
        print(f"训练集: {train_file}")
        print(f"测试集: {test_file}")
        
        # 加载CSV文件
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # 自动检测特征列
        possible_feature_prefixes = ['encoded_feature_', 'feature_', 'pca_', 'kpca_', 'cae_']
        feature_cols = []
        
        for prefix in possible_feature_prefixes:
            cols = [col for col in train_df.columns if col.startswith(prefix)]
            if cols:
                feature_cols = cols
                print(f"检测到特征列前缀: {prefix}, 特征数量: {len(cols)}")
                break
        
        if not feature_cols:
            # 如果没有找到特定前缀，使用除了'label'之外的所有数值列
            feature_cols = [col for col in train_df.columns if col != 'label' and train_df[col].dtype in ['int64', 'float64']]
            print(f"使用所有数值列作为特征，特征数量: {len(feature_cols)}")
        
        if not feature_cols:
            raise ValueError("未找到有效的特征列")
        
        # 提取特征和标签
        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values
        
        print(f"数据加载完成:")
        print(f"  训练集: {X_train.shape}, 类别: {len(np.unique(y_train))}")
        print(f"  测试集: {X_test.shape}, 类别: {len(np.unique(y_test))}")
        print(f"  训练集类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"  测试集类别分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        return X_train, X_test, y_train, y_test
    
    def load_features_from_directory(self, features_dir, dataset_name):
        """从目录自动加载特征文件"""
        print(f"\n从目录加载特征: {features_dir}")
        
        # 查找训练和测试文件
        possible_train_files = [
            f'train_features_{dataset_name}.csv',
            f'train_features_labels_{dataset_name}.csv',
            'train_features.csv',
            'encoded_features_train.csv'
        ]
        
        possible_test_files = [
            f'test_features_{dataset_name}.csv',
            f'test_features_labels_{dataset_name}.csv',
            'test_features.csv',
            'encoded_features_test.csv'
        ]
        
        train_file = None
        test_file = None
        
        for filename in possible_train_files:
            filepath = os.path.join(features_dir, filename)
            if os.path.exists(filepath):
                train_file = filepath
                break
        
        for filename in possible_test_files:
            filepath = os.path.join(features_dir, filename)
            if os.path.exists(filepath):
                test_file = filepath
                break
        
        if not train_file or not test_file:
            raise FileNotFoundError(f"在 {features_dir} 中未找到训练或测试特征文件")
        
        return self.load_features_from_files(train_file, test_file)
    
    def create_sequences(self, features, labels):
        """创建时序序列"""
        print(f"创建时序序列，序列长度: {self.seq_len}")
        
        X_seq = []
        y_seq = []
        
        # 按类别分组创建序列
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_features = features[label_indices]
            
            # 为每个类别创建序列
            for i in range(len(label_features) - self.seq_len + 1):
                X_seq.append(label_features[i:i+self.seq_len])
                y_seq.append(label)
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"序列创建完成: {X_seq.shape}")
        return X_seq, y_seq
    
    def prepare_data(self, X_train, X_test, y_train, y_test, test_size=0.2, random_state=42):
        """准备训练数据"""
        print("\n准备TCN训练数据...")
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 创建时序数据
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train_encoded)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test_encoded)
        
        # 从训练序列中分割验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_seq, y_train_seq, test_size=test_size, random_state=random_state, stratify=y_train_seq
        )
        
        print(f"数据准备完成:")
        print(f"  训练集: {X_train_final.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test_seq.shape}")
        print(f"  类别数量: {len(self.label_encoder.classes_)}")
        print(f"  类别映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return X_train_final, X_val, X_test_seq, y_train_final, y_val, y_test_seq
    
    def build_model(self, input_size, num_classes):
        """构建TCN模型"""
        self.model = TCNClassifier(
            input_size=input_size,
            num_channels=self.num_channels,
            num_classes=num_classes,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"TCN模型已创建:")
        print(f"  输入维度: {input_size}")
        print(f"  输出类别: {num_classes}")
        print(f"  通道配置: {self.num_channels}")
        print(f"  参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=5, batch_size=32, 
                   learning_rate=0.001, patience=15):
        """训练模型"""
        print(f"\n开始训练TCN模型...")
        print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).permute(0, 2, 1).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(batch_y.cpu().numpy())
            
            # 计算时间统计
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            estimated_remaining = avg_epoch_time * (epochs - epoch - 1)
            
            # 计算指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # 计算五个关键指标
            val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
            
            # 保存训练历史
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['five_metrics'].append({
                'epoch': epoch + 1,
                'accuracy': val_acc / 100,
                'precision': val_precision,
                'recall': val_recall,
                'f1_score': val_f1,
                'epoch_time': epoch_time
            })
            self.training_history['epoch_times'].append(epoch_time)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 每轮都输出详细信息（包含时间统计）
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"时间统计: 本轮={epoch_time:.2f}s | 累计={elapsed_time:.1f}s | 平均={avg_epoch_time:.2f}s/轮 | 预估剩余={estimated_remaining:.1f}s")
            print(f"损失: 训练={avg_train_loss:.4f} | 验证={avg_val_loss:.4f}")
            print(f"准确率: 训练={train_acc:.2f}% | 验证={val_acc:.2f}%")
            print(f"指标: Precision={val_precision:.4f} | Recall={val_recall:.4f} | F1={val_f1:.4f}")
            print(f"学习率: {optimizer.param_groups[0]['lr']:.2e} | 早停: {patience_counter}/{patience}")
            
            # 早停
            if patience_counter >= patience:
                print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        training_time = time.time() - start_time
        print(f"\n训练完成，总耗时: {training_time:.2f}秒")
        
        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            print("已加载最佳模型权重")
        
        return training_time
    
    def evaluate_model(self, X_test, y_test, batch_size=32):
        """评估模型"""
        print("\n开始模型评估...")
        
        # 转换为PyTorch张量
        X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # 创建数据加载器
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        test_time = time.time() - start_time
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        print(f"\n测试结果:")
        print(f"  测试时间: {test_time:.2f}秒")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # 详细分类报告
        class_names = self.label_encoder.classes_
        print("\n详细分类报告:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_time': test_time,
            'predictions': all_preds,
            'true_labels': all_labels,
            'classification_report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        }
    
    def save_model_and_results(self, test_results, training_time):
        """保存模型和结果"""
        # 创建更详细的输出路径结构
        if self.feature_source_name:
            # 如果有特征来源信息，创建更详细的路径
            output_path = os.path.join(self.output_dir, f'TCN_{self.dataset_name}_{self.feature_source_name}')
        else:
            output_path = os.path.join(self.output_dir, f'TCN_{self.dataset_name}')
        
        plots_path = os.path.join(output_path, 'plots')
        models_path = os.path.join(output_path, 'models')
        results_path = os.path.join(output_path, 'results')
        
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        
        # 生成详细的文件名
        if self.feature_source_name:
            file_prefix = f'TCN_{self.dataset_name}_{self.feature_source_name}'
        else:
            file_prefix = f'TCN_{self.dataset_name}'
        
        # 保存模型
        model_path = os.path.join(models_path, f'{file_prefix}_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.tcn.network[0].conv1.in_channels,
                'num_classes': len(self.label_encoder.classes_),
                'num_channels': self.num_channels,
                'kernel_size': self.kernel_size,
                'dropout': self.dropout,
                'seq_len': self.seq_len
            },
            'label_encoder': self.label_encoder,
            'dataset_name': self.dataset_name,
            'feature_source': self.feature_source_name
        }, model_path)
        
        # 保存训练历史
        for history_type, history_data in self.training_history.items():
            if history_data:
                if history_type == 'five_metrics':
                    history_df = pd.DataFrame(history_data)
                else:
                    history_df = pd.DataFrame({
                        'epoch': range(1, len(history_data) + 1),
                        history_type: history_data
                    })
                history_path = os.path.join(results_path, f'{file_prefix}_{history_type}.csv')
                history_df.to_csv(history_path, index=False)
        
        # 修复JSON序列化问题
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 保存测试结果
        test_results_path = os.path.join(results_path, f'{file_prefix}_test_results.json')
        with open(test_results_path, 'w') as f:
            results_to_save = convert_numpy_types(test_results.copy())
            json.dump(results_to_save, f, indent=2)
        
        # 保存综合统计
        stats = {
            'dataset_name': self.dataset_name,
            'feature_source': self.feature_source_name,
            'model_type': 'TCN',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': float(training_time),
            'test_time_seconds': float(test_results['test_time']),
            'total_epochs': len(self.training_history['train_loss']),
            'final_test_metrics': {
                'accuracy': float(test_results['accuracy']),
                'precision': float(test_results['precision']),
                'recall': float(test_results['recall']),
                'f1_score': float(test_results['f1_score']),
                'test_time': float(test_results['test_time'])
            },
            'model_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'model_config': {
                'num_channels': self.num_channels,
                'kernel_size': self.kernel_size,
                'dropout': self.dropout,
                'seq_len': self.seq_len
            }
        }
        
        stats_path = os.path.join(results_path, f'{file_prefix}_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n结果已保存到: {output_path}")
        print(f"- TCN模型: {model_path}")
        print(f"- 测试结果: {test_results_path}")
        print(f"- 统计信息: {stats_path}")
        
        return stats
    
    def process_dataset(self, features_source, epochs=5, batch_size=32, learning_rate=0.001):
        """处理完整的数据集流程"""
        print(f"\n{'='*80}")
        print(f"开始处理 {self.dataset_name} 数据集 - TCN")
        print(f"{'='*80}")
        
        try:
            # 加载数据
            if isinstance(features_source, tuple) and len(features_source) == 2:
                # 直接传入文件路径
                train_file, test_file = features_source
                X_train, X_test, y_train, y_test = self.load_features_from_files(train_file, test_file)
            elif isinstance(features_source, str):
                # 传入目录路径
                X_train, X_test, y_train, y_test = self.load_features_from_directory(features_source, self.dataset_name)
            else:
                raise ValueError("features_source 必须是 (train_file, test_file) 元组或目录路径字符串")
            
            # 准备数据
            X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = self.prepare_data(
                X_train, X_test, y_train, y_test
            )
            
            # 构建模型
            input_size = X_train_seq.shape[2]  # 特征维度
            num_classes = len(np.unique(y_train_seq))
            self.build_model(input_size, num_classes)
            
            # 训练模型
            training_time = self.train_model(
                X_train_seq, X_val_seq, y_train_seq, y_val_seq,
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
            )
            
            # 评估模型
            test_results = self.evaluate_model(X_test_seq, y_test_seq, batch_size=batch_size)
            
            # 保存结果
            stats = self.save_model_and_results(test_results, training_time)
            
            print(f"\n{'='*80}")
            print(f"{self.dataset_name} 数据集处理完成")
            print(f"最终测试准确率: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
            print(f"{'='*80}")
            
            return stats
            
        except Exception as e:
            print(f"\n处理 {self.dataset_name} 数据集时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == '__main__':
    # 使用示例
    datasets = ['USTC', 'CTU']
    
    # 特征文件路径配置
    feature_paths = {
        'USTC': {
            'adcae': 'USTC_result/adcae_features/layer_comparison/4_layers',
            'cae': 'USTC_result/cae_output/cae_features',
            'kpca': 'USTC_result/kpca_output/kpca_features',
            'pca': 'USTC_result/pca_output/pca_features'
        },
        'CTU': {
            'adcae': 'CTU_result/adcae_features/layer_comparison/4_layers',
            'cae': 'CTU_result/cae_output/cae_features',
            'kpca': 'CTU_result/kpca_output/kpca_features',
            'pca': 'CTU_result/pca_output/pca_features'
        }
    }
    
    for dataset in datasets:
        for feature_type, features_dir in feature_paths[dataset].items():
            print(f"\n\n{'='*100}")
            print(f"处理 {dataset} 数据集 - {feature_type.upper()} 特征")
            print(f"{'='*100}")
            
            # 创建TCN处理器
            tcn_processor = TCNProcessor(
                num_channels=[25, 25, 25, 25],
                kernel_size=7,
                dropout=0.1,
                seq_len=10,
                output_dir=f'{dataset}_result/tcn_output_{feature_type}',
                dataset_name=f'{dataset}_{feature_type}'
            )
            
            # 处理数据集
            stats = tcn_processor.process_dataset(
                features_source=features_dir,
                epochs=5,
                batch_size=32,
                learning_rate=0.001
            )
            
            if stats:
                print(f"\n{dataset}_{feature_type} 处理成功")
                print(f"最终准确率: {stats['final_test_metrics']['accuracy']:.4f}")
            else:
                print(f"\n{dataset}_{feature_type} 处理失败")
    
    print("\n所有数据集处理完成！")