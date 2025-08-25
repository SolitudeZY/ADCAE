import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# 添加父目录到路径以导入TCN模型
sys.path.append(r"D:\Python Project\ADCAE\ML+DL")
from tcn_models import TCNClassifier

class ADCAETCNComparison:
    """
    ADCAE特征的TCN对比分析器
    对比不同激活函数和注意力机制下的ADCAE特征性能
    支持多数据集对比
    """
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.results = {}
        
        # 数据路径配置
        if dataset_name == "CTU":
            self.base_path = "D:\\Python Project\\ADCAE\\CTU_result\\adcae_features"
        elif dataset_name == "USTC":
            self.base_path = "D:\\Python Project\\ADCAE\\USTC_result\\adcae_features"
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
            
        self.activation_path = os.path.join(self.base_path, "activation_comparison")
        self.attention_path = os.path.join(self.base_path, "attention_comparison")
        
        # 输出路径
        self.output_path = f"D:\\Python Project\\ADCAE\\ML+DL\\ADCAE_test\\TCN\\results_{dataset_name}"
        os.makedirs(self.output_path, exist_ok=True)
        
        print(f"使用设备: {self.device}")
        print(f"数据集: {dataset_name}")
        
        # 实验配置 - 可以通过修改这里来控制具体实验
        self.experiments = {
            'activation_comparison': {
                'elu': {'enabled': False},      # 启用/禁用 ELU 激活函数实验
                'relu': {'enabled': True},     # 启用/禁用 ReLU 激活函数实验
                'sigmod': {'enabled': True}   # 启用/禁用 sigmod 激活函数实验
            },
            'attention_comparison': {
                'with_cbam': {'enabled': False},        # 启用/禁用 CBAM 注意力机制实验
                'without_attention': {'enabled': False} # 启用/禁用 无注意力机制实验
            }
        }
        
        # 快速控制示例（通过注释来启用/禁用）:
        # self.experiments['activation_comparison']['elu']['enabled'] = False      # 禁用ELU实验
        # self.experiments['activation_comparison']['relu']['enabled'] = False     # 禁用ReLU实验
        # self.experiments['attention_comparison']['with_cbam']['enabled'] = False # 禁用CBAM实验
        
        # 获取数据集特定的超参数
        self.activation_hyperparameters = self._get_dataset_hyperparameters()['activation_comparison']
        self.attention_hyperparameters = self._get_dataset_hyperparameters()['attention_comparison']
    
    def _get_dataset_hyperparameters(self):
        """
        获取数据集特定的超参数配置
        """
        if self.dataset_name == "CTU":
            return {
                'activation_comparison': {
                    'elu': {
                        'num_channels': [32, 64, 128, 64],
                        'kernel_size': 3,
                        'dropout': 0.3,
                        'seq_len': 15,
                        'epochs': 1,
                        'batch_size': 64,
                        'learning_rate': 0.001,
                        'patience': 12,
                        'noise_std': 0.01
                    },
                    'relu': {
                        'num_channels': [64, 128, 256, 128],
                        'kernel_size': 5,
                        'dropout': 0.25,
                        'seq_len': 20,
                        'epochs': 1,
                        'batch_size': 32,
                        'learning_rate': 0.0001,
                        'patience': 15,
                        'noise_std': 0.05
                    },
                    'sigmod': {
                        'num_channels': [48, 96, 192, 96],
                        'kernel_size': 5,
                        'dropout': 0.1,
                        'seq_len': 10,
                        'epochs': 1,
                        'batch_size': 128,
                        'learning_rate': 0.001,
                        'patience': 10,
                        'noise_std': 0.02
                    }
                },
                'attention_comparison': {
                    'with_cbam': {
                        'num_channels': [48, 96, 192, 96],
                        'kernel_size': 4,
                        'dropout': 0.25,
                        'seq_len': 18,
                        'epochs': 1,
                        'batch_size': 48,
                        'learning_rate': 0.0008,
                        'patience': 18,
                        'noise_std': 0.008
                    },
                    'without_attention': {
                        'num_channels': [24, 48, 96, 48],
                        'kernel_size': 6,
                        'dropout': 0.35,
                        'seq_len': 12,
                        'epochs': 1,
                        'batch_size': 96,
                        'learning_rate': 0.0015,
                        'patience': 14,
                        'noise_std': 0.015
                    }
                }
            }
        elif self.dataset_name == "USTC":
            return {
                'activation_comparison': {
                    'elu': {
                        'num_channels': [40, 80, 160, 80],
                        'kernel_size': 4,
                        'dropout': 0.35,
                        'seq_len': 12,
                        'epochs': 1,
                        'batch_size': 48,
                        'learning_rate': 0.0012,
                        'patience': 10,
                        'noise_std': 0.012
                    },
                    'relu': {
                        'num_channels': [56, 112, 224, 112],
                        'kernel_size': 6,
                        'dropout': 0.25,
                        'seq_len': 16,
                        'epochs': 1,
                        'batch_size': 40,
                        'learning_rate': 0.0001,
                        'patience': 12,
                        'noise_std': 0.008
                    },
                    'sigmod': {
                        'num_channels': [64, 128, 256, 128],
                        'kernel_size': 3,
                        'dropout': 0.45,
                        'seq_len': 8,
                        'epochs': 2,
                        'batch_size': 96,
                        'learning_rate': 0.0025,
                        'patience': 8,
                        'noise_std': 0.025
                    }
                },
                'attention_comparison': {
                    'with_cbam': {
                        'num_channels': [52, 104, 208, 104],
                        'kernel_size': 5,
                        'dropout': 0.3,
                        'seq_len': 14,
                        'epochs': 1,
                        'batch_size': 56,
                        'learning_rate': 0.001,
                        'patience': 16,
                        'noise_std': 0.01
                    },
                    'without_attention': {
                        'num_channels': [28, 56, 112, 56],
                        'kernel_size': 7,
                        'dropout': 0.4,
                        'seq_len': 10,
                        'epochs': 1,
                        'batch_size': 72,
                        'learning_rate': 0.0018,
                        'patience': 12,
                        'noise_std': 0.018
                    }
                }
            }
    
    def create_sequences(self, features, labels, seq_len):
        """
        创建时序序列
        """
        X_seq = []
        y_seq = []
        
        # 确保输入数据是正确的数值类型
        if features.dtype != np.float32:
            features = features.astype(np.float32)
            print(f"已将特征数据转换为float32类型")
        
        # 按类别分组创建序列
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_features = features[label_indices]
            
            # 为每个类别创建序列
            for i in range(len(label_features) - seq_len + 1):
                sequence = label_features[i:i+seq_len]
                # 确保序列形状正确
                if sequence.shape[0] == seq_len:
                    X_seq.append(sequence)
                    y_seq.append(label)
        
        # 转换为numpy数组，确保数据类型一致
        if len(X_seq) > 0:
            X_seq = np.array(X_seq, dtype=np.float32)
            y_seq = np.array(y_seq, dtype=np.int64)
        else:
            raise ValueError("没有生成有效的序列数据")
        
        print(f"生成序列数量: {len(X_seq)}, 序列形状: {X_seq.shape if len(X_seq) > 0 else 'None'}")
        return X_seq, y_seq

    def load_data(self, data_path):
        """
        加载训练和测试数据
        """
        train_path = os.path.join(data_path, "train_features.csv")
        test_path = os.path.join(data_path, "test_features.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 读取数据，强制数值类型转换
        try:
            train_df = pd.read_csv(train_path, dtype={'label': str})  # 标签保持字符串
            test_df = pd.read_csv(test_path, dtype={'label': str})
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            # 如果指定dtype失败，尝试默认读取
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        
        # 分离特征和标签
        feature_columns = [col for col in train_df.columns if col != 'label']
        
        # 确保特征数据是数值类型
        print(f"处理特征列: {len(feature_columns)} 个")
        for col in feature_columns:
            # 转换为数值类型，无法转换的设为NaN
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        
        # 检查并处理NaN值
        train_nan_count = train_df[feature_columns].isnull().sum().sum()
        test_nan_count = test_df[feature_columns].isnull().sum().sum()
        
        if train_nan_count > 0 or test_nan_count > 0:
            print(f"发现NaN值 - 训练集: {train_nan_count}, 测试集: {test_nan_count}")
            # 用0填充NaN值
            train_df[feature_columns] = train_df[feature_columns].fillna(0)
            test_df[feature_columns] = test_df[feature_columns].fillna(0)
            print("已用0填充NaN值")
        
        # 提取特征和标签，确保数据类型
        X_train = train_df[feature_columns].values.astype(np.float32)
        X_test = test_df[feature_columns].values.astype(np.float32)
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print(f"数据加载完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        print(f"特征数据类型: {X_train.dtype}, 标签数据类型: {y_train.dtype}")
        
        # 验证数据质量
        if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
            print("警告: 数据中仍包含NaN值")
        if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
            print("警告: 数据中包含无穷值")
            # 处理无穷值
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
            print("已处理无穷值")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(self, X_train, X_test, y_train, y_test, seq_len, test_size=0.2):
        """
        准备时序数据
        """
        print(f"准备时序数据，序列长度: {seq_len}")
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 创建时序数据
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train_encoded, seq_len)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test_encoded, seq_len)
        
        # 从训练序列中分出验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_seq, y_train_seq, test_size=test_size, random_state=42, stratify=y_train_seq
        )
        
        print(f"训练集大小: {X_train_final.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"测试集大小: {X_test_seq.shape}")
        print(f"类别数量: {len(self.label_encoder.classes_)}")
        
        return X_train_final, X_val, X_test_seq, y_train_final, y_val, y_test_seq
    
    def add_noise_to_data(self, X, noise_std=0.01):
        """
        为训练数据添加高斯噪声
        """
        if noise_std > 0:
            noise = torch.randn_like(X) * noise_std
            return X + noise
        return X
    
    def train_tcn_model(self, X_train, X_val, y_train, y_val, hyperparams, experiment_name):
        """
        训练TCN模型
        """
        print(f"\n开始训练 {experiment_name} TCN模型...")
        print(f"超参数配置: {hyperparams}")
        
        # 获取输入维度和类别数
        input_size = X_train.shape[2]  # 特征维度
        num_classes = len(np.unique(y_train))
        
        # 创建模型
        model = TCNClassifier(
            input_size=input_size,
            num_channels=hyperparams['num_channels'],
            num_classes=num_classes,
            kernel_size=hyperparams['kernel_size'],
            dropout=hyperparams['dropout']
        ).to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 转换为PyTorch张量并调整维度
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).permute(0, 2, 1).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # 训练变量
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        print(f"\n开始训练循环...")
        
        for epoch in range(hyperparams['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                # 添加噪声（仅在训练时）
                if hyperparams['noise_std'] > 0:
                    batch_X = self.add_noise_to_data(batch_X, hyperparams['noise_std'])
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # 计算指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # 记录历史
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 简化的epoch输出
            if (epoch + 1) % max(1, hyperparams['epochs'] // 5) == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{hyperparams['epochs']}] - Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}% | Val: Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")
            
            # 早停
            if patience_counter >= hyperparams['patience']:
                print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
            print(f"\n已加载最佳模型（验证损失: {best_val_loss:.4f}）")
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总训练时间: {training_time:.2f}秒")
        
        return model, training_time, training_history
    
    def evaluate_tcn_model(self, model, X_test, y_test, batch_size=32):
        """
        评估TCN模型
        """
        print("\n开始模型评估...")
        
        # 转换为PyTorch张量并调整维度
        X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        model.eval()
        
        # 创建数据加载器
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        test_time = time.time() - start_time
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"\n测试结果:")
        print(f"测试时间: {test_time:.2f}秒")
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_loss': avg_test_loss,
            'test_time': test_time
        }
    
    def train_and_evaluate_tcn(self, X_train, X_test, y_train, y_test, hyperparams, experiment_name):
        """
        训练和评估TCN模型的完整流程
        """
        print(f"\n处理实验: {experiment_name}")
        print(f"原始数据 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 准备时序数据
        X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = self.prepare_data(
            X_train, X_test, y_train, y_test, hyperparams['seq_len']
        )
        
        # 训练模型
        model, training_time, training_history = self.train_tcn_model(
            X_train_seq, X_val_seq, y_train_seq, y_val_seq, hyperparams, experiment_name
        )
        
        # 评估模型
        test_results = self.evaluate_tcn_model(model, X_test_seq, y_test_seq, hyperparams['batch_size'])
        
        return {
            'experiment_name': experiment_name,
            'hyperparameters': hyperparams,
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1_score': test_results['f1_score'],
            'training_time': training_time,
            'testing_time': test_results['test_time'],
            'data_shape': {
                'train_seq': X_train_seq.shape,
                'test_seq': X_test_seq.shape
            },
            'training_history': training_history
        }
    
    def run_activation_comparison(self):
        """
        运行激活函数对比实验
        """
        print(f"\n{self.dataset_name} - 激活函数对比实验")
        print("="*60)
        
        activation_results = []
        
        for activation in ['elu', 'relu', 'sigmod']:
            if not self.experiments['activation_comparison'][activation]['enabled']:
                print(f"\n跳过激活函数: {activation} (已禁用)")
                continue
                
            try:
                print(f"\n处理激活函数: {activation}")
                data_path = os.path.join(self.activation_path, activation)
                
                # 加载数据
                X_train, X_test, y_train, y_test = self.load_data(data_path)
                print(f"数据加载完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
                
                # 获取超参数
                hyperparams = self.activation_hyperparameters[activation]
                
                # 训练和评估
                result = self.train_and_evaluate_tcn(
                    X_train, X_test, y_train, y_test, 
                    hyperparams, f"激活函数_{activation.upper()}"
                )
                
                activation_results.append(result)
                
            except Exception as e:
                print(f"处理激活函数 {activation} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return activation_results
    
    def run_attention_comparison(self):
        """
        运行注意力机制对比实验
        """
        print(f"\n{self.dataset_name} - 注意力机制对比实验")
        print("="*60)
        
        attention_results = []
        
        for attention in ['with_cbam', 'without_attention']:
            if not self.experiments['attention_comparison'][attention]['enabled']:
                print(f"\n跳过注意力机制: {attention} (已禁用)")
                continue
                
            try:
                print(f"\n处理注意力机制: {attention}")
                data_path = os.path.join(self.attention_path, attention)
                
                # 加载数据
                X_train, X_test, y_train, y_test = self.load_data(data_path)
                print(f"数据加载完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
                
                # 获取超参数
                hyperparams = self.attention_hyperparameters[attention]
                
                # 训练和评估
                result = self.train_and_evaluate_tcn(
                    X_train, X_test, y_train, y_test, 
                    hyperparams, f"注意力机制_{attention.replace('_', ' ').title()}"
                )
                
                attention_results.append(result)
                
            except Exception as e:
                print(f"处理注意力机制 {attention} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return attention_results
    
    def generate_comparison_report(self, activation_results, attention_results):
        """
        生成对比报告
        """
        print("\n" + "="*80)
        print("ADCAE TCN模型对比报告")
        print("="*80)
        
        # 激活函数对比报告
        if activation_results:
            print("\n1. 激活函数对比结果:")
            print("-" * 50)
            
            activation_df_data = []
            for result in activation_results:
                activation_df_data.append({
                    '实验': result['experiment_name'],
                    '准确率': f"{result['accuracy']:.4f}",
                    '精确率': f"{result['precision']:.4f}",
                    '召回率': f"{result['recall']:.4f}",
                    'F1分数': f"{result['f1_score']:.4f}",
                    '训练时间(秒)': f"{result['training_time']:.2f}",
                    '测试时间(秒)': f"{result['testing_time']:.2f}",
                    '序列长度': result['hyperparameters']['seq_len'],
                    '批次大小': result['hyperparameters']['batch_size']
                })
            
            activation_df = pd.DataFrame(activation_df_data)
            print(activation_df.to_string(index=False))
            
            # 找出最佳激活函数
            best_activation = max(activation_results, key=lambda x: x['f1_score'])
            print(f"\n最佳激活函数: {best_activation['experiment_name']} (F1分数: {best_activation['f1_score']:.4f})")
        
        # 注意力机制对比报告
        if attention_results:
            print("\n2. 注意力机制对比结果:")
            print("-" * 50)
            
            attention_df_data = []
            for result in attention_results:
                attention_df_data.append({
                    '实验': result['experiment_name'],
                    '准确率': f"{result['accuracy']:.4f}",
                    '精确率': f"{result['precision']:.4f}",
                    '召回率': f"{result['recall']:.4f}",
                    'F1分数': f"{result['f1_score']:.4f}",
                    '训练时间(秒)': f"{result['training_time']:.2f}",
                    '测试时间(秒)': f"{result['testing_time']:.2f}",
                    '序列长度': result['hyperparameters']['seq_len'],
                    '批次大小': result['hyperparameters']['batch_size']
                })
            
            attention_df = pd.DataFrame(attention_df_data)
            print(attention_df.to_string(index=False))
            
            # 找出最佳注意力机制
            best_attention = max(attention_results, key=lambda x: x['f1_score'])
            print(f"\n最佳注意力机制: {best_attention['experiment_name']} (F1分数: {best_attention['f1_score']:.4f})")
        
        # 保存结果到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if activation_results:
            activation_csv_path = os.path.join(self.output_path, f"activation_comparison_{timestamp}.csv")
            activation_df.to_csv(activation_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n激活函数对比结果已保存到: {activation_csv_path}")
        
        if attention_results:
            attention_csv_path = os.path.join(self.output_path, f"attention_comparison_{timestamp}.csv")
            attention_df.to_csv(attention_csv_path, index=False, encoding='utf-8-sig')
            print(f"注意力机制对比结果已保存到: {attention_csv_path}")
        
        # 保存详细结果到JSON
        detailed_results = {
            'timestamp': timestamp,
            'device': str(self.device),
            'activation_comparison': activation_results,
            'attention_comparison': attention_results
        }
        
        json_path = os.path.join(self.output_path, f"detailed_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"详细结果已保存到: {json_path}")
        
        return detailed_results
    
    def run_all_comparisons(self):
        """
        运行所有对比实验
        """
        print("开始ADCAE TCN对比实验")
        print(f"使用设备: {self.device}")
        print(f"输出路径: {self.output_path}")
        
        # 运行激活函数对比
        activation_results = self.run_activation_comparison()
        
        # 运行注意力机制对比
        attention_results = self.run_attention_comparison()
        
        # 生成对比报告
        detailed_results = self.generate_comparison_report(activation_results, attention_results)
        
        print("\n" + "="*80)
        print("所有对比实验完成!")
        print("="*80)
        
        return detailed_results

    def generate_cross_dataset_comparison(self, all_results):
        """
        生成跨数据集横向对比报告 - 统一表格格式
        """
        print("\n" + "="*100)
        print("跨数据集横向对比报告")
        print("="*100)
        
        # 收集所有实验结果到统一表格
        unified_results = []
        
        for dataset_name, results in all_results.items():
            # 处理激活函数实验结果
            if results['activation_comparison']:
                for result in results['activation_comparison']:
                    condition = result['experiment_name'].replace('激活函数_', '').upper()
                    unified_results.append({
                        'Dataset': dataset_name,
                        'Experiment_Type': 'Activation',
                        'Condition': condition,
                        'Accuracy': f"{result['accuracy']:.6f}",
                        'Precision': f"{result['precision']:.6f}",
                        'Recall': f"{result['recall']:.6f}",
                        'F1_Score': f"{result['f1_score']:.6f}",
                        'Training_Time(s)': f"{result['training_time']:.6f}",
                        'Test_Time(s)': f"{result['testing_time']:.6f}"
                    })
            
            # 处理注意力机制实验结果
            if results['attention_comparison']:
                for result in results['attention_comparison']:
                    condition = result['experiment_name'].replace('注意力机制_', '')
                    # 格式化条件名称
                    if 'cbam' in condition.lower():
                        condition = 'With Cbam'
                    elif 'without' in condition.lower() or 'no' in condition.lower():
                        condition = 'Without Attention'
                    else:
                        condition = condition.title()
                    
                    unified_results.append({
                        'Dataset': dataset_name,
                        'Experiment_Type': 'Attention',
                        'Condition': condition,
                        'Accuracy': f"{result['accuracy']:.6f}",
                        'Precision': f"{result['precision']:.6f}",
                        'Recall': f"{result['recall']:.6f}",
                        'F1_Score': f"{result['f1_score']:.6f}",
                        'Training_Time(s)': f"{result['training_time']:.6f}",
                        'Test_Time(s)': f"{result['testing_time']:.6f}"
                    })
        
        # 创建统一对比DataFrame
        if unified_results:
            unified_df = pd.DataFrame(unified_results)
            
            # 按数据集和实验类型排序
            unified_df = unified_df.sort_values(['Dataset', 'Experiment_Type', 'Condition'])
            
            print("\n统一跨数据集对比表格:")
            print("-" * 120)
            print(unified_df.to_string(index=False))
            
            # 找出各类实验的最佳表现
            print("\n\n最佳表现分析:")
            print("-" * 80)
            
            # 激活函数最佳表现
            activation_results = unified_df[unified_df['Experiment_Type'] == 'Activation']
            if not activation_results.empty:
                best_activation = activation_results.loc[activation_results['F1_Score'].astype(float).idxmax()]
                print(f"激活函数最佳: {best_activation['Dataset']} - {best_activation['Condition']} (F1: {best_activation['F1_Score']})")
            
            # 注意力机制最佳表现
            attention_results = unified_df[unified_df['Experiment_Type'] == 'Attention']
            if not attention_results.empty:
                best_attention = attention_results.loc[attention_results['F1_Score'].astype(float).idxmax()]
                print(f"注意力机制最佳: {best_attention['Dataset']} - {best_attention['Condition']} (F1: {best_attention['F1_Score']})")
            
            # 整体最佳表现
            overall_best = unified_df.loc[unified_df['F1_Score'].astype(float).idxmax()]
            print(f"整体最佳表现: {overall_best['Dataset']} - {overall_best['Experiment_Type']} - {overall_best['Condition']} (F1: {overall_best['F1_Score']})")
            
            # 保存统一对比结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unified_csv_path = os.path.join("TCN_outputs", f"unified_cross_dataset_comparison_{timestamp}.csv")
            
            # 确保输出目录存在
            os.makedirs("TCN_outputs", exist_ok=True)
            
            unified_df.to_csv(unified_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n统一跨数据集对比结果已保存到: {unified_csv_path}")
            
            # 保存详细分析
            unified_analysis = {
                'timestamp': timestamp,
                'datasets_compared': list(all_results.keys()),
                'unified_comparison': unified_results,
                'best_performances': {
                    'activation': best_activation.to_dict() if not activation_results.empty else None,
                    'attention': best_attention.to_dict() if not attention_results.empty else None,
                    'overall': overall_best.to_dict()
                },
                'detailed_results': all_results
            }
            
            unified_json_path = os.path.join("TCN_outputs", f"unified_cross_dataset_analysis_{timestamp}.json")
            with open(unified_json_path, 'w', encoding='utf-8') as f:
                json.dump(unified_analysis, f, indent=2, ensure_ascii=False, default=str)
            print(f"详细统一分析已保存到: {unified_json_path}")
            
            return unified_analysis
        else:
            print("\n警告: 没有找到任何实验结果进行对比！")
            return None

def main():
    """
    主函数
    """

    # 数据集控制配置 - 通过注释来启用/禁用数据集
    dataset_config = {
        "CTU": True,      # True: 启用, False: 禁用
        "USTC": True,     # True: 启用, False: 禁用
    }
    
    # 你也可以通过注释来快速控制:
    # dataset_config["CTU"] = False   # 禁用CTU数据集
    # dataset_config["USTC"] = False  # 禁用USTC数据集
    
    # 过滤启用的数据集
    enabled_datasets = [dataset for dataset, enabled in dataset_config.items() if enabled]
    
    if not enabled_datasets:
        print("警告: 没有启用任何数据集！")
        return
    
    print(f"启用的数据集: {enabled_datasets}")
    print(f"禁用的数据集: {[dataset for dataset, enabled in dataset_config.items() if not enabled]}")
    
    all_results = {}
    
    try:
        for dataset_name in enabled_datasets:
            print(f"\n{'='*100}")
            print(f"开始处理 {dataset_name} 数据集")
            print(f"{'='*100}")
            
            # 创建对比分析器
            comparator = ADCAETCNComparison(dataset_name)
            
            # 运行所有对比实验
            results = comparator.run_all_comparisons()
            all_results[dataset_name] = results
        
        # 生成跨数据集横向对比报告
        if len(all_results) > 1:
            print(f"\n{'='*100}")
            print("生成跨数据集横向对比报告")
            print(f"{'='*100}")
            
            # 创建一个临时的比较器实例来调用跨数据集对比方法
            temp_comparator = ADCAETCNComparison(list(enabled_datasets)[0])
            cross_dataset_analysis = temp_comparator.generate_cross_dataset_comparison(all_results)
        else:
            print("\n注意: 只有一个数据集启用，无法进行跨数据集对比。")
        
        # 实验总结
        print(f"\n{'='*100}")
        print("实验总结")
        print(f"{'='*100}")
        
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name} 数据集:")
            print(f"- 激活函数对比实验: {len(results['activation_comparison'])} 个")
            print(f"- 注意力机制对比实验: {len(results['attention_comparison'])} 个")
            print(f"- 使用设备: {results['device']}")
        
        print(f"\n处理的数据集总数: {len(all_results)}")
        
        if len(all_results) > 1:
            print("\n跨数据集横向对比报告已生成完成！")
        
    except Exception as e:
        print(f"实验过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()