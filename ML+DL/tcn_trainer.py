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

from tcn_models import TCNClassifier

class TCNTrainer:
    """TCN训练器"""
    def __init__(self, input_size, num_classes, num_channels=[25, 25, 25, 25], 
                 kernel_size=7, dropout=0.2, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TCNClassifier(input_size, num_channels, num_classes, kernel_size, dropout).to(self.device)
        self.num_classes = num_classes
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'five_metrics': [],
            'epoch_times': []
        }
        
        print(f"TCN模型已创建，使用设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    # 在TCNTrainer类中添加噪声方法
    def add_noise_to_data(self, X, noise_std=0.01):
        """为训练数据添加高斯噪声"""
        if noise_std > 0:
            noise = torch.randn_like(X) * noise_std
            return X + noise
        return X
    
    def prepare_data(self, features, labels, seq_len=10, test_size=0.2, random_state=42):
        """准备时序数据"""
        print("准备时序数据...")
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # 创建时序数据
        X_seq, y_seq = self.create_sequences(features, encoded_labels, seq_len)
        
        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=test_size, random_state=random_state, stratify=y_seq
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"类别数量: {len(self.label_encoder.classes_)}")
        print(f"类别映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return X_train, X_val, y_train, y_val
    
    def create_sequences(self, features, labels, seq_len):
        """创建时序序列"""
        X_seq = []
        y_seq = []
        
        # 按类别分组创建序列
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_features = features[label_indices]
            
            # 为每个类别创建序列
            for i in range(len(label_features) - seq_len + 1):
                X_seq.append(label_features[i:i+seq_len])
                y_seq.append(label)
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=100, batch_size=32, 
                   learning_rate=0.001, patience=15, noise_std=0.0):
        """训练模型"""
        print(f"\n开始训练TCN模型...")
        print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        if noise_std > 0:
            print(f"噪声标准差: {noise_std}")
        
        # 转换为PyTorch张量并调整维度
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).permute(0, 2, 1).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # 训练变量
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print("开始训练循环")
        print(f"{'='*80}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 使用tqdm显示训练进度
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
            
            for batch_X, batch_y in train_pbar:
                # 添加噪声（仅在训练时）
                if noise_std > 0:
                    batch_X = self.add_noise_to_data(batch_X, noise_std)
                
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
                
                # 更新进度条
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_preds = []
            all_val_labels = []
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            
            with torch.no_grad():
                for batch_X, batch_y in val_pbar:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(batch_y.cpu().numpy())
                    
                    # 更新进度条
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
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
            
            # 计算详细指标
            val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
            
            # 记录历史
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
            current_lr = optimizer.param_groups[0]['lr']
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 修改train_model方法中的输出部分
            # 将详细的epoch输出替换为简化版本
            # 简化的epoch输出
            if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}] - Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}% | Val: Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}% | Time: {epoch_time:.1f}s")
            
            # 早停
            if patience_counter >= patience:
                print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            print(f"\n已加载最佳模型（验证损失: {best_val_loss:.4f}）")
        
        total_training_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"训练完成！总训练时间: {total_training_time:.2f}秒")
        print(f"{'='*80}")
        
        return {
            'training_time': total_training_time,
            'best_val_loss': best_val_loss,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
            'epochs_completed': epoch + 1
        }
    
    def evaluate_model(self, X_test, y_test, batch_size=32):
        """评估模型"""
        print("\n开始模型评估...")
        
        # 转换为PyTorch张量并调整维度
        X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1).to(self.device)  # 转换为 (batch, features, seq_len)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        self.model.eval()
        
        # 创建数据加载器
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        # 使用tqdm显示测试进度
        test_pbar = tqdm(test_loader, desc='Testing', leave=False)
        
        with torch.no_grad():
            for batch_X, batch_y in test_pbar:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
                # 更新进度条
                test_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        test_time = time.time() - start_time
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"\n{'='*80}")
        print(f"测试结果")
        print(f"{'='*80}")
        print(f"测试时间: {test_time:.2f}秒")
        print(f"测试损失: {avg_test_loss:.4f}")
        print(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall):    {recall:.4f}")
        print(f"F1分数 (F1-Score):  {f1:.4f}")
        
        # 详细分类报告
        if hasattr(self, 'label_encoder'):
            class_names = self.label_encoder.classes_
            print(f"\n详细分类报告:")
            print(classification_report(all_labels, all_preds, target_names=[str(c) for c in class_names], zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_loss': avg_test_loss,
            'test_time': test_time,
            'predictions': all_preds,
            'true_labels': all_labels,
            'classification_report': classification_report(all_labels, all_preds, output_dict=True, zero_division=0) if len(set(all_labels)) > 1 else None
        }
    
    # 在文件开头添加导入
    import numpy as np
    
    # 修改save_model_and_results方法中的JSON保存部分
    def save_model_and_results(self, output_dir, test_results, dataset_name, feature_type, training_time):
        """保存模型和结果"""
        print(f"\n保存模型和结果到: {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = os.path.join(output_dir, f"tcn_model_{feature_type}_{dataset_name}_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.tcn.network[0].conv1.in_channels,
                'num_classes': self.num_classes,
                'num_channels': [layer.conv1.out_channels for layer in self.model.tcn.network if hasattr(layer, 'conv1')],
            },
            'label_encoder': self.label_encoder if hasattr(self, 'label_encoder') else None
        }, model_path)
        
        # 修复JSON序列化问题 - 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
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
            return obj
        
        # 保存训练历史
        history_path = os.path.join(output_dir, f"training_history_{feature_type}_{dataset_name}_{timestamp}.json")
        cleaned_history = convert_numpy_types(self.training_history)
        with open(history_path, 'w') as f:
            json.dump(cleaned_history, f, indent=2)
        
        # 保存测试结果
        results_data = {
            'dataset_name': dataset_name,
            'feature_type': feature_type,
            'model_type': 'TCN',
            'timestamp': timestamp,
            'training_time_seconds': float(training_time),
            'test_results': convert_numpy_types(test_results),
            'model_path': model_path,
            'history_path': history_path
        }
        
        results_path = os.path.join(output_dir, f"test_results_{feature_type}_{dataset_name}_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # 保存CSV格式的指标
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training_Time', 'Test_Time'],
            'Value': [
                float(test_results['accuracy']),
                float(test_results['precision']),
                float(test_results['recall']),
                float(test_results['f1_score']),
                float(training_time),
                float(test_results['test_time'])
            ]
        })
        
        metrics_path = os.path.join(output_dir, f"metrics_{feature_type}_{dataset_name}_{timestamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"模型已保存: {model_path}")
        print(f"训练历史已保存: {history_path}")
        print(f"测试结果已保存: {results_path}")
        print(f"指标CSV已保存: {metrics_path}")
        
        return {
            'model_path': model_path,
            'history_path': history_path,
            'results_path': results_path,
            'metrics_path': metrics_path
        }