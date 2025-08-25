import os
import time
import json
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from .model import ADCAE
from .preprocessor import BinaryDataPreprocessor

class ADCAETrainer:
    """ADCAE训练器"""
    
    def __init__(self, model_config, training_config, data_config):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = BinaryDataPreprocessor(
            file_length=data_config.file_length,
            image_size=data_config.image_size
        )
        
        self.training_history = {
            'loss': [],
            'metrics': [],
            'epoch_times': [],
            'five_metrics': []
        }
    
    def build_model(self):
        """构建ADCAE模型"""
        self.model = ADCAE(config=self.model_config).to(self.device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"ADCAE模型已构建:")
        print(f"  编码维度: {self.model_config.encoding_dim}")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数数: {trainable_params:,}")
        print(f"  使用设备: {self.device}")
    
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
    
    def train_model(self, X):
        """训练ADCAE模型"""
        print(f"开始训练ADCAE模型...")
        start_time = time.time()
        
        # 分割训练和验证集
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, X_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.training_config.batch_size, 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.training_config.batch_size, 
            shuffle=False
        )
        
        # 优化器和损失函数
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.training_config.learning_rate, 
            weight_decay=1e-4
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 25
        
        # 训练循环
        for epoch in range(self.training_config.epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            
            # 验证阶段
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 保存训练历史
            self._save_epoch_history(epoch, train_loss, val_loss, val_metrics, 
                                   epoch_time, optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            scheduler.step()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 输出详细信息
            self._print_epoch_info(epoch, train_loss, val_loss, val_metrics, 
                                 epoch_time, optimizer.param_groups[0]['lr'], 
                                 best_val_loss, patience_counter, early_stopping_patience)
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        training_time = time.time() - start_time
        print(f"\n训练完成，总耗时: {training_time:.2f}秒")
        print(f"平均每轮时间: {training_time/(epoch+1):.2f}秒")
        
        return training_time
    
    def _train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc='Training', leave=False, ncols=50)
        
        for batch_idx, (batch_data, _) in enumerate(train_pbar):
            optimizer.zero_grad()
            
            encoded, decoded = self.model(batch_data)
            loss = criterion(decoded, batch_data)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_original = []
        all_reconstructed = []
        
        val_pbar = tqdm(val_loader, desc='Validation', leave=False, ncols=50)
        
        with torch.no_grad():
            for batch_data, _ in val_pbar:
                encoded, decoded = self.model(batch_data)
                loss = criterion(decoded, batch_data)
                total_loss += loss.item()
                
                all_original.append(batch_data.cpu().numpy())
                all_reconstructed.append(decoded.cpu().numpy())
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # 计算重构指标
        original = np.vstack(all_original)
        reconstructed = np.vstack(all_reconstructed)
        metrics = self._calculate_reconstruction_metrics(original, reconstructed)
        
        return total_loss / len(val_loader), metrics
    
    def _calculate_reconstruction_metrics(self, original, reconstructed, threshold=0.05):
        """计算重构指标"""
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mse': mse,
            'mae': mae
        }
    
    def _save_epoch_history(self, epoch, train_loss, val_loss, val_metrics, epoch_time, lr):
        """保存epoch历史"""
        self.training_history['loss'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        self.training_history['metrics'].append({
            'epoch': epoch + 1,
            'train_mse': train_loss,  # 简化处理
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'learning_rate': lr
        })
        
        self.training_history['five_metrics'].append({
            'epoch': epoch + 1,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'epoch_time': epoch_time
        })
        
        self.training_history['epoch_times'].append({
            'epoch': epoch + 1,
            'time_seconds': epoch_time
        })
    
    def _print_epoch_info(self, epoch, train_loss, val_loss, val_metrics, epoch_time, 
                         lr, best_val_loss, patience_counter, early_stopping_patience):
        """打印epoch信息"""
        print(f"\n{'='*100}")
        print(f"Epoch [{epoch+1}/{self.training_config.epochs}] - 训练时间: {epoch_time:.2f}s")
        print(f"{'='*100}")
        print(f"损失指标:")
        print(f"  训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")
        print(f"  验证MSE: {val_metrics['mse']:.6f} | 验证MAE: {val_metrics['mae']:.6f}")
        print(f"五个关键指标:")
        print(f"  验证 - Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"其他信息:")
        print(f"  学习率: {lr:.2e} | 最佳验证损失: {best_val_loss:.6f}")
        print(f"  早停计数: {patience_counter}/{early_stopping_patience}")
        print(f"{'='*100}")
    
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
    
    def train_and_evaluate(self, train_dir, test_dir, output_dir):
        """完整的训练和评估流程"""
        # 构建模型
        self.build_model()
        
        # 准备训练数据
        train_X, train_labels, train_category_counts = self.prepare_data(train_dir)
        
        # 训练模型
        training_time = self.train_model(train_X)
        
        # 提取训练集特征
        train_encoded = self.encode_data(train_X)
        
        # 准备测试数据
        test_X, test_labels, test_category_counts = self.prepare_data(test_dir)
        
        # 提取测试集特征
        test_encoded = self.encode_data(test_X)
        
        # 保存结果
        results = self._save_results(
            output_dir, train_encoded, train_labels, train_category_counts,
            test_encoded, test_labels, test_category_counts, training_time
        )
        
        return results
    
    def _save_results(self, output_dir, train_encoded, train_labels, train_category_counts,
                     test_encoded, test_labels, test_category_counts, training_time):
        """保存训练结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(output_dir, 'adcae_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'data_config': self.data_config.__dict__
        }, model_path)
        
        # 保存训练集特征
        train_df = pd.DataFrame(train_encoded, 
                              columns=[f'feature_{i}' for i in range(train_encoded.shape[1])])
        train_df['label'] = train_labels
        train_df['split'] = 'train'
        train_path = os.path.join(output_dir, 'train_features.csv')
        train_df.to_csv(train_path, index=False)
        
        # 保存测试集特征
        test_df = pd.DataFrame(test_encoded, 
                             columns=[f'feature_{i}' for i in range(test_encoded.shape[1])])
        test_df['label'] = test_labels
        test_df['split'] = 'test'
        test_path = os.path.join(output_dir, 'test_features.csv')
        test_df.to_csv(test_path, index=False)
        
        # 保存训练历史
        for history_type, history_data in self.training_history.items():
            if history_data:
                history_df = pd.DataFrame(history_data)
                history_path = os.path.join(output_dir, f'{history_type}.csv')
                history_df.to_csv(history_path, index=False)
        
        # 保存统计信息
        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': training_time,
            'train_samples': len(train_encoded),
            'test_samples': len(test_encoded),
            'feature_dim': train_encoded.shape[1],
            'train_categories': train_category_counts,
            'test_categories': test_category_counts,
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__
        }
        
        stats_path = os.path.join(output_dir, 'experiment_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n实验结果已保存到: {output_dir}")
        print(f"- 模型文件: {model_path}")
        print(f"- 训练特征: {train_path}")
        print(f"- 测试特征: {test_path}")
        print(f"- 实验统计: {stats_path}")
        
        return stats