import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix

from tcn_trainer import TCNTrainer
from tcn_data_loader import TCNDataLoader
from tcn_config import TCNConfig

class TCNFeatureClassifier:
    """使用PCA、KPCA、CAE和ADCAE特征进行TCN分类的处理器"""
    
    def __init__(self, dataset_name, config_file=None, feature_types=None):
        self.dataset_name = dataset_name
        
        # 设置默认特征类型或使用用户指定的
        self.feature_types = feature_types if feature_types is not None else ['PCA', 'KPCA', 'CAE', 'ADCAE']
        
        # 设置输出路径
        base_path = f"D:\\Python Project\\ADCAE\\{dataset_name}_result"
        self.output_path = os.path.join(base_path, "tcn_output_features")
        os.makedirs(self.output_path, exist_ok=True)
        
        # 初始化组件
        cache_path = os.path.join(base_path, "tcn_cache")
        self.data_loader = TCNDataLoader(dataset_name, cache_path)
        self.config_manager = TCNConfig(config_file)
    
    def train_and_evaluate_tcn(self, X_train, X_test, y_train, y_test, feature_type):
        """训练和评估TCN模型"""
        print(f"\n开始训练{feature_type} TCN模型...")
        
        # 获取配置
        config = self.config_manager.get_config(self.dataset_name, feature_type)
        
        # 获取特征维度和类别数
        input_size = X_train.shape[1]
        num_classes = len(np.unique(np.concatenate([y_train, y_test])))
        
        # 创建TCN训练器
        tcn_trainer = TCNTrainer(
            input_size=input_size,
            num_classes=num_classes,
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        )
        
        # 准备数据
        X_train_seq, X_val_seq, y_train_seq, y_val_seq = tcn_trainer.prepare_data(
            X_train, y_train, seq_len=config.get('seq_len', 10)
        )
        
        # 训练模型（添加噪声参数）
        training_start = datetime.now()
        train_results = tcn_trainer.train_model(
            X_train_seq, X_val_seq, y_train_seq, y_val_seq,
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            learning_rate=config.get('learning_rate', 0.001),
            patience=config.get('patience', 15),
            noise_std=config.get('noise_std', 0.0)  # 添加噪声参数
        )
        training_time = (datetime.now() - training_start).total_seconds()
        
        # 评估模型
        X_test_seq, y_test_seq = tcn_trainer.create_sequences(X_test, y_test, config.get('seq_len', 10))
        test_results = tcn_trainer.evaluate_model(X_test_seq, y_test_seq)
        
        # 创建统计信息
        stats = {
            'dataset_name': self.dataset_name,
            'feature_type': feature_type,
            'model_type': 'TCN',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': training_time,
            'input_size': input_size,
            'num_classes': num_classes,
            'train_samples': len(X_train_seq),
            'test_samples': len(X_test_seq),
            'config': config,
            'train_results': train_results,
            'test_results': test_results
        }
        
        # 保存模型和结果
        output_dir = os.path.join(self.output_path, f"tcn_output_{feature_type.lower()}")
        os.makedirs(output_dir, exist_ok=True)
        tcn_trainer.save_model_and_results(output_dir, test_results, self.dataset_name, feature_type, training_time)
        
        return stats
    
    def run_all_classifications(self):
        """运行指定特征类型的分类任务"""
        results = {}
        
        for feature_type in self.feature_types:
            try:
                print(f"\n{'='*80}")
                print(f"开始处理 {feature_type} 特征")
                print(f"{'='*80}")
                
                # 加载数据
                if feature_type == 'PCA':
                    X_train, X_test, y_train, y_test = self.data_loader.load_pca_data()
                elif feature_type == 'KPCA':
                    X_train, X_test, y_train, y_test = self.data_loader.load_kpca_data()
                elif feature_type == 'CAE':
                    X_train, X_test, y_train, y_test = self.data_loader.load_cae_data()
                elif feature_type == 'ADCAE':
                    X_train, X_test, y_train, y_test = self.data_loader.load_adcae_data()
                
                # 训练和评估
                stats = self.train_and_evaluate_tcn(X_train, X_test, y_train, y_test, feature_type)
                results[feature_type] = stats
                
            except Exception as e:
                print(f"处理 {feature_type} 特征时出错: {e}")
                results[feature_type] = None
        
        # 对比结果
        self.compare_results(results)
        return results
    
    def compare_results(self, results):
        """对比不同特征类型的结果"""
        # ... 结果对比逻辑 ...
        pass
    
    def plot_confusion_matrix(self, true_labels, predictions, feature_type):
        """绘制混淆矩阵"""
        # ... 混淆矩阵绘制逻辑 ...
        pass