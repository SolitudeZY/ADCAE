import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import hashlib

class DTRawDataProcessor:
    def __init__(self, dataset_name="CTU"):
        """
        初始化决策树原始数据处理器（数据集特定参数版本）
        
        Args:
            dataset_name: 数据集名称 ("CTU" 或 "USTC")
        """
        self.dataset_name = dataset_name
        
        # 根据数据集设置不同的参数
        if dataset_name == "CTU":
            # CTU数据集参数 - 平衡性能
            self.max_packet_length = 256  # 适度增加特征维度
            self.max_packets_per_session = 8   # 适度增加数据量
            self.noise_std = 0.12  # 降低噪声强度
        elif dataset_name == "USTC":
            # USTC数据集参数 - 保持当前性能
            self.max_packet_length = 256  # 保持较低特征维度
            self.max_packets_per_session = 10   # 保持较低数据量
            self.noise_std = 0.05  # 适中噪声强度
        else:
            # 默认参数
            self.max_packet_length = 256
            self.max_packets_per_session = 10
            self.noise_std = 0.05
            
        self.label_encoder = LabelEncoder()
        
        # 设置数据路径
        self.base_path = Path(f"d:/Python Project/ADCAE/pcap_files/Dataset_{dataset_name}")
        self.train_path = self.base_path / "Train"
        self.test_path = self.base_path / "Test"
        
        # 输出路径
        self.output_path = Path(f"d:/Python Project/ADCAE/{dataset_name}_result/dt_output_raw")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 缓存路径
        self.cache_path = Path(f"d:/Python Project/ADCAE/{dataset_name}_result/dt_cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        print(f"初始化决策树原始数据处理器 - 数据集: {dataset_name}")
        print(f"参数设置: max_packet_length={self.max_packet_length}, max_packets_per_session={self.max_packets_per_session}, noise_std={self.noise_std}")
        print(f"训练数据路径: {self.train_path}")
        print(f"测试数据路径: {self.test_path}")
        print(f"输出路径: {self.output_path}")
        print(f"缓存路径: {self.cache_path}")
    
    def _get_cache_key(self, data_path, max_files_per_class):
        """
        生成缓存键，基于数据路径、参数和文件修改时间
        
        Args:
            data_path: 数据路径
            max_files_per_class: 每个类别最大文件数
            
        Returns:
            str: 缓存键
        """
        # 获取所有pcap文件的修改时间
        file_times = []
        if data_path.exists():
            for class_dir in data_path.iterdir():
                if class_dir.is_dir():
                    pcap_files = list(class_dir.glob("*.pcap"))
                    if max_files_per_class is not None:
                        pcap_files = pcap_files[:max_files_per_class]
                    for pcap_file in pcap_files:
                        file_times.append(str(pcap_file.stat().st_mtime))
        
        # 创建缓存键
        cache_data = {
            'dataset_name': self.dataset_name,
            'data_path': str(data_path),
            'max_packet_length': self.max_packet_length,
            'max_packets_per_session': self.max_packets_per_session,
            'noise_std': self.noise_std,
            'max_files_per_class': max_files_per_class,
            'file_times': sorted(file_times)
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        
        return cache_key
    
    def _save_cache(self, cache_key, data_type, features, labels, class_names):
        """
        保存数据到缓存
        
        Args:
            cache_key: 缓存键
            data_type: 数据类型 ('train' 或 'test')
            features: 特征数据
            labels: 标签数据
            class_names: 类别名称
        """
        cache_file = self.cache_path / f"{cache_key}_{data_type}.pkl"
        
        cache_data = {
            'features': features,
            'labels': labels,
            'class_names': class_names,
            'timestamp': datetime.now().isoformat(),
            'dataset_name': self.dataset_name,
            'data_type': data_type
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"数据已缓存到: {cache_file}")
    
    def _load_cache(self, cache_key, data_type):
        """
        从缓存加载数据
        
        Args:
            cache_key: 缓存键
            data_type: 数据类型 ('train' 或 'test')
            
        Returns:
            tuple: (features, labels, class_names) 或 None
        """
        cache_file = self.cache_path / f"{cache_key}_{data_type}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                print(f"从缓存加载{data_type}数据: {cache_file}")
                print(f"缓存时间: {cache_data['timestamp']}")
                
                return cache_data['features'], cache_data['labels'], cache_data['class_names']
            except Exception as e:
                print(f"加载缓存失败: {e}")
                return None
        
        return None
    
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
        将字节数据转换为特征向量（使用数据集特定噪声）
        """
        if len(byte_data) == 0:
            return np.zeros(self.max_packet_length, dtype=np.float32)
        
        # 将字节转换为0-255的整数数组
        features = np.frombuffer(byte_data, dtype=np.uint8).astype(np.float32)
        
        # 使用数据集特定的噪声强度
        noise = np.random.normal(0, self.noise_std, features.shape)
        features = features + noise
        
        # 使用标准归一化方法
        features = features / 255.0
        
        # 截断或填充到固定长度
        if len(features) > self.max_packet_length:
            features = features[:self.max_packet_length]
        else:
            padding = np.zeros(self.max_packet_length - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        
        return features
    
    def build_dt_model(self, num_classes):
        """
        构建决策树模型（数据集特定参数）
        
        Args:
            num_classes: 类别数量
            
        Returns:
            DecisionTreeClassifier: 决策树模型
        """
        if self.dataset_name == "CTU":
            # CTU数据集 - 平衡性能的参数
            model = DecisionTreeClassifier(
                max_depth=7,   # 适度增加深度
                min_samples_split=18,  # 降低分割要求
                min_samples_leaf=9,    # 降低叶节点要求
                max_features='sqrt',
                random_state=42,
                criterion='gini',
                splitter='best',
                class_weight=None,
                min_impurity_decrease=0.0002  # 降低纯度要求
            )
        elif self.dataset_name == "USTC":
            # USTC数据集 - 保持当前性能的参数
            model = DecisionTreeClassifier(
                max_depth=6,   # 适中的树深度
                min_samples_split=20,  # 适中的分割要求
                min_samples_leaf=8,    # 适中的叶节点要求
                max_features='sqrt',   # 使用sqrt特征选择
                random_state=42,
                criterion='gini',
                splitter='best',
                class_weight=None,
                min_impurity_decrease=0.0002
            )
        else:
            # 默认参数
            model = DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=6,
                max_features='sqrt',
                random_state=42,
                criterion='gini',
                splitter='best',
                class_weight='balanced',
                min_impurity_decrease=0.0002
            )
        
        return model
    
    def load_dataset(self, data_path, max_files_per_class=None):
        """
        加载数据集（带缓存功能）
        
        Args:
            data_path: 数据路径
            max_files_per_class: 每个类别最大文件数（None表示不限制）
            
        Returns:
            tuple: (features, labels, class_names)
        """
        # 生成缓存键
        cache_key = self._get_cache_key(data_path, max_files_per_class)
        data_type = 'train' if 'Train' in str(data_path) else 'test'
        
        # 尝试从缓存加载
        cached_data = self._load_cache(cache_key, data_type)
        if cached_data is not None:
            features, labels, class_names = cached_data
            print(f"\n从缓存加载{data_type}数据完成:")
            print(f"特征形状: {features.shape}")
            print(f"标签数量: {len(labels)}")
            print(f"类别: {set(labels)}")
            return features, labels, class_names
        
        # 缓存不存在，重新加载数据
        print(f"\n缓存不存在，从 {data_path} 重新加载数据...")
        
        features = []
        labels = []
        class_names = []
        
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
        self._save_cache(cache_key, data_type, features, labels, class_names)
        
        return features, labels, class_names
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        训练决策树模型（添加时间统计）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            tuple: (model, training_info)
        """
        print("\n开始训练决策树模型...")
        
        # 构建模型
        num_classes = len(np.unique(y_train))
        model = self.build_dt_model(num_classes)
        
        print(f"模型参数: max_depth={model.max_depth}, min_samples_split={model.min_samples_split}")
        
        # 训练模型并记录时间
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 验证集评估并记录时间
        val_start_time = time.time()
        val_pred = model.predict(X_val)
        val_prediction_time = time.time() - val_start_time
        val_accuracy = accuracy_score(y_val, val_pred)
        
        training_info = {
            'training_time': float(training_time),
            'validation_prediction_time': float(val_prediction_time),
            'val_accuracy': float(val_accuracy),
            'tree_depth': int(model.get_depth()),
            'n_leaves': int(model.get_n_leaves()),
            'max_depth': int(model.max_depth),
            'min_samples_split': int(model.min_samples_split),
            'min_samples_leaf': int(model.min_samples_leaf)
        }
        
        print(f"\n训练完成，耗时: {training_time:.2f} 秒")
        print(f"验证预测耗时: {val_prediction_time:.4f} 秒")
        print(f"验证准确率: {val_accuracy:.4f}")
        print(f"树深度: {model.get_depth()}")
        print(f"叶节点数: {model.get_n_leaves()}")
        
        return model, training_info
    
    def evaluate_model(self, model, X_test, y_test, class_names):
        """
        评估模型（添加时间统计）
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            class_names: 类别名称
            
        Returns:
            dict: 评估结果
        """
        print("\n评估模型...")
        
        # 预测并记录时间
        test_start_time = time.time()
        y_pred = model.predict(X_test)
        test_prediction_time = time.time() - test_start_time
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n测试结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"测试预测耗时: {test_prediction_time:.4f} 秒")
        print(f"平均每样本预测时间: {test_prediction_time/len(X_test)*1000:.4f} 毫秒")
        
        # 分类报告
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        print(f"\n分类报告:\n{report}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'test_prediction_time': float(test_prediction_time),
            'avg_prediction_time_per_sample': float(test_prediction_time/len(X_test)),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': [int(x) for x in y_pred.tolist()],
            'true_labels': [int(x) for x in y_test.tolist()]
        }
    
    def save_results(self, model, training_info, test_results):
        """
        保存结果（包含时间指标）
        
        Args:
            model: 训练好的模型
            training_info: 训练信息
            test_results: 测试结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = self.output_path / f"dt_raw_model_{self.dataset_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        print(f"模型已保存到: {model_path}")
        
        # 保存测试结果
        results_data = {
            'dataset': self.dataset_name,
            'timestamp': timestamp,
            'max_packet_length': self.max_packet_length,
            'max_packets_per_session': self.max_packets_per_session,
            'training_info': training_info,
            'test_results': test_results
        }
        
        results_path = self.output_path / f"dt_raw_results_{self.dataset_name}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # 保存性能指标CSV（包含时间指标）
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                      'Training_Time(s)', 'Test_Prediction_Time(s)', 
                      'Avg_Prediction_Time_Per_Sample(s)'],
            'Value': [
                test_results['accuracy'],
                test_results['precision'],
                test_results['recall'],
                test_results['f1_score'],
                training_info['training_time'],
                test_results['test_prediction_time'],
                test_results['avg_prediction_time_per_sample']
            ]
        })
        
        metrics_path = self.output_path / f"dt_raw_metrics_{self.dataset_name}_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"结果已保存到: {self.output_path}")
    
    def plot_confusion_matrix(self, test_results, class_names, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            test_results: 测试结果
            class_names: 类别名称
            save_path: 保存路径
        """
        cm = np.array(test_results['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.dataset_name} Dataset (Decision Tree)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"混淆矩阵图已保存到: {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, model, save_path=None):
        """
        绘制特征重要性
        
        Args:
            model: 训练好的模型
            save_path: 保存路径
        """
        feature_importance = model.feature_importances_
        
        # 只显示前20个最重要的特征
        top_indices = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_importance)), top_importance)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Index')
        plt.title(f'Top 20 Feature Importance - {self.dataset_name} Dataset (Decision Tree)')
        plt.yticks(range(len(top_importance)), [f'Feature {i}' for i in top_indices])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"特征重要性图已保存到: {save_path}")
        
        plt.close()
    
    def clear_cache(self):
        """
        清除所有缓存文件
        """
        if self.cache_path.exists():
            cache_files = list(self.cache_path.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            print(f"已清除 {len(cache_files)} 个缓存文件")
        else:
            print("缓存目录不存在")
    
    def list_cache_files(self):
        """
        列出所有缓存文件
        """
        if self.cache_path.exists():
            cache_files = list(self.cache_path.glob("*.pkl"))
            if cache_files:
                print(f"\n找到 {len(cache_files)} 个缓存文件:")
                for cache_file in cache_files:
                    file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
                    mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    print(f"  {cache_file.name} - {file_size:.2f}MB - {mod_time}")
            else:
                print("没有找到缓存文件")
        else:
            print("缓存目录不存在")
    
    def process_dataset(self, max_files_per_class=None):
        """
        处理完整的数据集（数据集特定参数版本）
        
        Args:
            max_files_per_class: 每个类别最大文件数（None表示不限制）
        """
        print(f"\n开始处理 {self.dataset_name} 数据集...")
        
        # 加载训练数据
        X_train, y_train, class_names = self.load_dataset(self.train_path, max_files_per_class)
        
        # 加载测试数据
        X_test, y_test, _ = self.load_dataset(self.test_path, max_files_per_class)
        
        # 编码标签
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 分割训练集和验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
        )
        
        print(f"\n数据分割完成:")
        print(f"训练集: {X_train_final.shape}")
        print(f"验证集: {X_val.shape}")
        print(f"测试集: {X_test.shape}")
        
        # 训练模型
        model, training_info = self.train_model(X_train_final, y_train_final, X_val, y_val)
        
        # 评估模型
        test_results = self.evaluate_model(model, X_test, y_test_encoded, class_names)
        
        # 保存结果
        self.save_results(model, training_info, test_results)
        
        # 绘制图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 混淆矩阵
        cm_path = self.output_path / f"dt_raw_confusion_matrix_{self.dataset_name}_{timestamp}.png"
        self.plot_confusion_matrix(test_results, class_names, cm_path)
        
        # 特征重要性
        fi_path = self.output_path / f"dt_raw_feature_importance_{self.dataset_name}_{timestamp}.png"
        self.plot_feature_importance(model, fi_path)
        
        print(f"\n{self.dataset_name} 数据集处理完成！")
        
        return model, training_info, test_results

# 主程序
if __name__ == "__main__":
    print("决策树原始数据处理器 (数据集特定参数版本)")
    print("=" * 50)
    
    # 处理CTU数据集 - 使用提升性能的参数
    print("处理CTU数据集...")
    ctu_processor = DTRawDataProcessor(dataset_name="CTU")
    
    try:
        ctu_model, ctu_training_info, ctu_results = ctu_processor.process_dataset(
            max_files_per_class=None
        )
        print("CTU数据集处理成功！")
    except Exception as e:
        print(f"CTU数据集处理失败: {e}")
    
    # 处理USTC数据集 - 使用保持当前性能的参数
    print("\n处理USTC数据集...")
    ustc_processor = DTRawDataProcessor(dataset_name="USTC")
    
    try:
        ustc_model, ustc_training_info, ustc_results = ustc_processor.process_dataset(
            max_files_per_class=None
        )
        print("USTC数据集处理成功！")
    except Exception as e:
        print(f"USTC数据集处理失败: {e}")
    
    print("\n所有数据集处理完成！")