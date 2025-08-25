import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import time
import hashlib
import pickle
import psutil
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

class RFFeatureClassifier:
    """
    使用PCA、KPCA、CAE和ADCAE特征进行随机森林分类的处理器
    """
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
        # 初始化标签编码器
        self.label_encoder = LabelEncoder()
        
        # 设置数据路径
        base_path = f"D:\\Python Project\\ADCAE\\{dataset_name}_result"
        self.pca_data_path = os.path.join(base_path, "pca_output", "pca_features")
        self.kpca_data_path = os.path.join(base_path, "kpca_output", "kpca_features")
        self.cae_data_path = os.path.join(base_path, "cae_output", "cae_features")
        self.adcae_data_path = os.path.join(base_path, "adcae_features", "activation_comparison", "elu")
        
        # 设置输出路径和缓存路径
        self.output_path = os.path.join(base_path, "rf_output_features")
        self.cache_path = os.path.join(base_path, "rf_cache")  # 添加这一行
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)  # 添加这一行
        
        # 获取数据集特定的超参数
        self.hyperparameters = self._get_dataset_hyperparameters()
        
    def _get_dataset_hyperparameters(self):
        # 获取CPU核心数，但限制使用量
        import multiprocessing
        max_cores = multiprocessing.cpu_count()
        # 使用75%的核心，为系统保留一些资源
        n_jobs = max(1, int(max_cores * 0.75))
        
        if self.dataset_name == "USTC":
            return {
                'PCA': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': None,  # 改为None，手动处理类别权重
                    'random_state': 42,
                    'n_jobs': n_jobs,
                    'max_samples': 0.8  # 添加这个参数减少内存使用
                },
                'KPCA': {
                    'n_estimators': 180,
                    'max_depth': 18,
                    'min_samples_split': 4,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': n_jobs
                },
                'CAE': {
                    'n_estimators': 100,
                    'max_depth': 25,
                    'min_samples_split': 4,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': n_jobs
                },
                'ADCAE': {
                    'n_estimators': 100,
                    'max_depth': 25,
                    'min_samples_split': 4,
                    'min_samples_leaf': 2,
                    'max_features': None,
                    'bootstrap': True,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': n_jobs
                }
            }
        elif self.dataset_name == "CTU":
            return {
                'PCA': {
                    'n_estimators': 250,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': None,
                    'random_state': 42,
                    'n_jobs': n_jobs
                },
                'KPCA': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': n_jobs
                },
                'CAE': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': None,
                    'random_state': 42,
                    'n_jobs': n_jobs
                },
                'ADCAE': {
                    'n_estimators': 100,
                    'max_depth': 25,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': n_jobs
                }
            }
        else:
            # 默认配置
            return {
                "PCA": {
                    "n_estimators": 100,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1
                },
                "KPCA": {
                    "n_estimators": 100,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1
                },
                "CAE": {
                    "n_estimators": 100,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1
                },
                "ADCAE": {
                    "n_estimators": 100,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1
                }
            }
    
    def _get_cache_key(self, feature_type):
        """生成缓存键"""
        cache_string = f"{self.dataset_name}_{feature_type}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _load_cached_data(self, feature_type):
        """加载缓存的数据"""
        cache_key = self._get_cache_key(feature_type)
        cache_file = os.path.join(self.cache_path, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            print(f"从缓存加载{feature_type}数据...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_cached_data(self, data, feature_type):
        """保存数据到缓存"""
        cache_key = self._get_cache_key(feature_type)
        cache_file = os.path.join(self.cache_path, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"{feature_type}数据已缓存")
    
    def load_pca_data(self):
        """
        加载PCA特征数据（从CSV文件）
        """
        print("加载PCA特征数据...")
        
        # 从CSV文件加载特征数据
        csv_path = os.path.join(self.pca_data_path, f"encoded_features_{self.dataset_name}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"PCA特征文件不存在: {csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        print(f"PCA数据形状: {df.shape}")
        print(f"PCA数据列名: {list(df.columns)}")
        
        # 排除非特征列
        columns_to_drop = ['label', 'dataset', 'split']  # 可能的非特征列
        
        # 检查实际存在的列
        actual_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        feature_columns = [col for col in df.columns if col not in actual_columns_to_drop]
        
        print(f"特征列数量: {len(feature_columns)}")
        print(f"排除的列: {actual_columns_to_drop}")
        
        # 分离特征和标签
        X = df[feature_columns].values
        
        # 获取标签
        if 'label' in df.columns:
            y = df['label'].values
        else:
            raise ValueError("PCA特征文件中未找到标签列 'label'")
        
        # 检查是否有split列来区分训练集和测试集
        if 'split' in df.columns:
            train_mask = df['split'] == 'train'
            test_mask = df['split'] == 'test'
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
        else:
            # 如果没有split列，使用train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        print(f"PCA训练集形状: {X_train.shape}")
        print(f"PCA测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        # 缓存数据
        data = (X_train, X_test, y_train, y_test)
        self._save_cached_data(data, 'PCA')
        
        return X_train, X_test, y_train, y_test
    
    def load_kpca_data(self):
        """
        加载KPCA特征数据（从CSV文件）
        """
        print("加载KPCA特征数据...")
        
        # 从CSV文件加载特征数据
        csv_path = os.path.join(self.kpca_data_path, f"encoded_features_{self.dataset_name}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"KPCA特征文件不存在: {csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        print(f"KPCA数据形状: {df.shape}")
        print(f"KPCA数据列名: {list(df.columns)}")
        
        # 排除非特征列
        columns_to_drop = ['label', 'dataset', 'split']  # 可能的非特征列
        
        # 检查实际存在的列
        actual_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        feature_columns = [col for col in df.columns if col not in actual_columns_to_drop]
        
        print(f"特征列数量: {len(feature_columns)}")
        print(f"排除的列: {actual_columns_to_drop}")
        
        # 分离特征和标签
        X = df[feature_columns].values
        
        # 获取标签
        if 'label' in df.columns:
            y = df['label'].values
        else:
            raise ValueError("KPCA特征文件中未找到标签列 'label'")
        
        # 检查是否有split列来区分训练集和测试集
        if 'split' in df.columns:
            train_mask = df['split'] == 'train'
            test_mask = df['split'] == 'test'
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
        else:
            # 如果没有split列，使用train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        print(f"KPCA训练集形状: {X_train.shape}")
        print(f"KPCA测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def load_cae_data(self):
        """
        加载CAE特征数据（从CSV文件）
        """
        print("加载CAE特征数据...")
        
        # 从CSV文件加载训练和测试特征
        train_path = os.path.join(self.cae_data_path, "train_features.csv")
        test_path = os.path.join(self.cae_data_path, "test_features.csv")
        
        # 检查文件是否存在
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"CAE训练特征文件不存在: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"CAE测试特征文件不存在: {test_path}")
        
        # 读取CSV文件
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"训练数据形状: {train_df.shape}")
        print(f"测试数据形状: {test_df.shape}")
        
        # 排除非特征列
        columns_to_drop = ['label', 'dataset']  # 添加dataset列到排除列表
        
        # 检查实际存在的列
        actual_columns_to_drop = [col for col in columns_to_drop if col in train_df.columns]
        feature_columns = [col for col in train_df.columns if col not in actual_columns_to_drop]
        
        print(f"特征列数量: {len(feature_columns)}")
        print(f"排除的列: {actual_columns_to_drop}")
        
        # 分离特征和标签
        X_train = train_df[feature_columns].values
        X_test = test_df[feature_columns].values
        
        # 获取标签
        if 'label' in train_df.columns:
            y_train = train_df['label'].values
            y_test = test_df['label'].values
        else:
            raise ValueError("CAE特征文件中未找到标签列 'label'")
        
        print(f"CAE训练集形状: {X_train.shape}")
        print(f"CAE测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def load_adcae_data(self):
        """
        加载ADCAE特征数据
        """
        print("加载ADCAE特征数据...")
        
        # 加载训练和测试特征
        train_path = os.path.join(self.adcae_data_path, "train_features.csv")
        test_path = os.path.join(self.adcae_data_path, "test_features.csv")
        
        # 读取CSV文件
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 打印列信息以便调试
        print(f"训练数据列名: {list(train_df.columns)}")
        print(f"训练数据形状: {train_df.shape}")
        
        # 检查并移除非数值列
        # 通常需要排除的列：'label', 'dataset', 可能还有索引列等
        columns_to_drop = []
        
        # 检查每一列的数据类型
        for col in train_df.columns:
            if train_df[col].dtype == 'object':  # 字符串类型列
                if col.lower() in ['label', 'dataset', 'index', 'id']:
                    columns_to_drop.append(col)
                else:
                    # 检查是否包含字符串值
                    sample_values = train_df[col].dropna().head(10)
                    if any(isinstance(val, str) for val in sample_values):
                        print(f"发现字符串列: {col}, 样本值: {sample_values.tolist()}")
                        columns_to_drop.append(col)
        
        print(f"将要删除的列: {columns_to_drop}")
        
        # 分离特征和标签
        if 'label' in train_df.columns:
            y_train = train_df['label'].copy()  # 使用copy()避免警告
            y_test = test_df['label'].copy()
        else:
            raise ValueError("未找到标签列 'label'")
        
        # 删除非特征列
        X_train = train_df.drop(columns=columns_to_drop, errors='ignore').copy()
        X_test = test_df.drop(columns=columns_to_drop, errors='ignore').copy()
        
        # 确保所有特征列都是数值类型
        for col in X_train.columns:
            X_train.loc[:, col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test.loc[:, col] = pd.to_numeric(X_test[col], errors='coerce')
        
        # 检查是否有NaN值
        if X_train.isnull().any().any():
            print("警告: 特征数据中存在NaN值，将用0填充")
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
        
        # 转换为numpy数组
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values
        
        print(f"ADCAE训练集形状: {X_train.shape}")
        print(f"ADCAE测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def encode_labels(self, y_train, y_test):
        """
        编码标签
        """
        # 如果标签是字符串，进行编码
        if y_train.dtype == 'object' or isinstance(y_train[0], str):
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            return y_train_encoded, y_test_encoded
        else:
            return y_train, y_test
    
    def build_rf_model(self, feature_type="CAE"):
        """
        构建随机森林模型
        
        Args:
            feature_type: 特征类型 ("PCA", "KPCA", "CAE" 或 "ADCAE")
        """
        # 获取对应的超参数
        params = self.hyperparameters[feature_type]
        
        print(f"\n{self.dataset_name}数据集 - {feature_type}特征随机森林超参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        rf_model = RandomForestClassifier(**params)
        
        return rf_model
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_type):
        """
        训练和评估随机森林模型（完全修复版本）
        """
        print(f"\n开始训练{feature_type}随机森林模型...")
        
        # 监控系统资源
        print(f"当前CPU使用率: {psutil.cpu_percent()}%")
        print(f"当前内存使用率: {psutil.virtual_memory().percent}%")
        
        try:
            # 编码标签
            y_train_encoded, y_test_encoded = self.encode_labels(y_train, y_test)
            
            # 构建模型
            rf_model = self.build_rf_model(feature_type)
            
            # 训练模型（修复warm_start警告）
            print("正在训练模型...")
            start_time = time.time()
            
            # 如果数据集很大，考虑使用增量训练
            if X_train.shape[0] > 100000:  # 大数据集
                print("检测到大数据集，使用增量训练...")
                
                # 手动计算类别权重以避免警告
                if rf_model.class_weight == 'balanced':
                    unique_classes = np.unique(y_train_encoded)
                    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_encoded)
                    class_weight_dict = dict(zip(unique_classes, class_weights))
                    rf_model.set_params(class_weight=class_weight_dict)
                
                # 修复warm_start问题：每次增加n_estimators
                original_n_estimators = rf_model.n_estimators
                rf_model.set_params(warm_start=True, n_estimators=0)  # 从0开始
                
                # 分批训练
                batch_size = 50000
                n_batches = (len(X_train) - 1) // batch_size + 1
                estimators_per_batch = max(1, original_n_estimators // n_batches)
                
                for i in range(0, len(X_train), batch_size):
                    batch_num = i // batch_size + 1
                    end_idx = min(i + batch_size, len(X_train))
                    X_batch = X_train[i:end_idx]
                    y_batch = y_train_encoded[i:end_idx]
                    
                    # 增加估计器数量
                    current_estimators = batch_num * estimators_per_batch
                    if batch_num == n_batches:  # 最后一批使用原始数量
                        current_estimators = original_n_estimators
                    
                    rf_model.set_params(n_estimators=current_estimators)
                    rf_model.fit(X_batch, y_batch)
                    print(f"完成批次 {batch_num}/{n_batches}，当前估计器数量: {current_estimators}")
            else:
                rf_model.fit(X_train, y_train_encoded)
            
            training_time = time.time() - start_time
            
            # 预测
            start_time = time.time()
            y_pred = rf_model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # 确保预测结果是一维数组
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            if y_test_encoded.ndim > 1:
                y_test_encoded = y_test_encoded.flatten()
            
            # 计算指标
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            # 平均每样本预测时间
            avg_prediction_time = prediction_time / len(y_test_encoded)
            
            print(f"\n{feature_type}随机森林模型评估结果:")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            print(f"训练时间: {training_time:.4f}秒")
            print(f"测试预测时间: {prediction_time:.4f}秒")
            print(f"平均每样本预测时间: {avg_prediction_time:.6f}秒")
            
            # 获取所有可能的类别
            all_train_classes = np.unique(y_train_encoded)
            all_test_classes = np.unique(y_test_encoded)
            all_pred_classes = np.unique(y_pred)
            
            # 合并所有类别并排序
            all_possible_classes = np.unique(np.concatenate([all_train_classes, all_test_classes, all_pred_classes]))
            
            print(f"训练集类别: {all_train_classes}")
            print(f"测试集类别: {all_test_classes}")
            print(f"预测类别: {all_pred_classes}")
            print(f"所有类别: {all_possible_classes}")
            
            # 创建target_names，确保与实际类别数量匹配
            if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                # 只使用实际存在的类别名称
                target_names = []
                for class_idx in all_possible_classes:
                    if class_idx < len(self.label_encoder.classes_):
                        target_names.append(self.label_encoder.classes_[class_idx])
                    else:
                        target_names.append(f"Class_{class_idx}")
            else:
                target_names = [f"Class_{i}" for i in all_possible_classes]
            
            print(f"目标类别名称: {target_names}")
            print(f"类别名称数量: {len(target_names)}，实际类别数量: {len(all_possible_classes)}")
            
            # 分类报告 - 使用labels参数确保一致性
            try:
                report = classification_report(
                    y_test_encoded, y_pred, 
                    labels=all_possible_classes,
                    target_names=target_names, 
                    zero_division=0,
                    output_dict=False  # 返回字符串格式
                )
                print(f"\n分类报告:\n{report}")
            except Exception as e:
                print(f"生成详细分类报告时出错: {e}")
                print("使用简化的分类报告...")
                try:
                    report = classification_report(y_test_encoded, y_pred, zero_division=0)
                    print(f"\n分类报告:\n{report}")
                except Exception as e2:
                    print(f"生成简化分类报告也失败: {e2}")
                    report = "分类报告生成失败"
            
            # 混淆矩阵 - 使用labels参数确保一致性
            try:
                cm = confusion_matrix(y_test_encoded, y_pred, labels=all_possible_classes)
                print(f"混淆矩阵形状: {cm.shape}")
            except Exception as e:
                print(f"生成混淆矩阵时出错: {e}")
                # 使用默认方式生成混淆矩阵
                try:
                    cm = confusion_matrix(y_test_encoded, y_pred)
                    print(f"使用默认混淆矩阵，形状: {cm.shape}")
                    # 重新调整target_names
                    target_names = [f"Class_{i}" for i in range(cm.shape[0])]
                except Exception as e2:
                    print(f"生成默认混淆矩阵也失败: {e2}")
                    # 创建一个空的混淆矩阵
                    cm = np.zeros((len(all_possible_classes), len(all_possible_classes)))
            
            # 创建结果字典
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'avg_prediction_time': avg_prediction_time,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'feature_type': feature_type,
                'hyperparameters': self.hyperparameters[feature_type]
            }
            
            # 保存结果
            self.save_results(rf_model, results)
            
            # 调用混淆矩阵绘制方法
            try:
                self.plot_confusion_matrix(cm, target_names, feature_type)
            except Exception as e:
                print(f"绘制混淆矩阵时出错: {e}")
            
            return results
            
        except Exception as e:
            print(f"训练和评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_confusion_matrix(self, cm, target_names, feature_type):
        """
        绘制混淆矩阵（作为类方法）
        """
        try:
            # 确保target_names与混淆矩阵维度匹配
            if len(target_names) != cm.shape[0]:
                print(f"警告：类别名称数量({len(target_names)})与混淆矩阵维度({cm.shape[0]})不匹配")
                target_names = [f"Class_{i}" for i in range(cm.shape[0])]
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.title(f'{feature_type} Random Forest - Confusion Matrix ({self.dataset_name})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.output_path, f"rf_{feature_type.lower()}_confusion_matrix_{self.dataset_name}_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"混淆矩阵图已保存到: {plot_path}")
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")
            plt.close()  # 确保关闭图形

        # 只返回结果字典，不返回模型
        return results
    
    def save_results(self, model, results):
        """
        保存模型和结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_type = results['feature_type']
        
        # 保存模型
        model_path = os.path.join(self.output_path, f"rf_{feature_type.lower()}_model_{self.dataset_name}_{timestamp}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存到: {model_path}")
        
        # 保存结果为JSON
        import json
        results_path = os.path.join(self.output_path, f"rf_{feature_type.lower()}_results_{self.dataset_name}_{timestamp}.json")
        
        # 转换numpy数组为列表以便JSON序列化
        json_results = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in results.items()}
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 保存性能指标CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                       'Training_Time', 'Prediction_Time', 'Avg_Prediction_Time_Per_Sample'],
            'Value': [
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['f1_score'],
                results['training_time'],
                results['prediction_time'],
                results['avg_prediction_time']
            ]
        })
        
        metrics_path = os.path.join(self.output_path, f"rf_{feature_type.lower()}_metrics_{self.dataset_name}_{timestamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"结果已保存到: {self.output_path}")
    
    def run_all_classifications(self, enabled_features=None):
        """
        运行所有特征类型的分类任务（优化版本）
        
        Args:
            enabled_features: 要运行的特征类型列表，如 ["PCA", "KPCA", "CAE", "ADCAE"]
                             如果为None，则运行所有特征类型
        """
        results = {}
        
        # 默认运行所有特征类型
        all_feature_types = ["PCA", "KPCA", "CAE", "ADCAE"]
        
        # 如果指定了enabled_features，则只运行指定的特征类型
        if enabled_features is not None:
            feature_types = [ft for ft in enabled_features if ft in all_feature_types]
            if not feature_types:
                print("警告：没有有效的特征类型被指定，将运行所有特征类型")
                feature_types = all_feature_types
        else:
            feature_types = all_feature_types
        
        print(f"将要运行的特征类型: {feature_types}")
        
        for feature_type in feature_types:
            print("=" * 60)
            print(f"{self.dataset_name} - {feature_type}特征随机森林分类")
            print("=" * 60)
            
            try:
                # 根据特征类型加载数据
                if feature_type == "PCA":
                    X_train, X_test, y_train, y_test = self.load_pca_data()
                elif feature_type == "KPCA":
                    X_train, X_test, y_train, y_test = self.load_kpca_data()
                elif feature_type == "CAE":
                    X_train, X_test, y_train, y_test = self.load_cae_data()
                elif feature_type == "ADCAE":
                    X_train, X_test, y_train, y_test = self.load_adcae_data()
                
                # 训练和评估
                feature_results = self.train_and_evaluate(X_train, X_test, y_train, y_test, feature_type)
                results[feature_type] = feature_results
                
                # 释放内存
                del X_train, X_test, y_train, y_test
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"{feature_type}分类失败: {e}")
                results[feature_type] = None
        
        # 对比结果
        self.compare_results(results)
        return results
    
    def compare_results(self, results_summary):
        """
        对比不同特征类型的结果
        """
        print("\n" + "=" * 60)
        print(f"{self.dataset_name} - 结果对比")
        print("=" * 60)
        
        # 过滤掉None值的结果
        valid_results = {k: v for k, v in results_summary.items() if v is not None}
        
        if not valid_results:
            print("没有有效的分类结果可供对比")
            return
        
        try:
            comparison_df = pd.DataFrame(valid_results).T
            
            # 确保所需的列存在
            required_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'prediction_time']
            available_columns = [col for col in required_columns if col in comparison_df.columns]
            
            if available_columns:
                print(comparison_df[available_columns])
            else:
                print("结果数据格式异常，无法显示对比表")
                print("可用的列名:", list(comparison_df.columns))
            
            # 保存对比结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = os.path.join(self.output_path, f"rf_feature_comparison_{self.dataset_name}_{timestamp}.csv")
            comparison_df.to_csv(comparison_path)
            print(f"\n对比结果已保存到: {comparison_path}")
            
        except Exception as e:
            print(f"创建对比表时出错: {e}")
            print("原始结果数据:")
            for feature_type, result in valid_results.items():
                if result:
                    print(f"{feature_type}: 准确率={result.get('accuracy', 'N/A'):.4f}")


def main():
    """
    主函数 - 处理USTC和CTU两个数据集
    """
    datasets = ["USTC", "CTU"]
    
    # ========== 配置区域 ==========
    # 你可以在这里配置每个数据集要运行的特征类型
    # 注释掉不想运行的特征类型即可
    
    dataset_config = {
        "USTC": [
            # "PCA",      # PCA特征
            # "KPCA",     # KPCA特征
            "CAE",      # CAE特征
            "ADCAE"     # ADCAE特征
        ],
        "CTU": [
            # "PCA",      # PCA特征
            "KPCA",     # KPCA特征
            # "CAE",    # 注释掉不运行CAE特征
            "ADCAE"     # ADCAE特征
        ]
    }
    
    # 或者你也可以使用这种更简洁的方式：
    # dataset_config = {
    #     "USTC": ["KPCA", "ADCAE"],  # 只运行KPCA和ADCAE
    #     "CTU": ["PCA", "KPCA"]      # 只运行PCA和KPCA
    # }
    
    # ========== 配置区域结束 ==========
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"开始处理 {dataset_name} 数据集")
        print(f"{'='*80}")
        
        # 获取该数据集的配置
        enabled_features = dataset_config.get(dataset_name, None)
        
        if enabled_features:
            print(f"{dataset_name} 数据集配置的特征类型: {enabled_features}")
        else:
            print(f"{dataset_name} 数据集将运行所有特征类型")
        
        try:
            # 创建分类器实例
            classifier = RFFeatureClassifier(dataset_name=dataset_name)
            
            # 运行指定的分类任务
            results = classifier.run_all_classifications(enabled_features=enabled_features)
            all_results[dataset_name] = results
            
            print(f"\n{dataset_name} 数据集分类任务完成！")
            
        except Exception as e:
            print(f"处理 {dataset_name} 数据集时出错: {e}")
            continue
    
    # 跨数据集结果对比
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("跨数据集结果对比")
        print(f"{'='*80}")
        
        # 创建跨数据集对比表
        comparison_data = []
        for dataset, features in all_results.items():
            if features:  # 确保features不为空
                for feature_type, metrics in features.items():
                    if metrics is not None:  # 确保metrics不为None
                        comparison_data.append({
                            'Dataset': dataset,
                            'Feature_Type': feature_type,
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1_Score': metrics['f1_score'],
                            'Training_Time': metrics['training_time'],
                            'Prediction_Time': metrics['prediction_time'],
                            'Avg_Prediction_Time': metrics['avg_prediction_time']
                        })
        
        if comparison_data:
            cross_comparison_df = pd.DataFrame(comparison_data)
            print(cross_comparison_df)
            
            # 保存跨数据集对比结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cross_comparison_path = f"D:\\Python Project\\ADCAE\\results\\rf_cross_dataset_comparison\\rf_cross_dataset_comparison_{timestamp}.csv"
            os.makedirs(os.path.dirname(cross_comparison_path), exist_ok=True)
            cross_comparison_df.to_csv(cross_comparison_path, index=False)
            print(f"\n跨数据集对比结果已保存到: {cross_comparison_path}")
        else:
            print("没有有效的结果数据可供跨数据集对比")
    
    print(f"\n{'='*80}")
    print("所有随机森林分类任务完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()