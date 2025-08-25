import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .cache_manager import RFCacheManager
from tqdm import tqdm

class RFDataLoader:
    def __init__(self, dataset_name, base_path, use_cache=True):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.label_encoder = LabelEncoder()
        self.use_cache = use_cache
        
        # 初始化缓存管理器
        if self.use_cache:
            self.cache_manager = RFCacheManager(base_path, dataset_name)
        
        # 设置数据路径
        self.pca_data_path = os.path.join(base_path, f"{dataset_name}_result", "pca_output", "pca_features")
        self.kpca_data_path = os.path.join(base_path, f"{dataset_name}_result", "kpca_output", "kpca_features")
        self.cae_data_path = os.path.join(base_path, f"{dataset_name}_result", "cae_output", "cae_features")
        self.adcae_data_path = os.path.join(base_path, f"{dataset_name}_result", "adcae_features", "activation_comparison", "elu")
    
    def load_pca_data(self):
        return self._load_with_cache("PCA", self.pca_data_path, self._load_standard_data)
    
    def load_kpca_data(self):
        return self._load_with_cache("KPCA", self.kpca_data_path, self._load_kpca_data)
    
    def load_cae_data(self):
        return self._load_with_cache("CAE", self.cae_data_path, self._load_standard_data)
    
    def load_adcae_data(self):
        return self._load_with_cache("ADCAE", self.adcae_data_path, self._load_adcae_data)
    
    def _load_with_cache(self, feature_type, data_path, load_func):
        """使用缓存加载数据"""
        if self.use_cache:
            # 尝试从缓存加载
            cached_data = self.cache_manager.get_cached_data(feature_type, data_path)
            if cached_data is not None:
                return cached_data
        
        # 缓存未命中，加载数据
        print(f"缓存未命中，正在加载{feature_type}数据...")
        if feature_type in ["PCA", "CAE"]:
            data = load_func(data_path, feature_type)
        else:
            data = load_func()
        
        # 缓存数据
        if self.use_cache:
            self.cache_manager.cache_data(feature_type, data_path, data)
        
        return data
    
    def _load_standard_data(self, data_path, feature_type):
        print(f"加载{feature_type}特征数据...")
        
        train_path = os.path.join(data_path, "train_features.csv")
        test_path = os.path.join(data_path, "test_features.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"{feature_type}特征文件不存在: {data_path}")
        
        # 使用进度条加载数据
        with tqdm(total=2, desc=f"加载{feature_type}数据") as pbar:
            train_df = pd.read_csv(train_path)
            pbar.update(1)
            test_df = pd.read_csv(test_path)
            pbar.update(1)
        
        print(f"训练数据形状: {train_df.shape}")
        print(f"测试数据形状: {test_df.shape}")
        
        # 处理特征和标签
        feature_columns = [col for col in train_df.columns if col not in ['label', 'dataset']]
        print(f"特征列数量: {len(feature_columns)}")
        
        X_train = train_df[feature_columns].values
        X_test = test_df[feature_columns].values
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print(f"{feature_type}训练集形状: {X_train.shape}")
        print(f"{feature_type}测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def _load_adcae_data(self):
        print("加载ADCAE特征数据...")
        
        train_path = os.path.join(self.adcae_data_path, "train_features.csv")
        test_path = os.path.join(self.adcae_data_path, "test_features.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"ADCAE特征文件不存在: {self.adcae_data_path}\n训练文件: {train_path}\n测试文件: {test_path}")
        
        print(f"从以下路径加载ADCAE数据: {self.adcae_data_path}")
        
        # 使用进度条加载ADCAE数据
        with tqdm(total=4, desc="加载ADCAE数据") as pbar:
            train_df = pd.read_csv(train_path)
            pbar.update(1)
            test_df = pd.read_csv(test_path)
            pbar.update(1)
            
            print(f"训练数据形状: {train_df.shape}")
            
            # 数据预处理
            columns_to_drop = []
            for col in train_df.columns:
                if train_df[col].dtype == 'object':
                    if col.lower() in ['label', 'dataset', 'index', 'id', 'split']:
                        columns_to_drop.append(col)
                    else:
                        sample_values = train_df[col].dropna().head(10)
                        if any(isinstance(val, str) for val in sample_values):
                            columns_to_drop.append(col)
            pbar.update(1)
            
            # 分离特征和标签
            if 'label' in train_df.columns:
                y_train = train_df['label'].copy()
                y_test = test_df['label'].copy()
            else:
                raise ValueError("未找到标签列 'label'")
            
            # 删除非特征列
            X_train = train_df.drop(columns=columns_to_drop, errors='ignore').copy()
            X_test = test_df.drop(columns=columns_to_drop, errors='ignore').copy()
            
            # 数值化处理
            for col in X_train.columns:
                X_train.loc[:, col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test.loc[:, col] = pd.to_numeric(X_test[col], errors='coerce')
            
            # 填充NaN值
            if X_train.isnull().any().any():
                print("警告: 特征数据中存在NaN值，将用0填充")
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)
            
            # 转换为numpy数组
            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values
            pbar.update(1)
        
        print(f"ADCAE训练集形状: {X_train.shape}")
        print(f"ADCAE测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def encode_labels(self, y_train, y_test):
        """编码标签"""
        if y_train.dtype == 'object' or isinstance(y_train[0], str):
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            return y_train_encoded, y_test_encoded
        else:
            return y_train, y_test
    
    def _load_kpca_data(self):
        print("加载KPCA特征数据...")
        
        # 优先尝试加载分离的npy文件
        train_features_path = os.path.join(self.kpca_data_path, f"train_encoded_features_{self.dataset_name}.npy")
        test_features_path = os.path.join(self.kpca_data_path, f"test_encoded_features_{self.dataset_name}.npy")
        train_labels_path = os.path.join(self.kpca_data_path, f"train_labels_{self.dataset_name}.npy")
        test_labels_path = os.path.join(self.kpca_data_path, f"test_labels_{self.dataset_name}.npy")
        
        # 检查npy文件是否都存在
        npy_files_exist = all(os.path.exists(f) for f in [train_features_path, test_features_path, train_labels_path, test_labels_path])
        
        if npy_files_exist:
            print(f"使用npy格式加载KPCA数据: {self.kpca_data_path}")
            try:
                X_train = np.load(train_features_path)
                X_test = np.load(test_features_path)
                y_train = np.load(train_labels_path)
                y_test = np.load(test_labels_path)
                
                print(f"KPCA训练集形状: {X_train.shape}")
                print(f"KPCA测试集形状: {X_test.shape}")
                print(f"类别数量: {len(np.unique(y_train))}")
                
                return X_train, X_test, y_train, y_test
                
            except Exception as e:
                print(f"加载npy文件失败: {str(e)}，尝试加载CSV文件")
        
        # 如果npy文件不存在或加载失败，尝试加载CSV文件
        csv_path = os.path.join(self.kpca_data_path, f"encoded_features_{self.dataset_name}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"KPCA特征文件不存在: {self.kpca_data_path}\n期望的文件: {csv_path} 或 npy文件组合")
        
        print(f"使用CSV格式加载KPCA数据: {csv_path}")
        
        try:
            # 加载CSV文件
            df = pd.read_csv(csv_path)
            print(f"CSV数据形状: {df.shape}")
            print(f"CSV列名: {list(df.columns)[:10]}...")  # 显示前10个列名
            
            # 检查是否有split列来区分训练和测试数据
            if 'split' in df.columns:
                train_df = df[df['split'] == 'train'].copy()
                test_df = df[df['split'] == 'test'].copy()
            elif 'dataset_split' in df.columns:
                train_df = df[df['dataset_split'] == 'train'].copy()
                test_df = df[df['dataset_split'] == 'test'].copy()
            else:
                # 如果没有split列，按比例分割（前80%作为训练，后20%作为测试）
                split_idx = int(len(df) * 0.8)
                train_df = df.iloc[:split_idx].copy()
                test_df = df.iloc[split_idx:].copy()
                print(f"警告: 未找到split列，按8:2比例分割数据")
            
            # 提取特征和标签
            feature_columns = [col for col in train_df.columns if col not in ['label', 'split', 'dataset_split', 'index', 'id']]
            
            X_train = train_df[feature_columns].values
            X_test = test_df[feature_columns].values
            y_train = train_df['label'].values if 'label' in train_df.columns else np.zeros(len(train_df))
            y_test = test_df['label'].values if 'label' in test_df.columns else np.zeros(len(test_df))
            
            print(f"KPCA训练集形状: {X_train.shape}")
            print(f"KPCA测试集形状: {X_test.shape}")
            print(f"特征列数量: {len(feature_columns)}")
            print(f"类别数量: {len(np.unique(y_train))}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise RuntimeError(f"加载KPCA CSV数据时出错: {str(e)}")