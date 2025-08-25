import pandas as pd
import numpy as np
import os
import pickle
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class TCNDataLoader:
    """TCN数据加载器"""
    
    def __init__(self, dataset_name, cache_path):
        self.dataset_name = dataset_name
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
        
        # 设置数据路径
        base_path = f"D:\\Python Project\\ADCAE\\{dataset_name}_result"
        self.pca_data_path = os.path.join(base_path, "pca_output", "pca_features")
        self.kpca_data_path = os.path.join(base_path, "kpca_output", "kpca_features")
        self.cae_data_path = os.path.join(base_path, "cae_output", "cae_features")
        # 修改ADCAE路径为with_cbam
        self.adcae_data_path = os.path.join(base_path, "adcae_features", "attention_comparison", "with_cbam")
    
    def _get_cache_key(self, data_path, feature_type):
        """生成缓存键"""
        path_hash = hashlib.md5(f"{data_path}_{feature_type}".encode()).hexdigest()[:16]
        return f"{self.dataset_name}_{feature_type.lower()}_{path_hash}"
    
    def _load_cached_data(self, cache_key):
        """加载缓存数据"""
        cache_file = os.path.join(self.cache_path, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载缓存失败: {e}")
        return None
    
    def _save_cached_data(self, cache_key, data):
        """保存数据到缓存"""
        cache_file = os.path.join(self.cache_path, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"数据已缓存: {cache_file}")
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def _load_standard_features(self, data_path, feature_type):
        """加载标准特征数据（PCA/KPCA/CAE/ADCAE）"""
        cache_key = self._get_cache_key(data_path, feature_type)
        cached_data = self._load_cached_data(cache_key)
        
        if cached_data is not None:
            print(f"从缓存加载{feature_type}数据")
            return cached_data
        
        print(f"加载{feature_type}特征数据...")
        
        try:
            # 对于PCA/KPCA，优先尝试加载CSV文件
            if feature_type in ['PCA', 'KPCA']:
                csv_file = os.path.join(data_path, f"encoded_features_{self.dataset_name}.csv")
                if os.path.exists(csv_file):
                    print(f"从CSV文件加载{feature_type}数据: {csv_file}")
                    # 读取CSV文件，指定第一行为列标题
                    data = pd.read_csv(csv_file, header=0)
                    
                    # 移除非特征列（label和dataset列）
                    feature_columns = [col for col in data.columns if col not in ['label', 'dataset']]
                    X = data[feature_columns].values
                    y = data['label'].values
                    
                    # 分割数据
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    raise FileNotFoundError(f"未找到{feature_type}的CSV特征文件: {csv_file}")
            
            # 对于CAE/ADCAE，使用分离的训练和测试文件
            else:
                train_file = os.path.join(data_path, "train_features.csv")
                test_file = os.path.join(data_path, "test_features.csv")
                
                if os.path.exists(train_file) and os.path.exists(test_file):
                    print(f"从分离文件加载{feature_type}数据: {train_file}, {test_file}")
                    
                    # 读取训练和测试数据
                    train_df = pd.read_csv(train_file, header=0)
                    test_df = pd.read_csv(test_file, header=0)
                    
                    # 对于ADCAE，需要特殊处理可能存在的非数值列
                    if feature_type.lower() == 'adcae':
                        # 识别数值列
                        numeric_columns = []
                        for col in train_df.columns:
                            if col.lower() not in ['label', 'dataset', 'split', 'type']:
                                try:
                                    # 尝试转换为数值类型
                                    pd.to_numeric(train_df[col], errors='raise')
                                    numeric_columns.append(col)
                                except (ValueError, TypeError):
                                    print(f"跳过非数值列: {col}")
                                    continue
                        
                        # 只保留数值特征列和标签列
                        feature_columns = [col for col in numeric_columns if col.lower() != 'label']
                        if 'label' in train_df.columns:
                            feature_columns.append('label')
                        
                        train_df = train_df[feature_columns]
                        test_df = test_df[feature_columns]
                    
                    # 分离特征和标签
                    if 'label' in train_df.columns:
                        X_train = train_df.drop('label', axis=1).values
                        y_train = train_df['label'].values
                        X_test = test_df.drop('label', axis=1).values
                        y_test = test_df['label'].values
                    else:
                        # 如果没有label列，假设最后一列是标签
                        X_train = train_df.iloc[:, :-1].values
                        y_train = train_df.iloc[:, -1].values
                        X_test = test_df.iloc[:, :-1].values
                        y_test = test_df.iloc[:, -1].values
                else:
                    raise FileNotFoundError(f"未找到{feature_type}特征文件: {train_file}, {test_file}")
            
            # 数据预处理
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            
            # 标签编码
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            result = (X_train, X_test, y_train_encoded, y_test_encoded)
            
            # 缓存数据
            self._save_cached_data(cache_key, result)
            
            print(f"{feature_type}数据加载完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
            print(f"标签范围: {np.min(y_train_encoded)} - {np.max(y_train_encoded)}")
            return result
            
        except Exception as e:
            print(f"加载{feature_type}数据时出错: {e}")
            raise

    def load_pca_data(self):
        """加载PCA特征数据"""
        return self._load_standard_features(self.pca_data_path, "PCA")
    
    def load_kpca_data(self):
        """加载KPCA特征数据"""
        return self._load_standard_features(self.kpca_data_path, "KPCA")
    
    def load_cae_data(self):
        """加载CAE特征数据"""
        return self._load_standard_features(self.cae_data_path, "CAE")
    
    def load_adcae_data(self):
        """加载ADCAE特征数据"""
        return self._load_standard_features(self.adcae_data_path, "ADCAE")