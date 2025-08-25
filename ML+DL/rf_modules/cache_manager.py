import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
import json

class RFCacheManager:
    def __init__(self, base_path, dataset_name):
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.cache_dir = os.path.join(base_path, f"{dataset_name}_result", "rf_cache")
        self.data_cache_dir = os.path.join(self.cache_dir, "data")
        self.model_cache_dir = os.path.join(self.cache_dir, "models")
        
        # 创建缓存目录
        os.makedirs(self.data_cache_dir, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # 缓存索引文件
        self.cache_index_file = os.path.join(self.cache_dir, "cache_index.json")
        self.load_cache_index()
    
    def load_cache_index(self):
        """加载缓存索引"""
        if os.path.exists(self.cache_index_file):
            with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {
                "data_cache": {},
                "model_cache": {},
                "created_time": datetime.now().isoformat()
            }
    
    def save_cache_index(self):
        """保存缓存索引"""
        self.cache_index["last_updated"] = datetime.now().isoformat()
        with open(self.cache_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
    
    def _generate_data_hash(self, feature_type, data_path):
        """生成数据哈希值"""
        hash_input = f"{feature_type}_{data_path}_{self.dataset_name}"
        if os.path.exists(data_path):
            # 添加文件修改时间到哈希中
            mtime = os.path.getmtime(data_path)
            hash_input += f"_{mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _generate_model_hash(self, feature_type, model_params, data_hash):
        """生成模型哈希值"""
        # 将模型参数转换为字符串
        params_str = json.dumps(model_params, sort_keys=True)
        hash_input = f"{feature_type}_{params_str}_{data_hash}_{self.dataset_name}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def get_cached_data(self, feature_type, data_path):
        """获取缓存的数据"""
        data_hash = self._generate_data_hash(feature_type, data_path)
        
        if data_hash in self.cache_index["data_cache"]:
            cache_info = self.cache_index["data_cache"][data_hash]
            cache_file = os.path.join(self.data_cache_dir, f"{data_hash}.pkl")
            
            if os.path.exists(cache_file):
                print(f"从缓存加载{feature_type}数据: {cache_info['feature_type']}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        return None
    
    def cache_data(self, feature_type, data_path, data):
        """缓存数据"""
        data_hash = self._generate_data_hash(feature_type, data_path)
        cache_file = os.path.join(self.data_cache_dir, f"{data_hash}.pkl")
        
        print(f"缓存{feature_type}数据...")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # 更新缓存索引
        self.cache_index["data_cache"][data_hash] = {
            "feature_type": feature_type,
            "data_path": data_path,
            "cached_time": datetime.now().isoformat(),
            "data_shapes": {
                "X_train": data[0].shape,
                "X_test": data[1].shape,
                "y_train": data[2].shape,
                "y_test": data[3].shape
            }
        }
        self.save_cache_index()
        return data_hash
    
    def get_cached_model(self, feature_type, model_params, data_hash):
        """获取缓存的模型"""
        model_hash = self._generate_model_hash(feature_type, model_params, data_hash)
        
        if model_hash in self.cache_index["model_cache"]:
            cache_info = self.cache_index["model_cache"][model_hash]
            cache_file = os.path.join(self.model_cache_dir, f"{model_hash}.pkl")
            
            if os.path.exists(cache_file):
                print(f"从缓存加载{feature_type}模型: {cache_info['feature_type']}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        return None
    
    def cache_model(self, feature_type, model_params, data_hash, model, training_time, metrics):
        """缓存模型"""
        model_hash = self._generate_model_hash(feature_type, model_params, data_hash)
        cache_file = os.path.join(self.model_cache_dir, f"{model_hash}.pkl")
        
        print(f"缓存{feature_type}模型...")
        model_data = {
            "model": model,
            "training_time": training_time,
            "metrics": metrics,
            "feature_type": feature_type
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 更新缓存索引
        self.cache_index["model_cache"][model_hash] = {
            "feature_type": feature_type,
            "model_params": model_params,
            "data_hash": data_hash,
            "cached_time": datetime.now().isoformat(),
            "training_time": training_time,
            "metrics": metrics
        }
        self.save_cache_index()
        return model_hash
    
    def clear_cache(self, cache_type="all"):
        """清理缓存"""
        if cache_type in ["all", "data"]:
            for file in os.listdir(self.data_cache_dir):
                os.remove(os.path.join(self.data_cache_dir, file))
            self.cache_index["data_cache"] = {}
        
        if cache_type in ["all", "models"]:
            for file in os.listdir(self.model_cache_dir):
                os.remove(os.path.join(self.model_cache_dir, file))
            self.cache_index["model_cache"] = {}
        
        self.save_cache_index()
        print(f"已清理{cache_type}缓存")
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        data_count = len(self.cache_index["data_cache"])
        model_count = len(self.cache_index["model_cache"])
        
        # 计算缓存大小
        data_size = sum(os.path.getsize(os.path.join(self.data_cache_dir, f)) 
                       for f in os.listdir(self.data_cache_dir) if f.endswith('.pkl'))
        model_size = sum(os.path.getsize(os.path.join(self.model_cache_dir, f)) 
                        for f in os.listdir(self.model_cache_dir) if f.endswith('.pkl'))
        
        return {
            "data_cache_count": data_count,
            "model_cache_count": model_count,
            "data_cache_size_mb": data_size / (1024 * 1024),
            "model_cache_size_mb": model_size / (1024 * 1024),
            "total_size_mb": (data_size + model_size) / (1024 * 1024)
        }