import json
import os

class TCNConfig:
    """TCN配置管理器"""
    
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.default_config = {
            'seq_len': 10,
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_channels': [50, 50, 50, 50],
            'kernel_size': 7,
            'dropout': 0.1,
            'patience': 15
        }
    
    def get_config(self, dataset_name=None, feature_type=None):
        """获取配置参数"""
        if self.config_file and os.path.exists(self.config_file):
            return self._load_from_file(dataset_name, feature_type)
        else:
            return self._get_default_config(dataset_name, feature_type)
    
    def _load_from_file(self, dataset_name, feature_type):
        """从配置文件加载"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            if dataset_name and feature_type:
                return config_data.get(dataset_name, {}).get(feature_type, self.default_config)
            elif dataset_name:
                return config_data.get(dataset_name, self.default_config)
            else:
                return self.default_config
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            return self.default_config
    
    def _get_default_config(self, dataset_name, feature_type):
        """获取默认配置"""
        config = self.default_config.copy()
        
        # 根据数据集调整配置
        if dataset_name == 'USTC':
            config.update({
                'seq_len': 15,
                'epochs': 80,
                'batch_size': 64,
                'learning_rate': 0.0005,
                'num_channels': [64, 64, 64, 64]
            })
        elif dataset_name == 'CTU':
            config.update({
                'seq_len': 10,
                'epochs': 60,
                'batch_size': 32,
                'learning_rate': 0.001
            })
        
        # 根据特征类型调整配置
        if feature_type == 'PCA':
            config.update({
                'seq_len': 8,
                'num_channels': [32, 32, 32],
                'learning_rate': 0.002
            })
        elif feature_type == 'ADCAE':
            config.update({
                'seq_len': 15,
                'num_channels': [64, 64, 64, 64, 64],
                'learning_rate': 0.0008
            })
        
        return config