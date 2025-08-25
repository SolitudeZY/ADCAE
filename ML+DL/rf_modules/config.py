import json
import os

class RFConfig:
    def __init__(self, config_path=None, dataset_name=None):
        self.dataset_name = dataset_name
        self.default_hyperparameters = {
            "PCA": {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': None,
                'random_state': 42,
                'n_jobs': 12
            },
            "KPCA": {
                'n_estimators': 100,
                'max_depth': 25,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': None,
                'random_state': 42,
                'n_jobs': 12
            },
            "CAE": {
                'n_estimators': 250,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': None,
                'bootstrap': True,
                'class_weight': None,
                'random_state': 42,
                'n_jobs': 12
            },
            "ADCAE": {
                'n_estimators': 300,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': None,
                'bootstrap': True,
                'class_weight': None,
                'random_state': 42,
                'n_jobs': 12
            }
        }
        
        self.hyperparameters = self.default_hyperparameters.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """从配置文件加载超参数"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                
            # 如果配置文件包含数据集特定的配置
            if 'hyperparameters' in loaded_config:
                config_hyperparams = loaded_config['hyperparameters']
                
                # 如果指定了数据集名称，优先使用数据集特定配置
                if self.dataset_name and self.dataset_name in config_hyperparams:
                    dataset_config = config_hyperparams[self.dataset_name]
                    # 合并数据集特定配置和默认配置
                    for feature_type in self.default_hyperparameters:
                        if feature_type in dataset_config:
                            self.hyperparameters[feature_type] = {
                                **self.default_hyperparameters[feature_type],
                                **dataset_config[feature_type]
                            }
                    print(f"已加载 {self.dataset_name} 数据集的专用配置")
                else:
                    # 如果没有数据集特定配置，使用通用配置
                    for feature_type in self.default_hyperparameters:
                        if feature_type in config_hyperparams:
                            self.hyperparameters[feature_type] = {
                                **self.default_hyperparameters[feature_type],
                                **config_hyperparams[feature_type]
                            }
                    print("已加载通用配置")
            else:
                # 旧格式兼容性
                self.hyperparameters = {**self.default_hyperparameters, **loaded_config}
                print("已加载配置（兼容模式）")
                
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            self.hyperparameters = self.default_hyperparameters
    
    def get_hyperparameters(self, feature_type):
        """获取指定特征类型的超参数"""
        params = self.hyperparameters.get(feature_type, self.default_hyperparameters[feature_type])
        
        # 显示使用的配置信息
        dataset_info = f" ({self.dataset_name})" if self.dataset_name else ""
        print(f"\n{feature_type}{dataset_info} 超参数配置:")
        for key, value in params.items():
            print(f"  {key}: {value}")
            
        return params
    
    def save_config(self, config_path):
        """保存当前配置到文件"""
        config_data = {
            "hyperparameters": {
                self.dataset_name: self.hyperparameters
            } if self.dataset_name else self.hyperparameters
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到: {config_path}")
        except Exception as e:
            print(f"保存配置失败: {e}")