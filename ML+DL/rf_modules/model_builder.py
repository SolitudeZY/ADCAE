from sklearn.ensemble import RandomForestClassifier

class RFModelBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_model(self, feature_type, y_train_encoded=None):
        """构建随机森林模型"""
        params = self.config.get_hyperparameters(feature_type)
        
        print(f"\n{feature_type}特征随机森林超参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 创建模型
        rf_model = RandomForestClassifier(**params)
        
        return rf_model