import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import threading

class RFTrainer:
    def __init__(self, model_builder, cache_manager=None):
        self.model_builder = model_builder
        self.cache_manager = cache_manager
        self.training_complete = False
    
    def train_model_with_cache(self, feature_type, model_params, data_hash, X_train, y_train_encoded):
        """使用缓存的模型训练"""
        if self.cache_manager:
            # 尝试从缓存加载模型
            cached_model_data = self.cache_manager.get_cached_model(feature_type, model_params, data_hash)
            if cached_model_data is not None:
                print(f"从缓存加载{feature_type}模型，跳过训练")
                return cached_model_data["model"], cached_model_data["training_time"]
        
        # 缓存未命中，训练新模型
        print(f"缓存未命中，开始训练{feature_type}模型...")
        rf_model = self.model_builder.build_model()
        rf_model.set_params(**model_params)
        
        return self.train_model(rf_model, X_train, y_train_encoded, feature_type)
    
    def train_model(self, rf_model, X_train, y_train_encoded, feature_type):
        """训练随机森林模型（带进度条）"""
        print(f"正在训练{feature_type}模型...")
        start_time = time.time()
        
        # 重置训练状态
        self.training_complete = False
        
        # 确保所有类别在训练数据中都存在
        unique_classes = np.unique(y_train_encoded)
        print(f"训练数据中的类别: {unique_classes}")
        
        # 创建进度条
        progress_bar = tqdm(total=100, desc=f"训练{feature_type}模型", unit="%")
        
        # 启动进度更新线程
        progress_thread = threading.Thread(
            target=self._update_progress, 
            args=(progress_bar, rf_model, start_time)
        )
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # 对于大数据集，使用改进的训练策略
            if X_train.shape[0] > 100000:
                print("检测到大数据集，使用优化训练策略...")
                
                # 手动计算类别权重
                if rf_model.class_weight == 'balanced' or isinstance(rf_model.class_weight, str):
                    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_encoded)
                    class_weight_dict = dict(zip(unique_classes, class_weights))
                    rf_model.set_params(class_weight=class_weight_dict)
                
                # 对于大数据集，调整参数
                if X_train.shape[0] > 150000:
                    print("数据集过大，调整参数以提高训练效率...")
                    original_n_estimators = rf_model.n_estimators
                    rf_model.set_params(n_estimators=min(50, original_n_estimators))
                    rf_model.set_params(max_samples=0.95)
            
            # 训练模型
            rf_model.fit(X_train, y_train_encoded)
            
        finally:
            # 标记训练完成
            self.training_complete = True
            
            # 等待进度线程完成（最多等待1秒）
            progress_thread.join(timeout=1.0)
            
            # 确保进度条完成
            progress_bar.n = 100
            progress_bar.refresh()
            progress_bar.close()
        
        training_time = time.time() - start_time
        print(f"\n{feature_type}模型训练完成，耗时: {training_time:.2f}秒")
        
        # 缓存训练好的模型
        if self.cache_manager:
            try:
                self.cache_manager.cache_model(feature_type, rf_model, training_time)
                print(f"缓存{feature_type}模型...")
            except Exception as e:
                print(f"缓存模型时出错: {e}")
        
        return rf_model, training_time
    
    def _update_progress(self, progress_bar, model, start_time):
        """更新训练进度（改进版）"""
        import time
        
        # 改进的进度更新逻辑
        update_count = 0
        while not self.training_complete and update_count < 1000:  # 最多更新1000次，防止无限循环
            time.sleep(0.05)  # 每50ms更新一次，减少CPU占用
            update_count += 1
            
            elapsed = time.time() - start_time
            
            # 基于时间的进度估算（更保守）
            if elapsed < 1.0:
                progress = min(10, int(elapsed * 10))  # 前1秒内最多10%
            elif elapsed < 5.0:
                progress = min(50, int(10 + (elapsed - 1) * 10))  # 1-5秒内到50%
            elif elapsed < 15.0:
                progress = min(90, int(50 + (elapsed - 5) * 4))  # 5-15秒内到90%
            else:
                progress = min(95, int(90 + (elapsed - 15) * 0.5))  # 15秒后缓慢增长到95%
            
            progress_bar.n = progress
            progress_bar.refresh()
            
            # 如果训练完成，退出循环
            if self.training_complete:
                break