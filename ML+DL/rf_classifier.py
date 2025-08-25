import os
import sys
import psutil
import gc
from datetime import datetime

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'rf_modules'))

from rf_modules.config import RFConfig
from rf_modules.data_loader import RFDataLoader
from rf_modules.model_builder import RFModelBuilder
from rf_modules.trainer import RFTrainer
from rf_modules.evaluator import RFEvaluator
from rf_modules.visualizer import RFVisualizer
from rf_modules.utils import RFUtils

class RFClassifier:
    def __init__(self, dataset_name, base_path="d:/Python Project/ADCAE", config_path=None):
        self.dataset_name = dataset_name
        self.base_path = base_path
        
        # 默认配置文件路径
        if config_path is None:
            config_path = os.path.join(base_path, "config", "rf_config.json")
        
        # 初始化各个模块（传递数据集名称给配置）
        self.config = RFConfig(config_path, dataset_name)
        self.data_loader = RFDataLoader(dataset_name, base_path)
        self.model_builder = RFModelBuilder(self.config)
        self.trainer = RFTrainer(self.model_builder)
        self.evaluator = RFEvaluator(self.data_loader.label_encoder)
        self.visualizer = RFVisualizer(dataset_name, base_path)
        self.utils = RFUtils(dataset_name, base_path)
        
        print(f"初始化 {dataset_name} 数据集的随机森林分类器")
        if hasattr(self.data_loader, 'cache_manager'):
            print(f"缓存目录: {self.data_loader.cache_manager.cache_dir}")
    
    def _print_cache_stats(self, cache_manager):
        """打印缓存统计信息"""
        try:
            stats = cache_manager.get_cache_stats()
            print(f"数据缓存: {stats['data_cache_count']} 个文件, {stats['data_cache_size_mb']:.2f} MB")
            print(f"模型缓存: {stats['model_cache_count']} 个文件, {stats['model_cache_size_mb']:.2f} MB")
            print(f"总缓存大小: {stats['total_size_mb']:.2f} MB")
        except Exception as e:
            print(f"获取缓存统计失败: {e}")
    
    def run_classification(self, feature_type):
        """运行单个特征类型的分类任务"""
        print("=" * 60)
        print(f"{self.dataset_name} - {feature_type}特征随机森林分类")
        print("=" * 60)
        
        try:
            # 显示缓存统计（如果有缓存管理器）
            if hasattr(self.data_loader, 'cache_manager'):
                self._print_cache_stats(self.data_loader.cache_manager)
            
            # 1. 加载数据（带缓存和进度条）
            print(f"\n正在加载 {feature_type} 数据...")
            if feature_type == "PCA":
                X_train, X_test, y_train, y_test = self.data_loader.load_pca_data()
            elif feature_type == "KPCA":
                X_train, X_test, y_train, y_test = self.data_loader.load_kpca_data()
            elif feature_type == "CAE":
                X_train, X_test, y_train, y_test = self.data_loader.load_cae_data()
            elif feature_type == "ADCAE":
                X_train, X_test, y_train, y_test = self.data_loader.load_adcae_data()
            else:
                raise ValueError(f"不支持的特征类型: {feature_type}")
            
            print(f"数据加载完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
            
            # 2. 编码标签
            y_train_encoded, y_test_encoded = self.data_loader.encode_labels(y_train, y_test)
            
            # 3. 监控系统资源
            print(f"当前CPU使用率: {psutil.cpu_percent()}%")
            print(f"当前内存使用率: {psutil.virtual_memory().percent}%")
            
            # 4. 构建模型
            rf_model = self.model_builder.build_model(feature_type, y_train_encoded)
            
            # 5. 训练模型（带进度条）
            print(f"\n开始训练 {feature_type} 模型...")
            rf_model, training_time = self.trainer.train_model(rf_model, X_train, y_train_encoded, feature_type)
            print(f"训练完成，耗时: {training_time:.2f}秒")
            
            # 6. 评估模型
            results = self.evaluator.evaluate_model(rf_model, X_test, y_test_encoded, feature_type)
            if results is None:
                return None
            
            # 添加训练时间
            results['training_time'] = training_time
            results['hyperparameters'] = self.config.get_hyperparameters(feature_type)
            
            # 7. 保存结果
            self.utils.save_results(rf_model, results)
            
            # 8. 可视化
            # 在run_classification方法中，找到并删除以下代码块：
            # if results['confusion_matrix'] is not None:
            #     self.visualizer.plot_confusion_matrix(
            #         results['confusion_matrix'],
            #         results['target_names'], 
            #         feature_type
            #     )
            
            # 修改为：
            print(f"\n{feature_type}模型训练和评估完成")
            print(f"准确率: {results['accuracy']:.4f}")
            print(f"F1分数: {results['f1_score']:.4f}")
            
            # 9. 释放内存
            del X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded
            gc.collect()
            
            return results
            
        except Exception as e:
            print(f"{feature_type}分类失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_classifications(self, enabled_features=None):
        """运行所有特征类型的分类任务"""
        results = {}
        
        # 默认运行所有特征类型
        all_feature_types = ["PCA", "KPCA", "CAE", "ADCAE"]
        
        if enabled_features is not None:
            feature_types = [ft for ft in enabled_features if ft in all_feature_types]
            if not feature_types:
                print("警告：没有有效的特征类型被指定，将运行所有特征类型")
                feature_types = all_feature_types
        else:
            feature_types = all_feature_types
        
        print(f"将要运行的特征类型: {feature_types}")
        
        for feature_type in feature_types:
            result = self.run_classification(feature_type)
            results[feature_type] = result
        
        # 对比结果
        self.utils.compare_results(results)
        
        # 显示最终缓存统计（如果有缓存管理器）
        if hasattr(self.data_loader, 'cache_manager'):
            print("\n" + "=" * 60)
            print("缓存使用统计")
            print("=" * 60)
            self._print_cache_stats(self.data_loader.cache_manager)
        
        return results
    
    def clear_cache(self):
        """清理缓存"""
        if hasattr(self.data_loader, 'cache_manager'):
            self.data_loader.cache_manager.clear_cache()
            print("缓存已清理")
        else:
            print("未启用缓存功能")
    
    def get_cache_info(self):
        """获取缓存信息"""
        if hasattr(self.data_loader, 'cache_manager'):
            return self.data_loader.cache_manager.get_cache_stats()
        else:
            return None

def main():
    """主函数"""
    datasets = ["USTC", "CTU"]
    
    # 配置每个数据集要运行的特征类型
    dataset_config = {
        "USTC": [
            # "PCA",      # PCA特征
            # "KPCA",     # KPCA特征
            # "CAE",      # CAE特征
            "ADCAE"     # ADCAE特征
        ],
        "CTU": [
            # "PCA",      # PCA特征
            # "KPCA",     # KPCA特征
            # "CAE",    # 注释掉不运行CAE特征
            "ADCAE"     # ADCAE特征
        ]
    }
    
    all_results = {}
    
    for dataset in datasets:
        print("=" * 80)
        print(f" 开始处理 {dataset} 数据集 ")
        print("=" * 80)
        
        enabled_features = dataset_config.get(dataset, None)
        print(f"{dataset} 数据集配置的特征类型: {enabled_features}")
        
        try:
            classifier = RFClassifier(dataset)
            results = classifier.run_all_classifications(enabled_features=enabled_features)
            all_results[dataset] = results
        except Exception as e:
            print(f"处理 {dataset} 数据集时出错: {e}")
            all_results[dataset] = None
    
    # 跨数据集对比
    print("\n" + "=" * 80)
    print(" 跨数据集结果对比 ")
    print("=" * 80)
    
    # 这里可以添加跨数据集的对比逻辑
    for dataset, results in all_results.items():
        if results:
            print(f"\n{dataset} 数据集结果:")
            for feature_type, result in results.items():
                if result:
                    print(f"  {feature_type}: 准确率={result['accuracy']:.4f}")

if __name__ == "__main__":
    main()