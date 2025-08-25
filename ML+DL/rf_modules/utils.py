import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime

class RFUtils:
    def __init__(self, dataset_name, base_path):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.output_path = os.path.join(base_path, f"{dataset_name}_result", "rf_output_features")
        
        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)
    
    def save_results(self, model, results):
        """
        保存模型和结果
        
        Args:
            model: 训练好的模型
            results: 结果字典
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_type = results['feature_type']
            
            # 保存模型
            model_path = os.path.join(self.output_path, f"rf_{feature_type.lower()}_model_{self.dataset_name}_{timestamp}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"模型已保存到: {model_path}")
            
            # 保存结果为JSON
            results_path = os.path.join(self.output_path, f"rf_{feature_type.lower()}_results_{self.dataset_name}_{timestamp}.json")
            
            # 转换numpy数组为列表以便JSON序列化
            json_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    json_results[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    json_results[k] = float(v)
                # 在save_results_to_json方法中，删除混淆矩阵的特殊处理：
                # elif k == 'confusion_matrix' and v is not None:
                # # 特殊处理混淆矩阵
                # results_dict[k] = v.tolist() if hasattr(v, 'tolist') else v
                
                # 保持其他代码不变
                else:
                    json_results[k] = v
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            # 保存性能指标CSV
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                           'Training_Time', 'Prediction_Time', 'Avg_Prediction_Time_Per_Sample'],
                'Value': [
                    results.get('accuracy', 0),
                    results.get('precision', 0),
                    results.get('recall', 0),
                    results.get('f1_score', 0),
                    results.get('training_time', 0),
                    results.get('prediction_time', 0),
                    results.get('avg_prediction_time', 0)
                ]
            })
            
            metrics_path = os.path.join(self.output_path, f"rf_{feature_type.lower()}_metrics_{self.dataset_name}_{timestamp}.csv")
            metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')
            
            print(f"结果已保存到: {self.output_path}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    def compare_results(self, results_summary):
        """
        对比不同特征类型的结果
        
        Args:
            results_summary: 包含所有特征类型结果的字典
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
            # 创建对比数据框
            comparison_data = {}
            for feature_type, result in valid_results.items():
                comparison_data[feature_type] = {
                    'accuracy': result.get('accuracy', 0),
                    'precision': result.get('precision', 0),
                    'recall': result.get('recall', 0),
                    'f1_score': result.get('f1_score', 0),
                    'training_time': result.get('training_time', 0),
                    'prediction_time': result.get('prediction_time', 0)
                }
            
            comparison_df = pd.DataFrame(comparison_data).T
            
            # 显示对比结果
            print(comparison_df.round(4))
            
            # 保存对比结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = os.path.join(self.output_path, f"rf_feature_comparison_{self.dataset_name}_{timestamp}.csv")
            comparison_df.to_csv(comparison_path, encoding='utf-8')
            print(f"\n对比结果已保存到: {comparison_path}")
            
            # 找出最佳结果
            best_accuracy = comparison_df['accuracy'].max()
            best_feature = comparison_df['accuracy'].idxmax()
            print(f"\n最佳准确率: {best_accuracy:.4f} ({best_feature})")
            
        except Exception as e:
            print(f"创建对比表时出错: {e}")
            print("原始结果数据:")
            for feature_type, result in valid_results.items():
                if result:
                    accuracy = result.get('accuracy', 'N/A')
                    if isinstance(accuracy, (int, float)):
                        print(f"{feature_type}: 准确率={accuracy:.4f}")
                    else:
                        print(f"{feature_type}: 准确率={accuracy}")
    
    def load_model(self, model_path):
        """
        加载保存的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的模型
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"模型已从 {model_path} 加载")
            return model
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return None
    
    def get_latest_model(self, feature_type):
        """
        获取指定特征类型的最新模型
        
        Args:
            feature_type: 特征类型
            
        Returns:
            最新模型的路径
        """
        try:
            pattern = f"rf_{feature_type.lower()}_model_{self.dataset_name}_"
            model_files = [f for f in os.listdir(self.output_path) if f.startswith(pattern) and f.endswith('.pkl')]
            
            if not model_files:
                print(f"未找到 {feature_type} 特征的模型文件")
                return None
            
            # 按时间戳排序，获取最新的
            model_files.sort(reverse=True)
            latest_model = os.path.join(self.output_path, model_files[0])
            
            return latest_model
            
        except Exception as e:
            print(f"获取最新模型时出错: {e}")
            return None