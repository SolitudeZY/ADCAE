import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

class RFVisualizer:
    def __init__(self, output_path, dataset_name):
        self.output_path = output_path
        self.dataset_name = dataset_name
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
    
    # 删除 plot_confusion_matrix 方法
    # 如果有其他可视化方法，保留它们
    
    def plot_feature_importance(self, feature_importance, feature_names, feature_type):
        """绘制特征重要性图（如果需要的话）"""
        try:
            plt.figure(figsize=(10, 6))
            indices = np.argsort(feature_importance)[::-1][:20]  # 显示前20个重要特征
            
            plt.title(f'{feature_type} - Top 20 Feature Importance')
            plt.bar(range(len(indices)), feature_importance[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.output_path, f"rf_{feature_type.lower()}_feature_importance_{self.dataset_name}_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"特征重要性图已保存到: {plot_path}")
        except Exception as e:
            print(f"绘制特征重要性图时出错: {e}")