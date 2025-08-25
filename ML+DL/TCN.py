"""TCN主入口文件"""
import os
import time
import json
from datetime import datetime

from tcn_classifier import TCNFeatureClassifier

# 修改main函数，添加跨数据集对比
def main():
    """主函数"""
    print("=" * 100)
    print("TCN (时序卷积网络) 训练和测试系统")
    print("使用PCA、KPCA、CAE和ADCAE特征")
    print("=" * 100)
    
    # 配置参数
    config_file = r"d:\Python Project\ADCAE\config\tcn_hyperparameters.json"
    base_output_dir = r"d:\Python Project\ADCAE\results\tcn_output"
    
    # 数据集列表
    datasets = ['USTC', 'CTU']
    
    # 记录总开始时间
    total_start_time = time.time()
    all_results = {}
    comparison_data = []
    
    # 处理每个数据集
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"开始处理 {dataset_name} 数据集")
        print(f"{'='*50}")
        
        try:
            # 创建分类器
            classifier = TCNFeatureClassifier(dataset_name, config_file)
            
            # 运行所有分类任务
            results = classifier.run_all_classifications()
            all_results[dataset_name] = results
            
            # 收集对比数据
            if results:
                for feature_type, result in results.items():
                    if result and 'test_results' in result:
                        test_res = result['test_results']
                        comparison_data.append({
                            'Dataset': dataset_name,
                            'Feature': feature_type,
                            'Accuracy': f"{test_res['accuracy']:.4f}",
                            'Precision': f"{test_res['precision']:.4f}",
                            'Recall': f"{test_res['recall']:.4f}",
                            'F1-Score': f"{test_res['f1_score']:.4f}",
                            'Training_Time(s)': f"{result['training_time_seconds']:.2f}",  # 修改这里
                            'Test_Time(s)': f"{test_res['test_time']:.2f}"
                        })
            
        except Exception as e:
            print(f"处理 {dataset_name} 数据集时出错: {e}")
            all_results[dataset_name] = None
    
    # 计算总时间
    total_time = time.time() - total_start_time
    
    # 显示跨数据集对比
    print(f"\n{'='*100}")
    print("跨数据集性能对比")
    print(f"{'='*100}")
    
    if comparison_data:
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # 保存对比结果
        comparison_path = os.path.join(base_output_dir, 'cross_dataset_comparison.csv')
        os.makedirs(base_output_dir, exist_ok=True)
        df.to_csv(comparison_path, index=False)
        print(f"\n跨数据集对比结果已保存到: {comparison_path}")
    
    # 保存综合统计
    summary_stats = {
        'total_processing_time_seconds': float(total_time),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'TCN',
        'datasets_processed': list(all_results.keys()),
        'individual_results': all_results,
        'comparison_summary': comparison_data
    }
    
    summary_path = os.path.join(base_output_dir, 'tcn_processing_summary.json')
    
    # 使用相同的类型转换函数
    def convert_numpy_types(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy_types(summary_stats), f, indent=2)
    
    print(f"\n{'='*100}")
    print("TCN处理完成！")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"处理的数据集: {list(all_results.keys())}")
    print(f"综合统计已保存到: {summary_path}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()