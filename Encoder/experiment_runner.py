import os
import json
import time
from datetime import datetime
from adcae.config import ExperimentConfigs, TrainingConfig, DataConfig
from ADCAE_main import ADCAEProcessor

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, base_data_path, results_base_path):
        self.base_data_path = base_data_path
        self.results_base_path = results_base_path
        self.experiment_results = {}
        
    def run_activation_comparison(self, dataset_name):
        """运行激活函数对比实验"""
        print("\n=== 激活函数对比实验 ===")
        activation_configs = ExperimentConfigs.get_all_activation_configs()
        
        for activation_name, config in activation_configs.items():
            print(f"\n运行 {activation_name} 激活函数实验...")
            
            experiment_name = f"{dataset_name}_activation_{activation_name.lower()}"
            results = self._run_single_experiment(experiment_name, config)
            
            self.experiment_results[f"activation_{activation_name}"] = results
            
    def run_layer_comparison(self, dataset_name):
        """运行层数对比实验"""
        print("\n=== 层数对比实验 ===")
        layer_configs = ExperimentConfigs.get_all_layer_configs()
        
        for layer_name, config in layer_configs.items():
            print(f"\n运行 {layer_name} 实验...")
            
            experiment_name = f"{dataset_name}_layers_{layer_name}"
            results = self._run_single_experiment(experiment_name, config)
            
            self.experiment_results[f"layers_{layer_name}"] = results
            
    def run_combined_experiments(self, dataset_name):
        """运行激活函数和层数的组合实验"""
        print("\n=== 激活函数和层数组合实验 ===")
        combined_configs = ExperimentConfigs.get_combined_experiment_configs()
        
        for experiment_name, config in combined_configs.items():
            print(f"\n运行 {experiment_name} 实验...")
            
            full_experiment_name = f"{dataset_name}_{experiment_name}"
            results = self._run_single_experiment(full_experiment_name, config)
            
            self.experiment_results[experiment_name] = results
            
    def _run_single_experiment(self, experiment_name, model_config):
        """运行单个实验"""
        start_time = time.time()
        
        try:
            # 创建处理器
            processor = ADCAEProcessor(
                model_config=model_config,
                training_config=TrainingConfig(),
                data_config=DataConfig()
            )
            
            # 运行实验
            train_path = os.path.join(self.base_data_path, "Train")
            test_path = os.path.join(self.base_data_path, "Test")
            results_path = os.path.join(self.results_base_path, experiment_name)
            
            # 处理训练集和测试集
            processor.process_train_test_separately(
                train_data_path=train_path,
                test_data_path=test_path,
                results_path=results_path
            )
            
            end_time = time.time()
            experiment_time = end_time - start_time
            
            # 收集结果
            results = {
                'experiment_name': experiment_name,
                'model_config': {
                    'activation': model_config.activation,
                    'num_layers': len(model_config.encoder_channels),
                    'encoder_channels': model_config.encoder_channels,
                    'use_attention': model_config.use_attention,
                    'attention_type': model_config.attention_type
                },
                'experiment_time': experiment_time,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"实验 {experiment_name} 完成，耗时: {experiment_time:.2f}秒")
            
        except Exception as e:
            end_time = time.time()
            experiment_time = end_time - start_time
            
            results = {
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': str(e),
                'experiment_time': experiment_time,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"实验 {experiment_name} 失败: {e}")
            
        return results
        
    def save_experiment_summary(self, output_file):
        """保存实验总结"""
        summary = {
            'total_experiments': len(self.experiment_results),
            'successful_experiments': len([r for r in self.experiment_results.values() if r['status'] == 'success']),
            'failed_experiments': len([r for r in self.experiment_results.values() if r['status'] == 'failed']),
            'experiments': self.experiment_results,
            'summary_timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"\n实验总结已保存到: {output_file}")
        print(f"总实验数: {summary['total_experiments']}")
        print(f"成功实验数: {summary['successful_experiments']}")
        print(f"失败实验数: {summary['failed_experiments']}")

# 使用示例
if __name__ == "__main__":
    # 配置路径
    base_data_path = "d:/Python Project/ADCAE/pcap_files/Dataset_CTU"  # 根据实际情况修改
    results_base_path = "d:/Python Project/ADCAE/results/experiments"
    dataset_name = "CTU"
    
    # 创建实验运行器
    runner = ExperimentRunner(base_data_path, results_base_path)
    
    # 运行实验（可以选择运行哪些实验）
    print("开始运行ADCAE对比实验...")
    
    # 1. 激活函数对比实验
    runner.run_activation_comparison(dataset_name)
    
    # 2. 层数对比实验
    runner.run_layer_comparison(dataset_name)
    
    # 3. 组合实验（可选，会运行很多实验）
    # runner.run_combined_experiments(dataset_name)
    
    # 保存实验总结
    summary_file = os.path.join(results_base_path, f"experiment_summary_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    runner.save_experiment_summary(summary_file)