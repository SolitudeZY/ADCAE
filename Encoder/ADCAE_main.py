import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adcae.config import ModelConfig, TrainingConfig, DataConfig, ExperimentConfigs
from adcae.model import ADCAE
from adcae.trainer import ADCAETrainer
from adcae.preprocessor import BinaryDataPreprocessor

def run_experiment(experiment_name, model_config, training_config, data_config, 
                  train_dir, test_dir, output_dir):
    """运行单个实验"""
    print(f"\n{'='*50}")
    print(f"开始实验: {experiment_name}")
    print(f"{'='*50}")
    
    # 创建实验专用输出目录
    exp_output_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # 创建训练器
    trainer = ADCAETrainer(model_config, training_config, data_config)
    
    # 训练和评估
    results = trainer.train_and_evaluate(train_dir, test_dir, exp_output_dir)
    
    print(f"实验 {experiment_name} 完成！")
    return results

def main():
    """主函数 - 运行对比实验"""
    # 基础配置
    base_output_dir = r"d:\Python Project\ADCAE\results\adcae_experiments"
    pcap_base_dir = r"d:\Python Project\ADCAE\pcap_files"
    
    # 数据集配置
    datasets = {
        'USTC': {
            'train_dir': os.path.join(pcap_base_dir, 'Dataset_USTC', 'Train'),
            'test_dir': os.path.join(pcap_base_dir, 'Dataset_USTC', 'Test')
        }
    }
    
    # 训练和数据配置
    training_config = TrainingConfig(epochs=50, batch_size=64)
    data_config = DataConfig()
    
    # 实验配置列表
    experiments = {
        'baseline_cbam_elu': ExperimentConfigs.get_base_config(),
        'no_attention': ExperimentConfigs.get_no_attention_config(),
        'channel_attention_only': ExperimentConfigs.get_channel_attention_only(),
        'spatial_attention_only': ExperimentConfigs.get_spatial_attention_only(),
        '3_layers': ExperimentConfigs.get_3_layer_config(),
        'relu_activation': ExperimentConfigs.get_relu_config(),
        'gelu_activation': ExperimentConfigs.get_gelu_config()
    }
    
    # 运行所有实验
    all_results = {}
    
    for dataset_name, dataset_paths in datasets.items():
        print(f"\n{'='*100}")
        print(f"开始处理数据集: {dataset_name}")
        print(f"{'='*100}")
        
        dataset_results = {}
        
        for exp_name, model_config in experiments.items():
            full_exp_name = f"{dataset_name}_{exp_name}"
            
            try:
                result = run_experiment(
                    experiment_name=full_exp_name,
                    model_config=model_config,
                    training_config=training_config,
                    data_config=data_config,
                    train_dir=dataset_paths['train_dir'],
                    test_dir=dataset_paths['test_dir'],
                    output_dir=base_output_dir
                )
                dataset_results[exp_name] = result
                
            except Exception as e:
                print(f"实验 {full_exp_name} 失败: {e}")
                dataset_results[exp_name] = {'error': str(e)}
        
        all_results[dataset_name] = dataset_results
    
    # 保存实验总结
    import json
    summary_path = os.path.join(base_output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*100}")
    print("所有实验完成！")
    print(f"实验总结保存到: {summary_path}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()