import os
import json
import time
from datetime import datetime
from adcae.config import ExperimentConfigs, TrainingConfig, DataConfig
from adcae.trainer import ADCAETrainer
from adcae.preprocessor import BinaryDataPreprocessor
import pickle
import hashlib
import torch  # 添加缺失的import

class FeatureGenerator:
    """ADCAE特征生成器 - 为后续检测模型生成训练数据"""
    
    def __init__(self, base_data_path, output_base_path):
        self.base_data_path = base_data_path
        self.output_base_path = output_base_path
        self.generation_log = []
        self.cache_dir = os.path.join(output_base_path, '.cache')
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(output_base_path, exist_ok=True)
    
    def _get_cache_key(self, dataset_dir):
        """生成数据集的缓存键"""
        # 基于数据集路径和修改时间生成唯一键
        path_hash = hashlib.md5(dataset_dir.encode()).hexdigest()[:8]
        return f"dataset_{path_hash}"
    
    def _load_cached_data(self, dataset_dir):
        """加载缓存的数据"""
        cache_key = self._get_cache_key(dataset_dir)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            print(f"发现缓存数据: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"成功加载缓存数据，样本数: {len(cached_data['X'])}")
                return cached_data
            except Exception as e:
                print(f"加载缓存失败: {e}，将重新处理数据")
                return None
        return None
    
    def _save_cached_data(self, dataset_dir, X, y, category_counts):
        """保存数据到缓存"""
        cache_key = self._get_cache_key(dataset_dir)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        cached_data = {
            'X': X,
            'y': y,
            'category_counts': category_counts,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"数据已缓存到: {cache_file}")
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def _check_experiment_exists(self, output_dir):
        """检查实验结果是否已存在"""
        required_files = [
            'train_features.csv',
            'test_features.csv',
            'adcae_model.pth',
            'experiment_stats.json'
        ]
        
        if not os.path.exists(output_dir):
            return False
        
        for file_name in required_files:
            if not os.path.exists(os.path.join(output_dir, file_name)):
                return False
        
        return True
    
    def _generate_features(self, experiment_name, model_config, output_dir):
        """生成单个实验的特征文件（带缓存和跳过机制）"""
        # 检查实验是否已完成
        if self._check_experiment_exists(output_dir):
            print(f"✓ {experiment_name} 实验结果已存在，跳过生成")
            print(f"  输出目录: {output_dir}")
            
            # 读取已有的统计信息
            stats_file = os.path.join(output_dir, 'experiment_stats.json')
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            log_entry = {
                'experiment_name': experiment_name,
                'output_directory': output_dir,
                'status': 'skipped_existing',
                'timestamp': datetime.now().isoformat(),
                'results': stats
            }
            self.generation_log.append(log_entry)
            return
        
        start_time = time.time()
        
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建训练器（使用新的模块化结构）
            trainer = ADCAETrainer(
                model_config=model_config,
                training_config=TrainingConfig(),
                data_config=DataConfig()
            )
            
            # 设置数据路径
            train_path = os.path.join(self.base_data_path, "Train")
            test_path = os.path.join(self.base_data_path, "Test")
            
            # 使用缓存机制的训练和评估方法
            results = self._train_and_evaluate_with_cache(trainer, train_path, test_path, output_dir)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 记录生成信息
            log_entry = {
                'experiment_name': experiment_name,
                'output_directory': output_dir,
                'model_config': {
                    'activation': model_config.activation,
                    'num_layers': len(model_config.encoder_channels),
                    'encoder_channels': model_config.encoder_channels,
                    'use_attention': model_config.use_attention,
                    'attention_type': model_config.attention_type if model_config.use_attention else 'none'
                },
                'generation_time': generation_time,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'output_files': {
                    'train_features': os.path.join(output_dir, 'train_features.csv'),
                    'test_features': os.path.join(output_dir, 'test_features.csv'),
                    'model_file': os.path.join(output_dir, 'adcae_model.pth'),
                    'config_file': os.path.join(output_dir, 'model_config.json')
                }
            }
            
            self.generation_log.append(log_entry)
            
            print(f"✓ {experiment_name} 特征生成完成，耗时: {generation_time:.2f}秒")
            print(f"  输出目录: {output_dir}")
            
            # 保存模型配置信息
            config_file = os.path.join(output_dir, 'model_config.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(log_entry['model_config'], f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            end_time = time.time()
            generation_time = end_time - start_time
            
            log_entry = {
                'experiment_name': experiment_name,
                'output_directory': output_dir,
                'status': 'failed',
                'error': str(e),
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.generation_log.append(log_entry)
            
            print(f"✗ {experiment_name} 特征生成失败: {e}")
            import traceback
            traceback.print_exc()
            
    def save_generation_summary(self):
        """保存特征生成总结"""
        summary = {
            'total_experiments': len(self.generation_log),
            'successful_generations': len([log for log in self.generation_log if log['status'] == 'success']),
            'failed_generations': len([log for log in self.generation_log if log['status'] == 'failed']),
            'generation_details': self.generation_log,
            'summary_timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.output_base_path, f"feature_generation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"\n特征生成总结已保存到: {summary_file}")
        print(f"总实验数: {summary['total_experiments']}")
        print(f"成功生成: {summary['successful_generations']}")
        print(f"生成失败: {summary['failed_generations']}")
        
        return summary_file

    def _train_and_evaluate_with_cache(self, trainer, train_dir, test_dir, output_dir):
        """带缓存的训练和评估流程"""
        # 构建模型
        trainer.build_model()
        
        # 尝试加载缓存的训练数据
        cached_train = self._load_cached_data(train_dir)
        if cached_train:
            train_X = torch.FloatTensor(cached_train['X']).to(trainer.device)
            train_labels = cached_train['y']
            train_category_counts = cached_train['category_counts']
        else:
            # 准备训练数据并缓存
            train_X, train_labels, train_category_counts = trainer.prepare_data(train_dir)
            self._save_cached_data(train_dir, train_X.cpu().numpy(), train_labels, train_category_counts)
        
        # 训练模型
        training_time = trainer.train_model(train_X)
        
        # 提取训练集特征
        train_encoded = trainer.encode_data(train_X)
        
        # 尝试加载缓存的测试数据
        cached_test = self._load_cached_data(test_dir)
        if cached_test:
            test_X = torch.FloatTensor(cached_test['X']).to(trainer.device)
            test_labels = cached_test['y']
            test_category_counts = cached_test['category_counts']
        else:
            # 准备测试数据并缓存
            test_X, test_labels, test_category_counts = trainer.prepare_data(test_dir)
            self._save_cached_data(test_dir, test_X.cpu().numpy(), test_labels, test_category_counts)
        
        # 提取测试集特征
        test_encoded = trainer.encode_data(test_X)
        
        # 保存结果
        results = trainer._save_results(
            output_dir, train_encoded, train_labels, train_category_counts,
            test_encoded, test_labels, test_category_counts, training_time
        )
        
        return results

    def generate_layer_comparison_features(self, dataset_name):
        """生成不同层数的ADCAE特征文件"""
        print("\n=== 生成层数对比实验特征文件 ===")
        
        # 获取层数配置
        layer_configs = ExperimentConfigs.generate_layer_configs()
        
        for layer_name, config in layer_configs.items():
            experiment_name = f"{dataset_name}_{layer_name}"
            output_dir = os.path.join(self.output_base_path, "layer_comparison", layer_name)
            
            print(f"\n处理 {layer_name} 配置...")
            self._generate_features(experiment_name, config, output_dir)
    
    def generate_activation_comparison_features(self, dataset_name):
        """生成不同激活函数的ADCAE特征文件"""
        print("\n=== 生成激活函数对比实验特征文件 ===")
        
        # 获取激活函数配置（使用4层作为基础）
        layer_configs = ExperimentConfigs.generate_layer_configs()
        base_config = layer_configs['4_layers']  # 使用4层作为基础配置
        
        # 修复：传递base_config参数
        activation_configs = ExperimentConfigs.generate_activation_configs(base_config)
        
        for activation_name, config in activation_configs.items():
            experiment_name = f"{dataset_name}_{activation_name}"
            output_dir = os.path.join(self.output_base_path, "activation_comparison", activation_name)
            
            print(f"\n处理 {activation_name} 激活函数...")
            self._generate_features(experiment_name, config, output_dir)
    
    def generate_attention_comparison_features(self, dataset_name):
        """生成不同注意力机制的ADCAE特征文件"""
        print("\n=== 生成注意力机制对比实验特征文件 ===")
        
        # 获取注意力机制配置（使用4层作为基础）
        layer_configs = ExperimentConfigs.generate_layer_configs()
        base_config = layer_configs['4_layers']  # 使用4层作为基础配置
        
        # 修复：传递base_config参数
        attention_configs = ExperimentConfigs.generate_attention_configs(base_config)
        
        for attention_name, config in attention_configs.items():
            experiment_name = f"{dataset_name}_{attention_name}"
            output_dir = os.path.join(self.output_base_path, "attention_comparison", attention_name)
            
            print(f"\n处理 {attention_name} 注意力配置...")
            self._generate_features(experiment_name, config, output_dir)
    
# 使用示例
if __name__ == "__main__":
    # 配置路径
    # base_data_path = "d:/Python Project/ADCAE/pcap_files/Dataset_CTU"  # 根据实际情况修改
    base_data_path = "d:/Python Project/ADCAE/pcap_files/Dataset_USTC"  # 根据实际情况修改
    # output_base_path = "d:/Python Project/ADCAE/CTU_result/adcae_features"
    output_base_path = "d:/Python Project/ADCAE/USTC_result/adcae_features"
    # dataset_name = "CTU"
    dataset_name = "USTC"
    
    # 创建特征生成器
    generator = FeatureGenerator(base_data_path, output_base_path)
    
    print("开始生成ADCAE特征文件用于后续检测模型训练...")
    
    # 1. 生成层数对比实验的特征文件
    generator.generate_layer_comparison_features(dataset_name)
    
    # 2. 生成激活函数对比实验的特征文件
    generator.generate_activation_comparison_features(dataset_name)
    
    # 3. 生成注意力机制对比实验的特征文件
    generator.generate_attention_comparison_features(dataset_name)
    
    # 保存生成总结
    summary_file = generator.save_generation_summary()
    
    print("\n所有ADCAE特征文件生成完成！")
    print("\n生成的文件结构:")
    print(f"{dataset_name}_result/adcae_features/")
    print("├── layer_comparison/")
    print("│   ├── 2_layers/")
    print("│   ├── 4_layers/")
    print("│   ├── 6_layers/")
    print("│   ├── 8_layers/")
    print("│   ├── 10_layers/")
    print("│   └── 12_layers/")
    print("├── activation_comparison/")
    print("│   ├── elu/")
    print("│   ├── relu/")
    print("│   └── sigmoid/")
    print("└── attention_comparison/")
    print("    ├── with_cbam/")
    print("    └── without_attention/")
    print("\n每个目录包含:")
    print("- train_features.csv (训练集特征)")
    print("- test_features.csv (测试集特征)")
    print("- adcae_model.pth (训练好的ADCAE模型)")
    print("- model_config.json (模型配置信息)")