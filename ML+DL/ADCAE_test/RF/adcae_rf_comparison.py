import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import time
from datetime import datetime
import json

class ADCAERFComparison:
    """
    ADCAE特征的随机森林对比分析器
    对比不同激活函数和注意力机制下的ADCAE特征性能
    支持多数据集对比
    """
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.label_encoder = LabelEncoder()
        self.results = {}
        
        # 数据路径配置
        if dataset_name == "CTU":
            self.base_path = "D:\\Python Project\\ADCAE\\CTU_result\\adcae_features"
        elif dataset_name == "USTC":
            self.base_path = "D:\\Python Project\\ADCAE\\USTC_result\\adcae_features"
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
            
        self.activation_path = os.path.join(self.base_path, "activation_comparison")
        self.attention_path = os.path.join(self.base_path, "attention_comparison")
        
        # 输出路径
        self.output_path = f"D:\\Python Project\\ADCAE\\ML+DL\\ADCAE_test\\RF\\results_{dataset_name}"
        os.makedirs(self.output_path, exist_ok=True)
        
        # 实验配置 - 可以通过修改这里来控制具体实验
        self.experiments = {
            'activation_comparison': {
                'elu': {'enabled': False},      # 启用/禁用 ELU 激活函数实验
                'relu': {'enabled': True},     # 启用/禁用 ReLU 激活函数实验
                'sigmoid': {'enabled': True}   # 启用/禁用 Sigmoid 激活函数实验
            },
            'attention_comparison': {
                'with_cbam': {'enabled': False},        # 启用/禁用 CBAM 注意力机制实验
                'without_attention': {'enabled': False} # 启用/禁用 无注意力机制实验
            }
        }
        
        # 快速控制示例（通过注释来启用/禁用）:
        # self.experiments['activation_comparison']['elu']['enabled'] = False      # 禁用ELU实验
        # self.experiments['activation_comparison']['relu']['enabled'] = False     # 禁用ReLU实验
        # self.experiments['attention_comparison']['with_cbam']['enabled'] = False # 禁用CBAM实验
    
    def _get_dataset_hyperparameters(self):
        """
        获取数据集特定的超参数配置
        """
        if self.dataset_name == "CTU":
            return {
                'activation_comparison': {
                    'elu': {
                        'n_estimators': 150,
                        'max_depth': 20,
                        'min_samples_split': 3,
                        'min_samples_leaf': 1,
                        'max_features': 'sqrt',
                        'bootstrap': True,
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'n_jobs': -1
                    },
                    'relu': {
                        'n_estimators': 500,
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'bootstrap': True,
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'n_jobs': -1
                    },
                    'sigmoid': {
                        'n_estimators': 500,
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': 'sqrt',
                        'bootstrap': True,
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'attention_comparison': {
                    'with_cbam': {
                        'n_estimators': 180,
                        'max_depth': 30,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': 'sqrt',
                        'bootstrap': True,
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'n_jobs': -1
                    },
                    'without_attention': {
                        'n_estimators': 120,
                        'max_depth': 18,
                        'min_samples_split': 3,
                        'min_samples_leaf': 2,
                        'max_features': 'log2',
                        'bootstrap': True,
                        'class_weight': 'balanced_subsample',
                        'random_state': 42,
                        'n_jobs': -1
                    }
                }
            }
        elif self.dataset_name == "USTC":
            return {
                'activation_comparison': {
                    'elu': {
                        'n_estimators': 160,
                        'max_depth': 22,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': 'sqrt',
                        'bootstrap': True,
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'n_jobs': -1
                    },
                    'relu': {
                        'n_estimators': 500,
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'bootstrap': True,
                        'class_weight': None,
                        'random_state': 42,
                        'n_jobs': -1
                    },
                    'sigmoid': {
                        'n_estimators': 500,
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'bootstrap': True,
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'attention_comparison': {
                    'with_cbam': {
                        'n_estimators': 190,
                        'max_depth': 32,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': 'sqrt',
                        'bootstrap': True,
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'n_jobs': -1
                    },
                    'without_attention': {
                        'n_estimators': 130,
                        'max_depth': 20,
                        'min_samples_split': 2,
                        'min_samples_leaf': 2,
                        'max_features': 'log2',
                        'bootstrap': True,
                        'class_weight': 'balanced_subsample',
                        'random_state': 42,
                        'n_jobs': -1
                    }
                }
            }
    
    def get_hyperparameters(self, experiment_type, condition):
        """
        获取指定实验条件的超参数
        """
        dataset_hyperparams = self._get_dataset_hyperparameters()
        return dataset_hyperparams[experiment_type][condition]
    
    def print_hyperparameters_summary(self):
        """
        打印超参数配置摘要
        """
        print(f"\n{self.dataset_name} 数据集超参数配置:")
        print("-" * 50)
        
        dataset_hyperparams = self._get_dataset_hyperparameters()
        
        for exp_type, conditions in dataset_hyperparams.items():
            print(f"\n{exp_type}:")
            for condition, params in conditions.items():
                print(f"  {condition}: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
    
    def load_data(self, data_path):
        """
        加载训练和测试数据
        """
        train_path = os.path.join(data_path, "train_features.csv")
        test_path = os.path.join(data_path, "test_features.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 读取数据
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 打印数据信息用于调试
        print(f"训练数据列: {train_df.columns.tolist()}")
        print(f"训练数据形状: {train_df.shape}")
        print(f"训练数据前几行:")
        print(train_df.head())
        
        # 检查并移除可能的索引列或无关列
        columns_to_drop = []
        for col in train_df.columns:
            if col.lower() in ['unnamed: 0', 'index', 'id']:
                columns_to_drop.append(col)
            elif train_df[col].dtype == 'object' and col != 'label':
                # 检查是否为字符串列（除了label列）
                unique_values = train_df[col].unique()
                print(f"发现字符串列 {col}: {unique_values[:5]}...")  # 只显示前5个值
                columns_to_drop.append(col)
        
        if columns_to_drop:
            print(f"移除列: {columns_to_drop}")
            train_df = train_df.drop(columns=columns_to_drop)
            test_df = test_df.drop(columns=columns_to_drop)
        
        # 分离特征和标签
        feature_columns = [col for col in train_df.columns if col != 'label']
        
        # 确保特征列都是数值型
        for col in feature_columns:
            if train_df[col].dtype == 'object':
                try:
                    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
                    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
                except:
                    print(f"无法转换列 {col} 为数值型，将其移除")
                    feature_columns.remove(col)
        
        # 检查是否有NaN值
        if train_df[feature_columns].isnull().any().any():
            print("发现NaN值，使用0填充")
            train_df[feature_columns] = train_df[feature_columns].fillna(0)
            test_df[feature_columns] = test_df[feature_columns].fillna(0)
        
        X_train = train_df[feature_columns].values
        X_test = test_df[feature_columns].values
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print(f"最终特征维度: {X_train.shape}")
        print(f"标签分布: {np.unique(y_train, return_counts=True)}")
        
        return X_train, X_test, y_train, y_test
    
    def encode_labels(self, y_train, y_test):
        """
        编码标签，确保训练集和测试集使用相同的编码器
        """
        if y_train.dtype == 'object' or isinstance(y_train[0], str):
            # 合并所有标签以确保编码器知道所有可能的标签
            all_labels = np.concatenate([y_train, y_test])
            unique_labels = np.unique(all_labels)
            
            # 使用所有唯一标签拟合编码器
            self.label_encoder.fit(unique_labels)
            
            # 转换标签
            y_train_encoded = self.label_encoder.transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            return y_train_encoded, y_test_encoded
        else:
            return y_train, y_test
    
    def train_and_evaluate_rf(self, X_train, X_test, y_train, y_test, hyperparams, experiment_name):
        """
        训练和评估随机森林模型
        """
        print(f"\n开始训练 {experiment_name} 随机森林模型...")
        print(f"数据集: {self.dataset_name}")
        print(f"超参数配置: {hyperparams}")
        
        # 编码标签
        y_train_encoded, y_test_encoded = self.encode_labels(y_train, y_test)
        
        # 创建模型
        rf_model = RandomForestClassifier(**hyperparams)
        
        # 训练模型
        start_time = time.time()
        rf_model.fit(X_train, y_train_encoded)
        training_time = time.time() - start_time
        
        # 预测
        start_time = time.time()
        y_pred = rf_model.predict(X_test)
        testing_time = time.time() - start_time
        
        # 计算指标
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        
        # 打印结果
        print(f"\n{experiment_name} 随机森林模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"训练时间: {training_time:.4f}秒")
        print(f"测试时间: {testing_time:.4f}秒")
        
        return {
            'dataset': self.dataset_name,
            'experiment_name': experiment_name,
            'hyperparameters': hyperparams,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'testing_time': testing_time,
            'data_shape': {
                'train': X_train.shape,
                'test': X_test.shape
            }
        }
    
    def run_activation_comparison(self):
        """
        运行激活函数对比实验
        """
        print(f"\n{self.dataset_name} - 激活函数对比实验")
        print("=" * 60)
        
        activation_results = []
        
        for activation in ['elu', 'relu', 'sigmoid']:
            if not self.experiments['activation_comparison'][activation]['enabled']:
                print(f"\n跳过激活函数: {activation} (已禁用)")
                continue
                
            try:
                print(f"\n处理激活函数: {activation}")
                data_path = os.path.join(self.activation_path, activation)
                
                # 加载数据
                X_train, X_test, y_train, y_test = self.load_data(data_path)
                print(f"数据加载完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
                
                # 获取数据集特定的超参数
                hyperparams = self.get_hyperparameters('activation_comparison', activation)
                
                # 训练和评估
                result = self.train_and_evaluate_rf(
                    X_train, X_test, y_train, y_test, 
                    hyperparams, f"激活函数_{activation.upper()}"
                )
                
                activation_results.append(result)
                
            except Exception as e:
                print(f"处理激活函数 {activation} 时出错: {str(e)}")
                continue
        
        return activation_results
    
    def run_attention_comparison(self):
        """
        运行注意力机制对比实验
        """
        print(f"\n{self.dataset_name} - 注意力机制对比实验")
        print("=" * 60)
        
        attention_results = []
        
        for attention in ['with_cbam', 'without_attention']:
            if not self.experiments['attention_comparison'][attention]['enabled']:
                print(f"\n跳过注意力机制: {attention} (已禁用)")
                continue
                
            try:
                print(f"\n处理注意力机制: {attention}")
                data_path = os.path.join(self.attention_path, attention)
                
                # 加载数据
                X_train, X_test, y_train, y_test = self.load_data(data_path)
                print(f"数据加载完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
                
                # 获取数据集特定的超参数
                hyperparams = self.get_hyperparameters('attention_comparison', attention)
                
                # 训练和评估
                result = self.train_and_evaluate_rf(
                    X_train, X_test, y_train, y_test, 
                    hyperparams, f"注意力机制_{attention.replace('_', ' ').title()}"
                )
                
                attention_results.append(result)
                
            except Exception as e:
                print(f"处理注意力机制 {attention} 时出错: {str(e)}")
                continue
        
        return attention_results
    
    def generate_comparison_report(self, activation_results, attention_results):
        """
        生成对比报告
        """
        print(f"\n{self.dataset_name} - ADCAE随机森林模型对比报告")
        print("=" * 80)
        
        # 激活函数对比报告
        if activation_results:
            print("\n1. 激活函数对比结果:")
            print("-" * 50)
            
            activation_df_data = []
            for result in activation_results:
                activation_df_data.append({
                    '数据集': result['dataset'],
                    '实验': result['experiment_name'],
                    '准确率': f"{result['accuracy']:.4f}",
                    '精确率': f"{result['precision']:.4f}",
                    '召回率': f"{result['recall']:.4f}",
                    'F1分数': f"{result['f1_score']:.4f}",
                    '训练时间(秒)': f"{result['training_time']:.4f}",
                    '测试时间(秒)': f"{result['testing_time']:.4f}"
                })
            
            activation_df = pd.DataFrame(activation_df_data)
            print(activation_df.to_string(index=False))
            
            # 找出最佳激活函数
            best_activation = max(activation_results, key=lambda x: x['f1_score'])
            print(f"\n最佳激活函数: {best_activation['experiment_name']} (F1分数: {best_activation['f1_score']:.4f})")
        
        # 注意力机制对比报告
        if attention_results:
            print("\n2. 注意力机制对比结果:")
            print("-" * 50)
            
            attention_df_data = []
            for result in attention_results:
                attention_df_data.append({
                    '数据集': result['dataset'],
                    '实验': result['experiment_name'],
                    '准确率': f"{result['accuracy']:.4f}",
                    '精确率': f"{result['precision']:.4f}",
                    '召回率': f"{result['recall']:.4f}",
                    'F1分数': f"{result['f1_score']:.4f}",
                    '训练时间(秒)': f"{result['training_time']:.4f}",
                    '测试时间(秒)': f"{result['testing_time']:.4f}"
                })
            
            attention_df = pd.DataFrame(attention_df_data)
            print(attention_df.to_string(index=False))
            
            # 找出最佳注意力机制
            best_attention = max(attention_results, key=lambda x: x['f1_score'])
            print(f"\n最佳注意力机制: {best_attention['experiment_name']} (F1分数: {best_attention['f1_score']:.4f})")
        
        # 保存结果到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if activation_results:
            activation_csv_path = os.path.join(self.output_path, f"activation_comparison_{timestamp}.csv")
            activation_df.to_csv(activation_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n激活函数对比结果已保存到: {activation_csv_path}")
        
        if attention_results:
            attention_csv_path = os.path.join(self.output_path, f"attention_comparison_{timestamp}.csv")
            attention_df.to_csv(attention_csv_path, index=False, encoding='utf-8-sig')
            print(f"注意力机制对比结果已保存到: {attention_csv_path}")
        
        # 保存详细结果到JSON
        detailed_results = {
            'dataset': self.dataset_name,
            'timestamp': timestamp,
            'activation_comparison': activation_results,
            'attention_comparison': attention_results
        }
        
        json_path = os.path.join(self.output_path, f"detailed_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"详细结果已保存到: {json_path}")
        
        return detailed_results
    
    def run_all_comparisons(self):
        """
        运行所有对比实验
        """
        print(f"开始 {self.dataset_name} 数据集的ADCAE随机森林对比实验")
        print(f"输出路径: {self.output_path}")
        
        # 打印超参数配置摘要
        self.print_hyperparameters_summary()
        
        # 运行激活函数对比
        activation_results = self.run_activation_comparison()
        
        # 运行注意力机制对比
        attention_results = self.run_attention_comparison()
        
        # 生成对比报告
        detailed_results = self.generate_comparison_report(activation_results, attention_results)
        
        print(f"\n{self.dataset_name} 数据集所有对比实验完成!")
        print("=" * 80)
        
        return detailed_results

def generate_cross_dataset_comparison(all_results):
    """
    生成跨数据集对比报告
    """
    print("\n" + "=" * 100)
    print("跨数据集ADCAE随机森林对比报告")
    print("=" * 100)
    
    # 收集所有结果
    comparison_data = []
    
    for dataset_name, results in all_results.items():
        # 激活函数对比结果
        for result in results.get('activation_comparison', []):
            comparison_data.append({
                'Dataset': dataset_name,
                'Experiment_Type': 'Activation',
                'Condition': result['experiment_name'].replace('激活函数_', ''),
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'Training_Time(s)': result['training_time'],
                'Test_Time(s)': result['testing_time']
            })
        
        # 注意力机制对比结果
        for result in results.get('attention_comparison', []):
            comparison_data.append({
                'Dataset': dataset_name,
                'Experiment_Type': 'Attention',
                'Condition': result['experiment_name'].replace('注意力机制_', ''),
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'Training_Time(s)': result['training_time'],
                'Test_Time(s)': result['testing_time']
            })
    
    # 创建DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        # 显示结果
        print("\n跨数据集对比结果:")
        print("-" * 80)
        print(comparison_df.to_string(index=False))
        
        # 保存结果
        output_dir = "D:\\Python Project\\ADCAE\\results\\rf_cross_dataset_comparison"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV
        csv_path = os.path.join(output_dir, f"rf_cross_dataset_comparison_{timestamp}.csv")
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n跨数据集对比结果已保存到: {csv_path}")
        
        # 保存JSON
        json_path = os.path.join(output_dir, f"rf_cross_dataset_comparison_{timestamp}.json")
        comparison_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        print(f"跨数据集对比结果已保存到: {json_path}")
        
        # 分析最佳配置
        print("\n=== 最佳配置分析 ===")
        
        # 按实验类型分组找最佳
        for exp_type in comparison_df['Experiment_Type'].unique():
            type_data = comparison_df[comparison_df['Experiment_Type'] == exp_type]
            best_result = type_data.loc[type_data['F1_Score'].idxmax()]
            print(f"\n{exp_type} 最佳配置:")
            print(f"  数据集: {best_result['Dataset']}")
            print(f"  条件: {best_result['Condition']}")
            print(f"  F1分数: {best_result['F1_Score']:.4f}")
            print(f"  准确率: {best_result['Accuracy']:.4f}")
    
    return comparison_df

def main():
    """
    主函数
    """
    # 数据集控制配置 - 通过注释来启用/禁用数据集
    dataset_config = {
        "CTU": True,      # True: 启用, False: 禁用
        "USTC": True,     # True: 启用, False: 禁用
    }
    
    # 你也可以通过注释来快速控制:
    # dataset_config["CTU"] = False   # 禁用CTU数据集
    # dataset_config["USTC"] = False  # 禁用USTC数据集
    
    # 过滤启用的数据集
    enabled_datasets = [dataset for dataset, enabled in dataset_config.items() if enabled]
    
    if not enabled_datasets:
        print("警告: 没有启用任何数据集！")
        return
    
    print(f"启用的数据集: {enabled_datasets}")
    print(f"禁用的数据集: {[dataset for dataset, enabled in dataset_config.items() if not enabled]}")
    
    all_results = {}
    
    try:
        for dataset_name in enabled_datasets:
            print(f"\n{'='*100}")
            print(f"开始处理 {dataset_name} 数据集")
            print(f"{'='*100}")
            
            # 创建对比分析器
            comparator = ADCAERFComparison(dataset_name)
            
            # 运行所有对比实验
            results = comparator.run_all_comparisons()
            all_results[dataset_name] = results
        
        # 只有在有多个数据集结果时才生成跨数据集对比报告
        if len(all_results) > 1:
            print(f"\n{'='*100}")
            print("生成跨数据集对比报告")
            print(f"{'='*100}")
            
            cross_comparison_df = generate_cross_dataset_comparison(all_results)
        else:
            print(f"\n{'='*100}")
            print("跳过跨数据集对比报告（只有一个数据集启用）")
            print(f"{'='*100}")
        
        # 实验总结
        print(f"\n{'='*100}")
        print("实验总结")
        print(f"{'='*100}")
        
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name} 数据集:")
            print(f"- 激活函数对比实验: {len(results['activation_comparison'])} 个")
            print(f"- 注意力机制对比实验: {len(results['attention_comparison'])} 个")
        
        if len(all_results) > 1:
            print(f"\n跨数据集对比结果: {len(cross_comparison_df)} 条记录")
        
        print(f"\n处理的数据集总数: {len(all_results)}")
        
    except Exception as e:
        print(f"实验过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()