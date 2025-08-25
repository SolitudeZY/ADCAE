import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
import time
import json

class ADCAEDTComparison:
    """
    ADCAE特征决策树对比实验类
    用于对比不同激活函数和注意力机制下的ADCAE特征性能
    支持每个数据集的独立超参数配置
    """
    
    def __init__(self, dataset_name="CTU"):
        self.dataset_name = dataset_name
        self.label_encoder = LabelEncoder()
        self.results = {}
        
        # 设置基础路径
        self.base_path = f"D:\\Python Project\\ADCAE\\{dataset_name}_result\\adcae_features"
        
        # 设置输出路径
        self.output_path = f"D:\\Python Project\\ADCAE\\ML+DL\\ADCAE_test\\DT\\results_{dataset_name}"
        os.makedirs(self.output_path, exist_ok=True)
        
        # 数据集特定的超参数配置
        self.dataset_hyperparameters = self._get_dataset_hyperparameters()
        
        # 实验配置
        self.experiments = {
            'activation_comparison': {
                'elu': {
                    'path': os.path.join(self.base_path, "activation_comparison", "elu"),
                    'enabled': True  # 设置为 False 可禁用此数据集
                },
                'relu': {
                    'path': os.path.join(self.base_path, "activation_comparison", "relu"),
                    'enabled': True  # 设置为 False 可禁用此数据集
                },
                'sigmoid': {
                    'path': os.path.join(self.base_path, "activation_comparison", "sigmoid"),
                    'enabled': True  # 设置为 False 可禁用此数据集
                }
            },
            'attention_comparison': {
                'with_cbam': {
                    'path': os.path.join(self.base_path, "attention_comparison", "with_cbam"),
                    'enabled': True  # 设置为 False 可禁用此数据集
                },
                'without_attention': {
                    'path': os.path.join(self.base_path, "attention_comparison", "without_attention"),
                    'enabled': True  # 设置为 False 可禁用此数据集
                }
            }
        }
    
    def _get_dataset_hyperparameters(self):
        """
        获取数据集特定的超参数配置
        每个数据集的每个实验条件都可以有独立的超参数
        """
        hyperparameters = {
            'CTU': {
                'activation_comparison': {
                    'elu': {
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'criterion': 'entropy',
                        'class_weight': None,
                        'random_state': 42,
                        'splitter': 'best'
                    },
                    'relu': {
                        'max_depth': None,
                        'min_samples_split': 3,
                        'min_samples_leaf': 2,
                        'max_features': 'sqrt',
                        'criterion': 'entropy',
                        'class_weight': None,
                        'random_state': 42,
                        'splitter': 'best'
                    },
                    'sigmoid': {
                        'max_depth': None,
                        'min_samples_split': 4,
                        'min_samples_leaf': 2,
                        'max_features': 'log2',
                        'criterion': 'entropy',
                        'class_weight': None,
                        'random_state': 42,
                        'splitter': 'random'
                    }
                },
                'attention_comparison': {
                    'with_cbam': {
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'criterion': 'entropy',
                        'class_weight': None,
                        'random_state': 42,
                        'splitter': 'best'
                    },
                    'without_attention': {
                        'max_depth': None,
                        'min_samples_split': 8,
                        'min_samples_leaf': 4,
                        'max_features': 'sqrt',
                        'criterion': 'gini',
                        'class_weight': None,
                        'random_state': 42,
                        'splitter': 'best'
                    }
                }
            },
            'USTC': {
                'activation_comparison': {
                    'elu': {
                        'max_depth': None,  # USTC数据集的ELU激活函数超参数
                        'min_samples_split': 3,
                        'min_samples_leaf': 2,
                        'max_features': 'sqrt',
                        'criterion': 'gini',
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'splitter': 'best'
                    },
                    'relu': {
                        'max_depth': None,  # USTC数据集的ReLU激活函数超参数
                        'min_samples_split': 4,
                        'min_samples_leaf': 3,
                        'max_features': 'log2',
                        'criterion': 'entropy',
                        'class_weight': None,
                        'random_state': 42,
                        'splitter': 'random'
                    },
                    'sigmoid': {
                        'max_depth': None,  # USTC数据集的Sigmoid激活函数超参数
                        'min_samples_split': 4,
                        'min_samples_leaf': 2,
                        'max_features': None,
                        'criterion': 'gini',
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'splitter': 'best'
                    }
                },
                'attention_comparison': {
                    'with_cbam': {
                        'max_depth': 35,  # USTC数据集的CBAM注意力机制超参数
                        'min_samples_split': 3,
                        'min_samples_leaf': 2,
                        'max_features': 'sqrt',
                        'criterion': 'entropy',
                        'class_weight': None,
                        'random_state': 42,
                        'splitter': 'best'
                    },
                    'without_attention': {
                        'max_depth': 22,  # USTC数据集的无注意力机制超参数
                        'min_samples_split': 7,
                        'min_samples_leaf': 5,
                        'max_features': 'log2',
                        'criterion': 'gini',
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'splitter': 'random'
                    }
                }
            }
        }
        
        return hyperparameters.get(self.dataset_name, hyperparameters['CTU'])
    
    def get_hyperparameters(self, experiment_type, condition):
        """
        获取指定实验类型和条件的超参数
        """
        try:
            return self.dataset_hyperparameters[experiment_type][condition]
        except KeyError:
            print(f"警告: 未找到 {self.dataset_name} 数据集 {experiment_type}-{condition} 的超参数配置，使用默认配置")
            # 返回默认配置
            return {
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': None,
                'criterion': 'gini',
                'class_weight': None,
                'random_state': 42,
                'splitter': 'best'
            }
    
    def print_hyperparameters_summary(self):
        """
        打印当前数据集的所有超参数配置摘要
        """
        print(f"\n{'='*80}")
        print(f"{self.dataset_name} 数据集超参数配置摘要")
        print(f"{'='*80}")
        
        for exp_type, conditions in self.dataset_hyperparameters.items():
            print(f"\n{exp_type.upper()}:")
            for condition, params in conditions.items():
                print(f"  {condition}:")
                for key, value in params.items():
                    print(f"    {key}: {value}")
        print(f"{'='*80}")

    def load_features(self, data_path):
        """
        从指定路径加载ADCAE特征数据
        """
        train_path = os.path.join(data_path, "train_features.csv")
        test_path = os.path.join(data_path, "test_features.csv")
        
        print(f"\n加载数据:")
        print(f"训练集: {train_path}")
        print(f"测试集: {test_path}")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"数据文件不存在: {data_path}")
            return None, None, None, None
        
        try:
            # 读取数据
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            print(f"原始训练数据形状: {train_df.shape}")
            print(f"原始测试数据形状: {test_df.shape}")
            
            # 检查并处理非数值列
            columns_to_drop = ['label']  # 保留label列用于后续处理
            
            for col in train_df.columns:
                if col.lower() != 'label':
                    # 检查是否为数值类型
                    if train_df[col].dtype == 'object':
                        print(f"发现非数值列: {col}，将被删除")
                        columns_to_drop.append(col)
            
            print(f"将要删除的列: {[col for col in columns_to_drop if col != 'label']}")
            
            # 分离特征和标签
            if 'label' in train_df.columns:
                y_train = train_df['label'].copy()
                y_test = test_df['label'].copy()
            else:
                raise ValueError("未找到标签列 'label'")
            
            # 删除非特征列
            feature_columns = [col for col in train_df.columns if col not in columns_to_drop]
            X_train = train_df[feature_columns].copy()
            X_test = test_df[feature_columns].copy()
            
            # 确保所有特征列都是数值类型
            for col in X_train.columns:
                X_train.loc[:, col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test.loc[:, col] = pd.to_numeric(X_test[col], errors='coerce')
            
            # 检查是否有NaN值
            if X_train.isnull().any().any():
                print("警告: 特征数据中存在NaN值，将用0填充")
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)
            
            # 转换为numpy数组
            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values
            
            print(f"处理后训练集形状: {X_train.shape}")
            print(f"处理后测试集形状: {X_test.shape}")
            print(f"类别数量: {len(np.unique(y_train))}")
            print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
            print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None, None, None, None
    
    def encode_labels(self, y_train, y_test):
        """
        编码标签，确保训练集和测试集标签一致性
        """
        # 合并所有标签以确保编码器看到所有可能的标签
        all_labels = np.concatenate([y_train, y_test])
        
        # 如果标签是字符串，进行编码
        if all_labels.dtype == 'object' or isinstance(all_labels[0], str):
            # 使用所有标签来拟合编码器
            self.label_encoder.fit(all_labels)
            y_train_encoded = self.label_encoder.transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            return y_train_encoded, y_test_encoded
        else:
            return y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, experiment_type, condition):
        """
        训练和评估决策树模型
        """
        print(f"\n开始训练 {experiment_type} - {condition} 决策树模型...")
        
        # 编码标签
        y_train_encoded, y_test_encoded = self.encode_labels(y_train, y_test)
        
        # 获取数据集特定的超参数
        hyperparameters = self.get_hyperparameters(experiment_type, condition)
        
        print(f"\n{self.dataset_name} 数据集 - {experiment_type} - {condition} 使用的超参数:")
        for key, value in hyperparameters.items():
            print(f"  {key}: {value}")
        
        # 构建模型
        dt_model = DecisionTreeClassifier(**hyperparameters)
        
        # 训练模型
        start_time = time.time()
        dt_model.fit(X_train, y_train_encoded)
        training_time = time.time() - start_time
        
        # 预测
        start_time = time.time()
        y_pred = dt_model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # 计算指标
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{self.dataset_name} - {experiment_type} - {condition} 决策树模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"训练时间: {training_time:.4f}秒")
        print(f"测试预测时间: {prediction_time:.4f}秒")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # 保存模型
        model_path = os.path.join(self.output_path, f"dt_model_{experiment_type}_{condition}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(dt_model, f)
        
        # 创建结果字典
        results = {
            'dataset': self.dataset_name,
            'experiment_type': experiment_type,
            'condition': condition,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'confusion_matrix': cm.tolist(),
            'hyperparameters': hyperparameters,
            'num_features': X_train.shape[1],
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'num_classes': len(np.unique(y_train_encoded)),
            'model_path': model_path
        }
        
        return results

    def run_activation_comparison(self):
        """
        运行激活函数对比实验
        """
        print("\n" + "="*80)
        print(f"开始 {self.dataset_name} 数据集激活函数对比实验")
        print("="*80)
        
        activation_results = {}
        
        for activation, config in self.experiments['activation_comparison'].items():
            # 检查是否启用此数据集
            if not config.get('enabled', True):
                print(f"\n跳过激活函数: {activation} (已禁用)")
                continue
                
            print(f"\n处理激活函数: {activation}")
            
            # 加载数据
            X_train, X_test, y_train, y_test = self.load_features(config['path'])
            
            if X_train is not None:
                # 训练和评估
                result = self.train_and_evaluate(X_train, X_test, y_train, y_test, 
                                                'activation_comparison', activation)
                activation_results[activation] = result
        
        self.results['activation_comparison'] = activation_results
        return activation_results
    
    def run_attention_comparison(self):
        """
        运行注意力机制对比实验
        """
        print("\n" + "="*80)
        print(f"开始 {self.dataset_name} 数据集注意力机制对比实验")
        print("="*80)
        
        attention_results = {}
        
        for attention, config in self.experiments['attention_comparison'].items():
            # 检查是否启用此数据集
            if not config.get('enabled', True):
                print(f"\n跳过注意力机制: {attention} (已禁用)")
                continue
                
            print(f"\n处理注意力机制: {attention}")
            
            # 加载数据
            X_train, X_test, y_train, y_test = self.load_features(config['path'])
            
            if X_train is not None:
                # 训练和评估
                result = self.train_and_evaluate(X_train, X_test, y_train, y_test, 
                                                'attention_comparison', attention)
                attention_results[attention] = result
        
        self.results['attention_comparison'] = attention_results
        return attention_results

    def generate_comparison_report(self):
        """
        生成对比报告
        """
        print("\n" + "="*80)
        print(f"生成 {self.dataset_name} 数据集对比报告")
        print("="*80)
        
        # 创建对比表格
        comparison_data = []
        
        # 激活函数对比
        if 'activation_comparison' in self.results:
            for activation, result in self.results['activation_comparison'].items():
                comparison_data.append({
                    'Dataset': self.dataset_name,
                    'Experiment_Type': 'Activation',
                    'Condition': activation,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}",
                    'F1_Score': f"{result['f1_score']:.4f}",
                    'Training_Time(s)': f"{result['training_time']:.4f}",
                    'Test_Time(s)': f"{result['prediction_time']:.4f}"
                })
        
        # 注意力机制对比
        if 'attention_comparison' in self.results:
            for attention, result in self.results['attention_comparison'].items():
                comparison_data.append({
                    'Dataset': self.dataset_name,
                    'Experiment_Type': 'Attention',
                    'Condition': attention,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}",
                    'F1_Score': f"{result['f1_score']:.4f}",
                    'Training_Time(s)': f"{result['training_time']:.4f}",
                    'Test_Time(s)': f"{result['prediction_time']:.4f}"
                })
        
        # 创建DataFrame并保存
        df_comparison = pd.DataFrame(comparison_data)
        
        # 打印对比表格
        print("\n" + "="*120)
        print(f"ADCAE特征决策树对比实验结果 - {self.dataset_name}数据集")
        print("="*120)
        print(df_comparison.to_string(index=False))
        print("="*120)
        
        # 找出最佳性能
        if not df_comparison.empty:
            best_accuracy_idx = df_comparison['Accuracy'].astype(float).idxmax()
            best_f1_idx = df_comparison['F1_Score'].astype(float).idxmax()
            
            print(f"\n最佳准确率: {df_comparison.loc[best_accuracy_idx, 'Condition']} ({df_comparison.loc[best_accuracy_idx, 'Experiment_Type']}) - {df_comparison.loc[best_accuracy_idx, 'Accuracy']}")
            print(f"最佳F1分数: {df_comparison.loc[best_f1_idx, 'Condition']} ({df_comparison.loc[best_f1_idx, 'Experiment_Type']}) - {df_comparison.loc[best_f1_idx, 'F1_Score']}")
        
        # 保存CSV
        csv_path = os.path.join(self.output_path, f"adcae_dt_comparison_{self.dataset_name}.csv")
        df_comparison.to_csv(csv_path, index=False)
        
        # 保存详细结果JSON
        json_path = os.path.join(self.output_path, f"detailed_results_{self.dataset_name}.json")
        with open(json_path, 'w') as f:
            # 转换numpy数组为列表以便JSON序列化
            json_results = {}
            for exp_type, exp_results in self.results.items():
                json_results[exp_type] = {}
                for condition, result in exp_results.items():
                    json_result = result.copy()
                    # 确保所有数据都可以JSON序列化
                    if 'confusion_matrix' in json_result:
                        json_result['confusion_matrix'] = result['confusion_matrix']
                    json_results[exp_type][condition] = json_result
            
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n对比报告已保存:")
        print(f"CSV文件: {csv_path}")
        print(f"详细结果: {json_path}")
        
        return df_comparison
    
    def run_all_experiments(self):
        """
        运行所有对比实验
        """
        print(f"\n开始ADCAE特征决策树对比实验 - {self.dataset_name}数据集")
        print(f"输出目录: {self.output_path}")
        
        # 打印超参数配置摘要
        self.print_hyperparameters_summary()
        
        # 运行激活函数对比
        self.run_activation_comparison()
        
        # 运行注意力机制对比
        self.run_attention_comparison()
        
        # 生成对比报告
        comparison_df = self.generate_comparison_report()
        
        print(f"\n所有实验完成！结果已保存到: {self.output_path}")
        
        return self.results, comparison_df


def main():
    """
    主函数 - 支持多个数据集，每个数据集使用独立的超参数配置
    """
    # 定义要处理的数据集
    datasets = ["CTU", "USTC"]
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*100}")
        print(f"开始处理 {dataset_name} 数据集")
        print(f"{'='*100}")
        
        try:
            # 创建实验对象（每个数据集使用独立的超参数配置）
            experiment = ADCAEDTComparison(dataset_name=dataset_name)
            
            # 运行所有实验
            results, comparison_df = experiment.run_all_experiments()
            
            # 保存结果
            all_results[dataset_name] = {
                'results': results,
                'comparison_df': comparison_df
            }
            
            print(f"\n{dataset_name} 数据集实验完成！")
            
        except Exception as e:
            print(f"\n处理 {dataset_name} 数据集时出错: {e}")
            continue
    
    # 生成跨数据集对比报告
    generate_cross_dataset_comparison(all_results)
    
    print(f"\n{'='*100}")
    print("所有数据集实验总结:")
    for dataset_name in datasets:
        if dataset_name in all_results:
            print(f"✓ {dataset_name} 数据集: 实验完成")
        else:
            print(f"✗ {dataset_name} 数据集: 实验失败")
    print("1. 激活函数对比实验已完成 (ELU, ReLU, Sigmoid)")
    print("2. 注意力机制对比实验已完成 (With CBAM, Without Attention)")
    print("3. 每个数据集的每个实验条件使用了独立的决策树超参数配置")
    print("4. 对比报告已生成，包含所有性能指标")
    print("5. 跨数据集对比报告已生成")
    print(f"{'='*100}")


def generate_cross_dataset_comparison(all_results):
    """
    生成跨数据集对比报告
    """
    print(f"\n{'='*80}")
    print("生成跨数据集对比报告")
    print(f"{'='*80}")
    
    cross_comparison_data = []
    
    for dataset_name, dataset_results in all_results.items():
        results = dataset_results['results']
        
        # 激活函数对比
        if 'activation_comparison' in results:
            for activation, result in results['activation_comparison'].items():
                cross_comparison_data.append({
                    'Dataset': dataset_name,
                    'Experiment_Type': 'Activation',
                    'Condition': activation,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}",
                    'F1_Score': f"{result['f1_score']:.4f}",
                    'Training_Time(s)': f"{result['training_time']:.4f}",
                    'Test_Time(s)': f"{result['prediction_time']:.4f}"
                })
        
        # 注意力机制对比
        if 'attention_comparison' in results:
            for attention, result in results['attention_comparison'].items():
                cross_comparison_data.append({
                    'Dataset': dataset_name,
                    'Experiment_Type': 'Attention',
                    'Condition': attention,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}",
                    'F1_Score': f"{result['f1_score']:.4f}",
                    'Training_Time(s)': f"{result['training_time']:.4f}",
                    'Test_Time(s)': f"{result['prediction_time']:.4f}"
                })
    
    if cross_comparison_data:
        # 创建DataFrame
        df_cross_comparison = pd.DataFrame(cross_comparison_data)
        
        # 打印跨数据集对比表格
        print(f"\n{'='*150}")
        print("ADCAE特征决策树跨数据集对比实验结果")
        print(f"{'='*150}")
        print(df_cross_comparison.to_string(index=False))
        print(f"{'='*150}")
        
        # 保存跨数据集对比CSV
        cross_csv_path = "D:\\Python Project\\ADCAE\\ML+DL\\ADCAE_test\\DT\\cross_dataset_comparison.csv"
        df_cross_comparison.to_csv(cross_csv_path, index=False)
        
        print(f"\n跨数据集对比报告已保存: {cross_csv_path}")
    else:
        print("\n没有可用的跨数据集对比数据")


if __name__ == "__main__":
    main()