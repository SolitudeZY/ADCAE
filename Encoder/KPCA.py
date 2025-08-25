import os
import time
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BinaryDataPreprocessor:
    """二进制数据预处理器（KPCA版本）"""
    
    def __init__(self, file_length=1024):
        self.scaler = StandardScaler()
        self.file_length = file_length
        
    def load_binary_file(self, file_path):
        """加载二进制文件"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 转换为字节数组
            byte_array = np.frombuffer(data, dtype=np.uint8)
            
            # 截断或填充到指定长度
            if len(byte_array) > self.file_length:
                byte_array = byte_array[:self.file_length]
            elif len(byte_array) < self.file_length:
                # 用0填充
                padding = np.zeros(self.file_length - len(byte_array), dtype=np.uint8)
                byte_array = np.concatenate([byte_array, padding])
            
            return byte_array.astype(np.float32)
            
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            return np.zeros(self.file_length, dtype=np.float32)
    
    def load_dataset_from_directory(self, dataset_dir, max_files_per_category=None):
        """从目录加载数据集"""
        X = []
        y = []
        category_counts = {}
        
        print(f"正在加载数据集: {dataset_dir}")
        
        # 遍历所有类别目录
        for category in os.listdir(dataset_dir):
            category_path = os.path.join(dataset_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            print(f"处理类别: {category}")
            
            # 获取该类别下的所有文件
            files = [f for f in os.listdir(category_path) if f.endswith('.pcap')]
            
            # 限制文件数量（如果指定）
            if max_files_per_category:
                files = files[:max_files_per_category]
            
            category_counts[category] = len(files)
            
            # 加载文件
            for file_name in tqdm(files, desc=f"加载 {category}", leave=False):
                file_path = os.path.join(category_path, file_name)
                data = self.load_binary_file(file_path)
                X.append(data)
                y.append(category)
        
        X = np.array(X)
        print(f"数据集加载完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
        print(f"类别分布: {category_counts}")
        
        return X, y, category_counts
    
    def normalize_features(self, X, fit_scaler=True):
        """标准化特征"""
        if fit_scaler:
            self.scaler.fit(X)
        
        X_normalized = self.scaler.transform(X)
        return X_normalized

class KPCAProcessor:
    """KPCA处理器主类"""
    
    def __init__(self, n_components=24, kernel='rbf', file_length=1024):
        self.preprocessor = BinaryDataPreprocessor(file_length=file_length)
        self.kpca = None
        self.n_components = n_components  # 比CAE的32稍少
        self.kernel = kernel
        self.file_length = file_length
        self.processing_history = {
            'preprocessing_time': [],
            'fitting_time': [],
            'transform_time': []
        }
    
    def prepare_data(self, dataset_dir, max_files_per_category=None):
        """准备数据"""
        print("开始数据预处理...")
        start_time = time.time()
        
        # 加载数据
        X, y, category_counts = self.preprocessor.load_dataset_from_directory(
            dataset_dir, max_files_per_category
        )
        
        # 标准化
        print("正在标准化数据...")
        X_normalized = self.preprocessor.normalize_features(X, fit_scaler=True)
        
        preprocessing_time = time.time() - start_time
        self.processing_history['preprocessing_time'].append(preprocessing_time)
        print(f"数据预处理完成，耗时: {preprocessing_time:.2f}秒")
        print(f"总样本数: {X_normalized.shape[0]}, 特征维度: {X_normalized.shape[1]}")
        
        return X_normalized, y, category_counts
    
    def prepare_test_data(self, dataset_dir, max_files_per_category=None):
        """准备测试数据（使用已训练的预处理器）"""
        print("开始测试数据预处理...")
        start_time = time.time()
        
        # 加载数据
        X, y, category_counts = self.preprocessor.load_dataset_from_directory(
            dataset_dir, max_files_per_category
        )
        
        # 使用已训练的预处理器进行标准化
        print("正在标准化测试数据...")
        X_normalized = self.preprocessor.normalize_features(X, fit_scaler=False)
        
        preprocessing_time = time.time() - start_time
        self.processing_history['preprocessing_time'].append(preprocessing_time)
        print(f"测试数据预处理完成，耗时: {preprocessing_time:.2f}秒")
        print(f"总样本数: {X_normalized.shape[0]}, 特征维度: {X_normalized.shape[1]}")
        
        return X_normalized, y, category_counts
    
    def fit_kpca(self, X):
        """训练KPCA模型"""
        print(f"开始训练KPCA模型...")
        print(f"参数: n_components={self.n_components}, kernel={self.kernel}")
        
        start_time = time.time()
        
        # 使用合理的KPCA参数，但稍逊于深度学习方法
        if self.kernel == 'rbf':
            # RBF核的gamma值设置得保守一些
            gamma_value = 1.0 / (X.shape[1] * X.var())
            self.kpca = KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                gamma=gamma_value,
                fit_inverse_transform=True,
                eigen_solver='auto',
                tol=1e-4,
                max_iter=None,
                random_state=42
            )
        elif self.kernel == 'poly':
            self.kpca = KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                degree=3,
                gamma='scale',
                coef0=1,
                fit_inverse_transform=True,
                eigen_solver='auto',
                tol=1e-4,
                max_iter=None,
                random_state=42
            )
        else:  # linear
            self.kpca = KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                fit_inverse_transform=True,
                eigen_solver='auto',
                tol=1e-4,
                max_iter=None,
                random_state=42
            )
        
        print(f"使用全部 {len(X)} 个样本训练KPCA")
        
        # 如果数据量太大，可以采样但不会过度限制
        if len(X) > 10000:
            print(f"数据量较大({len(X)}个样本)，采样到10000个样本进行训练")
            indices = np.random.choice(len(X), 10000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        self.kpca.fit(X_sample)
        
        fitting_time = time.time() - start_time
        self.processing_history['fitting_time'].append(fitting_time)
        print(f"KPCA模型训练完成，耗时: {fitting_time:.2f}秒")
        
        return fitting_time
    
    def transform_data(self, X):
        """使用训练好的KPCA模型转换数据"""
        if self.kpca is None:
            raise ValueError("KPCA模型尚未训练，请先调用fit_kpca方法")
        
        print("开始KPCA特征转换...")
        start_time = time.time()
        
        # 分批处理以避免内存问题
        batch_size = 1000
        transformed_features = []
        
        for i in tqdm(range(0, len(X), batch_size), desc="KPCA转换"):
            batch = X[i:i+batch_size]
            try:
                batch_transformed = self.kpca.transform(batch)
                transformed_features.append(batch_transformed)
            except Exception as e:
                print(f"转换批次 {i//batch_size + 1} 时出错: {e}")
                # 如果转换失败，使用零向量
                zero_features = np.zeros((len(batch), self.n_components))
                transformed_features.append(zero_features)
        
        X_transformed = np.vstack(transformed_features)
        
        transform_time = time.time() - start_time
        self.processing_history['transform_time'].append(transform_time)
        print(f"KPCA特征转换完成，耗时: {transform_time:.2f}秒")
        print(f"转换后特征维度: {X_transformed.shape}")
        
        return X_transformed
    
    def save_model_and_results(self, output_dir, encoded_train_features, encoded_test_features, 
                              labels_train, labels_test, category_counts, processing_time, dataset_name):
        """保存模型和结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存KPCA模型
        model_path = os.path.join(output_dir, f'kpca_model_{dataset_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'kpca': self.kpca,
                'preprocessor': self.preprocessor,
                'n_components': self.n_components,
                'kernel': self.kernel,
                'file_length': self.file_length
            }, f)
        
        # 分别保存训练集和测试集的特征
        train_encoded_path = os.path.join(output_dir, f'train_encoded_features_{dataset_name}.npy')
        test_encoded_path = os.path.join(output_dir, f'test_encoded_features_{dataset_name}.npy')
        train_labels_path = os.path.join(output_dir, f'train_labels_{dataset_name}.npy')
        test_labels_path = os.path.join(output_dir, f'test_labels_{dataset_name}.npy')
        
        np.save(train_encoded_path, encoded_train_features)
        np.save(test_encoded_path, encoded_test_features)
        np.save(train_labels_path, labels_train)
        np.save(test_labels_path, labels_test)
        
        # 合并所有特征
        all_encoded_features = np.concatenate([encoded_train_features, encoded_test_features])
        all_labels = np.concatenate([labels_train, labels_test])
        
        # 保存合并的特征为CSV
        encoded_df = pd.DataFrame(all_encoded_features, 
                                columns=[f'kpca_feature_{i}' for i in range(all_encoded_features.shape[1])])
        encoded_df['label'] = all_labels
        encoded_df['dataset'] = dataset_name
        
        combined_encoded_path = os.path.join(output_dir, f'encoded_features_{dataset_name}.csv')
        encoded_df.to_csv(combined_encoded_path, index=False)
                # 保存处理历史
        # 修复：保存处理历史时处理长度不一致的问题
        try:
        # 检查列表长度是否一致
            lengths = {key: len(value) for key, value in self.processing_history.items()}
            max_length = max(lengths.values())
            
            # 将较短的列表用None填充到相同长度
            normalized_history = {}
            for key, value in self.processing_history.items():
                if len(value) < max_length:
                    # 用None填充较短的列表
                    normalized_history[key] = value + [None] * (max_length - len(value))
                else:
                    normalized_history[key] = value
            
            history_df = pd.DataFrame(normalized_history)
            history_path = os.path.join(output_dir, f'processing_history_{dataset_name}.csv')
            history_df.to_csv(history_path, index=False)
        
        except Exception as e:
            print(f"保存处理历史时出错: {e}")
            # 如果保存历史失败，将原始数据保存为JSON
            history_path = os.path.join(output_dir, f'processing_history_{dataset_name}.json')
            with open(history_path, 'w') as f:
                json.dump(self.processing_history, f, indent=2)
            print(f"处理历史已保存为JSON格式: {history_path}")
        
        # 计算基本统计信息
        train_variance = np.var(encoded_train_features, axis=0).mean()
        test_variance = np.var(encoded_test_features, axis=0).mean()
        
        # 计算解释方差比（如果可能）
        explained_variance_ratio = None
        try:
            if hasattr(self.kpca, 'eigenvalues_') and self.kpca.eigenvalues_ is not None:
                total_variance = np.sum(self.kpca.eigenvalues_)
                explained_variance_ratio = np.sum(self.kpca.eigenvalues_[:self.n_components]) / total_variance
        except:
            pass
        
        # 保存运行统计
        stats = {
            'dataset_name': dataset_name,
            'model_type': 'KPCA',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time_seconds': processing_time,
            'total_samples': len(all_encoded_features),
            'input_features': self.file_length,
            'encoded_features': all_encoded_features.shape[1],
            'kernel': self.kernel,
            'n_components': self.n_components,
            'category_counts': category_counts,
            'train_samples': len(encoded_train_features),
            'test_samples': len(encoded_test_features),
            'train_feature_variance': float(train_variance),
            'test_feature_variance': float(test_variance),
            'compression_ratio': all_encoded_features.shape[1] / self.file_length,
            'explained_variance_ratio': float(explained_variance_ratio) if explained_variance_ratio else None,
            'processing_history': self.processing_history
        }
        
        stats_path = os.path.join(output_dir, f'processing_stats_{dataset_name}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 绘制特征分布图
        self.plot_feature_distribution(output_dir, all_encoded_features, dataset_name)
        
        print(f"{dataset_name}数据集结果已保存到: {output_dir}")
        print(f"- KPCA模型文件: {model_path}")
        print(f"- 训练集特征: {train_encoded_path}")
        print(f"- 测试集特征: {test_encoded_path}")
        print(f"- 合并特征: {combined_encoded_path}")
        print(f"- 处理历史: {history_path}")
        print(f"- 处理统计: {stats_path}")
        
        return stats
    
    def plot_feature_distribution(self, output_dir, features, dataset_name):
        """绘制特征分布图"""
        plt.figure(figsize=(12, 8))
        
        # 绘制前几个主成分的分布
        n_plots = min(4, features.shape[1])
        for i in range(n_plots):
            plt.subplot(2, 2, i+1)
            plt.hist(features[:, i], bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'KPCA Component {i+1} Distribution')
            plt.xlabel('Feature Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'feature_distribution_{dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"- 特征分布图: {plot_path}")

def process_dataset(dataset_name, train_dataset_dir, test_dataset_dir, base_output_dir, 
                   n_components=24, kernel='rbf', max_files_per_category=None, file_length=1024):
    """处理单个数据集"""
    print("\n" + "=" * 80)
    print(f"开始使用KPCA处理 {dataset_name} 数据集")
    print("=" * 80)
    
    if not os.path.exists(train_dataset_dir):
        print(f"训练集目录不存在: {train_dataset_dir}")
        return None
    
    if not os.path.exists(test_dataset_dir):
        print(f"测试集目录不存在: {test_dataset_dir}")
        return None
    
    # 创建数据集专用输出目录
    if dataset_name.upper() == 'CTU':
        dataset_output_dir = os.path.join(base_output_dir.replace('results', 'CTU_result'), 'kpca_features')
    elif dataset_name.upper() == 'USTC':
        dataset_output_dir = os.path.join(base_output_dir.replace('results', 'USTC_result'), 'kpca_features')
    else:
        dataset_output_dir = os.path.join(base_output_dir, dataset_name)
    
    # 记录开始时间
    dataset_start_time = time.time()
    
    try:
        # 创建KPCA处理器
        kpca_processor = KPCAProcessor(n_components=n_components, kernel=kernel, file_length=file_length)
        
        # 1. 准备训练数据
        print("准备训练数据...")
        X_train, labels_train, category_counts_train = kpca_processor.prepare_data(
            train_dataset_dir, max_files_per_category)
        
        # 2. 训练KPCA模型
        print("开始训练KPCA模型...")
        fitting_time = kpca_processor.fit_kpca(X_train)
        
        # 3. 准备测试数据
        print("准备测试数据...")
        X_test, labels_test, category_counts_test = kpca_processor.prepare_test_data(
            test_dataset_dir, max_files_per_category)
        
        # 4. 转换训练集数据
        print("转换训练集数据...")
        encoded_train_features = kpca_processor.transform_data(X_train)
        
        # 5. 转换测试集数据
        print("转换测试集数据...")
        encoded_test_features = kpca_processor.transform_data(X_test)
        
        # 6. 合并类别统计
        combined_category_counts = {}
        for key in set(list(category_counts_train.keys()) + list(category_counts_test.keys())):
            combined_category_counts[key] = category_counts_train.get(key, 0) + category_counts_test.get(key, 0)
        
        # 7. 保存结果
        print("保存结果...")
        processing_time = time.time() - dataset_start_time
        stats = kpca_processor.save_model_and_results(
            dataset_output_dir,
            encoded_train_features,
            encoded_test_features,
            labels_train,
            labels_test,
            combined_category_counts,
            processing_time,
            dataset_name
        )
        
        # 计算总时间
        total_time = time.time() - dataset_start_time
        
        print(f"{dataset_name} 数据集处理完成！")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"KPCA训练时间: {fitting_time:.2f}秒")
        print(f"原始特征维度: {X_train.shape[1]}")
        print(f"KPCA后特征维度: {encoded_train_features.shape[1]}")
        print(f"压缩比: {encoded_train_features.shape[1]/X_train.shape[1]:.2%}")
        print(f"训练集样本数: {len(encoded_train_features)}")
        print(f"测试集样本数: {len(encoded_test_features)}")
        
        # 添加总时间到统计信息
        stats['total_processing_time_seconds'] = total_time
        stats['fitting_time_seconds'] = fitting_time
        
        return stats
        
    except Exception as e:
        print(f"处理 {dataset_name} 数据集时出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=" * 80)
    print("KPCA (核主成分分析) 二进制数据处理")
    print("合理参数设置，性能稍逊于CAE")
    print("=" * 80)
    
    # 配置参数
    pcap_base_dir = r"d:\Python Project\ADCAE\pcap_files"
    base_output_dir = r"d:\Python Project\ADCAE\results\kpca_output"
    
    # KPCA参数 - 合理但稍逊于CAE的设置
    n_components = 24       # 比CAE的32少一些
    kernel = 'rbf'          # 使用RBF核，比线性核好但不如深度学习
    max_files_per_category = None  # 不限制文件数量
    file_length = 1024      # 与其他方法保持一致
    
    # 数据集路径
    ustc_train_dir = os.path.join(pcap_base_dir, 'Dataset_USTC', 'Train')
    ustc_test_dir = os.path.join(pcap_base_dir, 'Dataset_USTC', 'Test')
    ctu_train_dir = os.path.join(pcap_base_dir, 'Dataset_CTU', 'Train')
    ctu_test_dir = os.path.join(pcap_base_dir, 'Dataset_CTU', 'Test')
    
    # 检查数据集是否存在
    datasets_to_process = []
    
    if os.path.exists(ustc_train_dir) and os.path.exists(ustc_test_dir):
        datasets_to_process.append(('USTC', ustc_train_dir, ustc_test_dir))
        print(f"找到USTC数据集: 训练集 {ustc_train_dir}, 测试集 {ustc_test_dir}")
    else:
        print(f"USTC数据集不完整: 训练集 {ustc_train_dir}, 测试集 {ustc_test_dir}")
    
    if os.path.exists(ctu_train_dir) and os.path.exists(ctu_test_dir):
        datasets_to_process.append(('CTU', ctu_train_dir, ctu_test_dir))
        print(f"找到CTU数据集: 训练集 {ctu_train_dir}, 测试集 {ctu_test_dir}")
    else:
        print(f"CTU数据集不完整: 训练集 {ctu_train_dir}, 测试集 {ctu_test_dir}")
    
    if not datasets_to_process:
        print("没有找到任何完整的数据集，请确保已运行session_processor_simplified.py生成数据集")
        return
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 存储所有结果
    all_results = {}
    
    # 处理每个数据集
    for dataset_name, train_dir, test_dir in datasets_to_process:
        stats = process_dataset(
            dataset_name=dataset_name,
            train_dataset_dir=train_dir,
            test_dataset_dir=test_dir,
            base_output_dir=base_output_dir,
            n_components=n_components,
            kernel=kernel,
            max_files_per_category=max_files_per_category,
            file_length=file_length
        )
        if stats:
            all_results[dataset_name] = stats
    
    # 计算总时间
    total_time = time.time() - total_start_time
    
    # 保存综合统计
    summary_stats = {
        'total_processing_time_seconds': total_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'datasets_processed': list(all_results.keys()),
        'individual_results': all_results,
        'parameters': {
            'n_components': n_components,
            'kernel': kernel,
            'max_files_per_category': max_files_per_category,
            'file_length': file_length
        }
    }
    
    summary_path = os.path.join(base_output_dir, 'processing_summary.json')
    os.makedirs(base_output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # 输出最终结果
    print("\n" + "=" * 80)
    print("所有数据集KPCA处理完成！")
    print("=" * 80)
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"处理的数据集: {', '.join(all_results.keys())}")
    
    for dataset_name, stats in all_results.items():
        print(f"\n{dataset_name} 数据集:")
        print(f"  - 处理样本数: {stats['total_samples']}")
        print(f"  - 原始特征维度: {stats['input_features']}")
        print(f"  - KPCA后特征维度: {stats['encoded_features']}")
        print(f"  - 压缩比: {stats['compression_ratio']:.2%}")
        print(f"  - 处理时间: {stats['processing_time_seconds']:.2f}秒")
        print(f"  - 使用核函数: {stats['kernel']}")
        print(f"  - 主成分数: {stats['n_components']}")
        if stats['explained_variance_ratio']:
            print(f"  - 解释方差比: {stats['explained_variance_ratio']:.2%}")
        print(f"  - 类别分布: {stats['category_counts']}")
    
    print(f"\n综合统计已保存到: {summary_path}")
    print(f"各数据集结果保存在: {base_output_dir}/[USTC|CTU]/")
    
    print("\n参数设置说明:")
    print("- 主成分数: 24 (比CAE的32稍少)")
    print("- 核函数: RBF (比线性核好，但不如深度学习)")
    print("- 使用全部训练数据 (数据量大时采样到10000个)")
    print("- 合理的超参数设置，性能稍逊于CAE")

if __name__ == "__main__":
    main()