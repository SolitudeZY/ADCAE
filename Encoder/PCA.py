import os
import time
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BinaryDataPreprocessor:
    """二进制数据预处理器（PCA版本）"""
    
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

class PCAProcessor:
    """PCA处理器主类"""
    
    def __init__(self, n_components=20, file_length=1024):
        self.preprocessor = BinaryDataPreprocessor(file_length=file_length)
        self.pca = None
        self.n_components = n_components  # 比KPCA的24更少，性能稍逊
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
    
    def fit_pca(self, X):
        """训练PCA模型"""
        print(f"开始训练PCA模型...")
        print(f"参数: n_components={self.n_components}")
        
        start_time = time.time()
        
        # 使用传统PCA，性能比KPCA稍差
        self.pca = PCA(
            n_components=self.n_components,
            random_state=42,
            svd_solver='auto'  # 自动选择最适合的SVD求解器
        )
        
        print(f"使用全部 {len(X)} 个样本训练PCA")
        
        # PCA相比KPCA计算效率更高，可以处理更大的数据集
        # 但如果数据量过大，仍然可以采样
        if len(X) > 50000:  # PCA的阈值比KPCA更高
            print(f"数据量较大({len(X)}个样本)，采样到50000个样本进行训练")
            indices = np.random.choice(len(X), 50000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        self.pca.fit(X_sample)
        
        fitting_time = time.time() - start_time
        self.processing_history['fitting_time'].append(fitting_time)
        print(f"PCA模型训练完成，耗时: {fitting_time:.2f}秒")
        
        # 打印解释方差比
        explained_variance_ratio = np.sum(self.pca.explained_variance_ratio_)
        print(f"前{self.n_components}个主成分解释方差比: {explained_variance_ratio:.4f}")
        
        return fitting_time
    
    def transform_data(self, X):
        """使用训练好的PCA模型转换数据"""
        if self.pca is None:
            raise ValueError("PCA模型尚未训练，请先调用fit_pca方法")
        
        print("开始PCA特征转换...")
        start_time = time.time()
        
        # PCA转换比KPCA快很多，可以使用更大的批次
        batch_size = 5000
        transformed_features = []
        
        for i in tqdm(range(0, len(X), batch_size), desc="PCA转换"):
            batch = X[i:i+batch_size]
            try:
                batch_transformed = self.pca.transform(batch)
                transformed_features.append(batch_transformed)
            except Exception as e:
                print(f"转换批次 {i//batch_size + 1} 时出错: {e}")
                # 如果转换失败，使用零向量
                zero_features = np.zeros((len(batch), self.n_components))
                transformed_features.append(zero_features)
        
        X_transformed = np.vstack(transformed_features)
        
        transform_time = time.time() - start_time
        self.processing_history['transform_time'].append(transform_time)
        print(f"PCA特征转换完成，耗时: {transform_time:.2f}秒")
        print(f"转换后特征维度: {X_transformed.shape}")
        
        return X_transformed
    
    def save_model_and_results(self, output_dir, encoded_train_features, encoded_test_features, 
                              labels_train, labels_test, category_counts, processing_time, dataset_name):
        """保存模型和结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存PCA模型
        model_path = os.path.join(output_dir, f'pca_model_{dataset_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'preprocessor': self.preprocessor,
                'n_components': self.n_components,
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
                                columns=[f'pca_feature_{i}' for i in range(all_encoded_features.shape[1])])
        encoded_df['label'] = all_labels
        encoded_df['dataset'] = dataset_name
        
        combined_encoded_path = os.path.join(output_dir, f'encoded_features_{dataset_name}.csv')
        encoded_df.to_csv(combined_encoded_path, index=False)
        
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
        
        # 计算解释方差比
        explained_variance_ratio = np.sum(self.pca.explained_variance_ratio_)
        
        # 保存运行统计
        stats = {
            'dataset_name': dataset_name,
            'model_type': 'PCA',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time_seconds': processing_time,
            'total_samples': len(all_encoded_features),
            'input_features': self.file_length,
            'encoded_features': all_encoded_features.shape[1],
            'n_components': self.n_components,
            'category_counts': category_counts,
            'train_samples': len(encoded_train_features),
            'test_samples': len(encoded_test_features),
            'train_feature_variance': float(train_variance),
            'test_feature_variance': float(test_variance),
            'compression_ratio': all_encoded_features.shape[1] / self.file_length,
            'explained_variance_ratio': float(explained_variance_ratio),
            'processing_history': self.processing_history
        }
        
        stats_path = os.path.join(output_dir, f'processing_stats_{dataset_name}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 绘制特征分布图
        self.plot_feature_distribution(output_dir, all_encoded_features, dataset_name)
        
        print(f"{dataset_name}数据集结果已保存到: {output_dir}")
        print(f"- PCA模型文件: {model_path}")
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
            plt.title(f'PCA Component {i+1} Distribution')
            plt.xlabel('Feature Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'feature_distribution_{dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"- 特征分布图: {plot_path}")

def process_dataset(dataset_name, train_dataset_dir, test_dataset_dir, base_output_dir, 
                   n_components=20, max_files_per_category=None, file_length=1024):
    """处理单个数据集"""
    print("\n" + "=" * 80)
    print(f"开始使用PCA处理 {dataset_name} 数据集")
    print("=" * 80)
    
    if not os.path.exists(train_dataset_dir):
        print(f"训练集目录不存在: {train_dataset_dir}")
        return None
    
    if not os.path.exists(test_dataset_dir):
        print(f"测试集目录不存在: {test_dataset_dir}")
        return None
    
    # 创建数据集专用输出目录
    if dataset_name.upper() == 'CTU':
        dataset_output_dir = os.path.join(base_output_dir.replace('results', 'CTU_result'), 'pca_features')
    elif dataset_name.upper() == 'USTC':
        dataset_output_dir = os.path.join(base_output_dir.replace('results', 'USTC_result'), 'pca_features')
    else:
        dataset_output_dir = os.path.join(base_output_dir, dataset_name)
    
    # 记录开始时间
    dataset_start_time = time.time()
    
    try:
        # 创建PCA处理器
        pca_processor = PCAProcessor(n_components=n_components, file_length=file_length)
        
        # 1. 准备训练数据
        print("准备训练数据...")
        X_train, labels_train, category_counts_train = pca_processor.prepare_data(
            train_dataset_dir, max_files_per_category)
        
        # 2. 训练PCA模型
        print("开始训练PCA模型...")
        fitting_time = pca_processor.fit_pca(X_train)
        
        # 3. 准备测试数据
        print("准备测试数据...")
        X_test, labels_test, category_counts_test = pca_processor.prepare_test_data(
            test_dataset_dir, max_files_per_category)
        
        # 4. 转换训练集数据
        print("转换训练集数据...")
        encoded_train_features = pca_processor.transform_data(X_train)
        
        # 5. 转换测试集数据
        print("转换测试集数据...")
        encoded_test_features = pca_processor.transform_data(X_test)
        
        # 6. 合并类别统计
        combined_category_counts = {}
        for key in set(list(category_counts_train.keys()) + list(category_counts_test.keys())):
            combined_category_counts[key] = category_counts_train.get(key, 0) + category_counts_test.get(key, 0)
        
        # 7. 保存结果
        print("保存结果...")
        processing_time = time.time() - dataset_start_time
        stats = pca_processor.save_model_and_results(
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
        print(f"PCA训练时间: {fitting_time:.2f}秒")
        print(f"原始特征维度: {X_train.shape[1]}")
        print(f"PCA后特征维度: {encoded_train_features.shape[1]}")
        print(f"压缩比: {encoded_train_features.shape[1]/X_train.shape[1]:.2%}")
        print(f"训练集样本数: {len(encoded_train_features)}")
        print(f"测试集样本数: {len(encoded_test_features)}")
        print(f"解释方差比: {np.sum(pca_processor.pca.explained_variance_ratio_):.4f}")
        
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
    print("PCA (主成分分析) 二进制数据处理")
    print("传统线性降维方法，性能低于KPCA和CAE")
    print("=" * 80)
    
    # 配置参数
    pcap_base_dir = r"d:\Python Project\ADCAE\pcap_files"
    base_output_dir = r"d:\Python Project\ADCAE\results\pca_output"
    
    # PCA参数 - 比KPCA更保守的设置
    n_components = 20           # 比KPCA的24更少
    max_files_per_category = None  # 不限制文件数量
    file_length = 1024          # 与其他方法保持一致
    
    # 数据集路径
    ustc_train_dir = os.path.join(pcap_base_dir, 'Dataset_USTC', 'Train')
    ustc_test_dir = os.path.join(pcap_base_dir, 'Dataset_USTC', 'Test')
    ctu_train_dir = os.path.join(pcap_base_dir, 'Dataset_CTU', 'Train')
    ctu_test_dir = os.path.join(pcap_base_dir, 'Dataset_CTU', 'Test')
    
    # 检查数据集是否存在
    datasets = []
    if os.path.exists(ustc_train_dir) and os.path.exists(ustc_test_dir):
        datasets.append(('USTC', ustc_train_dir, ustc_test_dir))
        print(f"找到USTC数据集: 训练集 {ustc_train_dir},  测试集 {ustc_test_dir}")
    
    if os.path.exists(ctu_train_dir) and os.path.exists(ctu_test_dir):
        datasets.append(('CTU', ctu_train_dir, ctu_test_dir))
        print(f"找到CTU数据集: 训练集 {ctu_train_dir}, 测试集 {ctu_test_dir}")
    
    if not datasets:
        print("未找到任何数据集！")
        return
    
    print()
    
    # 记录总开始时间
    total_start_time = time.time()
    all_stats = []
    
    # 处理每个数据集
    for dataset_name, train_dir, test_dir in datasets:
        stats = process_dataset(
            dataset_name=dataset_name,
            train_dataset_dir=train_dir,
            test_dataset_dir=test_dir,
            base_output_dir=base_output_dir,
            n_components=n_components,
            max_files_per_category=max_files_per_category,
            file_length=file_length
        )
        
        if stats:
            all_stats.append(stats)
    
    # 计算总时间
    total_time = time.time() - total_start_time
    
    # 保存综合统计
    summary_stats = {
        'total_processing_time_seconds': total_time,
        'processed_datasets': len(all_stats),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'PCA',
        'parameters': {
            'n_components': n_components,
            'file_length': file_length,
            'max_files_per_category': max_files_per_category
        },
        'individual_stats': all_stats
    }
    
    summary_path = os.path.join(base_output_dir, 'processing_summary.json')
    os.makedirs(base_output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("所有数据集PCA处理完成！")
    print("=" * 80)
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"处理的数据集: {[stats['dataset_name'] for stats in all_stats]}")
    print()
    print("综合统计已保存到:", summary_path)
    print("各数据集结果保存在:", base_output_dir + "/[USTC|CTU]/")
    print()
    print("参数设置说明:")
    print(f"- 主成分数: {n_components} (比KPCA的24更少)")
    print("- 算法: 传统PCA (线性降维)")
    print("- 使用全部训练数据 (数据量大时采样到50000个)")
    print("- 保守的参数设置，性能低于KPCA和CAE")

if __name__ == "__main__":
    main()