import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import time

class DTFeatureClassifier:
    """
    使用PCA、KPCA和CAE特征进行决策树分类的处理器
    """
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
        # 初始化标签编码器
        self.label_encoder = LabelEncoder()
        
        # 设置数据路径
        base_path = f"D:\\Python Project\\ADCAE\\{dataset_name}_result"
        # self.pca_data_path = os.path.join(base_path, "pca_output", "pca_features")
        # self.kpca_data_path = os.path.join(base_path, "kpca_output", "kpca_features") # 若处理PCA和KPCA，解除注释即可
        self.cae_data_path = os.path.join(base_path, "cae_output", "cae_features")
        self.adcae_data_path = os.path.join(base_path, "adcae_features", "activation_comparison", "elu")
        
        # 设置输出路径
        self.output_path = os.path.join(base_path, "dt_output_features")
        os.makedirs(self.output_path, exist_ok=True)
        
        # 获取数据集特定的超参数
        self.hyperparameters = self._get_dataset_hyperparameters()
        
    def _get_dataset_hyperparameters(self):
        """
        根据数据集名称返回对应的决策树超参数
        """
        if self.dataset_name == "USTC":
            return {
                'PCA': {
                    'max_depth': 11,
                    'min_samples_split': 12,
                    'min_samples_leaf': 7,
                    'max_features': 'sqrt',
                    'criterion': 'gini',
                    'class_weight': None,
                    'random_state': 42
                },
                'KPCA': {
                    'max_depth': 12,
                    'min_samples_split': 12,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'criterion': 'gini',
                    'class_weight': None,
                    'random_state': 42
                },
                'CAE': { # USTC - 激进调优：从81.79%提升到89%
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': None,
                    'criterion': 'entropy',        # 改为entropy，对复杂模式更敏感
                    'class_weight': 'balanced',    # 添加类别平衡，这是关键！
                    'random_state': 42,
                    'splitter': 'best',
                    'min_impurity_decrease': 0.0,  # 确保无纯度约束
                    'ccp_alpha': 0.0               # 确保无剪枝约束
                },
                'ADCAE': {  # USTC - 从87.2%提升到88%-89%
                    'max_depth': None,  # 适度增加深度
                    'min_samples_split': 2,   # 降低分割约束
                    'min_samples_leaf': 1,    # 降低叶子节点约束
                    'max_features': None,
                    'criterion': 'entropy',
                    'class_weight': 'balanced',  # 处理类别不平衡
                    'random_state': 42,
                    'splitter': 'best',      # 确保使用最佳分割策略
                }
            }
        elif self.dataset_name == "CTU":
            return {
                'PCA': {
                    'max_depth': 40,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'criterion': 'gini',
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'KPCA': {
                    'max_depth': 30,
                    'min_samples_split': 4,
                    'min_samples_leaf': 2,
                    'max_features': 'log2',
                    'criterion': 'entropy',
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'CAE': {  # CTU - 从80.6%提升到88%-89%
                    'max_depth': None,          # 大幅增加深度以提高复杂度
                    'min_samples_split': 10,   # 降低分割约束
                    'min_samples_leaf': 8,    # 降低叶子节点约束
                    'max_features': None,
                    'criterion': 'entropy',
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'ADCAE': {  # CTU - 从85.8%提升到88%-89%
                    'max_depth': None,          # 进一步增加深度
                    'min_samples_split': 2,   # 更低的分割约束
                    'min_samples_leaf': 1,    # 最低的叶子节点约束
                    'max_features': None,
                    'criterion': 'entropy',
                    'class_weight': 'balanced', # 'balanced'
                    'random_state': 42
                }
            }
        else:
            # 默认配置
            return {
                "PCA": {
                    "max_depth": 12,
                    "min_samples_split": 8,
                    "min_samples_leaf": 3,
                    "max_features": "sqrt",
                    "criterion": "gini",
                    "class_weight": "balanced",
                    "random_state": 42
                },
                "KPCA": {
                    "max_depth": 10,
                    "min_samples_split": 10,
                    "min_samples_leaf": 4,
                    "max_features": "log2",
                    "criterion": "entropy",
                    "class_weight": "balanced",
                    "random_state": 42
                }
            }
    
    def load_pca_data(self):
        """
        加载PCA特征数据（从CSV文件）
        """
        print("加载PCA特征数据...")
        
        # 从CSV文件加载完整的编码特征
        csv_path = os.path.join(self.pca_data_path, f"encoded_features_{self.dataset_name}.csv")
        df = pd.read_csv(csv_path)
        
        # 排除非特征列
        columns_to_drop = ['label', 'dataset']  # 添加dataset列到排除列表
        feature_columns = [col for col in df.columns if col not in columns_to_drop]
        
        # 分离特征和标签
        X = df[feature_columns].values
        y = df['label'].values
        
        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        print(f"PCA训练集形状: {X_train.shape}")
        print(f"PCA测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def load_kpca_data(self):
        """
        加载KPCA特征数据（从CSV文件）
        """
        print("加载KPCA特征数据...")
        
        # 从CSV文件加载完整的编码特征
        csv_path = os.path.join(self.kpca_data_path, f"encoded_features_{self.dataset_name}.csv")
        df = pd.read_csv(csv_path)
        
        # 排除非特征列
        columns_to_drop = ['label', 'dataset']  # 添加dataset列到排除列表
        feature_columns = [col for col in df.columns if col not in columns_to_drop]
        
        # 分离特征和标签
        X = df[feature_columns].values
        y = df['label'].values
        
        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        print(f"KPCA训练集形状: {X_train.shape}")
        print(f"KPCA测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_feature_engineering(self, X_train, X_test, y_train, feature_type="CAE"):
        """
        对CAE特征应用特征工程优化
        
        Args:
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练标签
            feature_type: 特征类型
        
        Returns:
            优化后的训练和测试特征
        """
        print(f"\n对{feature_type}特征应用特征工程优化...")
        
        X_train_optimized = X_train.copy()
        X_test_optimized = X_test.copy()
        
        # 1. 特征缩放 - 使用RobustScaler对异常值更鲁棒
        print("步骤1: 特征标准化...")
        self.feature_scaler = RobustScaler()
        X_train_optimized = self.feature_scaler.fit_transform(X_train_optimized)
        X_test_optimized = self.feature_scaler.transform(X_test_optimized)
        print(f"标准化后训练集形状: {X_train_optimized.shape}")
        
        # 2. 特征选择 - 使用互信息选择最相关的特征
        print("步骤2: 特征选择...")
        # 选择前80%的特征
        k_features = int(X_train_optimized.shape[1] * 0.8)
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        X_train_optimized = self.feature_selector.fit_transform(X_train_optimized, y_train)
        X_test_optimized = self.feature_selector.transform(X_test_optimized)
        print(f"特征选择后训练集形状: {X_train_optimized.shape}")
        
        # 3. 递归特征消除 - 进一步优化特征
        print("步骤3: 递归特征消除...")
        # 使用轻量级决策树进行特征排序
        estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
        n_features_to_select = max(50, int(X_train_optimized.shape[1] * 0.7))
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        X_train_optimized = rfe.fit_transform(X_train_optimized, y_train)
        X_test_optimized = rfe.transform(X_test_optimized)
        print(f"RFE后训练集形状: {X_train_optimized.shape}")
        
        # 4. 可选：PCA降维（如果特征数量仍然很大）
        if X_train_optimized.shape[1] > 64:
            print("步骤4: PCA降维...")
            self.feature_reducer = PCA(n_components=64, random_state=42)
            X_train_optimized = self.feature_reducer.fit_transform(X_train_optimized)
            X_test_optimized = self.feature_reducer.transform(X_test_optimized)
            print(f"PCA降维后训练集形状: {X_train_optimized.shape}")
            print(f"保留的方差比例: {self.feature_reducer.explained_variance_ratio_.sum():.4f}")
        
        print(f"特征工程完成！最终特征维度: {X_train_optimized.shape[1]}")
        
        return X_train_optimized, X_test_optimized
    
    def enhanced_load_cae_data(self):
        """
        加载CAE特征数据并应用特征工程优化
        """
        print("加载CAE特征数据...")
        
        # 从CSV文件加载训练和测试特征
        train_path = os.path.join(self.cae_data_path, "train_features.csv")
        test_path = os.path.join(self.cae_data_path, "test_features.csv")
        
        # 检查文件是否存在
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"CAE训练特征文件不存在: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"CAE测试特征文件不存在: {test_path}")
        
        # 读取CSV文件
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"训练数据形状: {train_df.shape}")
        print(f"测试数据形状: {test_df.shape}")
        
        # 排除非特征列
        columns_to_drop = ['label', 'dataset']
        actual_columns_to_drop = [col for col in columns_to_drop if col in train_df.columns]
        feature_columns = [col for col in train_df.columns if col not in actual_columns_to_drop]
        
        print(f"特征列数量: {len(feature_columns)}")
        print(f"排除的列: {actual_columns_to_drop}")
        
        # 分离特征和标签
        X_train = train_df[feature_columns].values
        X_test = test_df[feature_columns].values
        
        # 获取标签
        if 'label' in train_df.columns:
            y_train = train_df['label'].values
            y_test = test_df['label'].values
        else:
            raise ValueError("CAE特征文件中未找到标签列 'label'")
        
        print(f"原始CAE训练集形状: {X_train.shape}")
        print(f"原始CAE测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        # 应用特征工程优化
        X_train_optimized, X_test_optimized = self.apply_feature_engineering(
            X_train, X_test, y_train, "CAE"
        )
        
        return X_train_optimized, X_test_optimized, y_train, y_test
    
    def load_adcae_data(self):
        """
        加载ADCAE特征数据
        """
        print("加载ADCAE特征数据...")
        
        # 加载训练和测试特征
        train_path = os.path.join(self.adcae_data_path, "train_features.csv")
        test_path = os.path.join(self.adcae_data_path, "test_features.csv")
        
        # 读取CSV文件
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 打印列信息以便调试
        print(f"训练数据列名: {list(train_df.columns)}")
        print(f"训练数据形状: {train_df.shape}")
        
        # 检查并移除非数值列
        # 通常需要排除的列：'label', 'dataset', 可能还有索引列等
        columns_to_drop = []
        
        # 检查每一列的数据类型
        for col in train_df.columns:
            if train_df[col].dtype == 'object':  # 字符串类型列
                if col.lower() in ['label', 'dataset', 'index', 'id']:
                    columns_to_drop.append(col)
                else:
                    # 检查是否包含字符串值
                    sample_values = train_df[col].dropna().head(10)
                    if any(isinstance(val, str) for val in sample_values):
                        print(f"发现字符串列: {col}, 样本值: {sample_values.tolist()}")
                        columns_to_drop.append(col)
        
        print(f"将要删除的列: {columns_to_drop}")
        
        # 分离特征和标签
        if 'label' in train_df.columns:
            y_train = train_df['label'].copy()  # 使用copy()避免警告
            y_test = test_df['label'].copy()
        else:
            raise ValueError("未找到标签列 'label'")
        
        # 删除非特征列
        X_train = train_df.drop(columns=columns_to_drop, errors='ignore').copy()
        X_test = test_df.drop(columns=columns_to_drop, errors='ignore').copy()
        
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
        
        print(f"ADCAE训练集形状: {X_train.shape}")
        print(f"ADCAE测试集形状: {X_test.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test
    
    def encode_labels(self, y_train, y_test):
        """
        编码标签
        """
        # 如果标签是字符串，进行编码
        if y_train.dtype == 'object' or isinstance(y_train[0], str):
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            return y_train_encoded, y_test_encoded
        else:
            return y_train, y_test
    
    def build_dt_model(self, feature_type="PCA"):
        """
        构建决策树模型
        
        Args:
            feature_type: 特征类型 ("PCA" 或 "KPCA")
        """
        # 获取对应的超参数
        params = self.hyperparameters[feature_type]
        
        print(f"\n{self.dataset_name}数据集 - {feature_type}特征决策树超参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        dt_model = DecisionTreeClassifier(**params)
        
        return dt_model
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_type):
        """
        训练和评估决策树模型
        """
        print(f"\n开始训练{feature_type}决策树模型...")
        
        # 编码标签
        y_train_encoded, y_test_encoded = self.encode_labels(y_train, y_test)
        
        # 构建模型
        dt_model = self.build_dt_model(feature_type)
        
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
        
        # 平均每样本预测时间
        avg_prediction_time = prediction_time / len(y_test_encoded)
        
        print(f"\n{feature_type}决策树模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"训练时间: {training_time:.4f}秒")
        print(f"测试预测时间: {prediction_time:.4f}秒")
        print(f"平均每样本预测时间: {avg_prediction_time:.6f}秒")
        
        # 分类报告
        if hasattr(self.label_encoder, 'classes_'):
            target_names = self.label_encoder.classes_
        else:
            target_names = [f"Class_{i}" for i in range(len(np.unique(y_train_encoded)))]
        
        report = classification_report(y_test_encoded, y_pred, target_names=target_names, zero_division=0)
        print(f"\n分类报告:\n{report}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # 创建结果字典
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'avg_prediction_time': avg_prediction_time,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_type': feature_type,
            'hyperparameters': self.hyperparameters[feature_type]
        }
        
        # 保存结果
        self.save_results(dt_model, results)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, target_names, feature_type)
        
        # 只返回结果字典，不返回模型
        return results
    
    def save_results(self, model, results):
        """
        保存模型和结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_type = results['feature_type']
        
        # 保存模型
        model_path = os.path.join(self.output_path, f"dt_{feature_type.lower()}_model_{self.dataset_name}_{timestamp}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存到: {model_path}")
        
        # 保存结果为JSON
        import json
        results_path = os.path.join(self.output_path, f"dt_{feature_type.lower()}_results_{self.dataset_name}_{timestamp}.json")
        
        # 转换numpy数组为列表以便JSON序列化
        json_results = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in results.items()}
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 保存性能指标CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                       'Training_Time', 'Prediction_Time', 'Avg_Prediction_Time_Per_Sample'],
            'Value': [
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['f1_score'],
                results['training_time'],
                results['prediction_time'],
                results['avg_prediction_time']
            ]
        })
        
        metrics_path = os.path.join(self.output_path, f"dt_{feature_type.lower()}_metrics_{self.dataset_name}_{timestamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"结果已保存到: {self.output_path}")
    
    def plot_confusion_matrix(self, cm, target_names, feature_type):
        """
        绘制混淆矩阵
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'{feature_type} Decision Tree - Confusion Matrix ({self.dataset_name})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_path, f"dt_{feature_type.lower()}_confusion_matrix_{self.dataset_name}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"混淆矩阵图已保存到: {plot_path}")
    
    def run_all_classifications(self):
        """
        运行所有特征类型的分类任务
        """
        results = {}
        
        # CAE特征分类 - 使用特征工程优化
        print("=" * 60)
        print(f"{self.dataset_name} - CAE特征决策树分类（特征工程优化）")
        print("=" * 60)
        try:
            X_train_cae, X_test_cae, y_train_cae, y_test_cae = self.enhanced_load_cae_data()
            cae_results = self.train_and_evaluate(X_train_cae, X_test_cae, y_train_cae, y_test_cae, "CAE")
            results['CAE'] = cae_results
        except Exception as e:
            print(f"CAE分类失败: {e}")
            results['CAE'] = None
        
        # ADCAE特征分类
        print("\n" + "=" * 60)
        print(f"{self.dataset_name} - ADCAE特征决策树分类")
        print("=" * 60)
        try:
            X_train_adcae, X_test_adcae, y_train_adcae, y_test_adcae = self.load_adcae_data()
            adcae_results = self.train_and_evaluate(X_train_adcae, X_test_adcae, y_train_adcae, y_test_adcae, "ADCAE")
            results['ADCAE'] = adcae_results
        except Exception as e:
            print(f"ADCAE分类失败: {e}")
            results['ADCAE'] = None
        
        # 对比结果
        self.compare_results(results)
        
        return results
    
    def compare_results(self, results_summary):
        """
        对比不同特征类型的结果
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
            comparison_df = pd.DataFrame(valid_results).T
            
            # 确保所需的列存在
            required_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'prediction_time']
            available_columns = [col for col in required_columns if col in comparison_df.columns]
            
            if available_columns:
                print(comparison_df[available_columns])
            else:
                print("结果数据格式异常，无法显示对比表")
                print("可用的列名:", list(comparison_df.columns))
            
            # 保存对比结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = os.path.join(self.output_path, f"dt_feature_comparison_{self.dataset_name}_{timestamp}.csv")
            comparison_df.to_csv(comparison_path)
            print(f"\n对比结果已保存到: {comparison_path}")
            
        except Exception as e:
            print(f"创建对比表时出错: {e}")
            print("原始结果数据:")
            for feature_type, result in valid_results.items():
                if result:
                    print(f"{feature_type}: 准确率={result.get('accuracy', 'N/A'):.4f}")


def main():
    """
    主函数 - 处理USTC和CTU两个数据集
    """
    datasets = ["USTC", "CTU"]
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"开始处理 {dataset_name} 数据集")
        print(f"{'='*80}")
        
        try:
            # 创建分类器实例
            classifier = DTFeatureClassifier(dataset_name=dataset_name)
            
            # 运行所有分类任务
            results = classifier.run_all_classifications()
            all_results[dataset_name] = results
            
            print(f"\n{dataset_name} 数据集分类任务完成！")
            
        except Exception as e:
            print(f"处理 {dataset_name} 数据集时出错: {e}")
            continue
    
    # 跨数据集结果对比
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("跨数据集结果对比")
        print(f"{'='*80}")
        
        # 创建跨数据集对比表
        comparison_data = []
        for dataset, features in all_results.items():
            if features:  # 确保features不为空
                for feature_type, metrics in features.items():
                    if metrics is not None:  # 确保metrics不为None
                        comparison_data.append({
                            'Dataset': dataset,
                            'Feature_Type': feature_type,
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1_Score': metrics['f1_score'],
                            'Training_Time': metrics['training_time'],
                            'Prediction_Time': metrics['prediction_time'],
                            'Avg_Prediction_Time': metrics['avg_prediction_time']
                        })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
            
            # 保存跨数据集对比结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "D:\\Python Project\\ADCAE\\results\\dt_cross_dataset_comparison"
            os.makedirs(output_dir, exist_ok=True)
            
            comparison_path = os.path.join(output_dir, f"dt_cross_dataset_comparison_{timestamp}.csv")
            comparison_df.to_csv(comparison_path, index=False)
            print(f"\n跨数据集对比结果已保存到: {comparison_path}")
    
    print("\n所有数据集的分类任务完成！")


if __name__ == "__main__":
    main()