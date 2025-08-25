import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class TCNRawDataProcessor:
    def __init__(self, dataset_name="CTU", max_packet_length=1024, max_packets_per_session=50):
        """
        初始化TCN原始数据处理器
        
        Args:
            dataset_name: 数据集名称 ("CTU" 或 "USTC")
            max_packet_length: 每个数据包的最大长度（字节）
            max_packets_per_session: 每个会话的最大数据包数量
        """
        self.dataset_name = dataset_name
        self.max_packet_length = max_packet_length
        self.max_packets_per_session = max_packets_per_session
        self.label_encoder = LabelEncoder()
        
        # 设置数据路径
        self.base_path = Path(f"d:/Python Project/ADCAE/pcap_files/Dataset_{dataset_name}")
        self.train_path = self.base_path / "Train"
        self.test_path = self.base_path / "Test"
        
        # 输出路径
        self.output_path = Path(f"d:/Python Project/ADCAE/{dataset_name}_result/tcn_output_raw")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"初始化TCN原始数据处理器 - 数据集: {dataset_name}")
        print(f"训练数据路径: {self.train_path}")
        print(f"测试数据路径: {self.test_path}")
        print(f"输出路径: {self.output_path}")
    
    def read_pcap_as_bytes(self, file_path, max_bytes=None):
        """
        读取pcap文件的原始字节数据
        
        Args:
            file_path: pcap文件路径
            max_bytes: 最大读取字节数
            
        Returns:
            bytes: 文件的字节数据
        """
        try:
            with open(file_path, 'rb') as f:
                if max_bytes:
                    data = f.read(max_bytes)
                else:
                    data = f.read()
            return data
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return b''
    
    def bytes_to_features(self, byte_data):
        """
        将字节数据转换为特征向量
        
        Args:
            byte_data: 字节数据
            
        Returns:
            np.array: 特征向量
        """
        if len(byte_data) == 0:
            return np.zeros(self.max_packet_length, dtype=np.float32)
        
        # 将字节转换为0-255的整数数组
        features = np.frombuffer(byte_data, dtype=np.uint8).astype(np.float32)
        
        # 归一化到0-1范围
        features = features / 255.0
        
        # 截断或填充到固定长度
        if len(features) > self.max_packet_length:
            features = features[:self.max_packet_length]
        else:
            padding = np.zeros(self.max_packet_length - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        
        return features
    
    def load_dataset(self, data_path, max_files_per_class=1000):
        """
        加载数据集
        
        Args:
            data_path: 数据路径
            max_files_per_class: 每个类别最大文件数
            
        Returns:
            tuple: (features, labels, class_names)
        """
        features = []
        labels = []
        class_names = []
        
        print(f"\n从 {data_path} 加载数据...")
        
        # 获取所有类别目录
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name.replace("-ALL", "")
            class_names.append(class_name)
            
            print(f"处理类别: {class_name}")
            
            # 获取该类别下的所有pcap文件
            pcap_files = list(class_dir.glob("*.pcap"))
            
            # 限制文件数量
            if len(pcap_files) > max_files_per_class:
                pcap_files = pcap_files[:max_files_per_class]
            
            print(f"  找到 {len(pcap_files)} 个文件")
            
            for i, pcap_file in enumerate(pcap_files):
                if i % 100 == 0:
                    print(f"  处理进度: {i+1}/{len(pcap_files)}")
                
                # 读取pcap文件的字节数据
                byte_data = self.read_pcap_as_bytes(pcap_file, self.max_packet_length)
                
                # 转换为特征向量
                feature_vector = self.bytes_to_features(byte_data)
                
                features.append(feature_vector)
                labels.append(class_name)
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"\n数据加载完成:")
        print(f"特征形状: {features.shape}")
        print(f"标签数量: {len(labels)}")
        print(f"类别: {set(labels)}")
        
        return features, labels, class_names
    
    def create_sequences(self, features, sequence_length=10):
        """
        创建时序序列（为了适配TCN）
        
        Args:
            features: 特征数组
            sequence_length: 序列长度
            
        Returns:
            np.array: 时序特征
        """
        # 将特征重塑为时序格式
        # 这里我们将每个特征向量分割成多个时间步
        n_samples, feature_dim = features.shape
        
        if feature_dim < sequence_length:
            # 如果特征维度小于序列长度，进行填充
            padding = np.zeros((n_samples, sequence_length - feature_dim))
            features = np.concatenate([features, padding], axis=1)
            feature_dim = sequence_length
        
        # 重塑为 (samples, timesteps, features)
        timesteps = feature_dim // sequence_length
        features_per_timestep = sequence_length
        
        # 截断特征以适应整数倍的时间步
        truncated_features = features[:, :timesteps * features_per_timestep]
        
        # 重塑为时序格式
        sequential_features = truncated_features.reshape(
            n_samples, timesteps, features_per_timestep
        )
        
        print(f"时序特征形状: {sequential_features.shape}")
        return sequential_features
    
    def build_tcn_model(self, input_shape, num_classes):
        """
        构建TCN模型
        
        Args:
            input_shape: 输入形状
            num_classes: 类别数量
            
        Returns:
            tf.keras.Model: TCN模型
        """
        model = Sequential([
            # 第一层卷积
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Dropout(0.2),
            
            # 第二层卷积
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            Dropout(0.2),
            
            # 第三层卷积
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            Dropout(0.3),
            
            # 全局最大池化
            GlobalMaxPooling1D(),
            
            # 全连接层
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            tuple: (model, history)
        """
        print("\n开始训练TCN模型...")
        
        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(y_train))
        
        model = self.build_tcn_model(input_shape, num_classes)
        
        print(f"模型结构:")
        model.summary()
        
        # 设置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 训练模型
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\n训练完成，耗时: {training_time:.2f} 秒")
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, class_names):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            class_names: 类别名称
            
        Returns:
            dict: 评估结果
        """
        print("\n评估模型...")
        
        # 预测
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n测试结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 分类报告
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(f"\n分类报告:\n{report}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }
    
    def save_results(self, model, history, test_results, training_time):
        """
        保存结果
        
        Args:
            model: 训练好的模型
            history: 训练历史
            test_results: 测试结果
            training_time: 训练时间
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = self.output_path / f"tcn_raw_model_{self.dataset_name}_{timestamp}.h5"
        model.save(model_path)
        print(f"模型已保存到: {model_path}")
        
        # 保存训练历史
        history_data = {
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'train_accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
        
        history_path = self.output_path / f"tcn_raw_history_{self.dataset_name}_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # 保存测试结果
        results_data = {
            'dataset': self.dataset_name,
            'timestamp': timestamp,
            'training_time': training_time,
            'max_packet_length': self.max_packet_length,
            'max_packets_per_session': self.max_packets_per_session,
            'test_results': test_results
        }
        
        results_path = self.output_path / f"tcn_raw_results_{self.dataset_name}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # 保存性能指标CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [
                test_results['accuracy'],
                test_results['precision'],
                test_results['recall'],
                test_results['f1_score']
            ]
        })
        
        metrics_path = self.output_path / f"tcn_raw_metrics_{self.dataset_name}_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"结果已保存到: {self.output_path}")
    
    def plot_training_history(self, history, save_path=None):
        """
        绘制训练历史
        
        Args:
            history: 训练历史
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 准确率曲线
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def process_dataset(self, max_files_per_class=1000, epochs=5, batch_size=32, sequence_length=10):
        """
        处理完整的数据集
        
        Args:
            max_files_per_class: 每个类别最大文件数
            epochs: 训练轮数
            batch_size: 批次大小
            sequence_length: 序列长度
        """
        print(f"\n开始处理 {self.dataset_name} 数据集...")
        
        # 加载训练数据
        X_train, y_train, class_names = self.load_dataset(self.train_path, max_files_per_class)
        
        # 加载测试数据
        X_test, y_test, _ = self.load_dataset(self.test_path, max_files_per_class)
        
        # 编码标签
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 创建时序序列
        X_train_seq = self.create_sequences(X_train, sequence_length)
        X_test_seq = self.create_sequences(X_test, sequence_length)
        
        # 分割训练集和验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_seq, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
        )
        
        print(f"\n数据分割完成:")
        print(f"训练集: {X_train_final.shape}")
        print(f"验证集: {X_val.shape}")
        print(f"测试集: {X_test_seq.shape}")
        
        # 训练模型
        start_time = time.time()
        model, history = self.train_model(
            X_train_final, y_train_final, X_val, y_val, epochs, batch_size
        )
        training_time = time.time() - start_time
        
        # 评估模型
        test_results = self.evaluate_model(model, X_test_seq, y_test_encoded, class_names)
        
        # 保存结果
        self.save_results(model, history, test_results, training_time)
        
        # 绘制训练历史
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_path / f"tcn_raw_training_history_{self.dataset_name}_{timestamp}.png"
        self.plot_training_history(history, plot_path)
        
        print(f"\n{self.dataset_name} 数据集处理完成！")
        
        return model, history, test_results

# 主程序
if __name__ == "__main__":
    print("TCN原始数据处理器")
    print("=" * 50)
    
    # 处理CTU数据集
    print("\n处理CTU数据集...")
    ctu_processor = TCNRawDataProcessor(
        dataset_name="CTU",
        max_packet_length=1024,
        max_packets_per_session=50
    )
    
    try:
        ctu_model, ctu_history, ctu_results = ctu_processor.process_dataset(
            max_files_per_class=None,  # 限制文件数量以加快处理速度
            epochs=5,
            batch_size=32,
            sequence_length=32
        )
        print("CTU数据集处理成功！")
    except Exception as e:
        print(f"CTU数据集处理失败: {e}")
    
    # 处理USTC数据集
    print("\n处理USTC数据集...")
    ustc_processor = TCNRawDataProcessor(
        dataset_name="USTC",
        max_packet_length=1024,
        max_packets_per_session=50
    )
    
    try:
        ustc_model, ustc_history, ustc_results = ustc_processor.process_dataset(
            max_files_per_class=None,  # 限制文件数量以加快处理速度
            epochs=5,
            batch_size=32,
            sequence_length=32
        )
        print("USTC数据集处理成功！")
    except Exception as e:
        print(f"USTC数据集处理失败: {e}")
    
    print("\n所有数据集处理完成！")