import os
import numpy as np
from sklearn.preprocessing import StandardScaler

class BinaryDataPreprocessor:
    """二进制数据预处理器（适配ADCAE）"""
    
    def __init__(self, file_length=1024, image_size=32):
        self.scaler = StandardScaler()
        self.file_length = file_length
        self.image_size = image_size
        
    def load_binary_file(self, file_path):
        """加载二进制文件并转换为32x32图像"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 确保数据长度一致
            if len(data) < self.file_length:
                data = data + b'\x00' * (self.file_length - len(data))
            elif len(data) > self.file_length:
                data = data[:self.file_length]
            
            # 转换为numpy数组并重塑为32x32
            features = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            image = features.reshape(self.image_size, self.image_size)
            
            # 归一化到[0,1]
            image = image / 255.0
            
            return image
            
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None
    
    def load_dataset_from_directory(self, dataset_dir, max_files_per_category=None):
        """从目录加载数据集"""
        print(f"正在加载数据集: {dataset_dir}")
        
        all_images = []
        all_labels = []
        category_counts = {}
        
        # 遍历所有类别目录
        for category_name in os.listdir(dataset_dir):
            category_path = os.path.join(dataset_dir, category_name)
            
            if not os.path.isdir(category_path):
                continue
                
            print(f"  处理类别: {category_name}")
            
            # 获取该类别下的所有文件
            files = [f for f in os.listdir(category_path) if f.endswith('.pcap')]
            
            if max_files_per_category:
                files = files[:max_files_per_category]
            
            category_images = []
            
            for i, filename in enumerate(files):
                if i == len(files) and i > 0:
                    print(f"    已处理 {i}/{len(files)} 个文件")
                
                file_path = os.path.join(category_path, filename)
                image = self.load_binary_file(file_path)
                
                if image is not None:
                    category_images.append(image)
                    all_labels.append(category_name)
            
            if category_images:
                all_images.extend(category_images)
                category_counts[category_name] = len(category_images)
                print(f"    {category_name}: {len(category_images)} 个文件")
        
        if not all_images:
            raise ValueError(f"在 {dataset_dir} 中没有找到有效的数据文件")
        
        # 转换为numpy数组，添加通道维度
        X = np.array(all_images)[:, np.newaxis, :, :]  # (N, 1, 32, 32)
        y = np.array(all_labels)
        
        print(f"数据加载完成:")
        print(f"  总样本数: {len(X)}")
        print(f"  图像形状: {X.shape}")
        print(f"  类别分布: {category_counts}")
        
        return X, y, category_counts