import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple

def main():
    # 配置参数
    SESSIONS_COUNT_LIMIT_MIN = 0
    SESSIONS_COUNT_LIMIT_MAX = 60000
    TRIMMED_FILE_LEN = 32 * 32 * 1  # 1024 bytes for 32x32x1
    TRAIN_RATIO = 0.9  # 90% 训练，10% 测试
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")
    
    # 切换到脚本目录
    os.chdir(script_dir)
    
    # 定义输入目录
    source_session_dirs = [
        "2_Session_USTC/AllLayers",
        "2_Session_CTU/AllLayers"
    ]
    
    print(f"Processing files with length limit: {TRIMMED_FILE_LEN} bytes")
    print(f"Train/Test split ratio: {TRAIN_RATIO:.1%}/{1-TRAIN_RATIO:.1%}")
    print(f"File count limit: {SESSIONS_COUNT_LIMIT_MIN} - {SESSIONS_COUNT_LIMIT_MAX}")
    
    # 处理每个数据集
    for source_dir in source_session_dirs:
        dataset_name = "USTC" if "USTC" in source_dir else "CTU"
        process_dataset_direct(source_dir, dataset_name, SESSIONS_COUNT_LIMIT_MIN, 
                              SESSIONS_COUNT_LIMIT_MAX, TRIMMED_FILE_LEN, TRAIN_RATIO)
    
    print("\nScript execution completed!")
    input("Press Enter to exit...")

def process_dataset_direct(source_dir: str, dataset_name: str, min_limit: int, 
                          max_limit: int, trimmed_len: int, train_ratio: float):
    """直接处理数据集生成最终的训练/测试数据"""
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Warning: Source directory {source_path} does not exist")
        return
    
    # 获取所有子目录（每个子目录代表一个类别）
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"Warning: No subdirectories found in {source_path}")
        return
    
    print(f"\nProcessing {dataset_name} dataset from {source_dir}")
    
    # 创建输出根目录
    output_root = Path(f"Dataset_{dataset_name}")
    train_root = output_root / "Train"
    test_root = output_root / "Test"
    
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)
    
    total_train_files = 0
    total_test_files = 0
    
    for subdir in subdirs:
        train_count, test_count = process_category_direct(
            subdir, train_root, test_root, min_limit, max_limit, 
            trimmed_len, train_ratio
        )
        total_train_files += train_count
        total_test_files += test_count
    
    print(f"\n{dataset_name} Dataset Summary:")
    print(f"  Total Train files: {total_train_files}")
    print(f"  Total Test files: {total_test_files}")
    print(f"  Output directory: {output_root}")

def process_category_direct(category_dir: Path, train_root: Path, test_root: Path,
                           min_limit: int, max_limit: int, trimmed_len: int, 
                           train_ratio: float) -> Tuple[int, int]:
    """直接处理类别目录，生成最终数据"""
    # 获取所有文件
    files = [f for f in category_dir.iterdir() if f.is_file()]
    file_count = len(files)
    
    if file_count <= min_limit:
        print(f"Skipping {category_dir.name}: only {file_count} files (minimum: {min_limit})")
        return 0, 0
    
    print(f"{category_dir.name}: {file_count} files")
    
    # 如果文件数量超过最大限制，选择最大的文件
    if file_count > max_limit:
        files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)[:max_limit]
        file_count = max_limit
        print(f"  Limited to {max_limit} largest files")
    
    # 随机分割为训练集和测试集
    random.shuffle(files)
    test_count = max(1, int(file_count * (1 - train_ratio)))
    test_files = files[:test_count]
    train_files = files[test_count:]
    
    print(f"  Train: {len(train_files)}, Test: {len(test_files)}")
    
    # 创建类别目录
    train_category_dir = train_root / category_dir.name
    test_category_dir = test_root / category_dir.name
    
    train_category_dir.mkdir(exist_ok=True)
    test_category_dir.mkdir(exist_ok=True)
    
    # 直接处理并保存文件
    process_and_save_files(train_files, train_category_dir, trimmed_len)
    process_and_save_files(test_files, test_category_dir, trimmed_len)
    
    return len(train_files), len(test_files)

def process_and_save_files(files: List[Path], dest_dir: Path, trimmed_len: int):
    """处理文件并直接保存到目标目录"""
    for file_path in files:
        try:
            # 读取并处理文件内容
            with open(file_path, 'rb') as f:
                content = f.read()
            
            current_len = len(content)
            
            if current_len > trimmed_len:
                # 截断文件
                content = content[:trimmed_len]
            elif current_len < trimmed_len:
                # 填充文件
                padding_len = trimmed_len - current_len
                content = content + b'\x00' * padding_len
            
            # 直接写入目标目录
            output_path = dest_dir / file_path.name
            with open(output_path, 'wb') as f:
                f.write(content)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# 可选：添加数据集验证功能
def validate_dataset(dataset_root: Path, expected_file_len: int):
    """验证生成的数据集"""
    print(f"\nValidating dataset: {dataset_root}")
    
    for split in ['Train', 'Test']:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found")
            continue
            
        total_files = 0
        categories = []
        
        for category_dir in split_dir.iterdir():
            if category_dir.is_dir():
                files = list(category_dir.glob('*'))
                file_count = len(files)
                total_files += file_count
                categories.append(category_dir.name)
                
                # 检查文件大小
                if files:
                    sample_file = files[0]
                    file_size = sample_file.stat().st_size
                    if file_size != expected_file_len:
                        print(f"Warning: {category_dir.name} file size {file_size} != expected {expected_file_len}")
        
        print(f"  {split}: {total_files} files, {len(categories)} categories")
        print(f"  Categories: {', '.join(categories)}")

if __name__ == "__main__":
    main()
    
    # 可选：验证生成的数据集
    # validate_dataset(Path("Dataset_USTC"), 32 * 32 * 1)
    # validate_dataset(Path("Dataset_CTU"), 32 * 32 * 1)