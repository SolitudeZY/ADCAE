import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple

def main():
    # 配置参数
    SESSIONS_COUNT_LIMIT_MIN = 0
    SESSIONS_COUNT_LIMIT_MAX = 60000
    TRIMMED_FILE_LEN = 32 * 32 * 1  # 768 for 16x16x3 RGB
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")
    
    # 切换到脚本目录
    os.chdir(script_dir)
    
    # 定义输入目录（来自pcap_processor.py的输出）
    source_session_dirs = [
        "2_Session_USTC/AllLayers",
        "2_Session_CTU/AllLayers"
    ]
    
    print(f"If Sessions more than {SESSIONS_COUNT_LIMIT_MAX} we only select the largest {SESSIONS_COUNT_LIMIT_MAX}.")
    print("Finally Selected Sessions:")
    
    # 处理每个数据集
    for source_dir in source_session_dirs:
        dataset_name = "USTC" if "USTC" in source_dir else "CTU"
        process_dataset(source_dir, dataset_name, SESSIONS_COUNT_LIMIT_MIN, 
                       SESSIONS_COUNT_LIMIT_MAX, TRIMMED_FILE_LEN)
    
    print("\nScript execution completed!")
    input("Press Enter to exit...")

def process_dataset(source_dir: str, dataset_name: str, min_limit: int, 
                   max_limit: int, trimmed_len: int):
    """处理单个数据集"""
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
    
    for subdir in subdirs:
        process_category(subdir, dataset_name, min_limit, max_limit, trimmed_len)

def process_category(category_dir: Path, dataset_name: str, min_limit: int, 
                    max_limit: int, trimmed_len: int):
    """处理单个类别目录"""
    # 获取所有文件
    files = [f for f in category_dir.iterdir() if f.is_file()]
    file_count = len(files)
    
    if file_count <= min_limit:
        print(f"Skipping {category_dir.name}: only {file_count} files (minimum: {min_limit})")
        return
    
    print(f"{category_dir.name}: {file_count} files")
    
    # 如果文件数量超过最大限制，选择最大的文件
    if file_count > max_limit:
        files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)[:max_limit]
        file_count = max_limit
        print(f"  Limited to {max_limit} largest files")
    
    # 随机分割为训练集和测试集（90% 训练，10% 测试）
    random.shuffle(files)
    test_count = max(1, file_count // 10)  # 至少1个测试文件
    test_files = files[:test_count]
    train_files = files[test_count:]
    
    print(f"  Train: {len(train_files)}, Test: {len(test_files)}")
    
    # 创建输出目录
    base_output_dir = Path(f"3_Processed_{dataset_name}")
    
    # 第一步：复制文件到FilteredSession目录
    filtered_test_dir = base_output_dir / "FilteredSession" / "Test" / category_dir.name
    filtered_train_dir = base_output_dir / "FilteredSession" / "Train" / category_dir.name
    
    filtered_test_dir.mkdir(parents=True, exist_ok=True)
    filtered_train_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    copy_files(test_files, filtered_test_dir)
    copy_files(train_files, filtered_train_dir)
    
    # 第二步：处理文件长度并保存到TrimedSession目录
    trimmed_test_dir = base_output_dir / "TrimedSession" / "Test" / category_dir.name
    trimmed_train_dir = base_output_dir / "TrimedSession" / "Train" / category_dir.name
    
    trimmed_test_dir.mkdir(parents=True, exist_ok=True)
    trimmed_train_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Trimming files to {trimmed_len} bytes...")
    
    # 处理测试集
    process_trimmed_files(filtered_test_dir, trimmed_test_dir, trimmed_len)
    
    # 处理训练集
    process_trimmed_files(filtered_train_dir, trimmed_train_dir, trimmed_len)

def copy_files(files: List[Path], destination_dir: Path):
    """复制文件到目标目录"""
    for file_path in files:
        try:
            shutil.copy2(file_path, destination_dir / file_path.name)
        except Exception as e:
            print(f"Error copying {file_path}: {e}")

def process_trimmed_files(source_dir: Path, dest_dir: Path, trimmed_len: int):
    """处理文件长度：截断或填充到指定长度"""
    for file_path in source_dir.iterdir():
        if not file_path.is_file():
            continue
            
        try:
            # 读取文件内容
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
            
            # 写入处理后的文件
            output_path = dest_dir / file_path.name
            with open(output_path, 'wb') as f:
                f.write(content)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()