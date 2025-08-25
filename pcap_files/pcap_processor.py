import os
import subprocess
import glob
from pathlib import Path

def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")
    
    # 切换到脚本目录
    os.chdir(script_dir)
    
    # 定义路径
    sampled_dir = script_dir / "sampled"
    ustc_dir = sampled_dir / "USTC"
    ctu_dir = sampled_dir / "CTU"
    splitcap_exe = script_dir / "0_Tool" / "SplitCap_2-1" / "SplitCap.exe"
    finddupe_exe = script_dir / "0_Tool" / "finddupe.exe"
    
    # 创建输出目录
    output_ustc = script_dir / "2_Session_USTC" / "AllLayers"
    output_ctu = script_dir / "2_Session_CTU" / "AllLayers"
    output_ustc.mkdir(parents=True, exist_ok=True)
    output_ctu.mkdir(parents=True, exist_ok=True)
    
    print(f"Sampled directory exists: {sampled_dir.exists()}")
    
    # 处理USTC数据集
    process_dataset("USTC", ustc_dir, output_ustc, splitcap_exe)
    
    # 处理CTU数据集
    process_dataset("CTU", ctu_dir, output_ctu, splitcap_exe)
    
    # 删除重复文件
    remove_duplicates(finddupe_exe, output_ustc, output_ctu)
    
    print("Script execution completed!")
    input("Press Enter to exit...")

def process_dataset(dataset_name, input_dir, output_dir, splitcap_exe):
    """处理数据集"""
    print(f"\nProcessing {dataset_name} dataset...")
    print(f"Checking {dataset_name} path: {input_dir}")
    print(f"{dataset_name} path exists: {input_dir.exists()}")
    
    if not input_dir.exists():
        print(f"Warning: {dataset_name} folder {input_dir} does not exist")
        return
    
    # 查找所有pcap文件
    pcap_files = list(input_dir.glob("*.pcap"))
    
    if not pcap_files:
        print(f"Warning: No PCAP files found in {input_dir} folder")
        return
    
    print(f"Found {len(pcap_files)} {dataset_name} pcap files")
    
    # 检查SplitCap工具
    if not splitcap_exe.exists():
        print(f"Error: SplitCap.exe not found at {splitcap_exe}")
        return
    
    # 处理每个pcap文件
    for pcap_file in pcap_files:
        print(f"Processing: {pcap_file.name}")
        
        # 创建输出目录
        file_output_dir = output_dir / f"{pcap_file.stem}-ALL"
        print(f"Output directory: {file_output_dir}")
        
        try:
            # 运行SplitCap
            cmd = [
                str(splitcap_exe),
                "-p", "50000",
                "-b", "50000",
                "-r", str(pcap_file),
                "-o", str(file_output_dir)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 删除空文件
            if file_output_dir.exists():
                remove_empty_files(file_output_dir)
            else:
                print(f"Warning: Output directory {file_output_dir} was not created")
                
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pcap_file.name}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {pcap_file.name}: {e}")

def remove_empty_files(directory):
    """删除指定目录中的空文件"""
    for file_path in Path(directory).rglob("*"):
        if file_path.is_file() and file_path.stat().st_size == 0:
            try:
                file_path.unlink()
                print(f"Removed empty file: {file_path}")
            except Exception as e:
                print(f"Error removing empty file {file_path}: {e}")

def remove_duplicates(finddupe_exe, ustc_dir, ctu_dir):
    """删除重复文件"""
    print("\nRemoving duplicate files...")
    
    if not finddupe_exe.exists():
        print(f"Warning: finddupe.exe not found at {finddupe_exe}, skipping duplicate removal")
        return
    
    # 处理USTC目录
    if ustc_dir.exists():
        print("Removing duplicates in USTC directory...")
        try:
            subprocess.run([str(finddupe_exe), "-del", str(ustc_dir)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error removing duplicates in USTC: {e}")
    else:
        print("Warning: USTC session directory does not exist")
    
    # 处理CTU目录
    if ctu_dir.exists():
        print("Removing duplicates in CTU directory...")
        try:
            subprocess.run([str(finddupe_exe), "-del", str(ctu_dir)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error removing duplicates in CTU: {e}")
    else:
        print("Warning: CTU session directory does not exist")
    
    print("Duplicate files removed successfully")

if __name__ == "__main__":
    main()