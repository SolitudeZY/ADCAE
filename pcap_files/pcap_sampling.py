import os
import random
from scapy.all import *
from scapy.utils import PcapWriter
import glob
import shutil
from datetime import datetime

class PcapSampler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ustc_dir = os.path.join(base_dir, 'USTC')
        self.ctu_dir = os.path.join(base_dir, 'CTU')
        self.ctu_normal_dir = os.path.join(self.ctu_dir, 'normal')
        
    def merge_ctu_normal_files(self):
        """合并CTU normal文件夹下的所有pcap文件为一个Normal.pcap"""
        print("开始合并CTU normal文件夹下的pcap文件...")
        
        normal_files = glob.glob(os.path.join(self.ctu_normal_dir, '*.pcap'))
        if not normal_files:
            print("CTU normal文件夹下没有找到pcap文件")
            return False
            
        merged_packets = []
        
        for file_path in normal_files:
            print(f"正在读取: {os.path.basename(file_path)}")
            try:
                packets = rdpcap(file_path)
                merged_packets.extend(packets)
                print(f"已读取 {len(packets)} 个数据包")
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                continue
        
        if merged_packets:
            # 按时间戳排序
            merged_packets.sort(key=lambda x: x.time)
            
            # 保存合并后的文件
            output_path = os.path.join(self.ctu_dir, 'Normal.pcap')
            wrpcap(output_path, merged_packets)
            print(f"合并完成！总共 {len(merged_packets)} 个数据包保存到: {output_path}")
            return True
        else:
            print("没有成功读取到任何数据包")
            return False
    
    def time_based_sampling(self, packets, sample_rate=0.5):
        """基于时间的随机采样方法"""
        if not packets:
            return []
            
        # 获取时间范围
        start_time = min(pkt.time for pkt in packets)
        end_time = max(pkt.time for pkt in packets)
        time_duration = end_time - start_time
        
        if time_duration == 0:
            # 如果所有包的时间戳相同，使用简单随机采样
            return random.sample(packets, int(len(packets) * sample_rate))
        
        # 创建时间窗口进行采样
        sampled_packets = []
        window_size = time_duration / (len(packets) * sample_rate)  # 动态窗口大小
        
        current_time = start_time
        packet_index = 0
        
        while current_time < end_time and packet_index < len(packets):
            # 在当前时间窗口内收集数据包
            window_packets = []
            while (packet_index < len(packets) and 
                   packets[packet_index].time < current_time + window_size):
                window_packets.append(packets[packet_index])
                packet_index += 1
            
            # 从窗口内随机选择数据包
            if window_packets:
                num_to_select = max(1, int(len(window_packets) * sample_rate))
                selected = random.sample(window_packets, 
                                       min(num_to_select, len(window_packets)))
                sampled_packets.extend(selected)
            
            current_time += window_size
        
        return sampled_packets
    
    def sample_pcap_file(self, input_path, output_path, sample_rate=0.5):
        """对单个pcap文件进行采样"""
        try:
            print(f"正在采样: {os.path.basename(input_path)}")
            
            # 读取原始数据包
            original_packets = rdpcap(input_path)
            original_count = len(original_packets)
            
            if original_count == 0:
                print(f"文件 {input_path} 中没有数据包")
                return False
            
            # 按时间戳排序
            original_packets.sort(key=lambda x: x.time)
            
            # 进行采样
            sampled_packets = self.time_based_sampling(original_packets, sample_rate)
            sampled_count = len(sampled_packets)
            
            # 保存采样后的数据包
            if sampled_packets:
                wrpcap(output_path, sampled_packets)
                print(f"采样完成: {original_count} -> {sampled_count} 数据包 "
                      f"(采样率: {sampled_count/original_count:.2%})")
                return True
            else:
                print(f"采样后没有数据包: {input_path}")
                return False
                
        except Exception as e:
            print(f"处理文件 {input_path} 时出错: {e}")
            return False
    
    def sample_dataset(self, dataset_dir, output_dir, sample_rate=0.5):
        """对整个数据集进行采样"""
        if not os.path.exists(dataset_dir):
            print(f"数据集目录不存在: {dataset_dir}")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有pcap文件
        pcap_files = glob.glob(os.path.join(dataset_dir, '*.pcap'))
        
        if not pcap_files:
            print(f"在 {dataset_dir} 中没有找到pcap文件")
            return
        
        print(f"\n开始对 {os.path.basename(dataset_dir)} 数据集进行采样...")
        print(f"找到 {len(pcap_files)} 个pcap文件")
        
        success_count = 0
        for pcap_file in pcap_files:
            filename = os.path.basename(pcap_file)
            output_path = os.path.join(output_dir, filename)
            
            if self.sample_pcap_file(pcap_file, output_path, sample_rate):
                success_count += 1
        
        print(f"\n{os.path.basename(dataset_dir)} 数据集采样完成: "
              f"{success_count}/{len(pcap_files)} 个文件成功处理")
    
    def run_sampling(self, sample_rate=0.5):
        """执行完整的采样流程"""
        print("=" * 60)
        print("开始流量数据集采样处理")
        print(f"采样率: {sample_rate:.0%}")
        print("=" * 60)
        
        # 设置随机种子以确保结果可重现
        random.seed(42)
        
        # 1. 合并CTU normal文件
        print("\n步骤1: 合并CTU normal文件")
        if self.merge_ctu_normal_files():
            print("CTU normal文件合并成功")
        else:
            print("CTU normal文件合并失败")
        
        # 2. 创建采样输出目录
        sampled_dir = os.path.join(self.base_dir, 'sampled')
        ustc_sampled_dir = os.path.join(sampled_dir, 'USTC')
        ctu_sampled_dir = os.path.join(sampled_dir, 'CTU')
        
        # 3. 对USTC数据集进行采样
        print("\n步骤2: 对USTC数据集进行采样")
        self.sample_dataset(self.ustc_dir, ustc_sampled_dir, sample_rate)
        
        # 4. 对CTU数据集进行采样
        print("\n步骤3: 对CTU数据集进行采样")
        self.sample_dataset(self.ctu_dir, ctu_sampled_dir, sample_rate)
        
        print("\n=" * 60)
        print("所有采样任务完成！")
        print(f"采样后的文件保存在: {sampled_dir}")
        print("=" * 60)

def main():
    # 设置基础目录
    base_dir = r"D:\\Python Project\ADCAE\\pcap_files"
    
    # 创建采样器实例
    sampler = PcapSampler(base_dir)
    
    # 执行采样，数据量减少50%
    sampler.run_sampling(sample_rate=0.5)
    
    print("\n处理统计:")
    print("- USTC数据集: 所有pcap文件采样至50%")
    print("- CTU数据集: normal文件夹合并 + 所有pcap文件采样至50%")
    print("- 采样方法: 基于时间的随机采样")
    print("- 输出目录: pcap_files/sampled/")

if __name__ == "__main__":
    main()