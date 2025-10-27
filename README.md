[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=plastic&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/SolitudeZY/ADCAE)
# ADCAE: 非对称深度卷积自编码器网络流量异常检测系统
## 项目简介
ADCAE (Asymmetric Deep Convolutional AutoEncoder) 是一个基于深度学习的网络流量异常检测系统。该项目实现了一种创新的非对称深度卷积自编码器架构，结合ELU激活函数和CBAM注意力机制，用于网络流量的特征提取和异常检测。

## 主要特性
<img width="983" height="875" alt="Fig1" src="https://github.com/user-attachments/assets/14f1c697-090f-4d53-a67e-9d6e848eb4a8" />

- 非对称深度卷积自编码器 : 采用非对称的编码器-解码器架构，提高特征提取能力
- CBAM注意力机制 : 集成通道注意力和空间注意力，增强关键特征感知能力
- ELU激活函数 : 缓解梯度消失问题，加速模型收敛
- 多种编码器支持 : 支持ADCAE、CAE、PCA、KPCA等多种特征提取方法
- 多种分类器 : 集成TCN、决策树、随机森林等分类算法
- 多数据集支持 : 支持USTC和CTU网络流量数据集
- 可配置架构 : 支持不同层数、激活函数和注意力机制的配置
## 项目结构
```
ADCAE/
├── Encoder/                    # 编码器模块
│   ├── ADCAE.py               # ADCAE主实现
│   ├── ADCAE_main.py          # ADCAE主程序入口
│   ├── CAE.py                 # 传统CAE实现
│   ├── PCA.py                 # PCA特征提取
│   ├── KPCA.py                # 核PCA特征提取
│   ├── cae_enhanced.py        # 增强版CAE
│   └── adcae/                 # ADCAE核心模块
│       ├── model.py           # 模型架构定义
│       ├── config.py          # 配置管理
│       ├── trainer.py         # 训练器
│       ├── preprocessor.py    # 数据预处理
│       ├── attention.py       # 注意力机制
│       └── blocks.py          # 网络模块
├── ML+DL/                     # 机器学习和深度学习模型
│   ├── TCN.py                 # 时间卷积网络
│   ├── DT.py                  # 决策树
│   ├── RF.py                  # 随机森林
│   ├── ADCAE_test/            # ADCAE测试模块
│   │   ├── TCN/               # TCN对比实验
│   │   ├── DT/                # 决策树对比实验
│   │   └── RF/                # 随机森林对比实验
│   └── rf_modules/            # 随机森林模块
├── pcap_files/                # 网络数据包文件
│   ├── Dataset_USTC/          # USTC数据集
│   └── Dataset_CTU/           # CTU数据集
├── results/                   # 实验结果
├── CTU_result/                # CTU数据集结果
├── USTC_result/               # USTC数据集结果
└── config/                    # 配置文件
```
## 环境要求
### Python版本
- Python 3.8+
### 依赖包
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
pillow>=8.0.0
joblib>=1.1.0
```
## 安装指南
### 1. 创建虚拟环境
```
# 创建虚拟环境（推荐使用Python 3.8）
python -m venv adcae_env

# 激活虚拟环境
# Windows:
adcae_env\Scripts\activate
# Linux/Mac:
source adcae_env/bin/activate
```
### 2. 安装依赖
```
# 安装项目依赖
pip install -r requirements.txt
```
### 3. 数据集准备
1. 1.
   下载并解压 Sampled_Datasets.zip 到 pcap_files/ 目录
2. 2.
   运行数据预处理脚本：
   ```
   # 切分PCAP文件
   python pcap_processor.py
   
   # 归一化并生成数据集
   python session_processor_simplified.py
   ```
## 使用方法
### 1. 训练ADCAE模型
```
from ADCAE.Encoder.ADCAE_main import main

# 运行ADCAE训练
main()
```
或直接运行：

```
cd Encoder
python ADCAE_main.py
```
### 2. 配置模型参数
在 adcae/config.py 中可以配置模型参数：

```
from adcae.config import ModelConfig, 
TrainingConfig, DataConfig

# 模型配置
model_config = ModelConfig(
    input_channels=1,
    encoding_dim=128,
    use_attention=True,
    attention_type='cbam',  # 'cbam', 'channel', 
    'spatial', 'none'
    activation='elu',       # 'elu', 'relu', 'gelu'
    encoder_channels=[64, 128, 256, 512, 1024],
    dropout_rate=0.1
)

# 训练配置
training_config = TrainingConfig(
    epochs=50,
    batch_size=64,
    learning_rate=0.001
)

# 数据配置
data_config = DataConfig(
    file_length=1024,
    image_size=32
)
```
### 3. 运行对比实验
```
# 运行不同配置的对比实验
from ADCAE.Encoder.experiment_runner import 
run_experiments

run_experiments()
```
### 4. 使用机器学习分类器
```
# 使用TCN分类器
from ML+DL.ADCAE_test.TCN.adcae_tcn_comparison 
import ADCAETCNComparison

# 创建TCN对比实验
tcn_experiment = ADCAETCNComparison
(dataset_name="USTC")
results = tcn_experiment.run_all_experiments()

# 使用随机森林分类器
from ML+DL.ADCAE_test.RF.adcae_rf_comparison import 
ADCAERFComparison

rf_experiment = ADCAERFComparison
(dataset_name="USTC")
results = rf_experiment.run_all_experiments()
```
## 数据集信息
### USTC数据集
- Cridex : 14,742 训练样本, 1,638 测试样本
- Geodo : 32,688 训练样本, 3,632 测试样本
- Htbot : 5,355 训练样本, 595 测试样本
- Miuref : 11,889 训练样本, 1,321 测试样本
- Neris : 28,512 训练样本, 3,168 测试样本
- Normal : 53,991 训练样本, 5,999 测试样本
- Shifu : 6,912 训练样本, 768 测试样本
- Tinba : 5,904 训练样本, 656 测试样本
- Virut : 29,574 训练样本, 3,286 测试样本
- Zeus : 9,522 训练样本, 1,058 测试样本
### CTU数据集
- Artemis : 7,776 训练样本, 864 测试样本
- Coinminer : 270 训练样本, 30 测试样本
- Dridex : 2,754 训练样本, 306 测试样本
- Htbot : 8,091 训练样本, 899 测试样本
- Miuref : 11,898 训练样本, 1,322 测试样本
- Normal : 45,918 训练样本, 5,102 测试样本
- Tinba : 40,185 训练样本, 4,465 测试样本
- Trickbot : 10,629 训练样本, 1,181 测试样本
- Ursnif : 22,140 训练样本, 2,460 测试样本
- Zeus : 2,079 训练样本, 231 测试样本
## 模型架构
### ADCAE数据处理流程
```
输入: PCAP网络流量文件
  ↓
预处理: 二进制数据 → One-hot编码 → Z-score归一化 → 32×32
图像
  ↓
编码器: Conv2D + ELU + CBAM → 特征提取
  ↓
瓶颈层: Flatten + Linear → 低维特征向量
  ↓
解码器恢复层: Linear + Reshape → 特征图恢复
  ↓
解码器: DeConv2D + ELU + CBAM → 图像重构
  ↓
输出: Sigmoid → 重构图像
```
### 支持的配置
配置类型 选项 说明 层数配置 2, 4, 6, 8, 10层 不同深度的网络架构 激活函数 ELU, ReLU, GELU 不同的激活函数选择 注意力机制 CBAM, Channel, Spatial, None 不同的注意力机制 编码维度 64, 128, 256, 512 瓶颈层特征维度

## 实验结果
实验结果将保存在以下目录：

- results/ : 主要实验结果
- USTC_result/ : USTC数据集结果
- CTU_result/ : CTU数据集结果
每个实验会生成：

- 训练损失曲线
- 特征可视化
- 性能指标报告
- 模型权重文件
- 提取的特征文件
## 性能指标
系统评估指标包括：

- 重构误差 : MSE, MAE
- 分类性能 : Accuracy, Precision, Recall, F1-Score
- 训练效率 : 训练时间, 收敛速度
- 特征质量 : 特征可分离性, 降维效果
## 贡献指南
1. 1.
   Fork 项目
2. 2.
   创建特性分支 ( git checkout -b feature/AmazingFeature )
3. 3.
   提交更改 ( git commit -m 'Add some AmazingFeature' )
4. 4.
   推送到分支 ( git push origin feature/AmazingFeature )
5. 5.
   打开 Pull Request
