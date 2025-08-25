import pandas as pd
import csv

# 根据终端输出的数据创建数据集信息
data = [
    # USTC数据集
    ['USTC', 'Cridex', 14742, 1638, 1638],
    ['USTC', 'Geodo', 32688, 3632, 3632],
    ['USTC', 'Htbot', 5355, 595, 595],
    ['USTC', 'Miuref', 11889, 1321, 1321],
    ['USTC', 'Neris', 28512, 3168, 3168],
    ['USTC', 'Normal', 53991, 5999, 5999],
    ['USTC', 'Shifu', 6912, 768, 768],
    ['USTC', 'Tinba', 5904, 656, 656],
    ['USTC', 'Virut', 29574, 3286, 3286],
    ['USTC', 'Zeus', 9522, 1058, 1058],
    
    # CTU数据集
    ['CTU', 'Artemis', 7776, 864, 864],
    ['CTU', 'Coinminer', 270, 30, 30],
    ['CTU', 'Dridex', 2754, 306, 306],
    ['CTU', 'Htbot', 8091, 899, 899],
    ['CTU', 'Miuref', 11898, 1322, 1322],
    ['CTU', 'Normal', 45918, 5102, 5102],
    ['CTU', 'Tinba', 40185, 4465, 4465],
    ['CTU', 'Trickbot', 10629, 1181, 1181],
    ['CTU', 'Ursnif', 22140, 2460, 2460],
    ['CTU', 'Zeus', 2079, 231, 231]
]

# 创建DataFrame
df = pd.DataFrame(data, columns=['数据集名', '类别名', '训练集数据量', '验证集数据量', '测试集数据量'])

# 保存为CSV文件
df.to_csv('dataset_summary.csv', index=False, encoding='utf-8-sig')

print("CSV文件已生成：dataset_summary.csv")
print("\n数据预览：")
print(df.head(10))

# 显示统计信息
print("\n数据集统计：")
print(f"USTC数据集总测试样本数: {df[df['数据集名']=='USTC']['测试集数据量'].sum():,}")
print(f"CTU数据集总测试样本数: {df[df['数据集名']=='CTU']['测试集数据量'].sum():,}")
print(f"USTC数据集总训练样本数: {df[df['数据集名']=='USTC']['训练集数据量'].sum():,}")
print(f"CTU数据集总训练样本数: {df[df['数据集名']=='CTU']['训练集数据量'].sum():,}")