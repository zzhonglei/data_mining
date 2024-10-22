import pandas as pd
import csv

# 读取 CSV 文件
file_path = '../data/GoodsTypes.csv'  # 替换为你的文件路径
output_file ='../data/GoodsTypes_deal.csv'
# 初始化一个空的二维数组
data = []
# 打开并读取 CSV 文件
with open(file_path, 'r', encoding='gbk') as file:
    for line in file:
        # 去除逗号并将每行转换为列表
        row = line.strip().split(',')  # 将行中的逗号移除，按逗号分割为列表
        data.append(row)  # 将每行列表添加到二维数组中

# 将处理后的数据写入新的 CSV 文件，使用 utf-8 编码
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)  # 将二维数组写入文件

print(f"数据已成功写入到 {output_file}")