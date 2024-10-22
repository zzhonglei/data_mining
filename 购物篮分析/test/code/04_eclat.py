import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules

def readData(inputfile):
    data = pd.read_csv(inputfile, encoding='gbk')
    if 'Goods' not in data.columns or 'id' not in data.columns:
        raise ValueError("输入的 CSV 文件必须包含 'id' 和 'Goods' 列。")
    data['Goods'] = data['Goods'].apply(lambda x: x.split(','))
    data = data.groupby('id')['Goods'].sum().reset_index()
    data['Goods'] = data['Goods'].apply(lambda x: [item.strip() for item in x if item.strip()])
    data_list = data['Goods'].tolist()
    return data_list

# 读取和预处理数据
data = readData('../data/GoodsOrder.csv')

# 一热编码
mlb = MultiLabelBinarizer()
df_encoded = pd.DataFrame(mlb.fit_transform(data), columns=mlb.classes_)
print(f"一热编码后的 DataFrame 形状: {df_encoded.shape}")

# 计算单个项的支持度
item_support = df_encoded.sum(axis=0) / len(df_encoded)
print("单个项的支持度:")
print(item_support.sort_values(ascending=False))

# 进行 Apriori 算法频繁项集挖掘
# 可以尝试不同的 min_support 值，例如 0.005
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True, max_len=3)
print(f"发现的频繁项集数量: {len(frequent_itemsets)}")
print(frequent_itemsets.head())

# 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)
print(f"生成的关联规则数量: {len(rules)}")
print(rules.head())

# 按支持度排序规则
rules.sort_values(by='support', ascending=False, inplace=True)

# 输出关联规则
for index, row in rules.iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    support = round(row['support'], 5)
    confidence = round(row['confidence'], 5)
    lift = round(row['lift'], 5)
    print(f"{{{antecedents}}} --> {{{consequents}}} 支持度 {support} 置信度： {confidence} lift值为： {lift}")