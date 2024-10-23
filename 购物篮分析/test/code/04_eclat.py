import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import association_rules
from itertools import combinations

# 读取数据
def readData(inputfile):
    data = pd.read_csv(inputfile, encoding='gbk')
    data['Goods'] = data['Goods'].apply(lambda x: x.split(','))
    data = data.groupby('id')['Goods'].sum().reset_index()
    data['Goods'] = data['Goods'].apply(lambda x: [item.strip() for item in x if item.strip()])
    data_list = data['Goods'].tolist()
    return data_list

# ECLAT算法函数，支持挖掘多个项集之间的关系
def eclat(df_encoded, min_support=0.02, max_len=2):
    itemsets = {}
    # 计算单项的支持度
    for column in df_encoded.columns:
        support = df_encoded[column].sum() / len(df_encoded)
        if support >= min_support:
            itemsets[frozenset([column])] = support
    # 生成多项组合，2项集及更高阶项集
    for k in range(2, max_len + 1):
        for item_comb in combinations(df_encoded.columns, k):
            combined = df_encoded[list(item_comb)].all(axis=1)
            support = combined.sum() / len(df_encoded)
            if support >= min_support:
                itemsets[frozenset(item_comb)] = support
    return pd.DataFrame(list(itemsets.items()), columns=['itemsets', 'support'])

if __name__ == '__main__':
    # 读取数据
    data = readData('../data/GoodsOrder.csv')
    # 对商品数据进行一热编码
    mlb = MultiLabelBinarizer()
    df_encoded = pd.DataFrame(mlb.fit_transform(data), columns=mlb.classes_)
    print(f"一热编码后的 DataFrame 形状: {df_encoded.shape}")
    # 计算单个项的支持度
    item_support = df_encoded.sum(axis=0) / len(df_encoded)
    print("单个项的支持度:")
    print(item_support.sort_values(ascending=False))
    # 生成频繁项集
    frequent_itemsets = eclat(df_encoded, min_support=0.02)
    print(f"发现的频繁项集数量: {len(frequent_itemsets)}")
    print(frequent_itemsets.head())
    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)
    print(f"生成的关联规则数量: {len(rules)}")
    print(rules.head())
    # 按支持度排序规则并输出
    rules.sort_values(by='support', ascending=False, inplace=True)
    for index, row in rules.iterrows():
        antecedent = frozenset(row['antecedents'])
        consequent = frozenset(row['consequents'])
        support = round(row['support'], 5)
        confidence = round(row['confidence'], 5)
        lift = round(row['lift'], 5)
        print(f"{antecedent} --> {consequent} 支持度 {support} 置信度： {confidence} lift值为： {lift}")