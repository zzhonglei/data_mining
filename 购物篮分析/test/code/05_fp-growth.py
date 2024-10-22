import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def readData(inputfile):
    data = pd.read_csv(inputfile,encoding = 'gbk')
    data['Goods'] = data['Goods'].apply(lambda x:','+x)
    data = data.groupby('id').sum().reset_index()
    data['Goods'] = data['Goods'].apply(lambda x :[x[1:]])
    data_list = list(data['Goods'])
    data_translation = []
    for i in data_list:
        p = i[0].split(',')
        data_translation.append(p)
    return data_translation

data = readData('../data/GoodsOrder.csv')

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)

for index, row in rules.iterrows():
    antecedent = frozenset(row['antecedents'])
    consequent = frozenset(row['consequents'])
    support = round(row['support'], 5)
    confidence = round(row['confidence'], 5)
    lift = round(row['lift'], 5)
    print(f"{antecedent} --> {consequent} 支持度 {support} 置信度： {confidence} lift值为： {lift}")