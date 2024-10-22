import pandas as pd
from sklearn.model_selection import train_test_split
from C45 import C45Classifier
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv('../../extent/environment_data.csv')
last_column = df.iloc[:, -1]
# 获取最后一列中的唯一值
unique_values = last_column.unique()
value_to_number = {value: idx + 1 for idx, value in enumerate(unique_values)}
# 用编号替换最后一列的原值
df.iloc[:, -1] = last_column.map(value_to_number)
# 打印修改后的 DataFrame
# 如果需要将修改后的DataFrame转换为数组
array = df.values.tolist()
labels = [row[-1] for row in array]
datas = [row[:-1] for row in array]

data_tr,data_te,label_tr,label_te = train_test_split(datas,labels,test_size=0.1)

# 构造C4.5决策树
model = C45Classifier()
model.fit(data_tr, label_tr)
model.evaluate(data_te, label_te)
print("\n")

pre_te = model.predict(data_te)
cm_te = confusion_matrix(label_te,pre_te)
print(cm_te)
print(accuracy_score(label_te,pre_te))
model.generate_tree_diagram(graphviz,"c4.5-tree")