import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import itertools
import math
from mlxtend.frequent_patterns.association_rules import association_rules
from mlxtend.frequent_patterns import fpcommon as fpc


def fpgrowth(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0):
    fpc.valid_input_check(df)
    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )
    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, _ = fpc.setup_fptree(df, min_support)
    minsup = math.ceil(min_support * len(df.index))  # min support as count
    generator = fpg_step(tree, minsup, colname_map, max_len, verbose)

    return fpc.generate_itemsets(generator, len(df.index), colname_map)


def fpg_step(tree, minsup, colnames, max_len, verbose):
    """
    Performs a recursive step of the fpgrowth algorithm.

    Parameters
    ----------
    tree : FPTree
    minsup : int

    Yields
    ------
    lists of strings
        Set of items that has occurred in minsup itemsets.
    """
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.cond_items + list(itemset)
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    if verbose:
        tree.print_status(count, colnames)

    # Generate conditional trees to generate frequent itemsets one item larger
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in fpg_step(cond_tree, minsup, colnames, max_len, verbose):
                yield sup, iset


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

if __name__ == '__main__':
    data = readData('../data/GoodsOrder.csv')
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = fpgrowth(df, min_support=0.02, use_colnames=True,max_len=3)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)
    for index, row in rules.iterrows():
        antecedent = frozenset(row['antecedents'])
        consequent = frozenset(row['consequents'])
        support = round(row['support'], 5)
        confidence = round(row['confidence'], 5)
        lift = round(row['lift'], 5)
        print(f"{antecedent} --> {consequent} 支持度 {support} 置信度： {confidence} lift值为： {lift}")