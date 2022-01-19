import sys
from typing import List
import sklearn.tree as tree
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

sys.path.append("../DataCreation/")
import dataCreation as dc



def train_decision_tree(df: pd.DataFrame) -> tree.DecisionTreeClassifier:
    classes = df["classes"]
    values = df.drop(columns=["classes"])
    decision_tree = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
    decision_tree.fit(values, classes)
    return decision_tree


def run(members: List[int]) -> None:
    df = dc.MaybeActualDataSet(members).data
    trained_tree = train_decision_tree(df)
    values = df.drop(columns="classes")
    test_val = values.iloc[[i*1000 for i in range(5)]]
    print(trained_tree.predict(test_val))
    dot_data = tree.export_graphviz(trained_tree, out_file=None)
    """
    plt.figure()
    plt.clf()
    graph = graphviz.Source(dot_data)
    graph.render("iris")

    #tree.plot_tree(trained_tree)
    plt.show()
    """
    dot_data = tree.export_graphviz(trained_tree, out_file=None,
                                    feature_names=values.columns,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()



if __name__ == "__main__":
    run(members=[1000 for _ in range(5)])

