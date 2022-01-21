import sys
from typing import List, Tuple
import sklearn.tree as tree
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import os

import DataCreation.dataCreation as dc
import DataCreation.visualization as vs



def train_decision_tree(df: pd.DataFrame, max_depth=5, min_samples_leaf=5) -> tree.DecisionTreeClassifier:
    classes = df["classes"]
    values = df.drop(columns=["classes"])
    decision_tree = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    decision_tree.fit(values, classes)
    return decision_tree


def predict_classes(trained_tree, data):
    values = data.drop(columns=["classes"])
    results = trained_tree.predict(values)
    data["predicted_classes"] = results


def save_predicted_data(members: List[int], depth=5, min_samples_leaf=5) -> Tuple:
    df = dc.MaybeActualDataSet(members).data
    trained_tree = train_decision_tree(df, max_depth=depth, min_samples_leaf=min_samples_leaf)
    predict_classes(trained_tree, df)
    df.to_csv(os.path.join(os.path.dirname(__file__), f"{members[0]}members_{depth}depth_{min_samples_leaf}min_samples.csv"))
    return df, trained_tree


def visualize(df, trained_tree):
    values = df.drop(columns="classes")
    values = values.drop(columns="predicted_classes")
    vs.visualize_2d(df, ("dim_00", "dim_02"), "classes", title="original")
    vs.visualize_2d(df, ("dim_00", "dim_02"), "predicted_classes", title="predicted")
    #vs.create_3d_gif(df=df, dims=("dim_00", "dim_01", "dim_04"), name="maybe_actual_data_original", class_column="classes", steps=30)
    #vs.create_3d_gif(df=df, dims=("dim_00", "dim_01", "dim_04"), name="maybe_actual_data_predicted", class_column="predicted_classes", steps=30)
    dot_data = tree.export_graphviz(trained_tree, out_file=None,
                                    feature_names=values.columns,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()


def load_df(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def get_number_of_changed_points(data: pd.DataFrame, fraction: bool = False) -> Tuple[int, int]:
    count = data.loc[data["classes"] != data["predicted_classes"]].count()["classes"]
    length = len(data)
    return count, length


def get_change_matrix(data) -> pd.DataFrame:
    present_classes = set(data["classes"].values)
    present_classes.union(set(data["predicted_classes"]))
    data_dict = {}
    index = []
    res_matrix = pd.DataFrame
    for class_ in present_classes:
        data_dict[class_] = []
    for org_class in present_classes:
        index.append(org_class)
        original_class_data = data.loc[data["classes"] == org_class]
        for new_class in present_classes:
            count = original_class_data.loc[original_class_data["predicted_classes"] == new_class].count()["classes"]
            data_dict[new_class].append(count)
        res_matrix = pd.DataFrame(data_dict, index=index)
    return res_matrix



def run():
    #data, trained_tree = save_predicted_data(members=[1000 for _ in range(5)])
    data = load_df("1000members_5depth_5min_samples.csv")
    #visualize(data, trained_tree)
    matrix = get_change_matrix(data)
    print(matrix)


if __name__ == "__main__":
    run()

