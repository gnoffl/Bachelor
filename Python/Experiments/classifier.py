import sys
from typing import List, Tuple
import sklearn.tree as tree
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import os

import Python.DataCreation.dataCreation as dc
import Python.DataCreation.visualization as vs



def train_decision_tree(df: pd.DataFrame, max_depth=5, min_samples_leaf=5) -> tree.DecisionTreeClassifier:
    """
    trains a decision tree for the given data
    :param df: data
    :param max_depth: max depth of the decision tree
    :param min_samples_leaf: minimum samples per leaf in the decision tree
    :return: the trained tree
    """
    classes = df["classes"]
    values = df.drop(columns=["classes"])
    decision_tree = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    decision_tree.fit(values, classes)
    return decision_tree


def predict_classes(trained_tree: tree.DecisionTreeClassifier, data: pd.DataFrame, pred_col_name: str) -> None:
    """
    uses a trained decision tree to predict the classes of the given data
    :param trained_tree: the trained decision tree
    :param data: the data
    :param pred_col_name: name for the column, in which the predicted classes will be stored
    """
    if pred_col_name in data.columns:
        raise dc.CustomError("Column already exists!")
    values = data.drop(columns=["classes"])
    results = trained_tree.predict(values)
    data[pred_col_name] = results


def save_predicted_data(dataset: dc.MaybeActualDataSet,
                        pred_col_name: str,
                        depth: int = 5,
                        min_samples_leaf: int = 5,
                        notes: str = "") -> Tuple[pd.DataFrame, tree.DecisionTreeClassifier]:
    """
    trains a decision tree on a MaybeActualDataSet, uses the tree to predict the classes and saves
    the resulting dataset
    :param dataset: the data on which predictions will be made
    :param pred_col_name: name for the column, in which the predicted classes will be stored
    :param depth: maximal depth of the decision tree
    :param min_samples_leaf: minimum number of data points per leaf in the decision tree
    :param notes: Notes to be saved in the description of the data class object
    :return: Tuple with the dataframe and the trained tree
    """
    df = dataset.data
    trained_tree = train_decision_tree(df, max_depth=depth, min_samples_leaf=min_samples_leaf)
    predict_classes(trained_tree=trained_tree, data=df, pred_col_name=pred_col_name)

    dataset.extend_notes(f"Predicted classes using Decision Tree in column \"{pred_col_name}\".")
    dataset.extend_notes(f"Parameters: max_depth={depth}, min_samples_leaf={min_samples_leaf}")
    dataset.save(notes=notes)
    return df, trained_tree


#todo: make this work from saved data
def visualize(df: pd.DataFrame, trained_tree: tree.DecisionTreeClassifier) -> None:
    """
    visualizes data (original vs predicted) as multiple 2d pictures as well as the tree as a diagram
    :param df: data to be visualized
    :param trained_tree: the trained tree
    """
    vs.visualize_2d(df, ("dim_00", "dim_04"), "classes", title="original")
    vs.visualize_2d(df, ("dim_00", "dim_04"), "predicted_classes", title="predicted")
    vs.visualize_2d(df, ("dim_01", "dim_04"), "classes", title="original")
    vs.visualize_2d(df, ("dim_01", "dim_04"), "predicted_classes", title="predicted")
    vs.visualize_2d(df, ("dim_02", "dim_03"), "classes", title="original")
    vs.visualize_2d(df, ("dim_02", "dim_03"), "predicted_classes", title="predicted")
    #vs.create_3d_gif(df=df, dims=("dim_00", "dim_01", "dim_04"), name="maybe_actual_data_original", class_column="classes", steps=30)
    #vs.create_3d_gif(df=df, dims=("dim_00", "dim_01", "dim_04"), name="maybe_actual_data_predicted", class_column="predicted_classes", steps=30)
    #visualization of the decision tree
    values = df.drop(columns="classes")
    values = values.drop(columns="predicted_classes")
    dot_data = tree.export_graphviz(trained_tree, out_file=None,
                                    feature_names=values.columns,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()


def run() -> None:
    """
    runs the training process and visualizes results
    """
    members_ = [1000 for _ in range(6)]
    data, trained_tree = save_predicted_data(members=members_, pred_col_name="predicted_classes")
    #data = load_df("1000members_5depth_5min_samples.csv")
    #visualize(data, trained_tree)
    matrix = vs.get_change_matrix(data, ("classes", "predicted_classes"))
    print(matrix)


if __name__ == "__main__":
    #run()
    #todo: run_hics darf nicht die Splate "predicted_classes" sehen
    """members_ = [1000 for _ in range(6)]
    set_ = dc.MaybeActualDataSet(members_)
    set_.run_hics()
    data, trained_tree_ = save_predicted_data(dataset=set_, pred_col_name="predicted_classes")"""
