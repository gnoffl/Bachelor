import sys
from typing import List, Tuple
import sklearn.tree as tree
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import os
import pickle

import Python.DataCreation.dataCreation as dc
import Python.DataCreation.visualization as vs


def load_tree(tree_path: str) -> tree.DecisionTreeClassifier:
    """
    loads a decisionTree from a given Path
    :param tree_path: path to the pickled tree
    :return: the loaded tree object
    """
    with open(tree_path, "rb") as f:
        print("loading tree!")
        decision_tree = pickle.load(f)
    return decision_tree


def train_decision_tree(dataset: dc.Data, max_depth=5, min_samples_leaf=5) -> tree.DecisionTreeClassifier:
    """
    trains a decision tree for the given data
    :param dataset: data
    :param max_depth: max depth of the decision tree
    :param min_samples_leaf: minimum samples per leaf in the decision tree
    :return: the trained tree
    """
    df = dataset.data
    tree_path = os.path.join(dataset.path, "tree_classifier.pkl")
    if not os.path.isfile(tree_path):
        classes = df["classes"]
        values = df[dataset.data_columns]
        decision_tree = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        decision_tree.fit(values, classes)
    else:
        decision_tree = load_tree(tree_path=tree_path)
    return decision_tree


def predict_classes(trained_tree: tree.DecisionTreeClassifier, dataset: dc.Data, pred_col_name: str) -> None:
    """
    uses a trained decision tree to predict the classes of the given data
    :param trained_tree: the trained decision tree
    :param dataset: the data
    :param pred_col_name: name for the column, in which the predicted classes will be stored
    """
    data = dataset.data
    if pred_col_name in data.columns:
        raise dc.CustomError("Column already exists!")
    values = data[dataset.data_columns]
    results = trained_tree.predict(values)
    data[pred_col_name] = results


def save_predicted_data(dataset: dc.MaybeActualDataSet,
                        pred_col_name: str,
                        depth: int = 5,
                        min_samples_leaf: int = 5,
                        notes: str = "") -> tree.DecisionTreeClassifier:
    """
    trains a decision tree on a MaybeActualDataSet, uses the tree to predict the classes and saves
    the resulting dataset. Also saves the resulting tree at the path of the dataset.
    :param dataset: the data on which predictions will be made
    :param pred_col_name: name for the column, in which the predicted classes will be stored
    :param depth: maximal depth of the decision tree
    :param min_samples_leaf: minimum number of data points per leaf in the decision tree
    :param notes: Notes to be saved in the description of the data class object
    :return: Trained decision tree
    """
    df = dataset.data
    trained_tree = train_decision_tree(dataset, max_depth=depth, min_samples_leaf=min_samples_leaf)
    predict_classes(trained_tree=trained_tree, dataset=dataset, pred_col_name=pred_col_name)

    dataset.extend_notes_by_one_line(f"Predicted classes using Decision Tree in column \"{pred_col_name}\".")
    dataset.extend_notes_by_one_line(f"Parameters: max_depth={depth}, min_samples_leaf={min_samples_leaf}")
    dataset.end_paragraph_in_notes()
    dataset.save(notes=notes)
    tree_path = os.path.join(dataset.path, "tree_classifier.pkl")
    if not os.path.isfile(tree_path):
        with open(tree_path, "wb") as f:
            pickle.dump(trained_tree, f)
    return trained_tree


def visualize_tree(dataset: dc.Data, trained_tree: tree.DecisionTreeClassifier) -> None:
    """
    creates a pdf file of the given decision tree in the folder of the dataset
    :param dataset: dataset, for which the decision tree is relevant. Attribute "path" will be used as the location for
    the resulting pdf file
    :param trained_tree: the decision tree that is to be visualized
    """
    df = dataset.data
    values = df[dataset.data_columns]
    visualization_path = os.path.join(dataset.path, "tree_visualization_data.gv")
    with open(visualization_path, "w") as f:
        tree.export_graphviz(trained_tree, out_file=f,
                             feature_names=values.columns,
                             filled=True, rounded=True,
                             special_characters=True)
    with open(visualization_path, "r") as f2:
        content = f2.read()
        graph = graphviz.Source(content, filename="tree_visualization", directory=dataset.path)
        graph.render()
    # clean up unnecessary files
    candidates = ["tree_visualization", "tree_visualization_data.gv"]
    files = os.listdir(dataset.path)
    for cand in candidates:
        if cand in files:
            os.remove(os.path.join(dataset.path, cand))


def visualize(dataset: dc.Data, trained_tree: tree.DecisionTreeClassifier) -> None:
    """
    visualizes data (original vs predicted) as multiple 2d pictures as well as the tree as a diagram
    :param dataset: data to be visualized
    :param trained_tree: the trained tree
    """
    df = dataset.data
    pics_path = os.path.join(dataset.path, "pics")
    if not os.path.isdir(pics_path):
        os.mkdir(pics_path)

    #check if all pictures are already in the folder:
    make_pics = False
    files_in_pics = os.listdir(pics_path)
    pics = ["00_04_org.png", "00_04_pred.png", "01_04_org.png", "01_04_pred.png", "02_03_org.png", "02_03_pred.png"]
    for pic in pics:
        if pic not in files_in_pics:
            make_pics = True
    if make_pics:
        vs.visualize_2d(df, ("dim_00", "dim_04"), "classes", title="original", path=os.path.join(pics_path, "00_04_org.png"))
        vs.visualize_2d(df, ("dim_00", "dim_04"), "predicted_classes", title="predicted", path=os.path.join(pics_path, "00_04_pred.png"))
        vs.visualize_2d(df, ("dim_01", "dim_04"), "classes", title="original", path=os.path.join(pics_path, "01_04_org.png"))
        vs.visualize_2d(df, ("dim_01", "dim_04"), "predicted_classes", title="predicted", path=os.path.join(pics_path, "01_04_pred.png"))
        vs.visualize_2d(df, ("dim_02", "dim_03"), "classes", title="original", path=os.path.join(pics_path, "02_03_org.png"))
        vs.visualize_2d(df, ("dim_02", "dim_03"), "predicted_classes", title="predicted", path=os.path.join(pics_path, "02_03_pred.png"))
    #visualization of the decision tree
    visualize_tree(dataset, trained_tree)


def run() -> None:
    """
    runs the training process and visualizes results
    """
    dataset = dc.MaybeActualDataSet.load("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220226_135403_MaybeActualDataSet")
    trained_tree = save_predicted_data(dataset=dataset, pred_col_name="predicted_classes")
    dataset.run_hics()
    """dataset = dc.MaybeActualDataSet.load("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220226_135403_MaybeActualDataSet")
    trained_tree = dataset.load_tree()"""
    visualize(dataset, trained_tree)
    #matrix = vs.get_change_matrix(data, ("classes", "predicted_classes"))
    #print(matrix)


if __name__ == "__main__":
    run()
    """members_ = [1000 for _ in range(6)]
    set_ = dc.MaybeActualDataSet(members_)
    set_.run_hics()
    data, trained_tree_ = save_predicted_data(dataset=set_, pred_col_name="predicted_classes")"""
