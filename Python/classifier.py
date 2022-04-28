import sklearn.tree as tree
import graphviz
import os
import pickle

import Python.dataCreation as dc
import Python.visualization as vs


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


def create_and_save_tree(dataset: dc.Data,
                         depth: int = 5,
                         min_samples_leaf: int = 5,
                         notes: str = "",
                         visualize_tree_par: bool = True,
                         pred_col_name: str = "") -> tree.DecisionTreeClassifier:
    """
    trains a decision tree on a DataSet, uses the tree to predict the classes and saves
    the resulting dataset. Also saves the resulting tree at the path of the dataset.
    :param dataset: the data on which predictions will be made
    :param depth: maximal depth of the decision tree
    :param min_samples_leaf: minimum number of data points per leaf in the decision tree
    :param notes: Notes to be saved in the description of the data class object
    :param visualize_tree_par: determines whether comparison pictures of the original classes of the dataset as well as
    pictures of the dataset with classes from the predictions of the tree will be generated. Also a graphical
    representation of the decisiontree will be generated
    :param pred_col_name: name for the column, in which the predicted classes will be stored. only necessary, if
    create_sample_pics is True
    :return: Trained decision tree
    """
    trained_tree = train_decision_tree(dataset, max_depth=depth, min_samples_leaf=min_samples_leaf)

    dataset.extend_notes_by_one_line(f"Trained DecisionTree!")
    dataset.extend_notes_by_one_line(f"Parameters: max_depth={depth}, min_samples_leaf={min_samples_leaf}")
    tree_path = os.path.join(dataset.path, "tree_classifier.pkl")
    if not os.path.isfile(tree_path):
        with open(tree_path, "wb") as f:
            pickle.dump(trained_tree, f)

    if visualize_tree_par:
        if not pred_col_name:
            raise dc.CustomError("pred_col_name wasnt given, so predicting is not possible! Pictures can not be generated!")
        predict_classes(trained_tree=trained_tree, dataset=dataset, pred_col_name=pred_col_name)
        dataset.extend_notes_by_one_line(f"Predicted classes using Decision Tree in column \"{pred_col_name}\".")
        visualize(dataset=dataset, trained_tree=trained_tree, pred_col_name=pred_col_name)

    dataset.end_paragraph_in_notes()
    dataset.save(notes=notes)

    return trained_tree


def visualize_tree(dataset: dc.Data, trained_tree: tree.DecisionTreeClassifier, tree_pics_path: str) -> None:
    """
    creates a pdf file of the given decision tree in the folder of the dataset
    :param dataset: dataset, for which the decision tree is relevant. Attribute "path" will be used as the location for
    the resulting pdf file
    :param trained_tree: the decision tree that is to be visualized
    :param tree_pics_path: Path were pictures for the visualization of the tree is supposed to be saved
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
        graph = graphviz.Source(content, filename="tree_visualization", directory=tree_pics_path)
        graph.render()
    # clean up unnecessary files
    vis_path = os.path.join(tree_pics_path, "tree_visualization")
    if os.path.isfile(vis_path):
        os.remove(vis_path)
    gv_path = os.path.join(dataset.path, "tree_visualization_data.gv")
    if os.path.isfile(gv_path):
        os.remove(gv_path)


def visualize(dataset: dc.Data, trained_tree: tree.DecisionTreeClassifier, pred_col_name: str) -> None:
    """
    visualizes data (original vs predicted) as multiple 2d pictures as well as the tree as a diagram
    :param dataset: data to be visualized
    :param trained_tree: the trained tree
    :param pred_col_name: name, where the predicted classes are stored
    """
    df = dataset.data
    pics_path = os.path.join(dataset.path, "pics")
    if not os.path.isdir(pics_path):
        os.mkdir(pics_path)
    tree_pics_path = os.path.join(pics_path, "Decision_Tree")
    if not os.path.isdir(tree_pics_path):
        os.mkdir(tree_pics_path)

    #check if all pictures are already in the folder:
    make_pics = False
    files_in_pics = os.listdir(tree_pics_path)
    if isinstance(dataset, dc.MaybeActualDataSet):
        pics = ["00_04_org.png", "00_04_pred.png", "01_04_org.png", "01_04_pred.png", "02_03_org.png", "02_03_pred.png"]
    elif isinstance(dataset, dc.IrisDataSet):
        pics = ["sepal_length_width_org.png", "sepal_length_width_pred.png",
                "petal_length_width_org.png", "petal_length_width_pred.png",
                "sepal_length_petal_width_org.png", "sepal_length_petal_width_pred.png"]
    else:
        raise dc.CustomError("unknown Dataset type!")
    for pic in pics:
        if pic not in files_in_pics:
            make_pics = True
            break

    if make_pics:
        if isinstance(dataset, dc.MaybeActualDataSet):
            dim0, dim1, dim2, dim3, dim4, dim5 = "dim_00", "dim_04", "dim_01", "dim_04", "dim_02", "dim_03"
        elif isinstance(dataset, dc.IrisDataSet):
            dim0, dim1, dim2 = "sepal_length", "sepal_width", "petal_length"
            dim3, dim4, dim5 = "petal_width", "sepal_length", "petal_width"
        vs.visualize_2d(df=df, dims=(dim0, dim1), class_column="classes", title="original",
                        path=os.path.join(tree_pics_path, pics[0]), class_names=dataset.class_names)
        vs.visualize_2d(df=df, dims=(dim0, dim1), class_column=pred_col_name, title="predicted",
                        path=os.path.join(tree_pics_path, pics[1]), class_names=dataset.class_names)
        vs.visualize_2d(df=df, dims=(dim2, dim3), class_column="classes", title="original",
                        path=os.path.join(tree_pics_path, pics[2]), class_names=dataset.class_names)
        vs.visualize_2d(df=df, dims=(dim2, dim3), class_column=pred_col_name, title="predicted",
                        path=os.path.join(tree_pics_path, pics[3]), class_names=dataset.class_names)
        vs.visualize_2d(df=df, dims=(dim4, dim5), class_column="classes", title="original",
                        path=os.path.join(tree_pics_path, pics[4]), class_names=dataset.class_names)
        vs.visualize_2d(df=df, dims=(dim4, dim5), class_column=pred_col_name, title="predicted",
                        path=os.path.join(tree_pics_path, pics[5]), class_names=dataset.class_names)
    #visualization of the decision tree
    visualize_tree(dataset, trained_tree, tree_pics_path)


def test() -> None:
    """
    runs the training process and visualizes results
    """
    #members = [1000 for _ in range(6)]
    #dataset = dc.MaybeActualDataSet(members)
    #dataset = dc.MaybeActualDataSet.load("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220226_135403_MaybeActualDataSet")
    dataset = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\220428_124321_IrisDataSet")
    dataset = dc.IrisDataSet()
    create_and_save_tree(dataset, pred_col_name="pred_tree")
    dataset.run_hics()
    """dataset = dc.MaybeActualDataSet.load("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220226_135403_MaybeActualDataSet")
    trained_tree = dataset.load_tree()"""
    #matrix = vs.get_change_matrix(data, ("classes", "predicted_classes"))
    #print(matrix)


if __name__ == "__main__":
    test()
    """members_ = [1000 for _ in range(6)]
    set_ = dc.MaybeActualDataSet(members_)
    set_.run_hics()
    data, trained_tree_ = save_predicted_data(dataset=set_, pred_col_name="predicted_classes")"""
