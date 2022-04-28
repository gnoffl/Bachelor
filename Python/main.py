import os
from typing import Dict, List
import random
import shutil
import pandas as pd
import sklearn.tree as tree

import dataCreation as dc
import visualization as vs
import classifier as cl
import QSM
import createDataSplits as cds


def add_result_matrices(matrix1: pd.DataFrame, matrix2: pd.DataFrame) -> pd.DataFrame:
    """
    adds the values of two dataframes. Missing values will be filled with zeroes
    :param matrix1: first summand matrix
    :param matrix2: second summand matrix
    :return: sum of the matrices
    """
    result_matrix = matrix1.add(other=matrix2, fill_value=0)
    #data type is automatically changed to float for some reason after adding the frames
    #even though fill_value is given, NaN can occur sometimes, so it needs to be replaced
    return result_matrix.fillna(0).astype("int32")


def recursion_end(curr_folder: str, dim: str, q: float,
                  subspace_list: List[pd.DataFrame], trained_tree: tree.DecisionTreeClassifier) -> pd.DataFrame:
    """
    Runs QSM on the dataset in the current folder. Results are visualized. The resulting data as well as the original
    values will be appended to a list of dataframes, in order to combine them again to a full dataset after shifting on
    the binned data.
    :param curr_folder: folder where the dataset lies
    :param dim: dim to shift
    :param q: quantile the data is supposed to be shifted
    :param subspace_list: in this list the resulting datasets from QSM on the data splits will be saved as dataframes to
    combine them later
    :param trained_tree: tree to do the predictions
    :return: the result matrix of the QSM
    """
    dataset = dc.Data.load(curr_folder)
    result = QSM.run_QSM_decisionTree(dataset=dataset,
                                      quantiles={dim: q},
                                      save_changes=True,
                                      trained_tree=trained_tree)
    new_dim_name = f"{dim}_shifted_by_{str(q)}"
    new_class_name = f"pred_with_{new_dim_name}"
    data = dataset.data.copy(deep=True)
    #copy values to avoid confusion when shifting the dimensions
    org_dim_data = data[dim].values.copy()
    org_pred_data = data["org_pred_classes_QSM"].values.copy()
    data[dim] = data[new_dim_name].values.copy()
    data = data[dataset.data_columns]
    data["classes"] = dataset.data["classes"]
    data["pred_classes"] = dataset.data[new_class_name]
    data["org_pred"] = org_pred_data
    data["source"] = curr_folder.split("\\")[-1]
    data[f"{dim}_org"] = org_dim_data
    subspace_list.append(data)
    #visualization with dim_04 is preferred
    if isinstance(dataset, dc.MaybeActualDataSet):
        pref_dim = "dim_04"
        secnd_choice = "dim_00"
    elif isinstance(dataset, dc.IrisDataSet):
        pref_dim = "petal_length"
        secnd_choice = "petal_width"
    else:
        raise dc.CustomError(f"class {type(dataset)} is unknown!")
    if dim != pref_dim:
        QSM.visualize_QSM(base_dim=pref_dim, dim_before_shift=dim, shift=q, dataset=dataset)
    else:
        QSM.visualize_QSM(base_dim=secnd_choice, dim_before_shift=dim, shift=q, dataset=dataset)
    matrix = result[dim]
    return matrix


def recursive_QSM(curr_folder: str, curr_name: str, dim: str, q: float,
                  subspace_list: List[pd.DataFrame], trained_tree: tree.DecisionTreeClassifier) -> pd.DataFrame:
    """
    checks if the curr_folder contains data splits. if so, the function is recursively called on the splits. If no
    further splits are present, recursion_end will be called, to do QSM and handle the results
    :param curr_folder: path to the folder that is inspected at the moment
    :param curr_name: name of the current folder
    :param dim: dimension to shift in QSM
    :param q: quantile to shift the data by
    :param subspace_list: list that contains the resulting data frames of the splits that were already shifted in QSM
    :param trained_tree: trained decision tree, to predict do predictions on the data
    :return: result matrix, that contains the sum of the result matrices from the QSMs executed on the data splits that
    are in the current folder
    """
    suffixes = ["0", "1"]
    count = 0
    result_matrix = pd.DataFrame()
    folders = [folder for folder in os.listdir(curr_folder) if os.path.isdir(os.path.join(curr_folder, folder))]
    #try to append 0 or 1 to the current folder name and see if those are subfolders --> if both exist, recursion
    # continues
    for suffix in suffixes:
        if curr_name:
            next_name = f"{curr_name}_{suffix}"
        else:
            next_name = suffix
        next_folder = os.path.join(curr_folder, next_name)
        if next_name in folders:
            #next layer of recursion
            count += 1
            curr_res = recursive_QSM(curr_folder=next_folder, curr_name=next_name, dim=dim, q=q,
                                     subspace_list=subspace_list, trained_tree=trained_tree)
            result_matrix = add_result_matrices(result_matrix, curr_res)

    if count == 2:
        pass
    elif count == 0:
        #if no further splits are present, end recursion (do QSM, save resulting dataset and result matrix)
        result_matrix = recursion_end(curr_folder=curr_folder, dim=dim, q=q, subspace_list=subspace_list,
                                      trained_tree=trained_tree)
    else:
        raise dc.CustomError(f"{count} number of splits detected! Should be either 0 or 2!")

    #create a local change matrix for the results from all splits of the current data split
    result_matrix.to_csv(os.path.join(curr_folder, "local_change_matrix.csv"))
    return result_matrix


def load_parameters():
    print("loading parameters..")
    path_here = os.path.dirname(__file__)
    param_path = os.path.join(path_here, "..", "Data", "Parameters.txt")
    parameters = {}
    with open(param_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "	" in line:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=")
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            if value == "True" or value == "true":
                                value = True
                            if value == "False" or value == "false":
                                value = False
                    parameters[key] = value
    return parameters, param_path


def run_from_file(quantiles: Dict[str, float], dataset: dc.Data):
    params, par_path = load_parameters()
    print("copying parameter file..")
    new_par_path = os.path.join(dataset.path, "Parameters.txt")
    shutil.copy2(par_path, new_par_path)
    dataset.extend_notes_by_one_line("Parameters for running the comparison between methods can be found in "
                                     "\"Parameters.txt\"")
    dataset.end_paragraph_in_notes()
    compare_vanilla_split(quantiles=quantiles, dataset=dataset, **params)



def QSM_on_binned_data(dataset: dc.Data, quantiles: Dict[str, float],
                       start_folders: Dict[str, str], trained_tree: tree.DecisionTreeClassifier or None = None)\
        -> pd.DataFrame:
    """
    runs recursive_QSM on the dataset for each dim/quantile pair in quantiles. The shifted splits will be combined into
    a full dataset, that is saved in the "Splits" folder of the dataset
    :param dataset: dataset to run the recursive QSM on
    :param quantiles: Dictionary with dimensions as key and a corresponding quantile, by which the data is supposed to
    be shifted in the key-dimension
    :param start_folders: Dictionary with dimensions as keys and the path to a folder containing split datasets.
    :param trained_tree: tree to do predictions on the data in QSM
    :return: result matrix of the QSM on the binned dataset
    """
    for dim, q in quantiles.items():
        start_folder = start_folders[dim]
        #subspace_list will contain a dataframe for each dataset that is not split further. Combining these Dataframes
        # will yield a dataframe containing all the original data points, after shifting them separately in their splits
        subspace_list = []
        #start recursion
        result_matrix = recursive_QSM(curr_folder=start_folder, curr_name="", dim=dim, q=q,
                                      subspace_list=subspace_list, trained_tree=trained_tree)
        #create new dataset from the results of QSM on the splits
        full_data = pd.concat(subspace_list)
        full_dataset = dataset.clone_meta_data()
        full_dataset.take_new_data(full_data)
        full_dataset.extend_notes_by_one_line("this dataset contains the full data, that was shifted "
                                              "in binned subsets.")
        full_dataset.extend_notes_by_one_line(f"Dimension {dim} was shifted by {q}.")
        full_dataset.end_paragraph_in_notes()
        full_dataset.save(start_folder)

        os.rename(os.path.join(start_folder, "local_change_matrix.csv"), os.path.join(start_folder, "binning_result_matrix.csv"))
        visualize_QSM_on_binned_data(full_dataset, dim)
        return result_matrix


def run_vanilla_qsm(dataset: dc.Data, quantiles: Dict[str, float],
                    trained_tree: tree.DecisionTreeClassifier or None = None) -> None:
    """
    runs the standard QSM on the given dataset. Results are visualized and the result matrix is saved.
    :param dataset: dataset to run QSM on
    :param quantiles: Dictionary with dimensions as key and a corresponding quantile, by which the data is supposed to
    be shifted in the key-dimension
    :param trained_tree: tree to do predictions on the data in QSM
    """
    results = QSM.run_QSM_decisionTree(dataset=dataset, quantiles=quantiles, save_changes=True,
                                       trained_tree=trained_tree)
    for dim, q in quantiles.items():
        #visualization
        if isinstance(dataset, dc.MaybeActualDataSet):
            pref_dim = "dim_04"
            secnd_choice = "dim_00"
        elif isinstance(dataset, dc.IrisDataSet):
            pref_dim = "petal_length"
            secnd_choice = "petal_width"
        else:
            raise dc.CustomError(f"class {type(dataset)} is unknown!")
        if dim != pref_dim:
            QSM.visualize_QSM(base_dim=pref_dim, dim_before_shift=dim, shift=q, dataset=dataset,
                              save_path=os.path.join(dataset.path, "pics", "QSM", "vanilla", dim),
                              class_names=dataset.class_names)
        else:
            QSM.visualize_QSM(base_dim=secnd_choice, dim_before_shift=dim, shift=q, dataset=dataset,
                              save_path=os.path.join(dataset.path, "pics", "QSM", "vanilla", dim),
                              class_names=dataset.class_names)

        #save result matrix
        results_folder = os.path.join(dataset.path, "vanilla_results")
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
        matrix = results[dim]
        matrix.to_csv(os.path.join(results_folder, f"{dim}_{q}.csv"))


def visualize_QSM_on_binned_data(dataset: dc.Data, shifted_dim: str, common_dim: str = "") -> None:
    """
    visualization of the combined results of doing QSM on split datasets. will create pairs of pictures from the data
    where the data points before and after the shift from QSM are compared.
    :param dataset: combined dataset from the QSM on split datasets
    :param shifted_dim: dim that was shifted in QSM
    :param common_dim: possible argument to define which Dimension will be the second on the 2d plots. If this argument
    is omitted a random data dim that is not the shifted dim will be selected
    """
    #pick a random dimension to display the data in 2d visualizations
    if not common_dim:
        data_dims = dataset.data_columns
        data_dims.remove(shifted_dim)
        common_dim = random.choice(data_dims)
    folder = os.path.join(dataset.path, "pics", "QSM_Binning")
    #compare how the final data bins lie compared to each other, and how they get shifted individually
    vs.compare_shift_2d(df=dataset.data, common_dim=common_dim, dims_to_compare=(f"{shifted_dim}_org", shifted_dim),
                        class_columns=("source", "source"),
                        path=os.path.join(folder, "source.png"))
    #compare the predictions of the classifier on the original data to the predictions of the classifier on the shifted
    # data
    vs.compare_shift_2d(df=dataset.data, common_dim=common_dim, dims_to_compare=(f"{shifted_dim}_org", shifted_dim),
                        class_columns=("pred_classes", "org_pred"),
                        path=os.path.join(folder, "predictions.png"))
    #see how the data is shifted by comparing the datapoints before and after the shift. displaed classes are the
    # original classes of the data points
    vs.compare_shift_2d(df=dataset.data, common_dim=common_dim, dims_to_compare=(f"{shifted_dim}_org", shifted_dim),
                        class_columns=("classes", "classes"),
                        path=os.path.join(folder, "classes.png"))


def compare_vanilla_split(quantiles: Dict[str, float], dataset: dc.Data, max_depth: int = 5,
                          min_samples_leaf: int = 5, nr_processes: int = 4, p_value: float = 0.05,
                          goodness_over_length: bool = True, threshold_fraction: float = 0.7,
                          max_split_nr: int = 3, HiCS_parameters: str = "") -> None:
    """
    runs QSM on the full dataset as well as the binned data. result matrices are saved as well as visualizations of the
    resulting shifted data for each approach
    :param quantiles: Dictionary with dimensions as key and a corresponding quantile, by which the data is supposed to
    be shifted in the key-dimension
    :param dataset: dataset to run QSM on
    :param max_depth: max depth of the decision tree
    :param min_samples_leaf: minimum samples per leaf in the decision tree
    :param nr_processes: determines the number of processes that are used to calculate the ks statistics
    :param p_value: if the best split has a p-value higher than this, the dataset will not be split further
    :param goodness_over_length: determines how the best HiCS is selected. If True, the subspace with the highest
    contrast will be selected. If False, the number of dimensions will also play a role. (For details see get_HiCS)
    :param threshold_fraction: determines the cutoff contrast value for "long" subspaces
    :param max_split_nr: max count of further splits (max count of splits in general when manually calling the
    function)
    :param HiCS_parameters: further parameters to be added to HiCS
    """
    print("training decision tree..")
    trained_tree = cl.create_and_save_tree(dataset, pred_col_name="test", depth=max_depth,
                                           min_samples_leaf=min_samples_leaf)
    print("start binning of data..")
    start_folder_dict = cds.data_binning(dataset=dataset, shifts=quantiles, max_split_nr=max_split_nr, visualize=True,
                                         nr_processes=nr_processes, max_p_for_split=p_value,
                                         threshold_fraction=threshold_fraction, HiCS_parameters=HiCS_parameters,
                                         goodness_over_length=goodness_over_length)
    print("running QSM on full dataset..")
    run_vanilla_qsm(dataset, quantiles, trained_tree)
    print("running QSM on split dataset..")
    QSM_on_binned_data(dataset=dataset, quantiles=quantiles, start_folders=start_folder_dict, trained_tree=trained_tree)



def main() -> None:
    quantiles = {
        "dim_04": 0.1,
        "dim_00": 0.05,
        "dim_01": -0.2
    }
    members = [50 for _ in range(6)]
    print("initializing dataset..")
    dataset = dc.MaybeActualDataSet(members)
    compare_vanilla_split(dataset=dataset, quantiles=quantiles)


def test():
    """
    quantiles = {
        "dim_04": 0.1,
        "dim_00": 0.05,
        "dim_01": -0.2
    }
    members = [50 for _ in range(6)]"""
    quantiles = {
        "sepal_length": 0.1,
        "petal_length": 0.05,
        "petal_width": -0.2
    }
    run_from_file(dataset=dc.IrisDataSet(), quantiles=quantiles)


def round_arr(arr: List[float]) -> None:
    for i, num in enumerate(arr):
        arr[i] = round(num, 3)


def add_arr(arr: List[float], summand) -> None:
    for i, num in enumerate(arr):
        arr[i] = num + summand


def test_iris_QSM():
    dataset = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\220428_195743_IrisDataSet")
    quantiles = {
        "sepal_length": 0.1,
        "petal_length": 0.05,
        "petal_width": -0.2
    }
    tree = cl.load_tree(os.path.join(r"D:\Gernot\Programmieren\Bachelor\Data\220428_195743_IrisDataSet", "tree_classifier.pkl"))
    QSM_on_binned_data(dataset=dataset, start_folders={"petal_length": r"D:\Gernot\Programmieren\Bachelor\Data\220428_195743_IrisDataSet\Splits\petal_length_005",
                                                       "petal_width": r"D:\Gernot\Programmieren\Bachelor\Data\220428_195743_IrisDataSet\Splits\petal_width_-02",
                                                       "sepal_length": r"D:\Gernot\Programmieren\Bachelor\Data\220428_195743_IrisDataSet\Splits\sepal_length_01"},
                       quantiles=quantiles, trained_tree=tree)
    """dataset = dc.IrisDataSet()
    quantiles = {
        "sepal_length": 0.1,
        "petal_length": 0.05,
        "petal_width": -0.2
    }
    trained = cl.create_and_save_tree(dataset, pred_col_name="test")
    run_vanilla_qsm(dataset=dataset, quantiles=quantiles, trained_tree=trained)"""
    """dataset = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\220428_142002_IrisDataSet")
    values, cumulative_frequencies = vs.get_cumulative_values(dataset.data["sepal_length"].values)
    val_shift, cum_shift = vs.get_cumulative_values(dataset.data["sepal_length_shifted_by_0.1"].values)
    add_arr(values, 0.001)
    add_arr(val_shift, 0.001)
    add_arr(cumulative_frequencies, 0.001)
    add_arr(cum_shift, 0.001)

    round_arr(values)
    round_arr(val_shift)
    round_arr(cumulative_frequencies)
    round_arr(cum_shift)
    print(values)
    print(cumulative_frequencies)
    print(val_shift)
    print(cum_shift)"""


if __name__ == "__main__":
    #main()
    test()
    #test_iris_QSM()
    #example_vis()
