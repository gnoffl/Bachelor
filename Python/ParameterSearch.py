import os
from typing import List, Tuple, Dict

import pandas as pd

import main
import dataCreation as dc


def create_parameters(tree: bool = True, max_depth: int = 5, min_samples_leaf: int = 5, lr: float = 0.001,
                      num_epochs: int = 100, batch_size: int = 64, shuffle: bool = True, nr_processes: int = 4,
                      max_split_nr: int = 2, p_value: float = .05, goodness_over_length: bool = True,
                      threshold_fraction: float = 0.7, HiCS_parameters: str = "", additionals: List[Tuple] = None):
    """
        tiebreaker fuer splits
    """
    content = "Classifier:\n"
    content += " " * 4 + f"tree={tree}\n\n"

    content += "DecisionTree:\n"
    content += " " * 4 + f"max_depth={max_depth}\n"
    content += " " * 4 + f"min_samples_leaf={min_samples_leaf}\n\n"

    content += f"NeuralNet:\n"
    content += " " * 4 + f"lr={lr}\n"
    content += " " * 4 + f"num_epochs={num_epochs}\n"
    content += " " * 4 + f"batch_size={batch_size}\n"
    content += " " * 4 + f"shuffle={shuffle}\n\n"

    content += f"CreateDataSplits:\n"
    content += " " * 4 + f"nr_processes={nr_processes}\n"
    content += " " * 4 + f"max_split_nr={max_split_nr}\n"
    content += " " * 4 + f"p_value={p_value}\n"
    content += " " * 4 + f"goodness_over_length={goodness_over_length}\n"
    content += " " * 4 + f"threshold_fraction={threshold_fraction}\n\n"

    content += f"dataCreation:\n"
    content += " " * 4 + f"HiCS_parameters={HiCS_parameters}\n\n"

    if additionals:
        content += f"Misc:\n"
        for name, value in additionals:
            content += " " * 4 + f"{name}={value}\n"

    with open("../Data/Parameters.txt", "w") as file:
        file.write(content)


def get_dataset(dataset_type: str, final_path: str, members: List[int] = None) -> Tuple[dc.Data, Dict[str, float]]:
    if dataset_type == "MaybeActualDataSet":
        if not members:
            raise dc.CustomError(f"members need to be given, when MaybeActualDataSet is supposed to be created!")
        dataset = dc.MaybeActualDataSet(members=members, path=final_path)
        quantiles = {"dim_04": 0.05}
    elif dataset_type == "IrisDataSet":
        dataset = dc.IrisDataSet(path=final_path)
        quantiles = {"petal_length": 0.05}
    elif dataset_type == "SoccerDataSet":
        dataset = dc.SoccerDataSet(path=final_path)
        quantiles = {"ps_Laufweite": 0.05}
    else:
        raise dc.CustomError(f"unknown dataset type ({dataset_type})!")
    return dataset, quantiles


def get_matrix(dataset_path: str) -> pd.DataFrame:
    folder = os.path.join(dataset_path, "Splits")
    folder = os.path.join(folder, os.listdir(folder)[0])
    matrix_path = os.path.join(folder, "binning_result_matrix.csv")
    matrix = pd.read_csv(matrix_path, index_col=0)
    return matrix


def calculate_diff_matrices():
    dataset_types = ["MaybeActualDataSet", "IrisDataSet", "SoccerDataSet"]
    classifiers = ["NN", "tree"]
    for dataset_type in dataset_types:
        path = os.path.join("..", "Data", "Parameters", dataset_type)
        for classifier in classifiers:
            new_path = os.path.join(path, classifier)
            default_path = os.path.join(new_path, "001")
            if os.path.isdir(default_path):
                default_matrix = get_matrix(default_path)
                datasets = os.listdir(new_path)
                for dataset_name in datasets:
                    curr_dataset_path = os.path.join(new_path, dataset_name)
                    curr_matrix = get_matrix(curr_dataset_path)
                    diff_matrix = curr_matrix - default_matrix
                    diff_matrix.to_csv(os.path.join(curr_dataset_path, "diff_matrix.csv"))



def create_paths(dataset_types):
    classifiers = ["NN", "tree"]
    for dataset_type in dataset_types:
        path = os.path.join("..", "Data", "Parameters", dataset_type)
        if not os.path.isdir(path):
            os.mkdir(path)
        for classifier in classifiers:
            new_path = os.path.join(path, classifier)
            if not os.path.isdir(new_path):
                os.mkdir(new_path)


def loop_core(dataset_type: str, parameter_set_nr: int, members: List[int], parameter_args: Dict, tree_path: str):
    create_parameters(**parameter_args)
    final_path = os.path.join(tree_path, str(parameter_set_nr).zfill(3))
    # don't recalculate a set of parameters, that was already calculated
    if os.path.isdir(final_path):
        print(f"{final_path} already exists!\n----------------------\n")
    else:
        dataset, quantiles = get_dataset(dataset_type, final_path, members)
        main.run_from_file(dataset=dataset, quantiles=quantiles)
        print(f"finished {str(parameter_set_nr).zfill(3)}!\n----------------------\n")


def parameter_search():
    members = [200 for _ in range(6)]
    #lists of options to iterate over
    dataset_types = ["MaybeActualDataSet", "IrisDataSet", "SoccerDataSet"]
    tree_args = [True, False]
    max_split_args = [3, 2, 4]
    p_val_args = [0.05, 0.01]
    goodness_args = [True, False]

    #nur wenn goodness_over_length = False
    threshold_args = [0.7, 0.55, 0.85]

    create_paths(dataset_types=dataset_types)

    #loop over all the options
    for dataset_type in dataset_types:
        path = os.path.join("..", "Data", "Parameters", dataset_type)
        parameter_args = {}
        for tree_arg in tree_args:
            #NNClassifier only implemented for SoccerDataSet, so unnecessary to test it for other datasets
            if dataset_type != "SoccerDataSet" and not tree_arg:
                continue
            tree_path = os.path.join(path, "tree") if tree_arg else os.path.join(path, "NN")
            #counter to distinguish datasets later
            i = 0
            parameter_args["tree"] = tree_arg
            for max_split_arg in max_split_args:
                parameter_args["max_split_nr"] = max_split_arg
                for p_val_arg in p_val_args:
                    parameter_args["p_value"] = p_val_arg
                    for goodness_arg in goodness_args:
                        parameter_args["goodness_over_length"] = goodness_arg
                        for threshold_arg in threshold_args:
                            parameter_args["threshold_fraction"] = threshold_arg
                            i += 1
                            loop_core(dataset_type=dataset_type, parameter_set_nr=i, members=members,
                                      parameter_args=parameter_args, tree_path=tree_path)
                            #threshold is only relevant, when goodness_over_length = False --> dont calculate
                            # combinations, where something is changed, that doesnt influece the results
                            if goodness_arg:
                                break


if __name__ == "__main__":
    #try:
    parameter_search()
    calculate_diff_matrices()
    #except Exception:
     #   pass
    #os.system("shutdown /s /t 1")
    #get_matrix(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters\MaybeActualDataSet\tree\001")
    #create_parameters()
