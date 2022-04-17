import os
from typing import Dict, List
import random

import pandas as pd

import dataCreation as dc
import visualization as vs
import classifier as cl
import QSM
import createDataSplits as cds


def add_result_matrices(matrix1: pd.DataFrame, matrix2: pd.DataFrame) -> pd.DataFrame:
    return matrix1.add(other=matrix2, fill_value=0)


def recursion_end(curr_folder: str, dim: str, q: float,
                  subspace_list: List[pd.DataFrame], trained_tree) -> pd.DataFrame:
    dataset = dc.MaybeActualDataSet.load(curr_folder)
    result = QSM.run_QSM_decisionTree(dataset=dataset,
                                      quantiles={dim: q},
                                      save_changes=True,
                                      trained_tree=trained_tree)
    new_dim_name = f"{dim}_shifted_by_{str(q)}"
    new_class_name = f"pred_with_{new_dim_name}"
    data = dataset.data.copy(deep=True)
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
    if dim != "dim_04":
        QSM.visualize_QSM(base_dim="dim_04", dim_before_shift=dim, shift=q, dataset=dataset)
    else:
        QSM.visualize_QSM(base_dim="dim_00", dim_before_shift=dim, shift=q, dataset=dataset)
    matrix = result[dim]
    return matrix


def recursive_QSM(curr_folder: str, curr_name: str, dim: str, q: float,
                  subspace_list: List[pd.DataFrame], trained_tree) -> pd.DataFrame:
    suffixes = ["0", "1"]
    count = 0
    result_matrix = pd.DataFrame()
    folders = [folder for folder in os.listdir(curr_folder) if os.path.isdir(os.path.join(curr_folder, folder))]
    for suffix in suffixes:
        if curr_name:
            next_name = f"{curr_name}_{suffix}"
        else:
            next_name = suffix
        next_folder = os.path.join(curr_folder, next_name)
        if next_name in folders:
            count += 1
            curr_res = recursive_QSM(curr_folder=next_folder, curr_name=next_name, dim=dim, q=q,
                                     subspace_list=subspace_list, trained_tree=trained_tree)
            result_matrix = add_result_matrices(result_matrix, curr_res)

    if count == 2:
        pass
    elif count == 0:
        result_matrix = recursion_end(curr_folder=curr_folder, dim=dim, q=q,
                             subspace_list=subspace_list, trained_tree=trained_tree)
    else:
        raise dc.CustomError(f"{count} number of splits detected! Should be either 0 or 2!")

    result_matrix.to_csv(os.path.join(curr_folder, "local_change_matrix.csv"))
    return result_matrix


def QSM_on_binned_data(dataset: dc.MaybeActualDataSet, quantiles: Dict[str, float],
                       start_folders: Dict[str, str], trained_tree=None):
    for dim, q in quantiles.items():
        start_folder = start_folders[dim]
        subspace_list = []
        result_matrix = recursive_QSM(curr_folder=start_folder, curr_name="", dim=dim, q=q,
                                      subspace_list=subspace_list, trained_tree=trained_tree)
        full_data = pd.concat(subspace_list)
        full_dataset = dc.MaybeActualDataSet.clone_meta_data(dataset)
        full_dataset.take_new_data(full_data)
        full_dataset.extend_notes_by_one_line("this dataset contains the full data, that was shifted in binned subsets.")
        full_dataset.extend_notes_by_one_line(f"Dimension {dim} was shifted by {q}.")
        full_dataset.end_paragraph_in_notes()
        full_dataset.save(start_folder)

        result_matrix.to_csv(os.path.join(start_folder, "binning_result_matrix.csv"))
        visualize_QSM_on_binned_data(full_dataset, dim)


def run_vanilla_qsm(dataset, quantiles, trained_tree=None):
    QSM.run_QSM_decisionTree(dataset=dataset,
                             quantiles=quantiles,
                             save_changes=True,
                             trained_tree=trained_tree)
    for dim, q in quantiles.items():
        if dim != "dim_04":
            QSM.visualize_QSM(base_dim="dim_04", dim_before_shift=dim, shift=q, dataset=dataset,
                              save_path=os.path.join(dataset.path, "pics", "QSM", "vanilla", dim))
        else:
            QSM.visualize_QSM(base_dim="dim_00", dim_before_shift=dim, shift=q, dataset=dataset,
                              save_path=os.path.join(dataset.path, "pics", "QSM", "vanilla", dim))

        print(f"change matrix {dim} shifted by {q}")
        results_folder = os.path.join(dataset.path, "vanilla_results")
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
        matrix = vs.get_change_matrix(dataset.data, ("org_pred_classes_QSM", f"pred_with_{dim}_shifted_by_{q}"))
        matrix.to_csv(os.path.join(results_folder, f"{dim}_{q}.csv"))


def visualize_QSM_on_binned_data(dataset: dc.Data, shifted_dim, common_dim: str = ""):
    #pick a random dimension to display the data in 2d visualizations
    if not common_dim:
        data_dims = dataset.data_columns
        data_dims.remove(shifted_dim)
        common_dim = random.choice(data_dims)
    folder = os.path.join(dataset.path, "pics", "QSM_Binning")
    vs.compare_shift_2d(df=dataset.data, common_dim=common_dim, dims_to_compare=(f"{shifted_dim}_org", shifted_dim),
                        class_columns=("source", "source"),
                        path=os.path.join(folder, "source.png"))
    vs.compare_shift_2d(df=dataset.data, common_dim=common_dim, dims_to_compare=(f"{shifted_dim}_org", shifted_dim),
                        class_columns=("pred_classes", "org_pred"),
                        path=os.path.join(folder, "predictions.png"))
    vs.compare_shift_2d(df=dataset.data, common_dim=common_dim, dims_to_compare=(f"{shifted_dim}_org", shifted_dim),
                        class_columns=("classes", "classes"),
                        path=os.path.join(folder, "classes.png"))


def main():
    quantiles = {
        "dim_04": 0.1,
        "dim_00": 0.05,
        "dim_01": -0.2
    }
    members = [50 for _ in range(6)]
    dataset = dc.MaybeActualDataSet(members)
    trained_tree = cl.create_and_save_tree(dataset, pred_col_name="test")
    start_folder_dict = cds.data_binning(dataset=dataset, shifts=quantiles, max_split_nr=2, visualize=True)
    run_vanilla_qsm(dataset, quantiles, trained_tree)
    QSM_on_binned_data(dataset=dataset, quantiles=quantiles, start_folders=start_folder_dict, trained_tree=trained_tree)


def test():
    dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220404_210124_MaybeActualDataSet\Splits\dim_04_01")
    visualize_QSM_on_binned_data(dataset, "dim_04")


if __name__ == "__main__":
    main()
    #test()
    #example_vis()
