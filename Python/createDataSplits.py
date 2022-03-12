from typing import List, Tuple
import subprocess

import pandas as pd

import dataCreation as dc
import os
import scipy.stats as stats
import visualization as vs


def run_R_script(additional_arguments: List,
                 path_to_script: str = ""):
    if not path_to_script:
        path_to_script = os.path.join(os.path.dirname(__file__), "..", "R", "Binning", "run_binning.R")
    command = "C:/Program Files/R/R-4.1.2/bin/Rscript.exe"
    additional_arguments = [str(arg) for arg in additional_arguments]
    x = subprocess.check_output([command, path_to_script] + additional_arguments)
    print(x)


def create_folder_for_splits(dataset: dc.Data) -> str:
    splits_folder = os.path.join(dataset.path, "Data_splits")
    if not os.path.isdir(splits_folder):
        os.mkdir(splits_folder)
    else:
        dc.CustomError("Splits already exist!")
    column_list = dataset.data_columns
    column_list.insert(0, "classes")
    vanilla_dataset = dataset.data[column_list]
    vanilla_dataset.to_csv(os.path.join(splits_folder, "dataset.csv"), index=False)
    return splits_folder


def save_new_datasets(data1: pd.DataFrame, data2: pd.DataFrame, dataset: dc.Data) -> Tuple[dc.Data, dc.Data]:
    split1 = dataset.clone_meta_data(path=os.path.join(dataset.path, "split1"))
    split2 = dataset.clone_meta_data(path=os.path.join(dataset.path, "split2"))
    split1.take_new_data(data1)
    split2.take_new_data(data2)
    split1.save()
    split2.save()
    return split1, split2


def create_optimal_split(dataset: dc.Data, dim_to_shift: str, dim_to_split: str, min_split_size: int)\
        -> Tuple[dc.Data, dc.Data] or None:
    length = len(dataset.data)
    if length < 2 * min_split_size:
        return None
    if min_split_size < 1:
        raise dc.CustomError("min split size needs to be larger than 1!")
    data = dataset.data.copy(deep=True)
    data = data.sort_values(by=[dim_to_split])
    ks_stat = [(0., 1.) for _ in range(min_split_size)]
    values = data[dim_to_shift].values
    for i in range(min_split_size, len(data) - min_split_size + 1):
        result = stats.kstest(values[:i], values[i:])
        ks_stat.append(result)
    ks_stat.extend([(0, 1) for _ in range(min_split_size - 1)])
    split_index = find_optimal_split_index(ks_stat=ks_stat)

    data1 = data.iloc[:split_index, :]
    data2 = data.iloc[split_index:, :]
    return save_new_datasets(data1, data2, dataset)



def find_optimal_split_index(ks_stat: List[stats.stats.KstestResult]) -> int:
    min_p_val = min(ks_stat, key=lambda elem: elem[1])
    cand_list = [res for res in ks_stat if res[1] == min_p_val[1]]
    # this is just a temporary fix: returns index of result with min p and max D

    return ks_stat.index(max(cand_list, key=lambda elem: elem[0]))


def create_binning_splits(dataset: dc.Data, dim_to_shift: str, dim_to_split: str, min_split_size: int):
    split1, split2 = create_optimal_split(dataset=dataset, dim_to_shift=dim_to_shift, dim_to_split=dim_to_split, min_split_size=min_split_size)

    vs.visualize_2d(split1.data, ("dim_01", "dim_04"))
    vs.visualize_2d(split2.data, ("dim_01", "dim_04"))


def create_data_splits(dataset: dc.Data, dim_to_shift: str, max_splits: int, min_number_of_points: int):
    folder_path = create_folder_for_splits(dataset=dataset)
    run_R_script(additional_arguments=[folder_path, dim_to_shift, max_splits, min_number_of_points])


def main(path: str, dim_to_shift: str, q: float):
    dataset = dc.MaybeActualDataSet.load(path)
    create_data_splits(dataset=dataset, dim_to_shift=dim_to_shift, max_splits=3, min_number_of_points=round(len(dataset.data) * q))


def test():
    dataset = dc.MaybeActualDataSet([1000 for _ in range(6)])
    columns = dataset.data.columns.values
    test_dim = columns[2]
    print(columns)
    print(f"test_dim: {test_dim}")
    print(f"cand: {columns[1]}")
    columns[2] = columns[1]
    columns[1] = test_dim
    dataset.data.columns = columns
    dataset.save_data_for_hics()
    dataset.save()


if __name__ == "__main__":
    _data = dc.MaybeActualDataSet([100 for _ in range(6)])
    create_binning_splits(_data, "dim_04", "dim_01", 30)
    #main(data.path, dim_to_shift="dim_04", q=0.05)
    #test()