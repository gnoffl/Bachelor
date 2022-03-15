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


def _create_data_splits(dataset: dc.Data, dim_to_shift: str, max_splits: int, min_number_of_points: int):
    folder_path = create_folder_for_splits(dataset=dataset)
    run_R_script(additional_arguments=[folder_path, dim_to_shift, max_splits, min_number_of_points])


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
    """
    tests every possible split of the data in the dim_to_split and determines the one which results in the strongest
    difference of the resulting sets in the kolomogorov smirnov test (ks-test) for the dim_to_shift. Only performs the
    split, if the result of the ks-test is meaningful. Definition for meaningfulness is in "find_optimal_split_index".
    :param dataset: Dataset to be split
    :param dim_to_shift: target dimension for the kolmogorov smirnov test
    :param dim_to_split: dimension in which the dataset will be split
    :param min_split_size: min number of points in a resulting split
    :return: the resulting datasets, if a split was performed, or None, if no split was performed
    """
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


def convert_indexes_to_column_names(dataset: dc.Data, indexes: List[str]) -> List[str]:
    return [dataset.data_columns[int(index)] for index in indexes]


def read_HiCS_results(dataset: dc.Data, dim_to_shift: str = ""):
    spaces = []
    if dim_to_shift:
        index_str = str(dataset.data_columns.index(dim_to_shift))
    with open(os.path.join(dataset.path, "HiCS_output.csv"), "r") as f:
        line = f.readline().strip()
        while line:
            val, dims = line.split("; ")
            dims = dims.split(";")
            # HiCS results only use numbers to refer to columns of the data set, instead of using the names of the columns
            # the dimension therefore has to be converted to the number of the column
            if not dim_to_shift or index_str in dims:
                spaces.append((float(val), dims))
            line = f.readline().strip()
    return spaces


def get_HiCS(dataset: dc.Data, dim_to_shift: str, goodness_over_length: bool) -> List[str]:
    spaces = read_HiCS_results(dataset, dim_to_shift)

    max_val_elem = max(spaces, key=lambda elem: elem[0])

    if goodness_over_length:
        return convert_indexes_to_column_names(dataset=dataset, indexes=max_val_elem[1])

    max_val = max_val_elem[0]
    max_length = len(max(spaces, key=lambda elem: len(elem[1]))[1])

    for length in range(max_length, len(max_val_elem[1]), -1):
        curr_spaces = [elem for elem in spaces if len(elem[1]) == length]
        curr_max = max(curr_spaces, key=lambda elem: elem[0])
        curr_max_val = curr_max[0]
        if curr_max_val > (.7 * max_val):
            return convert_indexes_to_column_names(dataset=dataset, indexes=curr_max[1])

    return convert_indexes_to_column_names(dataset=dataset, indexes=max_val_elem[1])


def find_dim_to_split(dataset: dc.Data, dim_to_shift: str) -> str:
    if not dataset.HiCS_dims:
        dataset.HiCS_dims = get_HiCS(dataset=dataset, dim_to_shift=dim_to_shift, goodness_over_length=True)
    spaces = read_HiCS_results(dataset=dataset, dim_to_shift=dim_to_shift)
    spaces = [elem for elem in spaces if len(elem[1]) == 2]
    dim_to_shift_index_str = str(dataset.data_columns.index(dim_to_shift))
    dim_0, dim_1 = max(spaces, key=lambda elem: elem[0])[1]
    if dim_0 == dim_to_shift_index_str:
        return dataset.data_columns[int(dim_1)]
    elif dim_1 == dim_to_shift_index_str:
        return dataset.data_columns[int(dim_0)]
    else:
        raise dc.CustomError("dim_to_shift was not in pair!")



def create_binning_splits(dataset: dc.Data, dim_to_shift: str, min_split_size: int, remaining_splits: int):
    if "HiCS_output.csv" not in os.listdir(dataset.path):
        dataset.run_hics()

    dim_to_split = find_dim_to_split(dataset=dataset, dim_to_shift=dim_to_shift)

    split1, split2 = create_optimal_split(dataset=dataset, dim_to_shift=dim_to_shift, dim_to_split=dim_to_split, min_split_size=min_split_size)

    vs.compare_splits_2d(split1.data, split2.data, (dim_to_split, dim_to_shift))
    vs.compare_splits_cumulative(split1.data, split2.data, dim_to_shift)


def main():
    _data = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220314_114453_MaybeActualDataSet")
    #dim_to_split = find_dim_to_split(_data, "dim_04")
    #print(dim_to_split)
    create_binning_splits(dataset=_data, dim_to_shift="dim_04", min_split_size=30, remaining_splits=3)
    #dims = get_HiCS(dataset=_data, dim_to_shift="dim_04", goodness_over_length=False)
    #_data.HiCS_dims = dims
    #_data.save()


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
    main()
    #main(data.path, dim_to_shift="dim_04", q=0.05)
    #test()