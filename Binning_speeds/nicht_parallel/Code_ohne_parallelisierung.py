import re
import time
from typing import List, Tuple
import subprocess
import classifier as cl
import pandas as pd

import dataCreation as dc
import os
import scipy.stats as stats
import visualization as vs


def run_R_script(additional_arguments: List,
                 path_to_script: str = "") -> None:
    """
    runs an R script. is not used atm.
    :param additional_arguments: additional arguments to be handed to the Script
    :param path_to_script: path to the script to be executed
    """
    if not path_to_script:
        path_to_script = os.path.join(os.path.dirname(__file__), "..", "R", "Binning", "run_binning.R")
    command = "C:/Program Files/R/R-4.1.2/bin/Rscript.exe"
    additional_arguments = [str(arg) for arg in additional_arguments]
    x = subprocess.check_output([command, path_to_script] + additional_arguments)
    print(x)


def create_folder_for_splits(dataset: dc.Data) -> str:
    """
    NOT USED
    """
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
    """
    NOT USED
    """
    folder_path = create_folder_for_splits(dataset=dataset)
    run_R_script(additional_arguments=[folder_path, dim_to_shift, max_splits, min_number_of_points])


def create_new_datasets(data1: pd.DataFrame, data2: pd.DataFrame, dataset: dc.Data) -> Tuple[dc.Data, dc.Data]:
    """
    creates new Datasets from given dataframes, by copying metadata from a given dataset and saves them to create the
    necessary folder structure
    :param data1: dataframe for dataset 1
    :param data2: dataframe for dataset 2
    :param dataset: "parent" dataset, from which metadata is copied, and where the resulting datasets paths will lead
    :return: Tuple of the created datasets
    """
    #naming of the datasets just takes the name of the parent dataset and appends "_0" or "_1"
    split1 = dataset.clone_meta_data(path=os.path.join(dataset.path, get_new_dataset_name(dataset, "0")))
    split2 = dataset.clone_meta_data(path=os.path.join(dataset.path, get_new_dataset_name(dataset, "1")))
    split1.take_new_data(data1)
    split2.take_new_data(data2)

    #saving the data, to create the folder structure and files
    split1.save()
    split2.save()
    return split1, split2


def get_new_dataset_name(dataset: dc.Data, suffix: str):
    parent_name = dataset.path.split("\\")[-1]
    if re.match(r"[01](_[01])*", parent_name):
        return f"{parent_name}_{suffix}"
    return suffix


def split_dataset(data: pd.DataFrame, dataset: dc.Data, dim_to_split: str, split_index: int):
    # split the dataframe at the resulting split point, create datasets from the dataframes and return them
    data1 = data.iloc[:split_index, :]
    data2 = data.iloc[split_index:, :]
    dataset1, dataset2 = create_new_datasets(data1, data2, dataset)
    dataset1.extend_notes_by_one_line(f"This dataset results from splitting a parent dataset.")
    dataset1.extend_notes_by_one_line(f"split criterion: {dim_to_split} < {data[dim_to_split].iloc[split_index]}")
    dataset1.extend_notes_by_one_line(f"number of data points: {len(dataset1.data)}")
    dataset1.end_paragraph_in_notes()
    dataset2.extend_notes_by_one_line(f"This dataset results from splitting a parent dataset.")
    dataset2.extend_notes_by_one_line(f"split criterion: {dim_to_split} >= {data[dim_to_split].iloc[split_index]}")
    dataset2.extend_notes_by_one_line(f"number of data points: {len(dataset2.data)}")
    dataset2.end_paragraph_in_notes()
    return dataset1, dataset2


def create_test_statistics_parallel(dataset: dc.Data, dim_to_shift: str, min_split_size: int, dim_to_split: str,
                                    ordered_data: pd.DataFrame = None):
    if ordered_data is not None:
        data = ordered_data
    else:
        data = dataset.data.copy(deep=True)
        data = data.sort_values(by=[dim_to_split])
    #ks_stat is supposed to be a list of the same length as the data of the dataset. At each index, is the result of the
    #ks test when comparing the datasets, that result from splitting the data at the index. Since no splits are to be
    #calculated where one dataset is smaller than min_split, dummy values are used in this range. Results have the shape
    #of (D-value, p-value), so (0, 1) means both datasets are the same.
    ks_stat = [(0., 1.) for _ in range(min_split_size)]
    values = data[dim_to_shift].values
    #range here is important. need to avoid off-by-one errors (dataset, whose length is exactly  2 * min_split can still
    #be split in the middle)
    for i in range(min_split_size, len(data) - min_split_size + 1):
        result = stats.kstest(values[:i], values[i:])
        ks_stat.append(result)
    #maybe unnecessary to also extend the list at the end. Is done to make the list actually have the same length as the
    #dataset
    ks_stat.extend([(0, 1) for _ in range(min_split_size - 1)])
    print(f"data length = {len(data)}")
    print(f"ks_stat length = {len(ks_stat)}")
    print(f"test creation length = {len([1 for _ in range(len(data))])}")
    return ks_stat


def create_test_statistics(dataset: dc.Data, dim_to_shift: str, min_split_size: int, dim_to_split: str,
                           ordered_data: pd.DataFrame = None):
    if ordered_data is not None:
        data = ordered_data
    else:
        data = dataset.data.copy(deep=True)
        data = data.sort_values(by=[dim_to_split])
    #ks_stat is supposed to be a list of the same length as the data of the dataset. At each index, is the result of the
    #ks test when comparing the datasets, that result from splitting the data at the index. Since no splits are to be
    #calculated where one dataset is smaller than min_split, dummy values are used in this range. Results have the shape
    #of (D-value, p-value), so (0, 1) means both datasets are the same.
    ks_stat = [(0., 1.) for _ in range(min_split_size)]
    values = data[dim_to_shift].values
    #range here is important. need to avoid off-by-one errors (dataset, whose length is exactly  2 * min_split can still
    #be split in the middle)
    for i in range(min_split_size, len(data) - min_split_size + 1):
        result = stats.kstest(values[:i], values[i:])
        ks_stat.append(result)
    #maybe unnecessary to also extend the list at the end. Is done to make the list actually have the same length as the
    #dataset
    ks_stat.extend([(0, 1) for _ in range(min_split_size - 1)])
    return ks_stat


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
    #cannot split data, if size is not at least 2 * min_split
    length = len(dataset.data)
    if length < 2 * min_split_size:
        #print(f"Dataset was not split again, because the number of points is less than twice "
        #      f"the min_split_size (min_split_size = {min_split_size})!")
        dataset.buffer_note(f"Dataset was not split again, because the number of points is less than twice "
                            f"the min_split_size (min_split_size = {min_split_size})!")
        return
    if min_split_size < 1:
        raise dc.CustomError("min split size needs to be larger than 0!")

    data = dataset.data.copy(deep=True)
    data = data.sort_values(by=[dim_to_split])


    start = time.perf_counter()


    ks_stat = create_test_statistics(dataset=dataset, dim_to_shift=dim_to_shift,
                                     min_split_size=min_split_size, dim_to_split=dim_to_split, ordered_data=data)


    print(f"time for create_test_statistics: {time.perf_counter() - start}")


    split_index = find_optimal_split_index(ks_stat=ks_stat)

    if split_index < 0:
        #print(f"Dataset was not split again, because no split lead to "
        #      f"significantly different distributions in the dim_to_shift!")
        dataset.buffer_note(f"Dataset was not split again, because no split lead to "
                            f"significantly different distributions in the dim_to_shift!")
        return

    dataset1, dataset2 = split_dataset(data=data, dataset=dataset, dim_to_split=dim_to_split, split_index=split_index)
    return dataset1, dataset2


def find_optimal_split_index(ks_stat: List[stats.stats.KstestResult]) -> int:
    """
    selects the index from a List with results from kolmogorov Smirnov tests. Currently only selects the value with
    the lowest p-value. If multiple values share the lowest p-value, the value with the highest D-value will be selected.
    functionality should be expanded to decide whether a split should be performed at all, and tie-breakers for equal
    p-values should be adjusted
    :param ks_stat: List of results from Kolmogorov Smirnov tests
    :return: index of best result
    """
    min_p_val = min(ks_stat, key=lambda elem: elem[1])
    if min_p_val[1] > .05:
        #print(ks_stat)
        return -1
    cand_list = [res for res in ks_stat if res[1] == min_p_val[1]]

    # this is just a temporary fix: returns index of result with min p and max D
    return ks_stat.index(max(cand_list, key=lambda elem: elem[0]))


def convert_indexes_to_column_names(dataset: dc.Data, indexes: List[str]) -> List[str]:
    """
    converts indexes of the list of columns to the names of the columns
    :param dataset: Dataset, for which the indexes are to be converted
    :param indexes: list of the indexes to be converted
    :return: List with column names instead of indexes
    """
    return [dataset.data_columns[int(index)] for index in indexes]


def convert_column_names_to_indexes(dataset: dc.Data, col_names: List[str]) -> List[str]:
    return [str(dataset.data_columns.index(col)) for col in col_names]


def read_HiCS_results(dataset: dc.Data, dim_to_shift: str = "") -> List[Tuple[float, List[str]]]:
    """
    parses an output file from HiCS. Returns List of spaces with their respective contrast value.
    :param dataset: Dataset for which the HiCS output was generated
    :param dim_to_shift: name of a dimension, that needs to be present in each space that is returned. If omitted, all
    spaces are returned.
    :return: List of all spaces containing the dim_to_shift (if given) together with their respective contrast value.
    List consists of Tuples. First element of the tuples is the contrast value, second element of the Tuples is a list
    of strings, that contains strings of the indices of the dimensions for the space.
    """
    spaces = []
    if "HiCS_output.csv" not in os.listdir(dataset.path):
        dataset.run_hics(silent=True)
    if dim_to_shift:
        # HiCS results only use numbers to refer to columns of the data set, instead of using the names of the columns
        # the dimension therefore has to be converted to the number of the column
        index_str = str(dataset.data_columns.index(dim_to_shift))
    #lines of the HiCS output consist of a number of leading spaces and then the contrast value. After the contrast,
    # a list of dimensions is given, that define the subspace. elements of the List are seperated by ";". the list is
    # seperated from the contrast value by "; ". each line stands for one subspace
    with open(os.path.join(dataset.path, "HiCS_output.csv"), "r") as f:
        line = f.readline().strip()
        while line:
            val, dims = line.split("; ")
            dims = dims.split(";")
            # if no dim_to_shift is given, all subspaces will be returned
            if not dim_to_shift or index_str in dims:
                spaces.append((float(val), dims))
            line = f.readline().strip()
    return spaces


def get_HiCS(dataset: dc.Data,
             dim_to_shift: str,
             goodness_over_length: bool,
             spaces: List[Tuple[float, List[str]]] = None,
             threshold_fraction: float = 0.7) -> List[str]:
    """
    finds and returns the best HiCS, that contains the dim_to_shift, for a given dataset.
    :param dataset: the dataset the HiCS is supposed to be found for.
    :param dim_to_shift: Dimension that needs to be present in the returned Subspace.
    :param goodness_over_length: if True, the Subspace with the highest contrast, that also contains dim_to_shift, will
    be selected. If False, the Subspace with the most dimensions will be selected, if its contrast value is not lower
    than threshold_fraction times the value of the subspace with the overall highest contrast.
    :param spaces: List with HiCS results. if not given will be read from disc
    :param threshold_fraction: determines the cutoff contrast value for "long" subspaces
    :return: List of strings with the names of the columns that make up the selected subspace.
    """
    if not spaces:
        spaces = read_HiCS_results(dataset, dim_to_shift)

    #element with the highest contrast value
    max_val_elem = max(spaces, key=lambda elem: elem[0])

    if goodness_over_length:
        dataset.HiCS_dims = convert_indexes_to_column_names(dataset=dataset, indexes=max_val_elem[1])
        return dataset.HiCS_dims

    max_val = max_val_elem[0]
    min_val = min(spaces, key=lambda elem: elem[0])[0]

    threshold = calculate_threshold(max_val=max_val, min_val=min_val, threshold_fraction=threshold_fraction)
    max_length = len(max(spaces, key=lambda elem: len(elem[1]))[1])

    #loops over length of subspaces starting from the max length. For each length the subspace with the best contrast
    # is picked. if the contrast is greater than 70% of the over all highest contrast, the subspace is selected as best
    # HiCS
    for length in range(max_length, len(max_val_elem[1]), -1):
        curr_spaces = [elem for elem in spaces if len(elem[1]) == length]
        curr_max = max(curr_spaces, key=lambda elem: elem[0])
        curr_max_val = curr_max[0]
        if curr_max_val > threshold:
            dataset.HiCS_dims = convert_indexes_to_column_names(dataset=dataset, indexes=curr_max[1])
            return dataset.HiCS_dims

    dataset.HiCS_dims = convert_indexes_to_column_names(dataset=dataset, indexes=max_val_elem[1])
    return dataset.HiCS_dims


def calculate_threshold(max_val: float, min_val: float, threshold_fraction: float):
    diff = max_val - min_val
    diff_fraction = threshold_fraction * diff
    threshold = diff_fraction + min_val
    print(f"threshold = {threshold}")
    return threshold


def find_dim_to_split(dataset: dc.Data, dim_to_shift: str) -> str:
    """
    selects the dimension that is most suited for splitting from the dataset.
    :param dataset: the dataset to select the dimension from
    :param dim_to_shift: name of the dimension to be shifted
    :return: name of the dimension to be split
    """
    spaces = read_HiCS_results(dataset=dataset, dim_to_shift=dim_to_shift)
    #transforming hics_dims from names of columns to their indices, excluding the dim_to_shift
    hics_dims = get_HiCS(dataset=dataset, dim_to_shift=dim_to_shift, goodness_over_length=True, spaces=spaces)
    hics_dims = [dim for dim in hics_dims if dim != dim_to_shift]
    hics_dims = convert_column_names_to_indexes(dataset, hics_dims)
    #selecting the dim, that is part of the datasets best HiCS and has the highest contrast value in a pair with the
    # dim_to_shift
    spaces = [elem for elem in spaces if len(elem[1]) == 2]
    spaces = [elem for elem in spaces
              if elem[1][0] in hics_dims or elem[1][1] in hics_dims]
    dim_to_shift_index_str = str(dataset.data_columns.index(dim_to_shift))
    dim_0, dim_1 = max(spaces, key=lambda elem: elem[0])[1]
    if dim_0 == dim_to_shift_index_str:
        return dataset.data_columns[int(dim_1)]
    elif dim_1 == dim_to_shift_index_str:
        return dataset.data_columns[int(dim_0)]
    else:
        raise dc.CustomError("dim_to_shift was not in pair!")


def recursive_splitting(dataset: dc.Data,
                        dim_to_shift: str,
                        min_split_size: int,
                        remaining_splits: int,
                        visualize: bool = True) -> None:
    """
    recursively splits a dataset in subsets.
    :param dataset: dataset to be split
    :param dim_to_shift: dimension that is to be shifted. the dataset will be split in a way, that the distribution of
    this dimension has the highest possible difference between the splits
    :param min_split_size: minimal number of data points in a split
    :param remaining_splits: max count of further splits (max count of splits in general when manually calling the
    function)
    :param visualize: determines whether the results will be displayed on the screen
    """
    dataset.buffer_note(f"recursive_splitting was called on this dataset with the following parameters:")
    dataset.buffer_note(f"dim_to_shift = {dim_to_shift}")
    dataset.buffer_note(f"min_split_size = {min_split_size}")
    dataset.buffer_note(f"remaining_splits = {remaining_splits}")
    dataset.buffer_note(f"visualize = {visualize}")
    if remaining_splits > 0:
        dim_to_split = find_dim_to_split(dataset=dataset, dim_to_shift=dim_to_shift)


        start = time.perf_counter()


        # if criteria for split are not met, None will be returned
        result = create_optimal_split(dataset=dataset,
                                      dim_to_shift=dim_to_shift,
                                      dim_to_split=dim_to_split,
                                      min_split_size=min_split_size)


        print(f"time for create_optimal_split: {time.perf_counter() - start}")
        #only proceed, if result actually contains new datasets
        if result:
            split1, split2 = result
            if visualize:
                #visualize the splits
                create_and_save_visualizations_for_splits(dataset, dim_to_shift, dim_to_split, split1, split2)

            #further split the resulting datasets
            name1 = split1.path.split('\\')[-1]
            #print(f"{(remaining_splits - 1) * '  '}{name1}: {len(split1.data)}")
            recursive_splitting(dataset=split1,
                                dim_to_shift=dim_to_shift,
                                min_split_size=min_split_size,
                                remaining_splits=remaining_splits - 1,
                                visualize=visualize)

            name2 = split2.path.split('\\')[-1]
            #print(f"{(remaining_splits - 1) * '  '}{name2}: {len(split2.data)}")
            recursive_splitting(dataset=split2,
                                dim_to_shift=dim_to_shift,
                                min_split_size=min_split_size,
                                remaining_splits=remaining_splits - 1,
                                visualize=visualize)
    else:
        dataset.buffer_note(f"data set not split further because maximum number of splits was reached!")
        #print("splitting terminated because max number of splits was reached!")

    dataset.create_buffered_notes()
    dataset.save()


def create_binning_splits(dataset: dc.Data,
                          dim_to_shift: str,
                          q: float,
                          remaining_splits: int,
                          visualize: bool = True) -> None:
    """
    wrapper to start recursive splitting of the data
    :param dataset: dataset to be split
    :param dim_to_shift: dimension that is to be shifted. the dataset will be split in a way, that the distribution of
    this dimension has the highest possible difference between the splits
    :param q: fraction by which the data will be shifted
    :param remaining_splits: max count of further splits (max count of splits in general when manually calling the
    function)
    :param visualize: determines whether the results will be displayed on the screen
    """
    min_split_size = max(int(len(dataset.data) * q), 1)
    recursive_splitting(dataset=dataset, dim_to_shift=dim_to_shift, min_split_size=min_split_size,
                        remaining_splits=remaining_splits, visualize=visualize)


def create_and_save_visualizations_for_splits(dataset, dim_to_shift, dim_to_split, split1, split2):
    folder_path = os.path.join(dataset.path, "pics")
    split_pics_folder = os.path.join(folder_path, "Binning")
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not os.path.isdir(split_pics_folder):
        os.mkdir(split_pics_folder)
    title = dataset.path.split("\\")[-1]
    vs.compare_splits_2d(df0=split1.data,
                         df1=split2.data,
                         dims=(dim_to_split, dim_to_shift),
                         title=title,
                         path=os.path.join(split_pics_folder, "splits_2d.png"))
    vs.compare_splits_cumulative(split1.data,
                                 split2.data,
                                 dim_to_shift,
                                 title=title,
                                 path=os.path.join(split_pics_folder, "splits_cum.png"))


def main():

    members = [100 for _ in range(6)]
    _data = dc.MaybeActualDataSet(members, save=True)

    #_data = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220325_153503_MaybeActualDataSet")
    #dim_to_split = find_dim_to_split(_data, "dim_04")
    #print(dim_to_split)
    remaining_splits = 10
    name = _data.path.split('\\')[-1]
    #print(f"{remaining_splits * '  '}{name}: {len(_data.data)}")
    create_binning_splits(dataset=_data, dim_to_shift="dim_04", q=.01, remaining_splits=remaining_splits, visualize=True)
    #dims = get_HiCS(dataset=_data, dim_to_shift="dim_04", goodness_over_length=False)
    #_data.HiCS_dims = dims
    #_data.save()


def test():
    #dataset = dc.MaybeActualDataSet([50 for _ in range(6)])
    dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220316_155110_MaybeActualDataSet\220316_155110_MaybeActualDataSet_1")
    print(find_dim_to_split(dataset, dim_to_shift="dim_04"))


def test_get_hics():
    dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220325_181838_MaybeActualDataSet\1")
    hics_dims = get_HiCS(dataset, dim_to_shift="dim_04", goodness_over_length=False)
    print(hics_dims)


def test_create_test_statistics_parallel():
    members = [250 for _ in range(6)]
    dataset = dc.MaybeActualDataSet(members)
    #dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220325_181838_MaybeActualDataSet\1")
    #create_test_statistics(dataset, "dim_04", 10, "dim_00")
    for i in range(5):

        start = time.perf_counter()


        create_binning_splits(dataset=dataset, dim_to_shift="dim_04", q=.01, remaining_splits=1,
                              visualize=False)


        print(f"overall time: {time.perf_counter() - start}")
        print("---------------------------\n")


def test_split_data():
    dataset = dc.MaybeActualDataSet.load(r"C:\Users\gerno\Programmieren\Bachelor\Data\220320_201805_MaybeActualDataSet\220320_201805_MaybeActualDataSet_1")
    print(dataset.data.describe()["dim_02"])


if __name__ == "__main__":
    #test_split_data()
    test_create_test_statistics_parallel()
    #main()
    #test()
    #main(data.path, dim_to_shift="dim_04", q=0.05)
    #test()