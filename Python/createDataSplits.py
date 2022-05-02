import re
import time
from typing import List, Tuple, Dict
import subprocess
import classifier as cl
import pandas as pd
import concurrent.futures
import math

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


def calculate_exponent(n: int, m: int, D: float) -> float:
    """
    calculates the exponent for the estimation of p for the ks test
    :param n: number of data points in the first sample
    :param m: number of data points in the second sample
    :param D: D statistic from ks test
    :return: calculated exponent
    """
    return -1 * (2 * (D**2) * n * m) / (n + m)


def calculate_alpha(n: int, m: int, D: float):
    """
    calculate alpha for a given result from the ks test. using an approximation.
    :param n: number of data points in the first sample
    :param m: number of data points in the second sample
    :param D: D statistic from ks test
    :return: calculated alpha
    """
    exponent = calculate_exponent(n, m, D)
    return 2 * math.exp(exponent)


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


def _create_data_splits(dataset: dc.Data, dim_to_shift: str, max_splits: int, min_number_of_points: int) -> None:
    """
    NOT USED
    """
    folder_path = create_folder_for_splits(dataset=dataset)
    run_R_script(additional_arguments=[folder_path, dim_to_shift, max_splits, min_number_of_points])


def get_new_dataset_name(dataset: dc.Data, suffix: str, dim_to_shift: str, q: float) -> str:
    """
    creates names for datasets that result from splitting a parent dataset. Resulting datasets will be named with a
    sequence of zeros and ones, depending on whether they are the "first" or "second" split of the parent dataset. The
    digits are seperated by "_".
    :param dataset: parent dataset to derive name from
    :param suffix: string to append to name to distinguish the created datasets. should be "0" or "1"
    :param dim_to_shift: dimension to shift in the QSM. needs to be part of the name
    :param q: quantile to shift the data. Needs to be part of the name
    :return: the new name
    """
    if suffix not in ["0", "1", ""]:
        raise dc.CustomError("suffix needs to be \"0\" or \"1\" or an empty string!")
    parent_name = dataset.path.split("\\")[-1]
    #if this is the first split from a "normal" dataset, the name will just be "0" or "1". Otherwise the digit will be
    # appended
    start = fr"{dim_to_shift}_{str(q).replace('.', '')}"
    splits_folder = os.path.join(dataset.path, "Splits")
    specific_folder = os.path.join(splits_folder, start)
    if dataset.path.find(os.path.join("Splits", start)) == -1:
        if not os.path.isdir(splits_folder):
            os.mkdir(splits_folder)
        if not os.path.isdir(specific_folder):
            os.mkdir(specific_folder)
        return os.path.join(specific_folder, suffix)
    else:
        return os.path.join(dataset.path, f"{parent_name}_{suffix}")


def split_dataset(data: pd.DataFrame, dataset: dc.Data, dim_to_split: str,
                  split_index: int, dim_to_shift: str, q: float) -> Tuple[dc.Data, dc.Data]:
    """
    splits a dataset into two, given the sorted data, a dimension to split and a split index. Resulting Datasets will
    be saved to create the necessary folder structure.
    :param data: dataframe ordered by the dim_to_split
    :param dataset: parent dataset, will be used as a template for the resulting datasets
    :param dim_to_split: dimension in which the data will be split
    :param split_index: index at which the data will be split
    :param dim_to_shift: dimension to shift in the QSM. needs to be part of the name
    :param q: quantile to shift the data. Necessary to create names of the resulting datasets
    :return: the two resulting datasets
    """
    # split the dataframe at the resulting split point, create datasets from the dataframes and return them
    data1 = data.iloc[:split_index, :]
    data2 = data.iloc[split_index:, :]

    #naming of the datasets just takes the name of the parent dataset and appends "_0" or "_1"
    dataset1 = dataset.clone_meta_data(get_new_dataset_name(dataset=dataset, suffix="0",
                                                            dim_to_shift=dim_to_shift, q=q))
    dataset2 = dataset.clone_meta_data(get_new_dataset_name(dataset=dataset, suffix="1",
                                                            dim_to_shift=dim_to_shift, q=q))
    dataset1.take_new_data(data1)
    dataset2.take_new_data(data2)

    #creating notes
    dataset1.extend_notes_by_one_line(f"This dataset results from splitting a parent dataset.")
    dataset1.extend_notes_by_one_line(f"split criterion: {dim_to_split} < {data[dim_to_split].iloc[split_index]}")
    dataset1.extend_notes_by_one_line(f"number of data points: {len(dataset1.data)}")
    dataset1.end_paragraph_in_notes()

    dataset2.extend_notes_by_one_line(f"This dataset results from splitting a parent dataset.")
    dataset2.extend_notes_by_one_line(f"split criterion: {dim_to_split} >= {data[dim_to_split].iloc[split_index]}")
    dataset2.extend_notes_by_one_line(f"number of data points: {len(dataset2.data)}")
    dataset2.end_paragraph_in_notes()

    #saving the data, to create the folder structure and files
    dataset1.save()
    dataset2.save()

    return dataset1, dataset2


def calculate_ks_tests(values: List[float], indices: List[int]) -> List[Tuple[int, Tuple[float, float]]]:
    """
    calculates the ks tests for splits of a dimension at given indices
    :param values: dimension to split. difference of the distribution of the values in the resulting splits will be
    tested.
    :param indices: indices at which the values are split
    :return: List of ks tests together with the respective index. the Tuples have the index in the first position and
    the result in the second position.
    """
    results = []
    for index in indices:
        result = stats.kstest(values[:index], values[index:])
        #also include index in the result, so it can be placed in the correct location of the full list
        results.append((index, result))
    return results


def create_sub_lists(nr_sub_lists: int, source_list: List) -> List[List]:
    """
    splits a list into a list of sublists of roughly even length.
    :param nr_sub_lists: number of sublists
    :param source_list: list to split
    :return:
    """
    task_splits = []

    #only an estimate for the best length for the list, will always be rounded down.
    task_len = round(len(source_list) / nr_sub_lists)

    for i in range(nr_sub_lists - 1):
        start = i * task_len
        end = (i + 1) * task_len
        task_splits.append(source_list[start:end])
    #last list will start at the end of the previous list and takes all remaining elements of the list. Length of this
    #list may be longer than the other lists.
    task_splits.append(source_list[(nr_sub_lists - 1) * task_len:])
    return task_splits


def create_test_statistics_parallel(dataset: dc.Data, dim_to_shift: str, min_split_size: int, dim_to_split: str,
                                    ordered_data: pd.DataFrame = None, nr_processes: int = 4)\
        -> List[Tuple[float, float]]:
    """
    creates the ks-test statistics for dataset. Data is sorted by dim_to_split and all allowed splits for the
    dim_to_shift are then compared using the ks-test. Results are returned in a list.
    :param dataset: the dataset to split
    :param dim_to_shift: dimension for which the splits will be evaluated using the ks-test
    :param min_split_size: min number of points per split
    :param dim_to_split: data points will be sorted by this dimension
    :param ordered_data: optional parameter. should contain the ordered data from the dataset. If given, the data wont
    be ordered again
    :param nr_processes: calculation of the ks-tests runs in parallel. this parameter determines the number of parallel
    processes
    :return: List of ks-test results. List has the length of the dataset. The Result at index i comes from comparing the
    splits of the dim_to_shift, where the first split has the first i elements, and the second split the remaining
    elements.
    """
    #only copy and sort data, if ordered_data is not given
    if ordered_data is not None:
        data = ordered_data
    else:
        data = dataset.data.copy(deep=True)
        data = data.sort_values(by=[dim_to_split])
    #ks_stat is supposed to be a list of the same length as the data of the dataset. At each index, is the result of the
    #ks test when comparing the datasets, that result from splitting the data at the index. Since no splits are to be
    #calculated where one dataset is smaller than min_split, dummy values are used in this range. Results have the shape
    #of (D-value, p-value), so (0, 1) means both datasets are the same (is used as dummy values, which can not distort
    #the results).
    ks_stat = [(0., 1.) for _ in range(len(data))]
    values = data[dim_to_shift].values
    #range here is important. need to avoid off-by-one errors (dataset, whose length is exactly  2 * min_split can still
    #be split in the middle)
    _, tests_to_calculate = vs.get_cumulative_values(data[dim_to_split].values, fraction=False)
    tests_to_calculate = [index for index in tests_to_calculate
                          if min_split_size <= index <= (len(data) - min_split_size)]
    tasks = create_sub_lists(nr_processes, tests_to_calculate)

    #parallel execution of code to speed the process up.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processes = [executor.submit(calculate_ks_tests, values, task) for task in tasks]

        for process in concurrent.futures.as_completed(processes):
            results = process.result()

            #results from the processes need to be placed at the correct spot in the list
            for index, ks_result in results:
                ks_stat[index] = ks_result
    return ks_stat


def create_test_statistics(dataset: dc.Data, dim_to_shift: str, min_split_size: int, dim_to_split: str,
                           ordered_data: pd.DataFrame = None) -> List[Tuple[float, float]]:
    """
    NOT USED ANYMORE
    creates the ks-test statistics for dataset. Data is sorted by dim_to_split and all allowed splits for the
    dim_to_shift are then compared using the ks-test. Results are returned in a list.
    :param dataset: the dataset to split
    :param dim_to_shift: dimension for which the splits will be evaluated using the ks-test
    :param min_split_size: min number of points per split
    :param dim_to_split: data points will be sorted by this dimension
    :param ordered_data: optional parameter. should contain the ordered data from the dataset. If given, the data wont
    be ordered again
    :return: List of ks-test results. List has the length of the dataset. The Result at index i comes from comparing the
    splits of the dim_to_shift, where the first split has the first i elements, and the second split the remaining
    elements.
    """
    #only copy and sort data, if ordered_data is not given
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
    ks_stat.extend([(0., 1.) for _ in range(min_split_size - 1)])
    return ks_stat


def find_optimal_split_index(ks_stat: List[stats.stats.KstestResult], max_p_for_split: float = .05) -> int:
    """
    selects the index from a List with results from kolmogorov Smirnov tests. Currently only selects the value with
    the lowest p-value. If multiple values share the lowest p-value, the D value will be used to calculate approximate
    an exponent for the calculation of p. Of the values with the lowest p, the one with the lowest exponent will be
    chosen.
    If the lowest p-value is larger than max_p_for_split, no valid index will be returned.
    :param ks_stat: List of results from Kolmogorov Smirnov tests
    :param max_p_for_split: a split index will only be returned, if the best p-value is lower than this
    :return: index of best result
    """
    min_p_elem = min(ks_stat, key=lambda elem: elem[1])
    if min_p_elem[1] > max_p_for_split:
        #print(ks_stat)
        return -1
    cand_list = [(res[0], i) for i, res in enumerate(ks_stat) if res[1] == min_p_elem[1]]

    # returns index of result with min p and max D, maybe think of other solution
    if len(cand_list) == 1:
        return cand_list[0][1]
    else:
        result_tuple = min(cand_list, key=lambda elem: calculate_exponent(n=elem[1],
                                                                          m=len(ks_stat) - elem[1],
                                                                          D=elem[0]))
        return result_tuple[1]


def create_optimal_split(dataset: dc.Data, dim_to_shift: str, dim_to_split: str, min_split_size: int, q: float,
                         nr_processes: int = 4, max_p_for_split: float = 0.05)\
        -> Tuple[dc.Data, dc.Data] or None:
    """
    tests every possible split of the data in the dim_to_split and determines the one which results in the strongest
    difference of the resulting sets in the kolomogorov smirnov test (ks-test) for the dim_to_shift. Only performs the
    split, if the result of the ks-test is meaningful. Definition for meaningfulness is in "find_optimal_split_index".
    :param dataset: Dataset to be split
    :param dim_to_shift: target dimension for the kolmogorov smirnov test
    :param dim_to_split: dimension in which the dataset will be split
    :param min_split_size: min number of points in a resulting split
    :param q: quantile to shift the data. Necessary to create names of the resulting datasets
    :param nr_processes: determines the number of processes that are used to calculate the ks statistics
    :param max_p_for_split: if the best split has a p-value higher than this, the dataset will not be split further
    :return: the resulting datasets, if a split was performed, or None, if no split was performed
    """
    #cannot split data, if size is not at least 2 * min_split
    length = len(dataset.data)
    if length < 2 * min_split_size:
        dataset.buffer_note(f"Dataset was not split again, because the number of points is less than twice "
                            f"the min_split_size (min_split_size = {min_split_size})!")
        return
    if min_split_size < 1:
        raise dc.CustomError("min split size needs to be larger than 0!")

    #order the data by the dim to split. Splitting the dataset at a certain index is now equivalent to splitting the
    #data at a certain value for this dim.
    data = dataset.data.copy(deep=True)
    data = data.sort_values(by=[dim_to_split])

    #calculate the differences between the splits for all allowed splits
    ks_stat = create_test_statistics_parallel(dataset=dataset, dim_to_shift=dim_to_shift,
                                              min_split_size=min_split_size, dim_to_split=dim_to_split,
                                              ordered_data=data, nr_processes=nr_processes)

    #find the index, where the difference between the split is highest
    split_index = find_optimal_split_index(ks_stat=ks_stat, max_p_for_split=max_p_for_split)

    #-1 will be returned by find_optimal_split, when no split gives significant differences
    if split_index < 0:
        dataset.buffer_note(f"Dataset was not split again, because no split lead to "
                            f"significantly different distributions in the dim_to_shift!")
        return

    #create new datasets with the split data
    dataset1, dataset2 = split_dataset(data=data, dataset=dataset, dim_to_split=dim_to_split, split_index=split_index,
                                       dim_to_shift=dim_to_shift, q=q)
    return dataset1, dataset2


def convert_indexes_to_column_names(dataset: dc.Data, indexes: List[str]) -> List[str]:
    """
    converts indexes of the list of columns to the names of the columns
    :param dataset: Dataset, for which the indexes are to be converted
    :param indexes: list of the indexes to be converted
    :return: List with column names instead of indexes
    """
    return [dataset.data_columns[int(index)] for index in indexes]


def convert_column_names_to_indexes(dataset: dc.Data, col_names: List[str]) -> List[str]:
    """
    converts the names of the columns to indexes of the list of columns
    :param dataset: Dataset, for which the column names are to be converted
    :param col_names: list of the column names to be converted
    :return: List with indexes instead of column names
    """
    return [str(dataset.data_columns.index(col)) for col in col_names]


def read_HiCS_results(dataset: dc.Data, dim_to_shift: str = "", HiCS_parameters: str = "") \
        -> List[Tuple[float, List[str]]]:
    """
    parses an output file from HiCS. Returns List of spaces with their respective contrast value.
    :param dataset: Dataset for which the HiCS output was generated
    :param dim_to_shift: name of a dimension, that needs to be present in each space that is returned. If omitted, all
    spaces are returned.
    :param HiCS_parameters: further parameters to be added to HiCS
    :return: List of all spaces containing the dim_to_shift (if given) together with their respective contrast value.
    List consists of Tuples. First element of the tuples is the contrast value, second element of the Tuples is a list
    of strings, that contains strings of the indices of the dimensions for the space.
    """
    spaces = []
    if "HiCS_output.csv" not in os.listdir(dataset.path):
        dataset.run_hics(silent=True, args_as_string=HiCS_parameters)
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


def calculate_threshold(max_val: float, min_val: float, threshold_fraction: float) -> float:
    """
    calculates the threshold p-value for given min, max and fraction for the get_HiCS method.
    :param max_val: max p-value
    :param min_val: min p-value
    :param threshold_fraction: fraction
    :return: min + fraction * (max - min)
    """
    diff = max_val - min_val
    diff_fraction = threshold_fraction * diff
    threshold = diff_fraction + min_val
    return threshold


def get_HiCS(dataset: dc.Data,
             dim_to_shift: str,
             goodness_over_length: bool,
             spaces: List[Tuple[float, List[str]]] = None,
             threshold_fraction: float = 0.7, HiCS_parameters: str = "") -> List[str]:
    """
    finds and returns the best HiCS, that contains the dim_to_shift, for a given dataset.
    :param dataset: the dataset the HiCS is supposed to be found for.
    :param dim_to_shift: Dimension that needs to be present in the returned Subspace.
    :param goodness_over_length: if True, the Subspace with the highest contrast, that also contains dim_to_shift, will
    be selected. If False, the Subspace with the most dimensions will be selected, if its contrast value is not lower
    than threshold_fraction times the value of the subspace with the overall highest contrast.
    :param spaces: List with HiCS results. if not given will be read from disc
    :param threshold_fraction: determines the cutoff contrast value for "long" subspaces
    :param HiCS_parameters: further parameters to be added to HiCS
    :return: List of strings with the names of the columns that make up the selected subspace.
    """
    if not spaces:
        spaces = read_HiCS_results(dataset, dim_to_shift, HiCS_parameters=HiCS_parameters)

    #element with the highest contrast value
    max_val_elem = max(spaces, key=lambda elem: elem[0])

    if goodness_over_length:
        HiCS_dims = convert_indexes_to_column_names(dataset=dataset, indexes=max_val_elem[1])
        return HiCS_dims

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
            HiCS_dims = convert_indexes_to_column_names(dataset=dataset, indexes=curr_max[1])
            return HiCS_dims

    HiCS_dims = convert_indexes_to_column_names(dataset=dataset, indexes=max_val_elem[1])
    return HiCS_dims


def find_dim_to_split(dataset: dc.Data, dim_to_shift: str, goodness_over_length: bool = True,
                      threshold_fraction: float = .7, HiCS_parameters: str = "") -> str:
    """
    selects the dimension that is most suited for splitting from the dataset.
    :param dataset: the dataset to select the dimension from
    :param dim_to_shift: name of the dimension to be shifted
    :param goodness_over_length: determines how the best HiCS is selected. If True, the subspace with the highest
    contrast will be selected. If False, the number of dimensions will also play a role. (For details see get_HiCS)
    :param threshold_fraction: determines the cutoff contrast value for "long" subspaces
    :param HiCS_parameters: further parameters to be added to HiCS
    :return: name of the dimension to be split
    """
    spaces = read_HiCS_results(dataset=dataset, dim_to_shift=dim_to_shift, HiCS_parameters=HiCS_parameters)

    hics_dims = get_HiCS(dataset=dataset, dim_to_shift=dim_to_shift, goodness_over_length=goodness_over_length,
                         spaces=spaces, threshold_fraction=threshold_fraction)
    #excluding the dim_to_shift
    hics_dims = [dim for dim in hics_dims if dim != dim_to_shift]
    hics_dims = convert_column_names_to_indexes(dataset, hics_dims)

    #selecting the dim, that is part of the datasets best HiCS and has the highest contrast value in a pair with the
    # dim_to_shift
    spaces = [elem for elem in spaces if len(elem[1]) == 2]
    spaces = [elem for elem in spaces
              if elem[1][0] in hics_dims or elem[1][1] in hics_dims]
    dim_to_shift_index_str = str(dataset.data_columns.index(dim_to_shift))

    #selecting Subspace with highest contrast
    dim_0, dim_1 = max(spaces, key=lambda elem: elem[0])[1]
    if dim_0 == dim_to_shift_index_str:
        return dataset.data_columns[int(dim_1)]
    elif dim_1 == dim_to_shift_index_str:
        return dataset.data_columns[int(dim_0)]
    else:
        raise dc.CustomError("dim_to_shift was not in pair!")


def create_and_save_visualizations_for_splits(dataset: dc.Data, dim_to_shift: str, dim_to_split: str,
                                              split1: dc.Data, split2: dc.Data) -> None:
    """
    creates pictures to visualize splits. The resulting pictures are saved at the path of the parent dataset.
    :param dataset: parent dataset
    :param dim_to_shift: the splits are calculted to maximize the difference between the splits when comparing the
    resulting distributions in this dimension
    :param dim_to_split: dimension in which the dataset is split
    :param split1: first split dataset
    :param split2: second split dataset
    """
    #create necessary folder structure
    folder_path = os.path.join(dataset.path, "pics")
    split_pics_folder = os.path.join(folder_path, "Binning")
    final_folder = os.path.join(split_pics_folder, dim_to_shift)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not os.path.isdir(split_pics_folder):
        os.mkdir(split_pics_folder)
    if not os.path.isdir(final_folder):
        os.mkdir(final_folder)
    title = dataset.path.split("\\")[-1]

    vs.compare_splits_2d(df0=split1.data,
                         df1=split2.data,
                         dims=(dim_to_split, dim_to_shift),
                         title=title,
                         path=os.path.join(final_folder, "splits_2d.png"))

    vs.compare_splits_cumulative(split1.data,
                                 split2.data,
                                 dim_to_shift,
                                 title=title,
                                 path=os.path.join(final_folder, "splits_cum.png"))


def recursive_splitting(dataset: dc.Data,
                        dim_to_shift: str,
                        min_split_size: int,
                        remaining_splits: int,
                        q: float,
                        visualize: bool = True,
                        nr_processes: int = 4,
                        max_p_for_split: float = 0.05,
                        goodness_over_length: bool = True,
                        threshold_fraction: float = 0.7,
                        HiCS_parameters: str = ""
                        ) -> None:
    """
    recursively splits a dataset in subsets.
    :param dataset: dataset to be split
    :param dim_to_shift: dimension that is to be shifted. the dataset will be split in a way, that the distribution of
    this dimension has the highest possible difference between the splits
    :param min_split_size: minimal number of data points in a split
    :param q: quantile to shift the data. Necessary to create names of the resulting datasets
    :param remaining_splits: max count of further splits (max count of splits in general when manually calling the
    function)
    :param visualize: determines whether the results will be displayed on the screen
    :param nr_processes: determines the number of processes that are used to calculate the ks statistics
    :param max_p_for_split: if the best split has a p-value higher than this, the dataset will not be split further
    :param goodness_over_length: determines how the best HiCS is selected. If True, the subspace with the highest
    contrast will be selected. If False, the number of dimensions will also play a role. (For details see get_HiCS)
    :param threshold_fraction: determines the cutoff contrast value for "long" subspaces
    :param HiCS_parameters: further parameters to be added to HiCS
    """
    #add notes to be put in the dataset later
    dataset.buffer_note(f"recursive_splitting was called on this dataset with the following parameters:")
    dataset.buffer_note(f"dim_to_shift = {dim_to_shift}")
    dataset.buffer_note(f"min_split_size = {min_split_size}")
    dataset.buffer_note(f"remaining_splits = {remaining_splits}")
    dataset.buffer_note(f"visualize = {visualize}")
    if remaining_splits > 0:
        dim_to_split = find_dim_to_split(dataset=dataset, dim_to_shift=dim_to_shift,
                                         threshold_fraction=threshold_fraction, HiCS_parameters=HiCS_parameters,
                                         goodness_over_length=goodness_over_length)

        # if criteria for split are not met, None will be returned
        result = create_optimal_split(dataset=dataset,
                                      dim_to_shift=dim_to_shift,
                                      dim_to_split=dim_to_split,
                                      min_split_size=min_split_size,
                                      q=q,
                                      nr_processes=nr_processes,
                                      max_p_for_split=max_p_for_split)

        #only proceed, if result actually contains new datasets
        if result:
            split1, split2 = result
            if visualize:
                #visualize the splits
                create_and_save_visualizations_for_splits(dataset, dim_to_shift, dim_to_split, split1, split2)

            #further split the resulting datasets
            recursive_splitting(dataset=split1, dim_to_shift=dim_to_shift, min_split_size=min_split_size,
                                remaining_splits=remaining_splits - 1, q=q, visualize=visualize,
                                nr_processes=nr_processes, max_p_for_split=max_p_for_split,
                                threshold_fraction=threshold_fraction, HiCS_parameters=HiCS_parameters,
                                goodness_over_length=goodness_over_length)

            recursive_splitting(dataset=split2, dim_to_shift=dim_to_shift, min_split_size=min_split_size,
                                remaining_splits=remaining_splits - 1, q=q, visualize=visualize,
                                nr_processes=nr_processes, max_p_for_split=max_p_for_split,
                                threshold_fraction=threshold_fraction, HiCS_parameters=HiCS_parameters,
                                goodness_over_length=goodness_over_length)
    else:
        dataset.buffer_note(f"data set not split further because maximum number of splits was reached!")
        #print("splitting terminated because max number of splits was reached!")

    dataset.create_buffered_notes()
    dataset.save()


def data_binning(dataset: dc.Data, shifts: Dict[str, float], max_split_nr: int, visualize: bool = True,
                 nr_processes: int = 4, max_p_for_split: float = 0.05, goodness_over_length: bool = True,
                 threshold_fraction: float = 0.7, HiCS_parameters: str = "") -> Dict[str, str]:
    """
    function to start the recursive splitting of a dataset. For each shift, that is given in shifts, a separate run of
    splits will be performed.
    :param dataset: dataset to be split
    :param shifts: dictionary, that has the dimensions, that are supposed to be shifted, as keys, and the quantiles by
    which the dimensions are supposed to be shifted as values
    :param max_split_nr: max count of further splits (max count of splits in general when manually calling the
    function)
    :param visualize: determines whether the results will be displayed on the screen
    :param nr_processes: determines the number of processes that are used to calculate the ks statistics
    :param max_p_for_split: if the best split has a p-value higher than this, the dataset will not be split further
    :param goodness_over_length: determines how the best HiCS is selected. If True, the subspace with the highest
    contrast will be selected. If False, the number of dimensions will also play a role. (For details see get_HiCS)
    :param threshold_fraction: determines the cutoff contrast value for "long" subspaces
    :param HiCS_parameters: further parameters to be added to HiCS
    :return: dictionary with dims that were shifted as keys. For each dim, the value is the path to the folder, in which
    the corresponding splits are saved.
    """
    new_dict = {}
    for dim, q in shifts.items():
        min_split_size = max(math.ceil(abs(len(dataset.data) * q)), 1)
        recursive_splitting(dataset=dataset, dim_to_shift=dim, min_split_size=min_split_size,
                            remaining_splits=max_split_nr, visualize=visualize, q=q, nr_processes=nr_processes,
                            max_p_for_split=max_p_for_split, threshold_fraction=threshold_fraction,
                            HiCS_parameters=HiCS_parameters,
                            goodness_over_length=goodness_over_length)
        new_dict[dim] = get_new_dataset_name(dataset=dataset, suffix="", dim_to_shift=dim, q=q)
    return new_dict


def main():
    """
    test function
    """
    members = [100 for _ in range(6)]
    _data = dc.MaybeActualDataSet(members, save=True)

    #_data = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220328_142537_MaybeActualDataSet")
    #dim_to_split = find_dim_to_split(_data, "dim_04")
    #print(dim_to_split)
    remaining_splits = 2
    quantiles = {"dim_04": -0.05}
    data_binning(dataset=_data, shifts=quantiles, max_split_nr=remaining_splits, visualize=True)
    #dims = get_HiCS(dataset=_data, dim_to_shift="dim_04", goodness_over_length=False)
    #_data.HiCS_dims = dims
    #_data.save()


def test():
    """
    test function
    """
    #quantiles = {
    #    "dim_04": 0.1,
    #    "dim_00": 0.05,
    #    "dim_01": -0.2
    #}
    quantiles = {
        "sepal_length": 0.1,
        "petal_length": 0.05,
        "petal_width": -0.2
    }
    #members = [50 for _ in range(6)]
    #dataset = dc.MaybeActualDataSet(members)
    dataset = dc.IrisDataSet()
    print(data_binning(dataset, shifts=quantiles, max_split_nr=2, visualize=True))


def test_sub_lists():
    create_test_statistics_parallel(dataset=dc.IrisDataSet(), dim_to_shift="sepal_length",
                                    dim_to_split="petal_width", min_split_size=10)
    """easy = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    weird = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4,  4,  5,  6,  7,  7,  7,  7,  7,  7,  8]
    print(vs.get_cumulative_values(easy, fraction=False))
    print(vs.get_cumulative_values(weird, fraction=False))"""
    #print("easy", create_sub_lists(4, easy))
    #print("weird", create_sub_lists(4, weird))


def test_get_hics():
    """
    test function
    """
    dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220325_181838_MaybeActualDataSet\1")
    hics_dims = get_HiCS(dataset, dim_to_shift="dim_04", goodness_over_length=False)
    print(hics_dims)


def test_create_test_statistics_parallel():
    """
    test function
    """
    #members = [100 for _ in range(6)]
    #dataset = dc.MaybeActualDataSet(members)
    dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220423_135313_MaybeActualDataSet")
    #create_test_statistics(dataset, "dim_04", 10, "dim_00")
    stats_ = create_test_statistics_parallel(dataset, "dim_04", 40, "dim_00")
    print(find_optimal_split_index(stats_))
    """for i, stat in enumerate(stats_):
        print(i, stat)"""


def test_split_data():
    """
    test function
    """
    dataset = dc.MaybeActualDataSet.load(r"C:\Users\gerno\Programmieren\Bachelor\Data\220320_201805_MaybeActualDataSet"
                                         r"\220320_201805_MaybeActualDataSet_1")
    print(dataset.data.describe()["dim_02"])


def test_get_name():
    dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220328_142537_MaybeActualDataSet")
    print(get_new_dataset_name(dataset, "", "dim_04", -0.5))


def test_alpha():
    n = 124
    print(calculate_alpha(n, 600-n, 0.4269449715370019))


if __name__ == "__main__":
    #test_split_data()
    #test_alpha()
    #test_create_test_statistics_parallel()
    #test_get_name()
    test_sub_lists()
    #main()
