import os.path
from typing import List, Dict, Tuple
import pandas as pd

import Python.treeClassifier as tcl
import Python.dataCreation as dc
import Python.visualization as vs
from collections.abc import Callable
import sklearn.tree as tree


def binary_search(arr: List[float], val: float, limits: Tuple[int, int] = None) -> List[int]:
    """
    searches a value on an ordered List. Returns index of the Value, if it is found. If the searched values lies between
    two values in the list, the indexes of the flanking values will be returned. If the searched value lies outside the
    range of the List, an empty list ist returned
    :param arr: List with sorted values. The value will be searched in here
    :param val: value to look for
    :param limits: index between which the value will be searched. If omitted, value will be searched on the whole List
    :return: returns a List. If the value is found, the List will contain the index, where the value was found. If the
    searched values lies between two values in the list, the indexes of the flanking values will be returned. If the
    searched value lies outside the range of the List (or outside the limits), an empty list ist returned
    """
    if limits:
        start, end = limits
    else:
        start = 0
        end = len(arr) - 1
    #stopping criteria:
    if start > end:
        if end < 0 or start >= len(arr):
            return []
        else:
            return [end, start]
    index = (start + end) // 2
    cand = arr[index]
    if cand == val:
        return [index]
    #if the candidate is larger than the searched value, search on the lower part of the search intervall
    elif cand > val:
        return binary_search(arr, val, (start, index - 1))
    #if the candidate is smaller than the searched value, search on the higher part of the search intervall
    else:
        return binary_search(arr, val, (index + 1, end))


def linear_interpolation(x0: float, x1: float, y0: float, y1: float, x: float) -> float:
    """
    NOT USED ANYMORE, SINCE VALUES SHOULD ONLY BE SHIFTED ONTO EXISTING VALUES
    takes to points with x and y values and an additional x value, that lies between the two points. Calculates the
    missing y value for the x value assuming a linear function between the two points
    :param x0: x value of the first point
    :param x1: x value of the second point
    :param y0: y value of the first point
    :param y1: y value of the second point
    :param x: x value between the points, for which the y value will be calculated
    :return: the missing y value
    """
    dx = x1 - x0
    dy = y1 - y0
    m = dy / dx
    b = y0 - m * x0
    return m * x + b


def get_value_at_shifted_quantile(quantile: float, shift: float, values: List[float], cum_frequencies: List[float])\
        -> float:
    """
    Calculates the value of a distribution at a given quantile. The distribution is given in the form of two lists,
    which contain all "measured" values, as well as their cumulative relative frequency. If the quantile, that is to be
    calculated is not in the cumulative frequencies, the corresponding value will be interpolated assuming a linear
    dependence between the next smaller and next larger value
    :param quantile: Original Quantile of the value
    :param shift: quantile the value is supposed to be shifted
    :param values: Ascendingly sorted list of all distinct values in the relevant dimension.
    :param cum_frequencies: Ascendingly sorted list of the cumulative frequencies. The entries are relative frequencies
    (between 0 and 1). Entry at an index gives the relative frequency of the value at the same index in "values".
    :return: the calculated value at the given quantile
    """
    # solves edge case and saves iterating over the whole list
    # edge case where shifted quantile == 0 will be solved in the first iteration of the loop with an exact match
    #quantile values must between 0 and 1
    quantile = max(min(quantile + shift, 1), 0)
    if quantile < cum_frequencies[0]:
        return values[0]
    indexes = binary_search(cum_frequencies, quantile)
    length = len(indexes)
    if length <= 0:
        raise dc.CustomError("Could not find the quantile you were looking for!"
                             f"\n quantile was {quantile}")
    if len(indexes) == 1:
        return values[indexes[0]]
    if shift < 0:
        return values[indexes[0]]
    elif shift > 0:
        return values[indexes[1]]
    else:
        raise dc.CustomError("shift must not be zero!")


def get_shifted_value(value_to_shift: float, shift: float, values: List[float], cum_frequencies: List[float]) -> float:
    """
    shifts a value from a dimension by a certain quantile in the distribution of that dimension.
    :param value_to_shift: the value that is supposed to be shifted
    :param shift: distance of the shift. Number between 0 and 1.
    :param values: Ascendingly sorted list of all distinct values in the relevant dimension. value_to_shift must be in
    the list.
    :param cum_frequencies: Ascendingly sorted list of the cumulative frequencies. The entries are relative frequencies
    (between 0 and 1). Entry at an index gives the relative frequency of the value at the same index in "values".
    :return: the value at the shifted quantile
    """
    indexes = binary_search(arr=values, val=value_to_shift)
    if len(indexes) != 1:
        raise dc.CustomError("Didnt find the value you were looking for!")
    index = indexes[0]
    return get_value_at_shifted_quantile(quantile=cum_frequencies[index], shift=shift, values=values,
                                         cum_frequencies=cum_frequencies)


def qsm(model,  # Model to use for the evaluation
        data_set: dc.Data,  # Data to use for manipulation
        quantiles: Dict[str, float],  # Manipulation for the features (as list),
        predict_fn: Callable,  # predictfunction to get predicted classes [ARGUMENTS : OBJECT, DATA, PREDICTION_COLUMNS]
        save_changes: bool = True) -> Dict[str, pd.DataFrame]:
    """
    implementation for the quantile shift method (QSM). Dimensions of a dataset are shifted by a given quantiles. For
    each shifted Dimension the resulting data (together with all other unshifted dimensions) is classified by a model,
    and the results are compared to the classification of the data before the shift. The dataset with the changed
    dimensions and predictions on the changed dimensions will be saved.
    :param model: model to classify the data
    :param data_set: the dataset (with meta info)
    :param quantiles: list of tuples with the name of the dimension that is to be shifted and the quantile the values of
    the dimension are to be shifted (value between 0 and 1)
    :param predict_fn: Function that uses the model to predict classes for the dataset. Has to have the signature
    (model, pd.Dataframe, List[str]) -> List[int].
    :param save_changes: determines if the dataset will be saved after running qsm or not
    :returns: Dictionary with the resulting change matrices. Key is the original dimension, value is the dataframe
    containing the matrix
    """
    pred_classes = predict_fn(model, data_set.data, data_set.data_columns)
    data = data_set.data
    data["org_pred_classes_QSM"] = pred_classes
    data_set.extend_notes_by_one_line("notes for QSM:")
    data_set.extend_notes_by_one_line(f"prediction on the original data in column \"org_pred_classes_QSM\"")
    results = {}
    for dim, shift in quantiles.items():
        new_dim_name = f"{dim}_shifted_by_{str(shift)}"
        prediction_dims = data_set.data_columns[:]
        prediction_dims.remove(dim)
        prediction_dims.append(new_dim_name)
        values, cumulative_frequencies = vs.get_cumulative_values(data[dim].values)
        data[new_dim_name] = data[dim].apply(lambda x: get_shifted_value(value_to_shift=x,
                                                                         shift=shift,
                                                                         values=values,
                                                                         cum_frequencies=cumulative_frequencies))
        new_class_name = f"pred_with_{new_dim_name}"
        data[new_class_name] = predict_fn(model, data, prediction_dims)
        data_set.extend_notes_by_one_line(f"shifted column \"{dim}\" by {str(shift)}. Shifted column is "
                                          f"\"{new_dim_name}\", corresponding predictions are in column "
                                          f"\"{new_class_name}\"")
        results[dim] = vs.get_change_matrix(data, ("org_pred_classes_QSM", new_class_name))
    if save_changes:
        data_set.save()
    return results


def run_QSM_decisionTree(dataset: dc.Data, quantiles: Dict, save_changes: bool = True,
                         trained_tree: tree.DecisionTreeClassifier = None) -> Dict[str, pd.DataFrame]:
    """
    runs QSM on a given Dataset, that has a trained DecisionTree
    :param dataset: the dataset to run qsm on
    :param quantiles: list of tuples with the name of the dimension that is to be shifted and the value the dimension
    is to be shifted (value between 0 and 1)
    :param save_changes: determines if the changes in the dataset are to be saved
    :param trained_tree: tree that will be used to make predictions in qsm. if none is given, Tree will be loaded from
    the dataset.
    :returns: Dictionary with the resulting change matrices. Key is the original dimension, value is the dataframe
    containing the matrix
    """
    if not trained_tree:
        trained_tree = dataset.load_tree()

    def predict_fn(trained_tree_: tree.DecisionTreeClassifier, data: pd.DataFrame, dims: List[str]) -> List[int]:
        """
        prediction function for trained decision Trees on set of given dimensions of a data set
        :param trained_tree_: the trained Decision Tree
        :param data: the Data
        :param dims: List of dimension names, that are to be used for the classification. Can vary in up to one spot
        from the names of the columns the decision tree was originally trained on
        :return: List of the predicted classes
        """

        # DecisionTree needs to get a dataframe that only contains columns, that were present when the tree was trained,
        # and the columns also need to be in the same order as when the tree was trained.
        # If a new dimension is in the given dimensions for the prediction, this dimension has to be identified, and it
        # has to be renamed to the name of the missing dimension
        prediction_dims = dataset.data_columns
        prediction_dims_set = set(prediction_dims)
        dims_set = set(dims)

        missing_dim_for_pred = prediction_dims_set - dims_set
        new_dim_name = dims_set - prediction_dims_set

        if len(missing_dim_for_pred) == 1 and len(new_dim_name) == 1:
            values = data[dims]
            cols = values.columns.values
            cols[dims.index(list(new_dim_name)[0])] = list(missing_dim_for_pred)[0]
        elif len(missing_dim_for_pred) == 0 and len(new_dim_name) == 0:
            values = data[dims]
        else:
            raise dc.CustomError("number of differing columns is unexpected!")

        values = values[prediction_dims]
        return trained_tree_.predict(values)

    return qsm(model=trained_tree, data_set=dataset, quantiles=quantiles, predict_fn=predict_fn,
               save_changes=save_changes)


def visualize_QSM(base_dim: str, dim_before_shift: str, shift: float, data_path: str = "",
                  dataset: dc.Data = None, save: bool = True, save_path: str = "", class_names: List[str] = None)\
        -> None:
    """
    visualizes results of a QSM. Creates a cumulative plot of the data in the shifted dimension before and after shift.
    Creates 2d visualization of the data with the shifted dimension and a the base_dim before and after shift.
    :param base_dim: second dimension to be depicted in 2d visualizations
    :param dim_before_shift: name of the dimension that was shifted
    :param shift: quantile by which the data were shifted
    :param data_path: path to where the dataset to be depicted is saved
    :param dataset: dataset to be depicted
    :param save: determines whether the pictures are shown or saved
    :param save_path: path where the pictures are supposed to be shifted. If omitted, pictures will be saved under
    /pics/QSM in the folder of the dataset
    :param class_names: List containing the names of the different classes. Should be used if the values in the classes
    column of df are only numbers coding for the actual class names.
    """
    if not data_path and not dataset:
        raise dc.CustomError("one of the parameters path or dataset needs to be given!")
    if data_path and dataset:
        raise dc.CustomError("only one of the parameters path or dataset must be given!")
    dim_after_shift = f"{dim_before_shift}_shifted_by_{str(shift)}"
    new_class_name = f"pred_with_{dim_after_shift}"
    if not dataset:
        dataset = dc.Data.load(data_path)
    if save:
        if not save_path:
            save_path = os.path.join(dataset.path, "pics", "QSM")
        vs.compare_shift_2d(df=dataset.data,
                            common_dim=base_dim,
                            dims_to_compare=(dim_before_shift, dim_after_shift),
                            class_columns=("org_pred_classes_QSM", new_class_name),
                            path=os.path.join(save_path, "compare_shift_2d.png"),
                            class_names=class_names)
        vs.compare_shift_cumulative(df=dataset.data, dims=(dim_before_shift, dim_after_shift),
                                    shift=shift, save_path=os.path.join(save_path, "compare_cumulative_shift.png"))
    else:
        vs.compare_shift_2d(df=dataset.data,
                            common_dim=base_dim,
                            dims_to_compare=(dim_before_shift, dim_after_shift),
                            class_columns=("org_pred_classes_QSM", new_class_name),
                            class_names=class_names)
        vs.compare_shift_cumulative(df=dataset.data, dims=(dim_before_shift, dim_after_shift), shift=shift)



def main():
    #quantiles = {
    #    "dim_04": 0.1,
    #    "dim_00": 0.05,
    #    "dim_01": -0.2
    #}
    quantiles = {
        "ps_Laufweite": 0.1,
        "Passprozente": 0.05,
        "ps_Fouls": -0.2
    }
    #dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220415_111316_MaybeActualDataSet")
    dataset = dc.SoccerDataSet()
    tcl.create_and_save_tree(dataset=dataset, visualize_tree_par=False)
    results = run_QSM_decisionTree(dataset=dataset,
                                   quantiles=quantiles,
                                   save_changes=False)
    visualize_QSM(base_dim="ps_Pass", dim_before_shift="ps_Laufweite", shift=0.1, dataset=dataset, save=False,
                  class_names=dataset.class_names)
    visualize_QSM(base_dim="ps_Pass", dim_before_shift="Passprozente", shift=0.05, dataset=dataset, save=False,
                  class_names=dataset.class_names)
    visualize_QSM(base_dim="ps_Pass", dim_before_shift="ps_Fouls", shift=-0.2, dataset=dataset, save=False,
                  class_names=dataset.class_names)
    for dim, matrix in results.items():
        print(dim)
        print(matrix)
        print("")


def test_shift_methods():
    """frame = pd.DataFrame()
    test_dist = [345, 24567, 8456, 24315, 123, 2456, 4678, 589, 1, 5, 90, 8]
    frame["only_dim"] = test_dist
    vs.create_cumulative_plot(frame, "only_dim")
    values, cum_frequencies = vs.get_cumulative_values(test_dist)
    print(f"shifted value 345 by 0.1 --> {get_shifted_value(4678, 0.1, values, cum_frequencies)}")"""
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    values, cumulative_frequencies = vs.get_cumulative_values(arr)
    res = get_shifted_value(2, .9, values=values, cum_frequencies=cumulative_frequencies)
    print(res)



if __name__ == "__main__":
    main()
    #test_shift_methods()
