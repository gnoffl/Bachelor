from typing import List, Dict
import pandas as pd

import Python.DataCreation.dataCreation as dc
import Python.DataCreation.visualization as vs
from collections.abc import Callable
import sklearn.tree as tree


#todo: make sure, that linear interpolation is valid here
def linear_interpolation(x0: float, x1: float, y0: float, y1: float, x: float) -> float:
    """
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


#todo: implement binary search
#todo: test edge cases
def get_value_at_quantile(quantile: float, values: List[float], cum_frequencies: List[float]) -> float:
    """
    Calculates the value of a distribution at a given quantile. The distribution is given in the form of two lists, which
    contain all "measured" values, as well as their cumulative relative frequency. If the quantile, that is to be calculated
    is not in the cumulative frequencies, the corresponding value will be interpolated assuming a linear dependence between
    the next smaller and next larger value
    :param quantile: Quantile the value is supposed to be calculated for
    :param values: Ascendingly sorted list of all distinct values in the relevant dimension.
    :param cum_frequencies: Ascendingly sorted list of the cumulative frequencies. The entries are relative frequencies
    (between 0 and 1). Entry at an index gives the relative frequency of the value at the same index in "values".
    :return: the calculated value at the given quantile
    """
    # solves edge case and saves iterating over the whole list
    # edge case where shifted quantile == 0 will be solved in the first iteration of the loop with an exact match
    if quantile == 1:
        return values[-1]
    next_larger_quant_index = 0
    exact_match = False
    for i, quant in enumerate(cum_frequencies):
        if quant < quantile:
            continue
        elif quant == quantile:
            next_larger_quant_index = i
            exact_match = True
        else:
            next_larger_quant_index = i
        break

    next_larger_value = values[next_larger_quant_index]
    # no need for interpolation, if value was hit perfectly
    if exact_match:
        return next_larger_value
    next_smaller_value = values[next_larger_quant_index - 1]
    next_larger_quant = cum_frequencies[next_larger_quant_index]
    next_smaller_quant = cum_frequencies[next_larger_quant_index - 1]
    return linear_interpolation(next_smaller_quant, next_larger_quant, next_smaller_value, next_larger_value, quantile)


#todo: implement binary search
#todo: test edge cases
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
    index = 0
    for i, val in enumerate(values):
        #since cumulative Values are calculated from the values of the dimension, every value that is searched for must
        #exactly be found
        if val == value_to_shift:
            index = i
            break
        if i == len(values) - 1:
            raise dc.CustomError("Didnt find the value you were looking for!")
    #quantile values must between 0 and 1
    shifted_quantile = max(min(cum_frequencies[index] + shift, 1), 0)
    return get_value_at_quantile(shifted_quantile, values, cum_frequencies)


#todo: how does prediction on the tree work with different column orders / column names?
# todo: results seem wrong
#todo: visualization
#todo: how to save results? --> explicit statements for neighborhoods, save matrices?
def qsm(model,  # Model to use for the evaluation
        data_set: dc.Data,  # Data to use for manipulation
        quantiles: Dict[str, float],  # Manipulation for the features (as list),
        predict_fn: Callable,  # predictfunction to get predicted classes [ARGUMENTS : OBJECT, DATA]
        ) -> None:
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
    """
    #todo: do i need a series here? --> looks like no
    pred_classes = predict_fn(model, data_set.data, data_set.data_columns)
    data = data_set.data
    data["org_pred_classes_QSM"] = pred_classes
    print(data["org_pred_classes_QSM"].value_counts())
    data_set.extend_notes_by_one_line("notes for QSM:")
    data_set.extend_notes_by_one_line(f"prediction on the original data in column \"org_pred_classes_QSM\"")
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
        data_set.extend_notes_by_one_line(f"shifted column \"{dim}\" by {str(shift)}. Shifted column is \"{new_dim_name}\", corresponding predictions are in column \"{new_class_name}\"")

    data_set.save()


def run_QSM_decisionTree(path_to_data, quantiles) -> None:
    """
    runs QSM on a given Dataset, that has a trained DecisionTree
    :param path_to_data: path to the save location of the dataset
    :param quantiles:  list of tuples with the name of the dimension that is to be shifted and the value the dimension
    is to be shifted (value between 0 and 1)
    """
    dataset = dc.MaybeActualDataSet.load(path_to_data)
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

    qsm(model=trained_tree, data_set=dataset, quantiles=quantiles, predict_fn=predict_fn)


def visualize_QSM_from_save(path: str, base_dim: str, dim_before_shift: str, shift: float):
    dim_after_shift = f"{dim_before_shift}_shifted_by_{str(shift)}"
    new_class_name = f"pred_with_{dim_after_shift}"
    dataset = dc.MaybeActualDataSet.load(path)
    vs.compare_shift_2d(df=dataset.data,
                        common_dim=base_dim,
                        dims_to_compare=(dim_before_shift, dim_after_shift),
                        class_columns=("org_pred_classes_QSM", new_class_name))
    vs.compare_shift_cumulative(df=dataset.data, dims=(dim_before_shift, dim_after_shift), shift=shift)


def main():
    quantiles = {
        "dim_04": 0.1,
        "dim_00": 0.05,
        "dim_01": -0.2
    }
    run_QSM_decisionTree(path_to_data="D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\MaybeActualDataSet",
                         quantiles=quantiles)
    #visualize_QSM_from_save("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\MaybeActualDataSet",
    #                        "dim_04",
    #                        "dim_01",
    #                        -0.2)


def test_shift_methods():
    frame = pd.DataFrame()
    test_dist = [345, 24567, 8456, 24315, 123, 2456, 4678, 589, 1, 5, 90, 8]
    frame["only_dim"] = test_dist
    vs.create_cumulative_plot(frame, "only_dim")
    values, cum_frequencies = vs.get_cumulative_values(test_dist)
    print(f"shifted value 345 by 0.1 --> {get_shifted_value(4678, 0.1, values, cum_frequencies)}")



if __name__ == "__main__":
    main()
    #test_shift_methods()
