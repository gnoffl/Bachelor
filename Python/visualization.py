from typing import List, Tuple, Dict


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import os
import dataCreation as dc

colors = [(0, 0, 1, 0.8),
          (1, 0, 0, 0.8),
          (0, 1, 0, 0.8),
          (1, 1, 0, 0.8),
          (1, 0, 1, 0.8),
          (0, 1, 1, 0.8),
          (0.6, 0.6, 0.6, 0.8),
          (0.1, 0.1, 0.1, 0.8),
          (.7, .7, .3, 0.8),
          (.7, .3, .7, 0.8),
          (.3, .7, .7, 0.8),
          (0, 0, 0, 0.8),
          (1, 1, 1, 0.8),
          (0, 0.5, 1, 0.8),
          (1, 0, 0.5, 0.8),
          (0.5, 1, 0, 0.8),
          (0.3, 0.3, 0.7, 0.8)
          ]


def calculate_visualized_area(x_axis: List[float] or np.ndarray, y_axis: List[float] or np.ndarray)\
        -> Tuple[float, float, float, float]:
    """
    takes the values for 2 axes as input, and calculates the min and max for each dimension + a small padding, to use as
    limits for the dimensions during visualization
    :param x_axis: values for the x_axis
    :param y_axis: values for the y_axis
    :return: tuple of (x_axis_min, x_axis_max, y_axis_min, y_axis_max)
    """
    x_min, x_max = min(x_axis), max(x_axis)
    y_min, y_max = min(y_axis), max(y_axis)
    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)
    x_axis_min = x_min - x_pad
    x_axis_max = x_max + x_pad
    y_axis_min = y_min - y_pad
    y_axis_max = y_max + y_pad
    return x_axis_min, x_axis_max, y_axis_min, y_axis_max


def find_common_area(x0_dim: List[float] or np.ndarray,
                     y0_dim: List[float] or np.ndarray,
                     x1_dim: List[float] or np.ndarray,
                     y1_dim: List[float] or np.ndarray) -> Tuple[float, float, float, float]:
    """
    to be used when two graphs are supposed to be comparable, in the sense that both show the same range on the depicted
    axes. calculates the min and max values for 2 dimensions for 2 graphs, and then combines the ranges for each
    dimension for both graphs.
    :param x0_dim: values for the x-dimension for the first graph
    :param y0_dim: values for the y-dimension for the first graph
    :param x1_dim: values for the x-dimension for the second graph
    :param y1_dim: values for the x-dimension for the second graph
    :return: Tuple of (x_axis_min, x_axis_max, y_axis_min, y_axis_max), that spans the values for both graphs
    """
    area_0 = calculate_visualized_area(x0_dim, y0_dim)
    area_1 = calculate_visualized_area(x1_dim, y1_dim)
    x_axis_min = min(area_0[0], area_1[0])
    x_axis_max = max(area_0[1], area_1[1])
    y_axis_min = min(area_0[2], area_1[2])
    y_axis_max = max(area_0[3], area_1[3])
    return x_axis_min, x_axis_max, y_axis_min, y_axis_max


def get_color_index(clas: int or str, class_names: List[str], i: int):
    """
    gives a color index for a class
    :param clas: class a color is needed for
    :param class_names: names of the classes in that dataset
    :param i: number of the class
    :return: index for color selection from colors
    """
    try:
        if class_names:
            color_ind = class_names.index(clas)
        else:
            raise ValueError
    except ValueError:
        try:
            if isinstance(clas, str) and "_" in clas:
                raise ValueError
            color_ind = int(clas)
        except ValueError:
            color_ind = i
    return color_ind


#first entry of dims will be x-axis, second will be y-axis
def visualize_2d(df: pd.DataFrame,
                 dims: Tuple[str, str],
                 class_column: str or None = None,
                 title: str = None,
                 path: str = None,
                 visualized_area: Tuple[float, float, float, float] = None,
                 class_names: List[str] = None) -> None:
    """
    Creates a 2d Image of the given data, showing the dimensions whose names are given in dims.
    :param df: Dataframe containing the data
    :param dims: Tuple of the names of the columns that are to be the axes of the plot
    :param class_column: Name of the column that contains class names. Will be used for labeling the data
    :param title: Title of the plot
    :param visualized_area: sets the limit of the x and y axes. order is x_min, x_max, y_min, y_max. If omitted, fitting
    values will be calculated based on the data.
    :param path: If path is given, the plot will not be shown but instead saved at the given location
    :param class_names: List containing the names of the different classes. Should be used if the values in the classes
    column of df are only numbers coding for the actual class names.
    """
    x_name, y_name = dims
    plt.figure(0, figsize=(6, 4.5))
    plt.clf()

    if visualized_area:
        x_axis_min, x_axis_max, y_axis_min, y_axis_max = visualized_area
    else:
        x_axis_min, x_axis_max, y_axis_min, y_axis_max = calculate_visualized_area(df[x_name].values, df[y_name].values)

    plt.xlim(x_axis_min, x_axis_max)
    plt.ylim(y_axis_min, y_axis_max)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    if title:
        plt.title(title)

    if class_column:
        classes = set(df[class_column].values)
        # plot classes seperately, to include a label, which then can be used to show the different classes in the plot
        for i, clas in enumerate(classes):
            color_ind = get_color_index(clas, class_names, i)
            plt.scatter(df.loc[df[class_column] == clas, [x_name]],
                        df.loc[df[class_column] == clas, [y_name]],
                        color=colors[color_ind],
                        edgecolor="k",
                        label=get_label(clas=clas, class_names=class_names))
        plt.legend(loc="upper left", frameon=True)
    else:
        plt.scatter(df[x_name], df[y_name], cmap=plt.cm.Set1, edgecolor="k")
    plt.subplots_adjust(right=0.87)
    if path:
        create_save_path(path)
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()


#first entry of dims will be x-axis, second will be y-axis
def visualize_2d_subplot(df: pd.DataFrame,
                         dims: Tuple[str, str],
                         subplot_location: (int, int, int, int, int, int),
                         class_column: str or None = None,
                         title: str = None,
                         visualized_area: Tuple[float, float, float, float] = None,
                         class_names: List[str] = None,
                         show_legend: bool = False,
                         bbox_to_anchor: (float, float) = None,
                         loc: str = "upper left",
                         map_color: bool = True,
                         map_label: bool = True,
                         axis_names: Tuple[str, str] = None,
                         frame_on=True,
                         ncol_legend=1):
    """
    Creates a 2d Image of the given data, showing the dimensions whose names are given in dims.
    :param df: Dataframe containing the data
    :param dims: Tuple of the names of the columns that are to be the axes of the plot
    :param class_column: Name of the column that contains class names. Will be used for labeling the data
    :param title: Title of the plot
    :param visualized_area: sets the limit of the x and y axes. order is x_min, x_max, y_min, y_max. If omitted, fitting
    values will be calculated based on the data.
    :param subplot_location: defines structure of the created subplot. first 2 numbers are rows and columns the grid
    will have. second 2 numbers are the location the subplot starts in rows and columns. third 2 numbers are the height
    and width of the subplot
    :param class_names: List containing the names of the different classes. Should be used if the values in the classes
    column of df are only numbers coding for the actual class names.
    :param show_legend: determines whether a legend will be shown
    :param bbox_to_anchor: location for a potential legend
    :param loc: location for a potential legend, bbox_to_anchor will override this, if both are given
    :param map_color: determines whether the classes will will be tried to be mapped to colors by their name or
    class_names, or if the colors will just be chosen by the order the classes apper in
    :param map_label: determines whether a label for the class name is searched, or if clas will just be returned
    :param axis_names: labels, that will be used for the axes (x-name, y-name). If not given, entries of dims will be
    used
    :param frame_on: determines whether the legend will have a border
    :param ncol_legend: Determines number of columns in legend
    :returns: created subplot
    """
    x_name, y_name = dims

    if visualized_area:
        x_axis_min, x_axis_max, y_axis_min, y_axis_max = visualized_area
    else:
        x_axis_min, x_axis_max, y_axis_min, y_axis_max = calculate_visualized_area(df[x_name].values, df[y_name].values)

    rows, columns, row_pos, col_pos, height, width = subplot_location

    plot = plt.subplot2grid((rows, columns), (row_pos, col_pos), rowspan=height, colspan=width)

    plot.axis(xmin=x_axis_min, xmax=x_axis_max, ymin=y_axis_min, ymax=y_axis_max)
    if axis_names:
        plot.set_xlabel(axis_names[0])
        plot.set_ylabel(axis_names[1])
    else:
        plot.set_xlabel(x_name)
        plot.set_ylabel(y_name)

    if title:
        plot.set_title(title)

    if class_column:
        classes = set(df[class_column].values)
        # plot classes seperately, to include a label, which then can be used to show the different classes in the plot
        for i, clas in enumerate(classes):
            if map_color:
                color_ind = get_color_index(clas, class_names, i)
            else:
                color_ind = i
            plot.scatter(df.loc[df[class_column] == clas, [x_name]],
                         df.loc[df[class_column] == clas, [y_name]],
                         color=colors[color_ind],
                         edgecolor="k",
                         label=get_label(clas=clas, class_names=class_names, map_label=map_label))
    else:
        plot.scatter(df[x_name], df[y_name], cmap=plt.cm.Set1, edgecolor="k")
    if show_legend:
        if bbox_to_anchor:
            plot.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, frameon=frame_on, ncol=ncol_legend)
        else:
            plot.legend(loc=loc, frameon=frame_on, ncol=ncol_legend)
    return plot
    #plot.subplots_adjust(right=0.87)


def compare_splits_2d(df0: pd.DataFrame,
                      df1: pd.DataFrame,
                      dims: Tuple[str, str],
                      title: str = None,
                      path: str = None) -> None:
    """
    Plots data from two datasets in the same 2d plot. Points from the dataset are colored in individual colors.
    :param df0: first dataset
    :param df1: second dataset
    :param dims: dims to be plotted
    :param title: title for the plot
    :param path: potential path to save plot
    """
    new_df0 = df0.copy()
    new_df1 = df1.copy()

    new_df0["vis_class"] = 0
    new_df1["vis_class"] = 1

    final_df = pd.concat([new_df0, new_df1])

    visualize_2d(df=final_df, dims=dims, class_column="vis_class", title=title, path=path)


def compare_shift_2d(df: pd.DataFrame,
                     common_dim: str,
                     dims_to_compare: Tuple[str, str],
                     class_columns: Tuple[str, str] = None,
                     titles: Tuple[str, str] = None,
                     path: str = None,
                     class_names: List[str] = None) -> None:
    """
    method to compare data before and after shift induced by QSM. Will create two 2d visualizations of the Data, but
    but the x and y limits will be the same for both graphs.
    :param df: Dataframe with data to be visualized
    :param common_dim: the Dimension that is not shifted
    :param dims_to_compare: first the shifted dimension before shift, then the shifted dimension after shift
    :param class_columns: column names of the predicted classification of the data before and after the shift
    :param titles: titles for the created graphs. titles will be the name of the shifted column before and after shift,
    if this argument is omitted
    :param path: path where graphics should be saved. Graphics will only be shown, not saved, if this argument is
    omitted
    :param class_names: List containing the names of the different classes. Should be used if the values in the classes
    column of df are only numbers coding for the actual class names.
    """
    x_axis_min, x_axis_max, y_axis_min, y_axis_max = find_common_area(df[common_dim].values,
                                                                      df[dims_to_compare[0]].values,
                                                                      df[common_dim].values,
                                                                      df[dims_to_compare[1]].values)

    visualized_area = (x_axis_min, x_axis_max, y_axis_min, y_axis_max)

    class_column_0 = class_columns[0] if class_columns else None
    class_column_1 = class_columns[1] if class_columns else None

    title_0 = titles[0] if titles else dims_to_compare[0]
    title_1 = titles[1] if titles else dims_to_compare[1]

    if path:
        actual_path = path.split(".png")[0]
        actual_path_0 = f"{actual_path}_0.png"
        actual_path_1 = f"{actual_path}_1.png"
    else:
        actual_path_0 = ""
        actual_path_1 = ""

    visualize_2d(df=df, dims=(common_dim, dims_to_compare[0]), class_column=class_column_0, title=title_0,
                 path=actual_path_0, visualized_area=visualized_area, class_names=class_names)
    visualize_2d(df=df, dims=(common_dim, dims_to_compare[1]), class_column=class_column_1, title=title_1,
                 path=actual_path_1, visualized_area=visualized_area, class_names=class_names)


def get_label(clas: int or str, class_names: List[str], map_label: bool = True):
    """
    gets the class name for a given entry in a class column. List containing the names of the different classes. The
    value clas is supposed to be an integer, or a string of an integer, that is used to return the class name at its
    index in the class_names array. If the conversion doesn't work, or the index is out of bounds for the array, clas
    will be returned
    :param clas: value from class column
    :param class_names: array with class names
    :param map_label: determines whether a label for the class name is searched, or if clas will just be returned
    :return: entry from class names, if conversion worked properly, otherwise clas.
    """
    if map_label:
        if class_names is None:
            return clas
        else:
            try:
                if isinstance(clas, str) and "_" in clas:
                    raise ValueError
                index = int(clas)
                return class_names[index]
            except (ValueError, IndexError):
                return clas
    else:
        return clas


def visualize_3d(df: pd.DataFrame,
                 dims: Tuple[str, str, str],
                 azim: int = 110,
                 elev: int = -150,
                 class_column: str or None = None,
                 path: str = "",
                 class_names: List[str] = None) -> None:
    """
    Creates a 3d Image of the given data, showing the dimensions whose names are given in dims.
    :param df: Dataframe containing the data
    :param dims: Tuple of the names of the columns that are to be the axes of the plot
    :param azim: not sure about the definition, but changes rotation
    :param elev: not sure about the definition, but changes rotation
    :param class_column: Name of the column that contains class names. Will be used for labeling the data
    :param path: If path is given, the plot will not be shown but instead saved at the given location
    :param class_names: List containing the names of the different classes. Should be used if the values in the classes
    column of df are only numbers coding for the actual class names.
    """
    x_name, y_name, z_name = dims
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)
    fig.add_axes(ax)
    if class_column:
        # plot classes separately, to include a label, which then can be used to show the different classes in the plot
        classes = set(df[class_column].values)
        for i, clas in enumerate(classes):
            ax.scatter(
                df.loc[df[class_column] == clas, [x_name]],
                df.loc[df[class_column] == clas, [y_name]],
                df.loc[df[class_column] == clas, [z_name]],
                color=colors[i],
                edgecolor="k",
                label=get_label(clas, class_names),
                s=40,
            )
    else:
        ax.scatter(
            df[x_name],
            df[y_name],
            df[z_name],
            cmap=plt.cm.Set1,
            edgecolor="k",
            s=40,
        )
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    plt.legend(bbox_to_anchor=(1.05, 1))
    if path:
        plt.savefig(path)
    else:
        plt.show()



def clear_temp(path_to_temp) -> None:
    """
    removes files from given folder
    :param path_to_temp: path to folder
    """
    files = [os.path.join(path_to_temp, file) for file in os.listdir(path_to_temp) if
             os.path.isfile(os.path.join(path_to_temp, file))]
    for file in files:
        os.remove(file)


#todo: maybe keep images in memory instead of saving them to disc.
#todo: maybe save as .mp4 or smth other than .gif
def create_3d_gif(df: pd.DataFrame, dims: Tuple[str, str, str], name: str, class_column: str or None = None,
                  steps: int = 36, duration: int = None, class_names: List[str] = None) -> None:
    """
    creates a gif of the data with the given dims being the shown axes of the 3d plot
    :param df: Dataframe containing the data
    :param dims: Name of the columns of df which will be displayed as the axes
    :param name: Will be the beginning of the filename. filename will also include the names of the dims.
    :param class_column: Column of df that contains class labels. Will be used to label classes in the plot.
    :param steps: Number of pictures per rotation of the data. The more steps, the more fluid the gif becomes
    :param duration: Milliseconds each frame is shown
    :param class_names: List containing the names of the different classes. Should be used if the values in the classes
    column of df are only numbers coding for the actual class names.
    """
    # keeps duration for the whole gif constant --> more fluid visual if steps is increased
    if not duration:
        duration = 36 * 100 / steps
    x_name, y_name, z_name = dims
    plots_folder = os.path.join(os.path.dirname(__file__), "../Plots")

    # folder where single pictures for the gif will be stored
    temp = os.path.join(plots_folder, "temp")
    if os.path.isdir(temp):
        # folder needs to be empty before
        clear_temp(temp)
    else:
        os.mkdir(temp)

    # amount that the angle needs to be changed every step
    stepsize = int(360 / steps)

    for i in range(steps):
        # file names for pictures are just numbers
        # zero padding to avoid confusion in the order of files with different name lengths
        filename = f"{i:03d}"
        visualize_3d(df, dims, class_column=class_column, azim=stepsize * i, elev=-164,
                     path=os.path.join(temp, f"{filename}.png"), class_names=class_names)
    for i in range(steps):
        # zero padding to avoid confusion in the order of files with different name lengths
        filename = f"{i + steps:03d}"
        visualize_3d(df, dims, class_column=class_column, azim=(steps - 1) * stepsize, elev=stepsize * i - 164,
                     path=os.path.join(temp, f"{filename}.png"), class_names=class_names)

    files = [os.path.join(temp, file) for file in os.listdir(temp) if os.path.isfile(os.path.join(temp, file))]
    # bring files in the order they were created in
    files = sorted(files)
    img = Image.open(files[0])
    imgs = [Image.open(f) for f in files]
    filename = os.path.join(plots_folder, f"{name}-{x_name}-{y_name}-{z_name}.gif")
    if os.path.isfile(filename):
        os.remove(filename)
    img.save(fp=filename, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=0)
    #danke stefan, hierfür kommst du in meine danksagung :)
    for img in imgs:
        img.close()
    clear_temp(temp)
    os.rmdir(temp)


def create_hist(series: pd.Series, number_of_bins: int = 20, x_label: str = "") -> None:
    """
    plots given data as a Histogram with evenly wide bins.
    :param series: Data to be plotted
    :param number_of_bins: number of bins in the plot
    :param x_label: label of the x-axis of the plot
    """
    plt.clf()
    n, _, _ = plt.hist(series, bins=number_of_bins)
    plt.grid(axis='y', alpha=0.75)
    if x_label:
        plt.xlabel(x_label)
    else:
        plt.xlabel('Value')
    plt.ylabel('Frequency')
    maxfreq = n.max() * 1.1
    plt.ylim(ymax=maxfreq)
    plt.show()


def get_cumulative_values(array: List[float], fraction: bool = True) -> Tuple[List[float], List[float]]:
    """
    Calculates the cumulative frequencies of values in an Array.
    :param array: Values for which the cumulative distribution is supposed to be calculated
    :param fraction: determines, whether the result will contain an array with the cumulative frequencies of the values
    (if fraction == True), or if the indices of the last occurrence of the values in the array will be returned.
    :return: Array containing all unique values sorted. Array containing the frequency of values lower or equal to the
    value at the same index in the other array.
    """
    #avoid destroying data --> safety copy
    new_array = array.copy()
    new_array.sort()
    prev_val = new_array[0]
    values = []
    cum_frequencies = []
    for i, value in enumerate(new_array):
        # count how many values were <= the value for each value
        # count is given by pos in sorted array
        if value != prev_val:
            values.append(prev_val)
            cum_frequencies.append(i)
            prev_val = value
    # append data for last value (doesnt match condition value != prev_val)
    values.append(new_array[-1])
    length = len(new_array)
    cum_frequencies.append(length)

    if fraction:
        cum_frequencies = [cf / length for cf in cum_frequencies]
    return values, cum_frequencies


def apply_constraints(constraints: Dict[str, List[Tuple[bool, float]]], df: pd.DataFrame) -> pd.DataFrame:
    """
    applies constraints to a dataframe. Mins and Maxs for multiple dimensions can be used to limit the data that is
    shown. Does change the given Dataframe
    :param constraints: possible constraints for the data. The keys are columns of df, for which the constraints are to
    be applied. For each key, a list of Tuples can be supplied. Within the Tuple, the bool value determines, if the
    int-value will be interpreted as a maximum or a minimum value (True --> max, False --> min)
    :param df: the dataframe the constraints are applied to
    :return: the changed dataframe
    """
    for dimension, constraint_list in constraints.items():
        for constraint in constraint_list:
            max_, value = constraint
            if max_:
                df = df.loc[df[dimension] <= value]
            else:
                df = df.loc[df[dimension] >= value]
    return df


def create_cumulative_plot(df: pd.DataFrame,
                           dim: str,
                           constraints: Dict[str, List[Tuple[bool, float]]] = None,
                           path_name: None or str = None,
                           x_axis_label: str = "",
                           title: str = "") -> None:
    """
    plots 1d cumulative plot of data from df in direction of dim under the given constraints
    :param df: Data to be plotted
    :param dim: Dimension, along which the plot will be created
    :param constraints: possible constraints for the data. The keys are columns of df, for which the constraints are to
    be applied. For each key, a list of Tuples can be supplied. Within the Tuple, the bool value determines, if the
    int-value will be interpreted as a maximum or a minimum value (True --> max, False --> min)
    :param path_name: if given, the plot will be saved at this location (starting from Bachelor/DataCreation/). Plot
    will only be saved, not shown.
    :param x_axis_label: Optional label for the x_axis. If omitted, label will be dim
    :param title: Title for the plot
    """
    new_df = df.copy()
    if constraints:
        # apply the constaints to the data by limiting the dimensions
        new_df = apply_constraints(constraints, new_df)

    values, cum_frequencies = get_cumulative_values(new_df[dim].values)
    plt.clf()
    plt.title(title)

    if x_axis_label:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(dim)
    plt.ylabel("Cumulative Frequency")
    plt.plot(values, cum_frequencies)
    if path_name:
        path_here = os.path.dirname(__file__)
        plt.savefig(os.path.join(path_here, path_name))
    else:
        plt.show()


def create_save_path(path):
    path = os.path.normpath(path)
    folders = path.split(os.sep)
    for i in range(1, len(folders)):
        curr_path = os.sep.join(folders[:i])
        if not os.path.isdir(curr_path):
            os.mkdir(curr_path)


def visualize_model_predictions(dataset: dc.Data, pred_col_name: str) -> str:
    """
    visualizes data (original vs predicted) as multiple 2d pictures as well as the tree as a diagram
    :param dataset: data to be visualized
    :param pred_col_name: name, where the predicted classes are stored
    """
    df = dataset.data
    pics_path = os.path.join(dataset.path, "pics")
    if not os.path.isdir(pics_path):
        os.mkdir(pics_path)
    final_pics_path = os.path.join(pics_path, "Classifier")
    if not os.path.isdir(final_pics_path):
        os.mkdir(final_pics_path)

    # check if all pictures are already in the folder:
    make_pics = False
    files_in_pics = os.listdir(final_pics_path)
    if isinstance(dataset, dc.MaybeActualDataSet):
        pics = ["00_04_org.png", "00_04_pred.png", "01_04_org.png", "01_04_pred.png", "02_03_org.png", "02_03_pred.png"]
        dim0, dim1, dim2, dim3, dim4, dim5 = "dim_00", "dim_04", "dim_01", "dim_04", "dim_02", "dim_03"
        dims = [(dim0, dim1), (dim2, dim3), (dim4, dim5)]
    elif isinstance(dataset, dc.IrisDataSet):
        pics = ["sepal_length_width_org.png", "sepal_length_width_pred.png",
                "petal_length_width_org.png", "petal_length_width_pred.png",
                "sepal_length_petal_width_org.png", "sepal_length_petal_width_pred.png"]
        dim0, dim1, dim2 = "sepal_length", "sepal_width", "petal_length"
        dim3, dim4, dim5 = "petal_width", "sepal_length", "petal_width"
        dims = [(dim0, dim1), (dim2, dim3), (dim4, dim5)]
    elif isinstance(dataset, dc.SoccerDataSet):
        pics = ["gefoult_laufweite_org.png", "gefoult_laufweite_pred.png",
                "pass_laufweite_org.png", "pass_laufweite_pred.png",
                "pass_zweikampfprozente_org.png", "pass_zweikampfprozente_pred.png"]
        dim0, dim1, dim2 = "ps_Gefoult", "ps_Laufweite", "ps_Pass"
        dim3, dim4, dim5 = "ps_Laufweite", "ps_Pass", "Zweikampfprozente"
        dims = [(dim0, dim1), (dim2, dim3), (dim4, dim5)]
    elif isinstance(dataset, dc.BinningDataSet):
        pics = ["test.png", "test_pred.png"]
        dims = [("X", "Y")]
    else:
        raise dc.CustomError(f"unknown Dataset type: {type(dataset)}")
    for pic in pics:
        if pic not in files_in_pics:
            make_pics = True
            break

    if make_pics:
        for i, pair in enumerate(dims):
            visualize_2d(df=df, dims=(pair[0], pair[1]), class_column="classes", title="original",
                         path=os.path.join(final_pics_path, pics[2 * i]), class_names=dataset.class_names)
            visualize_2d(df=df, dims=(pair[0], pair[1]), class_column=pred_col_name, title="predicted",
                         path=os.path.join(final_pics_path, pics[(2 * i) + 1]), class_names=dataset.class_names)

    return final_pics_path


def compare_shift_cumulative(df: pd.DataFrame,
                             dims: Tuple[str, str],
                             shift: float,
                             constraints: Dict[str, List[Tuple[bool, float]]] = None,
                             save_path: None or str = None,
                             x_axis_labels: Tuple[str] = None,
                             titles: Tuple[str, str] = None) -> None:
    """
    compare the cumulated distribution functions for one dimension before and after being shifted.
    :param df: Dataframe containing the data
    :param dims: names of the columns to compare before and after the shift
    :param shift: "distance" the data was shifted (as quantile)
    :param constraints: optional constraints to apply to the data
    :param save_path: Path where the resulting figures will be saved. If omitted, the figures will only be shown, not
    saved
    :param x_axis_labels: labels for the x_axes in both plots
    :param titles: title for the plots
    """
    new_df = df.copy()
    if constraints:
        # apply the constaints to the data by limiting the dimensions
        new_df = apply_constraints(constraints, new_df)

    values_0, cum_frequencies_0 = get_cumulative_values(new_df[dims[0]].values)
    values_1, cum_frequencies_1 = get_cumulative_values(new_df[dims[1]].values)

    cum_frequencies_1 = [elem + shift for elem in cum_frequencies_1]

    if shift < 0:
        new_first = values_0[0] - 0.000001 * abs(values_0[0])
        values_1.insert(0, new_first)
        cum_frequencies_1.insert(0, shift)

    values = [values_0, values_1]
    cum_frequencies = [cum_frequencies_0, cum_frequencies_1]

    x_axis_min, x_axis_max, y_axis_min, y_axis_max = find_common_area(values_0, cum_frequencies_0, values_1,
                                                                      cum_frequencies_1)
    for i in range(2):
        plt.clf()
        if titles:
            plt.title(titles[i])

        if x_axis_labels:
            plt.xlabel(x_axis_labels[i])
        else:
            plt.xlabel(dims[i])

        plt.xlim(x_axis_min, x_axis_max)
        plt.ylim(y_axis_min, y_axis_max)

        plt.ylabel("Cumulative Frequency")
        plt.plot(values[i], cum_frequencies[i])

        if save_path:
            create_save_path(save_path + str(i))
            actual_path = save_path.split(".png")[0]
            actual_path = f"{actual_path}_{str(i)}.png"
            plt.savefig(actual_path)
        else:
            plt.show()


def compare_splits_cumulative(df0: pd.DataFrame,
                              df1: pd.DataFrame,
                              dim: str,
                              path: None or str = None,
                              x_axis_label: str = None,
                              title: str = None
                              ) -> None:
    """
    Plots a cumulative graph for a dimension from both datasets in a common plot.
    :param df0: first dataset
    :param df1: second dataset
    :param dim: dimension for which the cumulative graphs are generated
    :param path: potential name and path for a location to save the resulting plot
    :param x_axis_label: label for the x-axis. If no label is given, dim will be used as the label
    :param title: title for the plot
    """
    new_df0 = df0.copy()
    new_df1 = df1.copy()

    values_0, cum_frequencies_0 = get_cumulative_values(new_df0[dim].values)
    values_1, cum_frequencies_1 = get_cumulative_values(new_df1[dim].values)

    x_axis_min, x_axis_max, y_axis_min, y_axis_max = find_common_area(values_0, cum_frequencies_0, values_1,
                                                                      cum_frequencies_1)

    #plotting starts here
    plt.clf()
    if title:
        plt.title(title)

    if x_axis_label:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(dim)

    plt.ylabel("Cumulative Frequency")

    plt.xlim(x_axis_min, x_axis_max)
    plt.ylim(y_axis_min, y_axis_max)

    plt.plot(values_0, cum_frequencies_0, color=colors[0])
    plt.plot(values_1, cum_frequencies_1, color=colors[1])

    if path:
        path_here = os.path.dirname(__file__)
        plt.savefig(os.path.join(path_here, path))
    else:
        plt.show()


def get_number_of_changed_points(data: pd.DataFrame, dims: Tuple[str, str]) -> int:
    """
    Compares the columns given in dims and returns the count of differences.
    :param data: data
    :param dims: column names for the columns to be compared
    :return: count of rows in which the two columns dont shwo the same value
    """
    count = data.loc[data[dims[0]] != data[dims[1]]].count()["classes"]
    return count


def get_change_matrix(data: pd.DataFrame, dims: Tuple[str, str], class_names: List[str] = None) -> pd.DataFrame:
    """
    returns a matrix showing the migration between classes.
    :param data: data
    :param dims: names of columns that represent different classifications of the data
    :param class_names: names of the classes, in order to sort the results and add empty rows and columns for classes,
    that are not present
    :return: Migration Matrix
    """
    col1, col2 = dims
    if class_names:
        present_classes = class_names
    else:
        present_classes = set(data[col1].values)
        present_classes = present_classes.union(set(data[col2].values))
    index = []

    # initialize dictionary
    data_dict = {}
    for class_ in present_classes:
        data_dict[class_] = []

    for org_class in present_classes:
        index.append(org_class)
        #filter data for their value in first col
        original_class_data = data.loc[data[col1] == org_class]
        for new_class in present_classes:
            #count occurrences of the classes in the 2. column
            count = original_class_data.loc[original_class_data[col2] == new_class].count()[col1]
            data_dict[new_class].append(count)
    res_matrix = pd.DataFrame(data_dict, index=index)
    return res_matrix


def get_unique_handles(all_handles: List):
    labels = []
    result = []
    for handle in all_handles:
        label = handle.get_label()
        if label not in labels:
            labels.append(label)
            result.append(handle)
    return result


def main() -> None:
    """
    just a test function
    """
    d_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\220715_153628_BinningDataSet")
    compare_shift_2d(df=d_set.data, common_dim="Y", dims_to_compare=("X", "X_shifted_by_0.1"), class_columns=("classes", "classes"),
                     path=r"D:\Gernot\Programmieren\Bachelor\Data\220715_153628_BinningDataSet\pics\QSM\vanilla\X\shift_test.png",
                     class_names=d_set.class_names)
    # visualize_3d(df, ("dim_01", "dim_02", "dim_03"))
    # for i in range(36):
    #    visualize_3d(df, ("dim_01", "dim_02", "dim_03"), class_column="classes", azim=10*i, elev=-150)


if __name__ == "__main__":
    #create_save_path(r"D:\Gernot\Programmieren\Bachelor\Data\220330_220119_MaybeActualDataSet\pics\QSM\test.csv")
    #maybeActualDataSet_figure()
    #soccerDataSet_figure()
    #Iris_p_val_figure()
    main()
    #soccer_QSM_comparison_figure_2()
    #Iris_QSM_comparison_figure()
    #parallelization_figure()
    #print("\\".join())
