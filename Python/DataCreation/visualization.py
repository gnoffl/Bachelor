from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import random
import glob
from PIL import Image
import os

colors = [(1, 0, 0, 0.9),
          (0, 1, 0, 0.9),
          (0, 0, 1, 0.9),
          (1, 1, 0, 0.9),
          (1, 0, 1, 0.9),
          (0, 1, 1, 0.9),
          (0.6, 0.6, 0.6, 0.9),
          (0.1, 0.1, 0.1, 0.9),
          (.7, .7, .3, 0.9),
          (.7, .3, .7, 0.9),
          (.3, .7, .7, 0.9),
          ]


#todo: scheint nicht zu funktionieren mit mehreren klassen?
#first entry of dims will be x-axis, second will be y-axis
def visualize_2d(df: pd.DataFrame, dims: Tuple[str, str], class_column: str or None = None, title: str = None, path: str = None) -> None:
    """
    Creates a 2d Image of the given data, showing the dimensions whose names are given in dims.
    :param df: Dataframe containing the data
    :param dims: Tuple of the names of the columns that are to be the axes of the plot
    :param class_column: Name of the column that contains class names. Will be used for labeling the data
    :param title: Title of the plot
    :param path: If path is given, the plot will not be shown but instead saved at the given location
    """
    x_name, y_name = dims
    plt.figure(0, figsize=(8, 6))
    plt.clf()

    x_min, x_max = df[x_name].min(), df[x_name].max()
    y_min, y_max = df[y_name].min(), df[y_name].max()

    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)

    x_axis_min = x_min - x_pad
    x_axis_max = x_max + x_pad
    y_axis_min = y_min - y_pad
    y_axis_max = y_max + y_pad

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
            plt.scatter(df.loc[df[class_column] == clas, [x_name]],
                        df.loc[df[class_column] == clas, [y_name]],
                        c=colors[i],
                        edgecolor="k",
                        label=clas)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    else:
        plt.scatter(df[x_name], df[y_name], cmap=plt.cm.Set1, edgecolor="k")
    plt.subplots_adjust(right=0.87)
    if path:
        plt.savefig(path)
    else:
        plt.show()


def visualize_3d(df: pd.DataFrame,
                 dims: Tuple[str, str, str],
                 azim: int = 110,
                 elev: int = -150,
                 class_column: str or None = None,
                 path: str = "") -> None:
    """
    Creates a 3d Image of the given data, showing the dimensions whose names are given in dims.
    :param df: Dataframe containing the data
    :param dims: Tuple of the names of the columns that are to be the axes of the plot
    :param azim: not sure about the definition, but changes rotation
    :param elev: not sure about the definition, but changes rotation
    :param class_column: Name of the column that contains class names. Will be used for labeling the data
    :param path: If path is given, the plot will not be shown but instead saved at the given location
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
                c=colors[i],
                edgecolor="k",
                label=clas,
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
    files = [os.path.join(path_to_temp, file) for file in os.listdir(path_to_temp) if os.path.isfile(os.path.join(path_to_temp, file))]
    for file in files:
        os.remove(file)


#todo: maybe keep images in memory instead of saving them to disc.
#todo: maybe save as .mp4 or smth other than .gif
def create_3d_gif(df: pd.DataFrame, dims: Tuple[str, str, str], name: str, class_column: str or None = None,
                  steps: int = 36, duration: int = None) -> None:
    """
    creates a gif of the data with the given dims being the shown axes of the 3d plot
    :param df: Dataframe containing the data
    :param dims: Name of the columns of df which will be displayed as the axes
    :param name: Will be the beginning of the filename. filename will also include the names of the dims.
    :param class_column: Column of df that contains class labels. Will be used to label classes in the plot.
    :param steps: Number of pictures per rotation of the data. The more steps, the more fluid the gif becomes
    :param duration: Milliseconds each frame is shown
    """
    # keeps duration for the whole gif constant --> more fluid visual if steps is increased
    if not duration:
        duration = 36 * 100 / steps
    x_name, y_name, z_name = dims
    plots_folder = os.path.join(os.path.dirname(__file__), "Plots")

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
                     path=os.path.join(temp, f"{filename}.png"))
    for i in range(steps):
        # zero padding to avoid confusion in the order of files with different name lengths
        filename = f"{i + steps:03d}"
        visualize_3d(df, dims, class_column=class_column, azim=(steps - 1) * stepsize, elev=stepsize * i - 164,
                     path=os.path.join(temp, f"{filename}.png"))

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


def get_cumulative_values(array: List[int]) -> Tuple[List[int], List[float]]:
    """
    Calculates the cumulative frequencies of values in an Array.
    :param array: Values for which the cumulative distribution is supposed to be calculated
    :return: Array containing all unique values sorted. Array containing the frequency of values lower or equal to the
    value at the same index in the other array.
    """
    #avoid destroying data --> safety copy
    new_array = array[:]
    new_array.sort()
    last_val = new_array[0]
    values = []
    cum_frequencies = []
    for i, value in enumerate(new_array):
        # count how many values were <= the value for each value
        # count is given by pos in sorted array
        if value != last_val:
            values.append(last_val)
            cum_frequencies.append(i)
            last_val = value
    # append data for last value (doesnt match condition value != last_val)
    values.append(new_array[-1])
    length = len(new_array)
    cum_frequencies.append(length)

    cum_frequencies = [cf / length for cf in cum_frequencies]
    return values, cum_frequencies


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
    :param constraints: possible constraints for the data. The keys are columns of df, for which the constraints are to be applied.
    For each key, a list of Tuples can be supplied. Within the Tuple, the bool value determines, if the int-value will be interpreted
    as a maximum or a minimum value (True --> max, False --> min)
    :param path_name: if given, the plot will be saved at this location (starting from Bachelor/DataCreation/). Plot will only be saved, not shown.
    :param x_axis_label: Optional label for the x_axis. If omitted, label will be dim
    :param title: Title for the plot
    """
    new_df = df.copy()
    if constraints:
        # apply the constaints to the data by limiting the dimensions
        for dimension, constraint_list in constraints.items():
            for constraint in constraint_list:
                max_, value = constraint
                if max_:
                    new_df = new_df.loc[new_df[dimension] <= value]
                else:
                    new_df = new_df.loc[new_df[dimension] >= value]

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


def get_number_of_changed_points(data: pd.DataFrame, dims: Tuple[str, str]) -> int:
    """
    Compares the columns given in dims and returns the count of differences.
    :param data: data
    :param dims: column names for the columns to be compared
    :return: count of rows in which the two columns dont shwo the same value
    """
    count = data.loc[data[dims[0]] != data[dims[1]]].count()["classes"]
    return count


def get_change_matrix(data, dims: Tuple[str, str]) -> pd.DataFrame:
    """
    returns a matrix showing the migration between classes.
    :param data: data
    :param dims: names of columns that represent dirrent classifications of the data
    :return: Migration Matrix
    """
    col1, col2 = dims
    present_classes = set(data[col1].values)
    present_classes.union(set(data[col2]))
    index = []
    res_matrix = pd.DataFrame

    # initialize dictionary
    data_dict = {}
    for class_ in present_classes:
        data_dict[class_] = []

    for org_class in present_classes:
        index.append(org_class)
        #filter data for their value in first col
        original_class_data = data.loc[data[col1] == org_class]
        for new_class in present_classes:
            #count occurences of the classes in the 2. column
            count = original_class_data.loc[original_class_data[col2] == new_class].count()[col1]
            data_dict[new_class].append(count)
    res_matrix = pd.DataFrame(data_dict, index=index)
    return res_matrix


def main() -> None:
    """
    just a test function
    """
    df = pd.DataFrame()
    classes = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    dim_01 = [1, 2, 3, 4, 5, 12, 13, 15, 12, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    dim_02 = [5, 6, 7, 9, 4, .1, .4, .5, .6, .2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    dim_03 = [3, 5, 8, 1, 4, 7, 9, 5, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    df["classes"] = classes
    df["dim_01"] = dim_01
    df["dim_02"] = dim_02
    df["dim_03"] = dim_03
    # visualize_2d(df, ("dim_01", "dim_02"))
    visualize_2d(df, ("dim_01", "dim_03"), "classes")
    # visualize_3d(df, ("dim_01", "dim_02", "dim_03"))
    # for i in range(36):
    #    visualize_3d(df, ("dim_01", "dim_02", "dim_03"), class_column="classes", azim=10*i, elev=-150)


if __name__ == "__main__":
    # main()
    print(os.listdir("Plots/temp/"))
