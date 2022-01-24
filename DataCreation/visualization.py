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
# first entry of dims will be x-axis, second will be y-axis
def visualize_2d(df: pd.DataFrame, dims: Tuple[str, str], class_column: str or None = None, title: str = None):
    x_name, y_name = dims
    plt.figure(0, figsize=(6, 4))
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
        for i, clas in enumerate(classes):
            plt.scatter(df.loc[df[class_column] == clas, [x_name]],
                        df.loc[df[class_column] == clas, [y_name]],
                        c=colors[i],
                        edgecolor="k",
                        label=clas)
        plt.legend(bbox_to_anchor=(1, 1.05))
    else:
        plt.scatter(df[x_name], df[y_name], cmap=plt.cm.Set1, edgecolor="k")
    plt.show()


def create_3d_plot(df: pd.DataFrame,
                   dims: Tuple[str, str, str],
                   azim: int = 110,
                   elev: int = -150,
                   class_column: str or None = None,
                   save: bool = False,
                   path: str = ""):
    x_name, y_name, z_name = dims
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)
    fig.add_axes(ax)
    if class_column:
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
    if save:
        plt.savefig(path)
    else:
        plt.show()


def visualize_3d(df: pd.DataFrame, dims: Tuple[str, str, str], azim: int or None = None, elev: int = -150,
                 class_column: str or None = None, save: bool = False, path: str = ""):
    if isinstance(azim, int):
        create_3d_plot(df, dims, azim, elev, class_column, save, path)
    else:
        azim = 110
        create_3d_plot(df, dims, azim, elev, class_column, save, path)
        create_3d_plot(df, dims, azim - 90, elev, class_column, save, path)


def clear_temp(path_to_temp):
    files = [os.path.join(path_to_temp, file) for file in os.listdir(path_to_temp) if os.path.isfile(os.path.join(path_to_temp, file))]
    for file in files:
        os.remove(file)


def find_correct_path_to_():
    pass


#todo: maybe keep images in memory instead of saving them to disc.
#todo: maybe save as .mp4 or smth other than .gif
def create_3d_gif(df: pd.DataFrame, dims: Tuple[str, str, str], name: str, class_column: str or None = None,
                  steps: int = 36, duration=None):
    if not duration:
        duration = 36 * 100 / steps
    x_name, y_name, z_name = dims
    plots_folder = os.path.join(os.path.dirname(__file__), "Plots")

    temp = os.path.join(plots_folder, "temp")
    if os.path.isdir(temp):
        clear_temp(temp)
    else:
        os.mkdir(temp)

    stepsize = int(360 / steps)

    for i in range(steps):
        filename = f"{i:03d}"
        create_3d_plot(df, dims, class_column=class_column, azim=stepsize * i, elev=-164, save=True,
                       path=os.path.join(temp, f"{filename}.png"))
    for i in range(steps):
        filename = f"{i + steps:03d}"
        create_3d_plot(df, dims, class_column=class_column, azim=(steps - 1) * stepsize, elev=stepsize * i - 164,
                       save=True,
                       path=os.path.join(temp, f"{filename}.png"))

    files = [os.path.join(temp, file) for file in os.listdir(temp) if os.path.isfile(os.path.join(temp, file))]
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


def create_hist(series: pd.Series) -> None:
    plt.clf()
    n, _, _ = plt.hist(series, bins=20)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    maxfreq = n.max() * 1.1
    plt.ylim(ymax=maxfreq)
    plt.show()


def get_cumulative_values(array: List[int]) -> Tuple[List[int], List[int]]:
    array.sort()
    last_val = array[0]
    values = []
    cum_frequencies = []
    for i, value in enumerate(array):
        if value != last_val:
            values.append(last_val)
            cum_frequencies.append(i)
            last_val = value
    values.append(array[-1])
    length = len(array)
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


def get_number_of_changed_points(data: pd.DataFrame) -> Tuple[int, int]:
    count = data.loc[data["classes"] != data["predicted_classes"]].count()["classes"]
    length = len(data)
    return count, length


def get_change_matrix(data) -> pd.DataFrame:
    present_classes = set(data["classes"].values)
    present_classes.union(set(data["predicted_classes"]))
    data_dict = {}
    index = []
    res_matrix = pd.DataFrame
    for class_ in present_classes:
        data_dict[class_] = []
    for org_class in present_classes:
        index.append(org_class)
        original_class_data = data.loc[data["classes"] == org_class]
        for new_class in present_classes:
            count = original_class_data.loc[original_class_data["predicted_classes"] == new_class].count()["classes"]
            data_dict[new_class].append(count)
        res_matrix = pd.DataFrame(data_dict, index=index)
    return res_matrix


def main():
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


#todo: create something to visualize 1d Data distributions (histograms or cumulative thingies)
if __name__ == "__main__":
    # main()
    print(os.listdir("Plots/temp/"))
