from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import random


#first entry of dims will be x-axis, second will be y-axis
def visualize_2d(df: pd.DataFrame, dims: Tuple[str, str], class_column: str or None=None):
    x_name, y_name = dims
    plt.figure(0, figsize=(5.5, 4))
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
    if class_column:
        plt.scatter(df[x_name], df[y_name], c=df[class_column], cmap=plt.cm.Set1, edgecolor="k")
    else:
        plt.scatter(df[x_name], df[y_name], cmap=plt.cm.Set1, edgecolor="k")
    plt.show()


def create_3d_plot(df: pd.DataFrame, dims: Tuple[str, str, str], azim: int = 110, elev: int = -150, class_column: str or None = None):
    x_name, y_name, z_name = dims
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim, auto_add_to_figure=False)
    fig.add_axes(ax)
    if class_column:
        ax.scatter(
            df[x_name],
            df[y_name],
            df[z_name],
            c=df[class_column],
            cmap=plt.cm.Set1,
            edgecolor="k",
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
    plt.show()


def visualize_3d(df: pd.DataFrame, dims: Tuple[str, str, str], azim: int or None = None, elev: int = -150,  class_column: str or None=None):
    if isinstance(azim, int):
        create_3d_plot(df, dims, azim, elev, class_column)
    else:
        azim = 110
        create_3d_plot(df, dims, azim, elev, class_column)
        create_3d_plot(df, dims, azim - 90, elev, class_column)


def main():
    df = pd.DataFrame()
    classes = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    dim_01 = [1, 2, 3, 4, 5, 12, 13, 15, 12, 11]
    dim_02 = [5, 6, 7, 9, 4, .1, .4, .5, .6, .2]
    dim_03 = [3, 5, 8, 1, 4, 7, 9, 5, 3, 2]
    df["classes"] = classes
    df["dim_01"] = dim_01
    df["dim_02"] = dim_02
    df["dim_03"] = dim_03
    #visualize_2d(df, ("dim_01", "dim_02"))
    #visualize_2d(df, ("dim_01", "dim_03"), "classes")
    #visualize_3d(df, ("dim_01", "dim_02", "dim_03"))
    for i in range(36):
        visualize_3d(df, ("dim_01", "dim_02", "dim_03"), class_column="classes", azim=10*i, elev=-150)




if __name__ == "__main__":
    main()

