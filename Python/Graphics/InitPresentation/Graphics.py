from typing import List, Tuple

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import numpy as np


def segment_new_plot(iris):
    plt.figure(0, figsize=(5.5, 4))
    plt.clf()
    ax = plt.axes()
    rect1 = mpl.patches.Rectangle((-1, 0), 4, 2.3, linewidth=0, facecolor=(1, 0, 0, 0.2))
    ax.add_patch(rect1)
    rect2 = mpl.patches.Rectangle((-1, 2.3), 2.75, 2.65, linewidth=0, facecolor=(1, 1, 0, 0.2))
    ax.add_patch(rect2)
    rect3 = mpl.patches.Rectangle((-1, 4.95), 2.75, 3, linewidth=0, facecolor=(0, 0, 0, 0.2))
    ax.add_patch(rect3)
    rect4 = mpl.patches.Rectangle((1.75, 2.3), 3, 8, linewidth=0, facecolor=(0, 0, 0, 0.2))
    ax.add_patch(rect4)
    X: pd.DataFrame = iris.data[['petal length (cm)', 'petal width (cm)']]


    x_pad = 0.2
    y_pad = 1
    length_min, length_max = X["petal length (cm)"].min() - 0.5, X["petal length (cm)"].max() + y_pad
    width_min, width_max = X["petal width (cm)"].min() - x_pad, X["petal width (cm)"].max() + x_pad

    plt.xlim(width_min, width_max)
    plt.ylim(length_min, length_max)
    plt.xlabel("Petal width (cm)")
    plt.ylabel("Petal length (cm)")


def create_binning_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(1)
    data1 = pd.DataFrame()
    data1["X"] = [random.uniform(0, 1) for i in range(100)]
    data1["Y"] = [random.uniform(0, 2) for i in range(100)]
    data2 = pd.DataFrame()
    data2["X"] = [random.uniform(1, 2) for i in range(100)]
    data2["Y"] = [random.uniform(-1, 0) for i in range(100)]
    data3 = pd.DataFrame()
    data3["X"] = [random.uniform(2, 3) for i in range(100)]
    data3["Y"] = [random.uniform(0, 2) for i in range(100)]
    return data1, data2, data3


def binning_plot():
    data = pd.concat(create_binning_data())
    plt.figure(0, figsize=(5.5, 4))
    plt.clf()

    pad = 0.3
    x_min, x_max = data["X"].min() - pad, data["X"].max() + pad
    y_min, y_max = data["Y"].min() - pad, data["Y"].max() + pad

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(data["X"], data["Y"])
    plt.show()
    #plt.savefig("C:/Users/gerno/OneDrive/Bachelorarbeit/Intro_Vortrag/Grafiken/binning.png")


def binning_plot_adv():
    data = pd.concat(create_binning_data())
    plt.clf()
    plt.figure(0, figsize=(5.5, 7.5))

    plot1 = plt.subplot2grid((32, 1), (0, 0), rowspan=18)
    plot2 = plt.subplot2grid((32, 1), (22, 0), rowspan=10)

    pad = 0.3
    x_min, x_max = data["X"].min() - pad, data["X"].max() + pad
    y_min, y_max = data["Y"].min() - pad, data["Y"].max() + pad

    #fill first plot
    plot1.axis(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max)
    plot1.set_xlabel("X")
    plot1.set_ylabel("Y")
    plot1.scatter(data["X"], data["Y"])
    plot1.set_title("A")
    # Create a Rectangle patch
    rect = mpl.patches.Rectangle((-0.1, .5), 3.2, 1, linewidth=1, edgecolor='none', facecolor=(.3, .3, .3, .3))
    # Add the patch to plot1
    plot1.add_patch(rect)

    #fill second plot
    plot2.set_title("B")
    plot2.axis(xmin=x_min, xmax=x_max, ymin=-0.3, ymax=1.3)
    plot2.set_yticks([0, 0.5, 1], labels=["0", "", "1"])
    plot2.set_xlabel("X")
    plot2.set_ylabel("Dichte")
    plot2.plot([0, 3], [1, 1])
    plot2.plot([0, 1, 1, 2, 2, 3], [.5, .5, 0, 0, .5, .5], color=(.3, .3, .3, .6))
    plt.show()
    #plt.savefig(r"D:\Gernot\Programmieren\Bachelor\Plots\BA_Grafiken/binning_adv.png")


def binning_plot_actually_binned():
    data1, data2, data3 = create_binning_data()
    data = pd.concat([data1, data2, data3])
    plt.clf()
    plt.figure(0, figsize=(5.5, 7.5))

    plot1 = plt.subplot2grid((32, 1), (0, 0), rowspan=18)
    plot2 = plt.subplot2grid((32, 1), (22, 0), rowspan=10)

    pad = 0.3
    x_min, x_max = data["X"].min() - pad, data["X"].max() + pad
    y_min, y_max = data["Y"].min() - pad, data["Y"].max() + pad

    #fill first plot
    plot1.axis(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max)
    plot1.set_xlabel("X")
    plot1.set_ylabel("Y")
    plot1.scatter(data1["X"], data1["Y"], color=(0, 0, 1, 1))
    plot1.scatter(data3["X"], data3["Y"], color=(0, 0, 1, 1))
    plot1.scatter(data2["X"], data2["Y"], color=(1, 0, 0, 1))
    plot1.set_title("A")

    #fill second plot
    plot2.set_title("B")
    plot2.axis(xmin=x_min, xmax=x_max, ymin=-0.3, ymax=1.3)
    plot2.set_yticks([0, 0.5, 1], labels=["0", "", "1"])
    plot2.set_xlabel("X")
    plot2.set_ylabel("Dichte")
    plot2.plot([0, 1.01, 1.01, 2.01, 2.01, 3], [1, 1, 0, 0, 1, 1], color=(0, 0, 1, 1))
    plot2.plot([0, 1, 1, 2, 2, 3], [0, 0, 1, 1, 0, 0], color=(1, 0, 0, 1))
    plt.show()
    #plt.savefig(r"D:\Gernot\Programmieren\Bachelor\Plots\BA_Grafiken/binning_actually_binned.png")


def org_iris(iris):
    y = iris.target
    X: pd.DataFrame = iris.data[['petal width (cm)', 'petal length (cm)']]

    segment_new_plot(iris)

    plt.scatter(X["petal width (cm)"], X["petal length (cm)"], c=y, cmap=plt.cm.Set1, edgecolor="k")
    #plt.show()
    plt.savefig("C:/Users/gerno/OneDrive/Bachelorarbeit/Abschluss_Vortrag/iris_segmented.png", bbox_inches='tight')


def shifted_iris(iris):
    y = iris.target
    X: pd.DataFrame = iris.data[['petal width (cm)', 'petal length (cm)']]

    segment_new_plot(iris)

    X["petal length (cm)"] += 0.5
    plt.scatter(X["petal width (cm)"], X["petal length (cm)"], c=y, cmap=plt.cm.Set1, edgecolor="k")
    #plt.show()
    plt.savefig("C:/Users/gerno/OneDrive/Bachelorarbeit/Abschluss_Vortrag/iris_segmented_shifted.png", bbox_inches='tight')


def create_binning_data_paper_plot() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(1)
    data1 = pd.DataFrame()
    data1["X"] = [random.uniform(0, 1) for i in range(100)]
    data1["Y"] = [random.uniform(1.04, 3) for i in range(100)]
    data2 = pd.DataFrame()
    data2["X"] = [random.uniform(1, 2) for i in range(100)]
    data2["Y"] = [random.uniform(0, 0.96) for i in range(100)]
    data3 = pd.DataFrame()
    data3["X"] = [random.uniform(2, 3) for i in range(100)]
    data3["Y"] = [random.uniform(1.04, 3) for i in range(100)]
    return data1, data2, data3


def fill_scatter_subplot(scatter_plot, data_separated, colors):
    data = pd.concat(data_separated)
    bin1 = pd.concat([data_separated[0], data_separated[2]])
    bin2 = data_separated[1]
    pad = 0.1
    x_min, x_max = data["X"].min() - pad, data["X"].max() + pad
    y_min, y_max = data["Y"].min() - pad, data["Y"].max() + pad

    scatter_plot.set_xlim(x_min, x_max)
    scatter_plot.set_ylim(y_min, y_max)
    scatter_plot.set_xlabel("X", fontsize=7)#, labelpad=1)
    scatter_plot.set_ylabel("Y", fontsize=7)#, labelpad=1)
    scatter_plot.set_title("example data set", fontsize=7)

    scatter_plot.scatter(bin1["X"], bin1["Y"], s=10, color=colors[0], label="bin 1", zorder=1)
    scatter_plot.scatter(bin2["X"], bin2["Y"], s=10, color=colors[1], label="bin 2", zorder=1)
    #scatter_plot.legend(fontsize=7, bbox_to_anchor=(0.5, 0.8), frameon=True)

    x1, y1 = [-5, 5], [1, 1]
    scatter_plot.plot(x1, y1, marker='o', color="black", linewidth=1, linestyle="dashed")

    scatter_plot.arrow(0.8, 2.25, 0.9, 0, width=0.005, color="black", head_width=0.075, length_includes_head=True, zorder=20)
    scatter_plot.arrow(0.8, 2.15, 1.6, 0, width=0.005, color="blue", head_width=0.075, length_includes_head=True, zorder=20)

    scatter_plot.scatter([1.75], [2.25], s=10, color=colors[3], zorder=10)
    scatter_plot.scatter([2.45], [2.15], s=10, color=colors[2], zorder=10)

    #scatter_plot.arrow(1.2, 2.25, 0.6, 0, width=0.005, color="black", head_width=0.075, length_includes_head=True)
    #scatter_plot.arrow(0.8, 2.15, 1.4, 0, width=0.005, color=colors[0], head_width=0.075, length_includes_head=True)

    scatter_plot.set_yticks([0, 1, 2, 3])
    scatter_plot.set_xticks([0, 1, 2, 3])
    return scatter_plot


def fill_ecdf_subplot(ecdf_plot, colors):
    ecdf_plot.plot([0, 1, 2, 3], [0, 0.5, 0.5, 1], zorder=5, label="bin 1")
    ecdf_plot.plot([0, 1, 2, 3], [0, 0, 1, 1], zorder=0, label="bin 2")
    ecdf_plot.plot([0, 3], [0, 1], color="black", zorder=1, label="full dataset")
    ecdf_plot.legend(fontsize=7, frameon=False)

    ecdf_plot.arrow(0.81, 0.26, 0.9, 0, color="black", width=0.009, length_includes_head=True, head_width=0.02, head_length=0.08, linewidth=0.001, zorder=10)
    ecdf_plot.arrow(1.71, 0.26, 0, 0.3, color="black", width=0.03, length_includes_head=True, head_width=0.07, head_length=0.023, linewidth=0.001, zorder=10)

    ecdf_plot.arrow(0.81, 0.393, 1.6, 0, color="blue", width=0.009, length_includes_head=True, head_width=0.02, head_length=0.08, linewidth=0.001, zorder=10)
    ecdf_plot.arrow(2.41, 0.393, 0, 0.3, color="blue", width=0.03, length_includes_head=True, head_width=0.07, head_length=0.023, linewidth=0.001, zorder=10)

    """ecdf_plot.arrow(1.2, 0.4, 0.6, 0, color="black", width=0.005, length_includes_head=True, head_width=0.01,
                    head_length=0.1, linewidth=0.001, zorder=10)
    ecdf_plot.arrow(1.8, 0.4, 0, 0.2, color="black", width=0.03, length_includes_head=True, head_width=0.07,
                    head_length=0.013, linewidth=0.001, zorder=10)

    ecdf_plot.arrow(0.8, 0.4, 1.4, 0, color=colors[0], width=0.005, length_includes_head=True, head_width=0.01,
                    head_length=0.1, linewidth=0.001, zorder=10)
    ecdf_plot.arrow(2.2, 0.4, 0, 0.2, color=colors[0], width=0.03, length_includes_head=True, head_width=0.07,
                    head_length=0.013, linewidth=0.001, zorder=10)"""
    ecdf_plot.set_title("ecdf", fontsize=7)
    ecdf_plot.set_xlabel("X", fontsize=7, labelpad=1)
    ecdf_plot.set_ylabel("Cumulative Frequency", fontsize=7, labelpad=1)
    ecdf_plot.set_xticks([0, 1, 2, 3])
    ecdf_plot.set_yticks([0, .2, .4, .6, .8, 1])  # , ["0", ".2", ".4", ".6", ".8", "1"])


def binning_plot_paper():
    plt.clf()
    plt.figure(0, figsize=(5, 2.25))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    plt.rcParams["font.family"] = "Times New Roman"
    mpl.rc('xtick', labelsize=7)
    mpl.rc('ytick', labelsize=7)

    scatter_plot = plt.subplot2grid((1, 100), (0, 0), rowspan=1, colspan=43)
    ecdf_plot = plt.subplot2grid((1, 100), (0, 57), rowspan=1, colspan=43)

    data_raw = create_binning_data_paper_plot()
    fill_scatter_subplot(scatter_plot, data_raw, colors)
    fill_ecdf_subplot(ecdf_plot, colors)
    #plt.show()
    plt.savefig("../../../Plots/Paper_Grafiken/binning.pdf", bbox_inches='tight')


def get_dataset_split() -> pd.DataFrame:
    frame = pd.DataFrame()
    first_dim = [0.01*i for i in range(100)]
    random.seed(1)
    first_dim.extend([random.uniform(1, 2) for _ in range(100)])
    second_dim = [random.uniform(0, 1) for _ in range(100)]
    second_dim.extend([random.uniform(1, 2) for _ in range(100)])
    frame["dim_to_split"] = first_dim
    frame["dim_to_shift"] = second_dim
    return frame


def fill_scatter_plot_splitting_1(colors, dataset, scatter_plot):
    classes = [0 for _ in range(50)]
    classes.extend([1 for _ in range(150)])
    dataset["class"] = classes
    pad = 0.1
    x_min, x_max = dataset["dim_to_split"].min() - pad, dataset["dim_to_split"].max() + pad
    y_min, y_max = dataset["dim_to_shift"].min() - pad, dataset["dim_to_shift"].max() + pad
    scatter_plot.set_xlim(x_min, x_max)
    scatter_plot.set_ylim(y_min, y_max)
    scatter_plot.set_xlabel("dim_to_split", fontsize=10)  # , labelpad=1)
    scatter_plot.set_ylabel("dim_to_shift", fontsize=10)  # , labelpad=1)
    scatter_plot.set_title("example data set", fontsize=10)
    scatter_plot.scatter(dataset["dim_to_split"][0:50], dataset["dim_to_shift"][0:50], s=15, color=colors[0],
                         label="bin 0", zorder=1)
    scatter_plot.scatter(dataset["dim_to_split"][50:200], dataset["dim_to_shift"][50:200], s=15, color=colors[1],
                         label="bin 1", zorder=1)
    scatter_plot.set_yticks([0, 0.5, 1, 1.5, 2], fontsize=10)
    scatter_plot.legend(fontsize=10, bbox_to_anchor=(.37, .97), frameon=True)


def fill_ecdf_subplot_splitting_1(ecdf_plot, colors):
    ecdf_plot.plot([0, 1, 2], [0, 1, 1], zorder=5, label="bin 0")
    ecdf_plot.plot([0, 1, 2], [0, 0.3333, 1], zorder=0, label="bin 1")
    ecdf_plot.legend(fontsize=10, frameon=True, bbox_to_anchor=(.37, .97))

    """ecdf_plot.arrow(1.2, 0.4, 0.6, 0, color="black", width=0.005, length_includes_head=True, head_width=0.01,
                    head_length=0.1, linewidth=0.001, zorder=10)
    ecdf_plot.arrow(1.8, 0.4, 0, 0.2, color="black", width=0.03, length_includes_head=True, head_width=0.07,
                    head_length=0.013, linewidth=0.001, zorder=10)

    ecdf_plot.arrow(0.8, 0.4, 1.4, 0, color=colors[0], width=0.005, length_includes_head=True, head_width=0.01,
                    head_length=0.1, linewidth=0.001, zorder=10)
    ecdf_plot.arrow(2.2, 0.4, 0, 0.2, color=colors[0], width=0.03, length_includes_head=True, head_width=0.07,
                    head_length=0.013, linewidth=0.001, zorder=10)"""
    ecdf_plot.set_title("ecdf", fontsize=10)
    ecdf_plot.set_xlabel("dim_to_shift", fontsize=10, labelpad=1)
    ecdf_plot.set_ylabel("Cumulative Frequency", fontsize=10, labelpad=3)
    ecdf_plot.set_xticks([0, 1, 2], fontsize=10)
    ecdf_plot.set_yticks([0, .2, .4, .6, .8, 1])  # , ["0", ".2", ".4", ".6", ".8", "1"])


def example_splitting_presentation_split_1():
    plt.clf()
    plt.figure(0, figsize=(8, 3.5))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    scatter_plot = plt.subplot2grid((1, 100), (0, 0), rowspan=1, colspan=43)
    ecdf_plot = plt.subplot2grid((1, 100), (0, 57), rowspan=1, colspan=43)

    dataset = get_dataset_split()

    fill_scatter_plot_splitting_1(colors, dataset, scatter_plot)
    fill_ecdf_subplot_splitting_1(ecdf_plot, colors)
    #plt.show()
    plt.savefig("../../../Plots/Paper_Grafiken/splitting_1.png", bbox_inches='tight', dpi=900)


def fill_scatter_plot_splitting_2(colors, dataset, scatter_plot):
    classes = [0 for _ in range(100)]
    classes.extend([1 for _ in range(100)])
    dataset["class"] = classes
    pad = 0.1
    x_min, x_max = dataset["dim_to_split"].min() - pad, dataset["dim_to_split"].max() + pad
    y_min, y_max = dataset["dim_to_shift"].min() - pad, dataset["dim_to_shift"].max() + pad
    scatter_plot.set_xlim(x_min, x_max)
    scatter_plot.set_ylim(y_min, y_max)
    scatter_plot.set_xlabel("dim_to_split", fontsize=10)  # , labelpad=1)
    scatter_plot.set_ylabel("dim_to_shift", fontsize=10)  # , labelpad=1)
    scatter_plot.set_title("example data set", fontsize=10)
    scatter_plot.scatter(dataset["dim_to_split"][0:100], dataset["dim_to_shift"][0:100], s=15, color=colors[0],
                         label="bin 0", zorder=1)
    scatter_plot.scatter(dataset["dim_to_split"][100:200], dataset["dim_to_shift"][100:200], s=15, color=colors[1],
                         label="bin 1", zorder=1)
    scatter_plot.set_yticks([0, 0.5, 1, 1.5, 2], fontsize=10)
    scatter_plot.legend(fontsize=10, bbox_to_anchor=(.37, .97), frameon=True)


def fill_ecdf_subplot_splitting_2(ecdf_plot, colors):
    ecdf_plot.plot([0, 1, 2], [0, 1, 1], zorder=5, label="bin 0")
    ecdf_plot.plot([0, 1, 2], [0, 0, 1], zorder=0, label="bin 1")
    ecdf_plot.legend(fontsize=10, frameon=True, bbox_to_anchor=(.37, .97))

    """ecdf_plot.arrow(1.2, 0.4, 0.6, 0, color="black", width=0.005, length_includes_head=True, head_width=0.01,
                    head_length=0.1, linewidth=0.001, zorder=10)
    ecdf_plot.arrow(1.8, 0.4, 0, 0.2, color="black", width=0.03, length_includes_head=True, head_width=0.07,
                    head_length=0.013, linewidth=0.001, zorder=10)

    ecdf_plot.arrow(0.8, 0.4, 1.4, 0, color=colors[0], width=0.005, length_includes_head=True, head_width=0.01,
                    head_length=0.1, linewidth=0.001, zorder=10)
    ecdf_plot.arrow(2.2, 0.4, 0, 0.2, color=colors[0], width=0.03, length_includes_head=True, head_width=0.07,
                    head_length=0.013, linewidth=0.001, zorder=10)"""
    ecdf_plot.set_title("ecdf", fontsize=10)
    ecdf_plot.set_xlabel("dim_to_shift", fontsize=10, labelpad=1)
    ecdf_plot.set_ylabel("Cumulative Frequency", fontsize=10, labelpad=3)
    ecdf_plot.set_xticks([0, 1, 2], fontsize=10)
    ecdf_plot.set_yticks([0, .2, .4, .6, .8, 1])  # , ["0", ".2", ".4", ".6", ".8", "1"])


def example_splitting_presentation_split_2():
    plt.clf()
    plt.figure(0, figsize=(8, 3.5))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    scatter_plot = plt.subplot2grid((1, 100), (0, 0), rowspan=1, colspan=43)
    ecdf_plot = plt.subplot2grid((1, 100), (0, 57), rowspan=1, colspan=43)

    dataset = get_dataset_split()

    fill_scatter_plot_splitting_2(colors, dataset, scatter_plot)
    fill_ecdf_subplot_splitting_2(ecdf_plot, colors)
    #plt.show()
    plt.savefig("../../../Plots/Paper_Grafiken/splitting_2.png", bbox_inches='tight', dpi=900)


if __name__ == "__main__":
    """iris = datasets.load_iris(as_frame=True)
    org_iris(iris)
    shifted_iris(iris)"""
    #binning_plot_paper()
    example_splitting_presentation_split_1()
