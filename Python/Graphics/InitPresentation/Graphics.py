from typing import List, Tuple

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random


def segment_new_plot(iris):
    plt.figure(0, figsize=(5.5, 4))
    plt.clf()
    ax = plt.axes()
    rect1 = mpl.patches.Rectangle((0, -1), 2.5, 7, linewidth=0, facecolor=(1, 0, 0, 0.2))
    ax.add_patch(rect1)
    rect2 = mpl.patches.Rectangle((2.5, -1), 7, 2.75, linewidth=0, facecolor=(1, 1, 0, 0.2))
    ax.add_patch(rect2)
    rect3 = mpl.patches.Rectangle((2.5, 1.75), 7, 7, linewidth=0, facecolor=(0, 0, 0, 0.2))
    ax.add_patch(rect3)
    X: pd.DataFrame = iris.data[['petal length (cm)', 'petal width (cm)']]


    pad = 0.3
    length_min, length_max = X["petal length (cm)"].min() - pad, X["petal length (cm)"].max() + pad
    width_min, width_max = X["petal width (cm)"].min() - pad, X["petal width (cm)"].max() + pad

    plt.xlim(length_min, length_max)
    plt.ylim(width_min, width_max)
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")


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
    #plt.show()
    plt.savefig("C:/Users/gerno/OneDrive/Bachelorarbeit/Intro_Vortrag/Grafiken/binning.png")


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
    #plt.show()
    plt.savefig(r"D:\Gernot\Programmieren\Bachelor\Plots\BA_Grafiken/binning_actually_binned.png")


def org_iris(iris):
    y = iris.target
    X: pd.DataFrame = iris.data[['petal length (cm)', 'petal width (cm)']]

    segment_new_plot(iris)

    plt.scatter(X["petal length (cm)"], X["petal width (cm)"], c=y, cmap=plt.cm.Set1, edgecolor="k")
    #plt.show()
    plt.savefig("C:/Users/gerno/OneDrive/Bachelorarbeit/Intro_Vortrag/Grafiken/iris_segmented.png")


def shifted_iris(iris):
    y = iris.target
    X: pd.DataFrame = iris.data[['petal length (cm)', 'petal width (cm)']]

    segment_new_plot(iris)

    X["petal width (cm)"] += 0.17
    plt.scatter(X["petal length (cm)"], X["petal width (cm)"], c=y, cmap=plt.cm.Set1, edgecolor="k")
    #plt.show()
    plt.savefig("C:/Users/gerno/OneDrive/Bachelorarbeit/Intro_Vortrag/Grafiken/iris_segmented_shifted.png")





if __name__ == "__main__":
    #iris = datasets.load_iris(as_frame=True)
    #org_iris(iris)
    #shifted_iris(iris)
    binning_plot_actually_binned()
