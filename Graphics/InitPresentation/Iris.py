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


def create_binning_data() -> pd.DataFrame:
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
    return pd.concat([data1, data2, data3])


def binning_plot():
    data = create_binning_data()
    plt.figure(0, figsize=(5.5, 4))
    plt.clf()
    ax = plt.axes()
    """
    rect1 = mpl.patches.Rectangle((0, 0), 1, 2, linewidth=1, edgecolor="black", facecolor="none")
    ax.add_patch(rect1)
    rect2 = mpl.patches.Rectangle((1, -1), 1, 1, linewidth=1, edgecolor="black", facecolor="none")
    ax.add_patch(rect2)
    rect3 = mpl.patches.Rectangle((2, 0), 1, 2, linewidth=1, edgecolor="black", facecolor="none")
    ax.add_patch(rect3)
    """

    pad = 0.3
    x_min, x_max = data["X"].min() - pad, data["X"].max() + pad
    y_min, y_max = data["Y"].min() - pad, data["Y"].max() + pad

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(data["X"], data["Y"])
    plt.savefig("C:/Users/gerno/OneDrive/Bachelorarbeit/Intro_Vortrag/Grafiken/binning.png")


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
    binning_plot()
