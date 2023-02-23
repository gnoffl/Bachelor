import dataCreation as dc
import visualization as vs
import matplotlib.pyplot as plt
from scipy.stats import sem
from matplotlib.lines import Line2D


def IrisDataSet_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 8))
    ids = dc.IrisDataSet()
    subplot_locs = [
        (100, 100, 0, 0, 43, 43),
        (100, 100, 0, 57, 43, 43),
        (100, 100, 57, 0, 43, 43)
    ]
    titles = ["A", "B", "C"]
    dims = [
        ("petal_width", "petal_length"),
        ("sepal_width", "sepal_length"),
        ("petal_width", "sepal_length")
    ]
    axis_names = [("petal_width / cm", "petal_length / cm"),
                  ("sepal_width / cm", "sepal_length / cm"),
                  ("petal_width / cm", "sepal_length / cm")]

    for i, (plot_loc, dim, title, axis_name) in enumerate(zip(subplot_locs, dims, titles, axis_names)):
        if i == 0:
            vs.visualize_2d_subplot(df=ids.data, dims=dim, subplot_location=plot_loc, class_column="classes",
                                    class_names=ids.class_names, show_legend=True, loc="upper left", title=title,
                                    axis_names=axis_name)
        else:
            vs.visualize_2d_subplot(df=ids.data, dims=dim, subplot_location=plot_loc, class_column="classes",
                                    class_names=ids.class_names, title=title, axis_names=axis_name)

    new_df = ids.data.copy()

    values, cum_frequencies = vs.get_cumulative_values(new_df["petal_length"].values)

    plot = plt.subplot2grid((100, 100), (57, 57), rowspan=43, colspan=43)
    plot.set_title("D")
    plot.set_xlabel("petal_length / cm")
    plot.set_ylabel("Cumulative Frequency")
    plot.plot(values, cum_frequencies)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Iris_intro.png", bbox_inches='tight')


def maybeActualDataSet_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 8))
    mads = dc.MaybeActualDataSet([200 for _ in range(6)])
    subplot_locs = [
        (100, 100, 0, 0, 43, 43),
        (100, 100, 0, 57, 43, 43),
        (100, 100, 57, 0, 43, 43)
    ]
    titles = ["A", "B", "C"]
    dims = [
        ("dim_00", "dim_04"),
        ("dim_01", "dim_04"),
        ("dim_02", "dim_03")
    ]

    for i, (plot_loc, dim, title) in enumerate(zip(subplot_locs, dims, titles)):
        if i == 1:
            vs.visualize_2d_subplot(df=mads.data, dims=dim, subplot_location=plot_loc, class_column="classes",
                                    class_names=mads.class_names, show_legend=True, bbox_to_anchor=(1.01, 1),
                                    title=title)
        else:
            vs.visualize_2d_subplot(df=mads.data, dims=dim, subplot_location=plot_loc, class_column="classes",
                                    class_names=mads.class_names, title=title)

    new_df = mads.data.copy()

    values, cum_frequencies = vs.get_cumulative_values(new_df["dim_04"].values)

    plot = plt.subplot2grid((100, 100), (57, 57), rowspan=43, colspan=43)
    plot.set_title("D")
    plot.set_xlabel("dim_04")
    plot.set_ylabel("Cumulative Frequency")
    plot.plot(values, cum_frequencies)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/MaybeActual_intro.png", bbox_inches='tight')


def soccerDataSet_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 12))
    sds = dc.SoccerDataSet()
    subplot_locs = [
        (100, 100, 0, 0, 28, 43),
        (100, 100, 0, 57, 28, 43),
        (100, 100, 36, 0, 28, 43),
        (100, 100, 36, 57, 28, 43),
        (100, 100, 72, 0, 28, 43),
        (100, 100, 72, 57, 28, 43)
    ]
    titles = ["A", "B", "C", "D", "E", "F"]
    dims = [
        ("ps_Pass", "Passprozente"),
        ("ps_Zweikampf", "Zweikampfprozente"),
        ("ps_Fouls", "ps_Gefoult"),
        ("ps_Laufweite", "ps_Abseits"),
        ("ps_Assists", "ps_Fusstore"),
        ("ps_Kopftore", "ps_Pass")
    ]
    axis_namess = [("ps_Pass", "Passprozente / %"),
                   ("ps_Zweikampf", "Zweikampfprozente / %"),
                   ("ps_Fouls", "ps_Gefoult"),
                   ("ps_Laufweite / km", "ps_Abseits"),
                   ("ps_Assists", "ps_Fusstore"),
                   ("ps_Kopftore", "ps_Pass")]

    for i, (plot_loc, dim, title, axis_names) in enumerate(zip(subplot_locs, dims, titles, axis_namess)):
        if i == 1:
            vs.visualize_2d_subplot(df=sds.data, dims=dim, subplot_location=plot_loc, class_column="classes",
                                    class_names=sds.class_names, show_legend=True, bbox_to_anchor=(1.01, 1),
                                    title=title,
                                    axis_names=axis_names)
        else:
            vs.visualize_2d_subplot(df=sds.data, dims=dim, subplot_location=plot_loc, class_column="classes",
                                    class_names=sds.class_names, title=title, axis_names=axis_names)

    plt.savefig("../Plots/BA_Grafiken/Soccer_intro.png", bbox_inches='tight')


def Iris_max_split_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 8))

    set_09 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\009\Splits\petal_length_005")
    set_17 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\017\Splits\petal_length_005")
    visualized_area = vs.find_common_area(set_09.data["petal_width"].values, set_09.data["petal_length_org"].values,
                                          set_09.data["petal_width"].values, set_09.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 0, 43, 43),
        (100, 100, 57, 0, 43, 43)
    ]
    titles = ["A", "C"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length")
    ]
    axis_names = ("petal_width / cm", "petal_length / cm")

    class_names = set(set_17.data["source"].unique())
    class_names.update(set(set_09.data["source"].unique()))
    class_names = sorted(list(class_names))

    for i, (plot_loc, dim, title) in enumerate(zip(subplot_locs, dims, titles)):
        vs.visualize_2d_subplot(df=set_09.data, dims=dim, subplot_location=plot_loc, class_column="source",
                                show_legend=True, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    visualized_area = vs.find_common_area(set_17.data["petal_width"].values, set_17.data["petal_length_org"].values,
                                          set_17.data["petal_width"].values, set_17.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 57, 43, 43),
        (100, 100, 57, 57, 43, 43)
    ]
    titles = ["B", "D"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length")
    ]

    for i, (plot_loc, dim, title) in enumerate(zip(subplot_locs, dims, titles)):
        vs.visualize_2d_subplot(df=set_17.data, dims=dim, subplot_location=plot_loc, class_column="source",
                                show_legend=True, bbox_to_anchor=(1.01, 1), title=title,
                                visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Iris_results/Iris_max_split.png", bbox_inches='tight')


def Iris_p_val_figure():
    plt.clf()
    plt.figure(0, figsize=(4, 8))

    set_17 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\017\Splits\petal_length_005")
    visualized_area = vs.find_common_area(set_17.data["petal_width"].values, set_17.data["petal_length_org"].values,
                                          set_17.data["petal_width"].values, set_17.data["petal_length"].values)

    set_21 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\021\Splits\petal_length_005")
    axis_names = ("petal_width / cm", "petal_length / cm")

    class_names = set(set_17.data["source"].unique())
    class_names.update(set(set_21.data["source"].unique()))
    class_names = list(class_names)
    class_names = sorted(class_names)
    subplot_locs = [
        (100, 100, 0, 0, 43, 100)
    ]
    titles = ["A"]
    dims = [
        ("petal_width", "petal_length_org")
    ]

    for i, (plot_loc, dim, title) in enumerate(zip(subplot_locs, dims, titles)):
        vs.visualize_2d_subplot(df=set_17.data, dims=dim, subplot_location=plot_loc, class_column="source",
                                show_legend=True, bbox_to_anchor=(1.01, 1), title=title,
                                visualized_area=visualized_area,
                                class_names=class_names, map_label=False, axis_names=axis_names)

    visualized_area = vs.find_common_area(set_21.data["petal_width"].values, set_21.data["petal_length_org"].values,
                                          set_21.data["petal_width"].values, set_21.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 57, 0, 43, 100)
    ]
    titles = ["B"]
    dims = [
        ("petal_width", "petal_length_org")
    ]

    for i, (plot_loc, dim, title) in enumerate(zip(subplot_locs, dims, titles)):
        vs.visualize_2d_subplot(df=set_21.data, dims=dim, subplot_location=plot_loc, class_column="source",
                                show_legend=True, bbox_to_anchor=(1.01, 1), title=title,
                                visualized_area=visualized_area,
                                class_names=class_names, map_label=False, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Iris_results/Iris_p_val.png", bbox_inches='tight')


def Iris_HiCS_figure():
    plt.clf()
    plt.figure(0, figsize=(9, 4))

    set_01 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\001\Splits\petal_length_005")
    visualized_area = vs.find_common_area(set_01.data["petal_width"].values, set_01.data["petal_length_org"].values,
                                          set_01.data["petal_width"].values, set_01.data["petal_length"].values)

    set_03 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\003\Splits\petal_length_005")

    class_names = set(set_01.data["source"].unique())
    class_names.update(set(set_03.data["source"].unique()))
    class_names = list(class_names)
    class_names = sorted(class_names)
    subplot_locs = [
        (100, 100, 0, 0, 100, 43)
    ]
    titles = ["A"]
    dims = [
        ("petal_width", "petal_length_org")
    ]
    axis_names = ("petal_width / cm", "petal_length / cm")

    for i, (plot_loc, dim, title) in enumerate(zip(subplot_locs, dims, titles)):
        vs.visualize_2d_subplot(df=set_01.data, dims=dim, subplot_location=plot_loc, class_column="source",
                                show_legend=True, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, map_label=False, axis_names=axis_names)

    visualized_area = vs.find_common_area(set_03.data["petal_width"].values, set_03.data["petal_length_org"].values,
                                          set_03.data["petal_width"].values, set_03.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 57, 100, 43)
    ]
    titles = ["B"]
    dims = [
        ("petal_width", "petal_length_org")
    ]

    for i, (plot_loc, dim, title) in enumerate(zip(subplot_locs, dims, titles)):
        vs.visualize_2d_subplot(df=set_03.data, dims=dim, subplot_location=plot_loc, class_column="source",
                                show_legend=True, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, map_label=False, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Iris_results/Iris_HiCS.png", bbox_inches='tight')


def Iris_tree_pred_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 8))

    set_09 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\009\Splits\petal_length_005")
    visualized_area = vs.find_common_area(set_09.data["petal_width"].values, set_09.data["petal_length_org"].values,
                                          set_09.data["petal_width"].values, set_09.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 0, 43, 43),
        (100, 100, 0, 57, 43, 43),
        (100, 100, 57, 0, 43, 43),
        (100, 100, 57, 57, 43, 43)
    ]
    titles = ["A", "B", "C", "D"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length"),
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length")
    ]
    class_columns = ["classes", "classes", "org_pred", "pred_classes"]
    show_legend = [True, False, False, False]
    axis_names = ("petal_width / cm", "petal_length / cm")

    class_names = set_09.class_names

    for i, (plot_loc, dim, title, class_column, legend) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend)):
        vs.visualize_2d_subplot(df=set_09.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Iris_results/Iris_example_pred.png", bbox_inches='tight')


def Iris_NN_pred_figure():
    plt.clf()
    plt.figure(0, figsize=(9, 4))

    set_09 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\NN\009\Splits\petal_length_005")
    visualized_area = vs.find_common_area(set_09.data["petal_width"].values, set_09.data["petal_length_org"].values,
                                          set_09.data["petal_width"].values, set_09.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 0, 100, 43),
        (100, 100, 0, 57, 100, 43)
    ]
    titles = ["A", "B", "C", "D"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length_org")
    ]
    class_columns = ["classes", "org_pred"]
    show_legend = [True, False]
    axis_names = ("petal_width / cm", "petal_length / cm")

    class_names = set_09.class_names

    for i, (plot_loc, dim, title, class_column, legend) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend)):
        vs.visualize_2d_subplot(df=set_09.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Iris_results/Iris_NN_pred.png", bbox_inches='tight')


def Iris_NN_QSM_figure():
    plt.clf()
    plt.figure(0, figsize=(9, 4))

    set_09 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\NN\009\Splits\petal_length_005")
    visualized_area = vs.find_common_area(set_09.data["petal_width"].values, set_09.data["petal_length_org"].values,
                                          set_09.data["petal_width"].values, set_09.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 0, 100, 43),
        (100, 100, 0, 57, 100, 43)
    ]
    titles = ["A", "B", "C", "D"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length")
    ]
    class_columns = ["org_pred", "pred_classes"]
    show_legend = [True, False]
    axis_names = ("petal_width / cm", "petal_length / cm")

    class_names = set_09.class_names

    for i, (plot_loc, dim, title, class_column, legend) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend)):
        vs.visualize_2d_subplot(df=set_09.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Iris_results/Iris_NN_QSM.png", bbox_inches='tight')


def Iris_QSM_comparison_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 12))

    qsm_tree_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\004")
    improved_tree_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\004\Splits\petal_length_005")
    visualized_area = vs.find_common_area(improved_tree_set.data["petal_width"].values,
                                          improved_tree_set.data["petal_length_org"].values,
                                          improved_tree_set.data["petal_width"].values,
                                          improved_tree_set.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 0, 28, 43),
        (100, 100, 36, 0, 28, 43)
    ]
    titles = ["A", "B"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length")
    ]
    class_columns = ["org_pred", "pred_classes"]
    show_legend = [True, False]
    axis_names = ("petal_width / cm", "petal_length / cm")

    class_names = improved_tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend)):
        vs.visualize_2d_subplot(df=improved_tree_set.data, dims=dim, subplot_location=plot_loc,
                                class_column=class_column,
                                show_legend=legend, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)
    vs.visualize_2d_subplot(df=qsm_tree_set.data, dims=("petal_width", "petal_length_shifted_by_0.05"),
                            subplot_location=(100, 100, 72, 0, 28, 43),
                            class_column="pred_with_petal_length_shifted_by_0.05",
                            show_legend=False, title="C", visualized_area=visualized_area, class_names=class_names,
                            axis_names=axis_names)

    qsm_NN_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\NN\004")
    improved_NN_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\NN\004\Splits\petal_length_005")

    subplot_locs = [
        (100, 100, 0, 57, 28, 43),
        (100, 100, 36, 57, 28, 43)
    ]
    titles = ["D", "E"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length")
    ]
    class_columns = ["org_pred", "pred_classes"]
    show_legend = [False, False]

    class_names = improved_tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend)):
        vs.visualize_2d_subplot(df=improved_NN_set.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)
    subplot = vs.visualize_2d_subplot(df=qsm_NN_set.data, dims=("petal_width", "petal_length_shifted_by_0.05"),
                                      subplot_location=(100, 100, 72, 57, 28, 43),
                                      class_column="pred_with_petal_length_shifted_by_0.05",
                                      show_legend=False, title="F", visualized_area=visualized_area,
                                      class_names=class_names, axis_names=axis_names)

    subplot.scatter([0.2],
                    [3.3],
                    color=vs.colors[0],
                    edgecolor="k",
                    label="setosa")

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Iris_results/Iris_QSM_comparison.png", bbox_inches='tight')


def Iris_QSM_comparison_figure_quer():
    plt.clf()
    plt.figure(0, figsize=(12, 8))

    qsm_tree_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\004")
    improved_tree_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\tree\004\Splits\petal_length_005")
    visualized_area = vs.find_common_area(improved_tree_set.data["petal_width"].values,
                                          improved_tree_set.data["petal_length_org"].values,
                                          improved_tree_set.data["petal_width"].values,
                                          improved_tree_set.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 0, 43, 28),
        (100, 100, 0, 72, 43, 28)
    ]
    titles = ["A", "C"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length")
    ]
    class_columns = ["org_pred", "pred_classes"]
    show_legend = [False, False]
    axis_names = ("petal_width / cm", "petal_length / cm")

    class_names = improved_tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend)):
        vs.visualize_2d_subplot(df=improved_tree_set.data, dims=dim, subplot_location=plot_loc,
                                class_column=class_column,
                                show_legend=legend, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)
    vs.visualize_2d_subplot(df=qsm_tree_set.data, dims=("petal_width", "petal_length_shifted_by_0.05"),
                            subplot_location=(100, 100, 0, 36, 43, 28),
                            class_column="pred_with_petal_length_shifted_by_0.05",
                            show_legend=False, title="B", visualized_area=visualized_area, class_names=class_names,
                            axis_names=axis_names)

    qsm_NN_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\NN\004")
    improved_NN_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\IrisDataSet\NN\004\Splits\petal_length_005")

    subplot_locs = [
        (100, 100, 57, 0, 43, 28),
        (100, 100, 57, 72, 43, 28)
    ]
    titles = ["D", "F"]
    dims = [
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length")
    ]
    class_columns = ["org_pred", "pred_classes"]
    show_legend = [False, False]

    class_names = improved_tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend)):
        vs.visualize_2d_subplot(df=improved_NN_set.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, loc="upper left", title=title, visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)
    subplot = vs.visualize_2d_subplot(df=qsm_NN_set.data, dims=("petal_width", "petal_length_shifted_by_0.05"),
                                      subplot_location=(100, 100, 57, 36, 43, 28),
                                      class_column="pred_with_petal_length_shifted_by_0.05",
                                      show_legend=False, title="E", visualized_area=visualized_area,
                                      class_names=class_names, axis_names=axis_names)

    subplot.scatter([0.2],
                    [3.3],
                    color=vs.colors[0],
                    edgecolor="k",
                    label="setosa")

    # plt.show()
    plt.savefig(r"C:\Users\gerno\OneDrive\Bachelorarbeit\Abschluss_Vortrag\Iris_QSM_comparison_quer_1.png",
                bbox_inches='tight')


def hole_parameter_figure():
    plt.clf()
    plt.figure(0, figsize=(10, 10))

    datasets = [
        dc.Data.load(
            r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\001\Splits\dim_04_005"),
        dc.Data.load(
            r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\009\Splits\dim_04_005"),
        dc.Data.load(
            r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\005\Splits\dim_04_005"),
        dc.Data.load(
            r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\003\Splits\dim_04_005")
    ]
    # visualized_area = vs.find_common_area(set_09.data["dim_00"].values, set_09.data["dim_04_org"].values,
    #                                   set_09.data["dim_00"].values, set_09.data["dim_04"].values)

    subplot_locs = [
        (100, 100, 0, 0, 45, 45),
        (100, 100, 0, 55, 45, 45),
        (100, 100, 55, 0, 45, 45),
        (100, 100, 55, 55, 45, 45)
    ]
    titles = ["A", "B", "C", "D"]
    dims = [
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04_org")
    ]
    class_columns = ["source", "source", "source", "source"]
    show_legend = [True, True, True, True]
    axis_names = ("dim_00", "dim_04")

    class_names = set()
    for dataset in datasets:
        new_class_names = set(dataset.data["source"].unique())
        class_names.update(new_class_names)
    class_names = sorted(list(class_names))
    print(len(class_names))

    legend_elements = []
    for clas in class_names:
        color_ind = vs.get_color_index(clas, class_names=class_names, i=1)
        curr = Line2D([0], [0], marker='o', color=vs.colors[color_ind], label=clas)
        legend_elements.append(curr)

    subplots = []
    all_handles = []

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        subplots.append(
            vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                    show_legend=legend, bbox_to_anchor=(1.01, 1), title=title, class_names=class_names,
                                    axis_names=axis_names))
        legend = subplots[i].legend()
        all_handles.extend(legend.legendHandles)

    final_handles = vs.get_unique_handles(all_handles)
    for subplot in subplots:
        subplot.get_legend().remove()

    subplots[1].legend(handles=final_handles, bbox_to_anchor=(1.01, 1))

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Hole_results/Hole_parameters.png", bbox_inches='tight')


def hole_classification_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 8))

    tree_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\001\Splits\dim_04_005")
    NN_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\NN\001\Splits\dim_04_005")

    datasets = [tree_set, tree_set, NN_set]
    # visualized_area = vs.find_common_area(set_09.data["dim_00"].values, set_09.data["dim_04_org"].values,
    #                                   set_09.data["dim_00"].values, set_09.data["dim_04"].values)

    subplot_locs = [
        (100, 100, 0, 28, 45, 45),
        (100, 100, 55, 0, 45, 45),
        (100, 100, 55, 55, 45, 45)
    ]
    titles = ["A", "B", "C"]
    dims = [
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04_org")
    ]
    class_columns = ["classes", "org_pred", "org_pred"]
    show_legend = [True, False, False]
    axis_names = ("dim_00", "dim_04")

    class_names = tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=(1.01, 1), title=title, class_names=class_names,
                                axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Hole_results/Hole_classifiers.png", bbox_inches='tight')


def hole_maxsplit_shift_comp_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 8))

    tree_set_3 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\004\Splits\dim_04_005")
    tree_set_2 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\012\Splits\dim_04_005")
    tree_set_4 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\020\Splits\dim_04_005")

    datasets = [tree_set_2, tree_set_3, tree_set_4]
    visualized_area = vs.find_common_area(tree_set_2.data["dim_00"].values, tree_set_2.data["dim_04"].values,
                                          tree_set_4.data["dim_00"].values, tree_set_4.data["dim_04"].values)

    subplot_locs = [
        (100, 100, 0, 28, 45, 45),
        (100, 100, 55, 0, 45, 45),
        (100, 100, 55, 55, 45, 45)
    ]
    titles = ["A", "B", "C"]
    dims = [
        ("dim_00", "dim_04"),
        ("dim_00", "dim_04"),
        ("dim_00", "dim_04")
    ]
    class_columns = ["pred_classes", "pred_classes", "pred_classes"]
    show_legend = [True, False, False]

    class_names = tree_set_2.class_names

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=(1.01, 1), title=title, class_names=class_names,
                                visualized_area=visualized_area)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Hole_results/Hole_shift_comp_max_split.png", bbox_inches='tight')


def hole_QSM_comparison_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 12))

    qsm_tree_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\020")
    improved_tree_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\012\Splits\dim_04_005")
    qsm_NN_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\NN\020")
    improved_NN_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\NN\012\Splits\dim_04_005")

    visualized_area = vs.find_common_area(improved_tree_set.data["dim_00"].values,
                                          improved_tree_set.data["dim_04_org"].values,
                                          improved_tree_set.data["dim_00"].values,
                                          improved_tree_set.data["dim_04"].values)

    subplot_locs = [
        (100, 100, 0, 0, 28, 43),
        (100, 100, 36, 0, 28, 43),
        (100, 100, 72, 0, 28, 43),
        (100, 100, 0, 57, 28, 43),
        (100, 100, 36, 57, 28, 43),
        (100, 100, 72, 57, 28, 43)
    ]
    titles = ["A", "B", "C", "D", "E", "F"]
    dims = [
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04"),
        ("dim_00", "dim_04_shifted_by_0.05"),
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04"),
        ("dim_00", "dim_04_shifted_by_0.05")
    ]
    class_columns = ["org_pred",
                     "pred_classes",
                     "pred_with_dim_04_shifted_by_0.05",
                     "org_pred",
                     "pred_classes",
                     "pred_with_dim_04_shifted_by_0.05"]
    show_legend = [False, False, False, True, False, False]
    datasets = [improved_tree_set, improved_tree_set, qsm_tree_set, improved_NN_set, improved_NN_set, qsm_NN_set]
    axis_names = ("dim_00", "dim_04")

    class_names = improved_tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=(1.01, 1), title=title,
                                visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    plt.show()
    # plt.savefig("../Plots/BA_Grafiken/Hole_results/Hole_QSM_comparison.png", bbox_inches='tight')


def hole_QSM_comparison_figure_quer():
    plt.clf()
    plt.figure(0, figsize=(12, 8))

    qsm_tree_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\020")
    improved_tree_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\tree\012\Splits\dim_04_005")
    qsm_NN_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\NN\020")
    improved_NN_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\MaybeActualDataSet\NN\012\Splits\dim_04_005")

    visualized_area = vs.find_common_area(improved_tree_set.data["dim_00"].values,
                                          improved_tree_set.data["dim_04_org"].values,
                                          improved_tree_set.data["dim_00"].values,
                                          improved_tree_set.data["dim_04"].values)

    subplot_locs = [
        (100, 100, 0, 0, 43, 28),
        (100, 100, 0, 72, 43, 28),
        (100, 100, 0, 36, 43, 28),
        (100, 100, 57, 0, 43, 28),
        (100, 100, 57, 72, 43, 28),
        (100, 100, 57, 36, 43, 28)
    ]
    titles = ["A", "C", "B", "D", "F", "E"]
    dims = [
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04"),
        ("dim_00", "dim_04_shifted_by_0.05"),
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04"),
        ("dim_00", "dim_04_shifted_by_0.05")
    ]
    class_columns = ["org_pred",
                     "pred_classes",
                     "pred_with_dim_04_shifted_by_0.05",
                     "org_pred",
                     "pred_classes",
                     "pred_with_dim_04_shifted_by_0.05"]
    show_legend = [False, True, False, False, False, False]
    datasets = [improved_tree_set, improved_tree_set, qsm_tree_set, improved_NN_set, improved_NN_set, qsm_NN_set]
    axis_names = ("dim_00", "dim_04")

    class_names = improved_tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=(1.01, 1), title=title,
                                visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    # plt.show()
    plt.savefig(r"C:\Users\gerno\OneDrive\Bachelorarbeit\Abschluss_Vortrag\Hole_QSM_comparison_quer_1.png",
                bbox_inches='tight')


def soccer_parameter_figure():
    plt.clf()
    plt.figure(0, figsize=(9, 9))

    datasets = [
        dc.Data.load(
            r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\023\Splits\ps_Laufweite_005"),
        dc.Data.load(
            r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\015\Splits\ps_Laufweite_005"),
        dc.Data.load(
            r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\019\Splits\ps_Laufweite_005"),
        dc.Data.load(
            r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\021\Splits\ps_Laufweite_005")
    ]
    # visualized_area = vs.find_common_area(set_09.data["dim_00"].values, set_09.data["dim_04_org"].values,
    #                                   set_09.data["dim_00"].values, set_09.data["dim_04"].values)

    subplot_locs = [
        (100, 100, 0, 0, 45, 45),
        (100, 100, 0, 55, 45, 45),
        (100, 100, 55, 0, 45, 45),
        (100, 100, 55, 55, 45, 45)
    ]
    titles = ["A", "B", "C", "D"]
    dims = [
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite_org")
    ]
    class_columns = ["source", "source", "source", "source"]
    show_legend = [True, True, True, True]
    axis_names = ("Zweikampfprozente / %", "ps_Laufweite / km")

    class_names = set()
    for dataset in datasets:
        new_class_names = set(dataset.data["source"].unique())
        class_names.update(new_class_names)
    class_names = sorted(list(class_names))

    legend_elements = []
    for clas in class_names:
        color_ind = vs.get_color_index(clas, class_names=class_names, i=1)
        curr = Line2D([0], [0], marker='o', color=vs.colors[color_ind], label=clas)
        legend_elements.append(curr)

    subplots = []
    all_handles = []

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        subplots.append(
            vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                    show_legend=legend, bbox_to_anchor=(1.01, 1), title=title, class_names=class_names,
                                    map_label=False, axis_names=axis_names))
        legend = subplots[i].legend()
        handles = legend.legendHandles
        all_handles.extend(handles)

    final_handles = vs.get_unique_handles(all_handles)
    for subplot in subplots:
        subplot.get_legend().remove()

    subplots[1].legend(handles=final_handles, bbox_to_anchor=(1.01, 1))

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Soccer_results/Soccer_parameters.png", bbox_inches='tight')


def soccer_classification_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 8))

    tree_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\001\Splits\ps_Laufweite_005")
    NN_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\NN\001\Splits\ps_Laufweite_005")

    datasets = [tree_set, tree_set, NN_set]
    # visualized_area = vs.find_common_area(set_09.data["dim_00"].values, set_09.data["dim_04_org"].values,
    #                                   set_09.data["dim_00"].values, set_09.data["dim_04"].values)

    subplot_locs = [
        (100, 100, 0, 28, 45, 45),
        (100, 100, 55, 0, 45, 45),
        (100, 100, 55, 55, 45, 45)
    ]
    titles = ["A", "B", "C"]
    dims = [
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite_org")
    ]
    class_columns = ["classes", "org_pred", "org_pred"]
    show_legend = [True, False, False]
    axis_names = ("Zweikampfprozente / %", "ps_Laufweite / km")

    class_names = tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=(1.01, 1), title=title, class_names=class_names,
                                axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Soccer_results/Soccer_classifiers.png", bbox_inches='tight')


def soccer_QSM_comparison_figure():
    plt.clf()
    plt.figure(0, figsize=(8, 12))

    qsm_tree_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\004")
    improved_tree_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\004\Splits\ps_Laufweite_005")
    qsm_NN_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\NN\004")
    improved_NN_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\NN\004\Splits\ps_Laufweite_005")

    visualized_area = vs.find_common_area(improved_tree_set.data["Zweikampfprozente"].values,
                                          improved_tree_set.data["ps_Laufweite_org"].values,
                                          improved_tree_set.data["Zweikampfprozente"].values,
                                          improved_tree_set.data["ps_Laufweite"].values)

    subplot_locs = [
        (100, 100, 0, 0, 28, 43),
        (100, 100, 36, 0, 28, 43),
        (100, 100, 72, 0, 28, 43),
        (100, 100, 0, 57, 28, 43),
        (100, 100, 36, 57, 28, 43),
        (100, 100, 72, 57, 28, 43)
    ]
    titles = ["A", "B", "C", "D", "E", "F"]
    dims = [
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite"),
        ("Zweikampfprozente", "ps_Laufweite_shifted_by_0.05"),
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite"),
        ("Zweikampfprozente", "ps_Laufweite_shifted_by_0.05")
    ]
    class_columns = ["org_pred",
                     "pred_classes",
                     "pred_with_ps_Laufweite_shifted_by_0.05",
                     "org_pred",
                     "pred_classes",
                     "pred_with_ps_Laufweite_shifted_by_0.05"]
    show_legend = [False, False, False, True, False, False]
    datasets = [improved_tree_set, improved_tree_set, qsm_tree_set, improved_NN_set, improved_NN_set, qsm_NN_set]
    axis_names = ("Zweikampfprozente / %", "ps_Laufweite / km")

    class_names = improved_tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=(1.01, 1), title=title,
                                visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Soccer_results/Soccer_QSM_comparison.png", bbox_inches='tight')


def soccer_QSM_comparison_figure_2():
    plt.clf()
    plt.figure(0, figsize=(8, 12))

    qsm_tree_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\004")
    improved_tree_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\tree\004\Splits\ps_Laufweite_005")
    qsm_NN_set = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\NN\004")
    improved_NN_set = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\NN\004\Splits\ps_Laufweite_005")

    visualized_area = vs.find_common_area(improved_tree_set.data["ps_Zweikampf"].values,
                                          improved_tree_set.data["ps_Laufweite_org"].values,
                                          improved_tree_set.data["ps_Zweikampf"].values,
                                          improved_tree_set.data["ps_Laufweite"].values)

    subplot_locs = [
        (100, 100, 0, 0, 28, 43),
        (100, 100, 36, 0, 28, 43),
        (100, 100, 72, 0, 28, 43),
        (100, 100, 0, 57, 28, 43),
        (100, 100, 36, 57, 28, 43),
        (100, 100, 72, 57, 28, 43)
    ]
    titles = ["A", "B", "C", "D", "E", "F"]
    dims = [
        ("ps_Zweikampf", "ps_Laufweite_org"),
        ("ps_Zweikampf", "ps_Laufweite"),
        ("ps_Zweikampf", "ps_Laufweite_shifted_by_0.05"),
        ("ps_Zweikampf", "ps_Laufweite_org"),
        ("ps_Zweikampf", "ps_Laufweite"),
        ("ps_Zweikampf", "ps_Laufweite_shifted_by_0.05")
    ]
    class_columns = ["org_pred",
                     "pred_classes",
                     "pred_with_ps_Laufweite_shifted_by_0.05",
                     "org_pred",
                     "pred_classes",
                     "pred_with_ps_Laufweite_shifted_by_0.05"]
    show_legend = [False, False, False, True, False, False]
    datasets = [improved_tree_set, improved_tree_set, qsm_tree_set, improved_NN_set, improved_NN_set, qsm_NN_set]
    axis_names = ("Zweikampfprozente / %", "ps_Laufweite / km")

    class_names = improved_tree_set.class_names

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=(1.01, 1), title=title,
                                visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/BA_Grafiken/Soccer_results/Soccer_QSM_comparison_2.png", bbox_inches='tight')


def parallelization_figure():
    plt.figure(0, figsize=(5, 4))
    plt.clf()

    plt.xlabel("Anzahl Datenpunkte")
    plt.ylabel("Laufzeit / s")

    number_of_points_per_class = [50, 100, 150, 200, 250, 500, 1000]
    non_parallel_data = {
        50: [1.0041487, 0.7653119999999998, 0.7623885999999995, 0.7916483000000003, 0.7480089000000012],
        100: [3.1073082, 2.779576399999999, 3.0132773000000004, 2.8404352, 2.6871526999999986],
        150: [6.7146821, 6.0561393, 6.0204731, 6.0338478, 6.162561499999999],
        200: [11.729178, 10.826789, 10.8760227, 10.795521, 10.823170300000001],
        250: [19.626736200000003, 17.8267194, 18.0243379, 17.785090900000007, 17.43169499999999],
        500: [81.05624279999999, 78.87529980000001, 74.7995133, 74.74279660000002, 74.65827630000001],
        1000: [379.0309271, 372.42410409999997, 373.5247737000001, 373.2645138999999, 372.95511520000014]
    }

    parallel_data = {
        50: [4.5908374, 4.9390367, 4.226475199999999, 3.8868492000000003, 3.778698000000002],
        100: [4.7841138, 4.7933691000000005, 4.7221858, 5.409774599999999, 4.869699100000002],
        150: [5.920484700000001, 5.5034621, 5.528682, 5.463392300000002, 5.4891383000000005],
        200: [7.605921, 6.955283999999999, 6.941353499999998, 6.839835600000001, 6.9377566],
        250: [9.731842199999999, 9.723643800000001, 8.6968392, 8.689061899999999, 9.970855100000001],
        500: [27.0167894, 25.674273200000002, 25.3907757, 25.461330200000006, 25.4961957],
        1000: [105.231246, 111.80721190000001, 118.46165230000003, 119.85438250000004, 121.58854930000007]
    }

    sems_non_parallel = [sem(non_parallel_data[points]) for points in number_of_points_per_class]
    sems_parallel = [sem(parallel_data[points]) for points in number_of_points_per_class]

    means_non_parallel = [sum(non_parallel_data[points]) / len(non_parallel_data[points]) for points in
                          number_of_points_per_class]
    means_parallel = [sum(parallel_data[points]) / len(parallel_data[points]) for points in number_of_points_per_class]

    number_of_points_per_class = [6 * points for points in number_of_points_per_class]
    print("means non:", means_non_parallel)
    print("sem nicht parallel:", sems_non_parallel)

    print("means parallel:", means_parallel)
    print("sem parallel:", sems_parallel)

    plt.errorbar(x=number_of_points_per_class, y=means_non_parallel, yerr=sems_non_parallel,
                 label="nicht parallelisiert", fmt="o")
    plt.errorbar(x=number_of_points_per_class, y=means_parallel, yerr=sems_parallel, label="parallelisiert",
                 fmt="o")

    plt.yscale('log')

    plt.legend(loc="upper left", frameon=True)
    plt.savefig("../Plots/BA_Grafiken/Parallelisierung.png", bbox_inches='tight')
    # plt.show()


def big_figure_for_paper():
    plt.clf()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(0, figsize=(8, 15))

    qsm_NN_set_hole = dc.Data.load(r"..\Data\Parameters2\MaybeActualDataSet\NN\020")
    improved_NN_set_hole = dc.Data.load(
        r"..\Data\Parameters2\MaybeActualDataSet\NN\012\Splits\dim_04_005")

    visualized_area_hole = vs.find_common_area(improved_NN_set_hole.data["dim_00"].values,
                                               improved_NN_set_hole.data["dim_04_org"].values,
                                               improved_NN_set_hole.data["dim_00"].values,
                                               improved_NN_set_hole.data["dim_04"].values)

    qsm_NN_set_iris = dc.Data.load(r"..\Data\Parameters2\IrisDataSet\NN\004")
    improved_NN_set_iris = dc.Data.load(
        r"..\Data\Parameters2\IrisDataSet\NN\004\Splits\petal_length_005")

    visualized_area_iris = vs.find_common_area(improved_NN_set_iris.data["petal_width"].values,
                                               improved_NN_set_iris.data["petal_length_org"].values,
                                               improved_NN_set_iris.data["petal_width"].values,
                                               improved_NN_set_iris.data["petal_length"].values)

    subplot_locs = [
        (100, 100, 0, 57, 20, 43),
        (100, 100, 26, 57, 20, 43),
        (100, 100, 52, 57, 20, 43),
        (100, 100, 78, 57, 20, 43),
        (100, 100, 0, 0, 20, 43),
        (100, 100, 26, 0, 20, 43),
        (100, 100, 52, 0, 20, 43),
        (100, 100, 78, 0, 20, 43)
    ]
    titles = ["original hole data set", "Hole data set labelled by NN",
              "QSM results on Hole data set",
              "Improved QSM results on Hole data set", "original iris data set",
              "Iris data set labelled by NN", "QSM results on Iris data set",
              "Improved QSM results on Iris data set"]
    dims = [
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04_org"),
        ("dim_00", "dim_04_shifted_by_0.05"),
        ("dim_00", "dim_04"),
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length_org"),
        ("petal_width", "petal_length_shifted_by_0.05"),
        ("petal_width", "petal_length")
    ]
    class_columns = ["classes",
                     "org_pred",
                     "pred_with_dim_04_shifted_by_0.05",
                     "pred_classes",
                     "classes",
                     "org_pred",
                     "pred_with_petal_length_shifted_by_0.05",
                     "pred_classes"]
    show_legend = [True, False, False, False, True, False, False, False]
    datasets = [improved_NN_set_hole, improved_NN_set_hole, qsm_NN_set_hole, improved_NN_set_hole,
                improved_NN_set_iris, improved_NN_set_iris, qsm_NN_set_iris, improved_NN_set_iris]
    axis_names = [("dim_00", "dim_04"), ("dim_00", "dim_04"), ("dim_00", "dim_04"), ("dim_00", "dim_04"),
                  ("petal_width / cm", "petal_length / cm"), ("petal_width / cm", "petal_length / cm"),
                  ("petal_width / cm", "petal_length / cm"), ("petal_width / cm", "petal_length / cm")]

    class_name_hole = improved_NN_set_hole.class_names
    class_name_iris = improved_NN_set_iris.class_names

    class_names = [["C0", "C1", "C2", "C3", "C4", "C5"], class_name_hole, class_name_hole, class_name_hole, class_name_iris, class_name_iris,
                   class_name_iris, class_name_iris]

    visualized_areas = [visualized_area_hole, visualized_area_hole, visualized_area_hole, visualized_area_hole,
                        visualized_area_iris, visualized_area_iris, visualized_area_iris, visualized_area_iris]

    legend_locs = [(-0.03, 1.45), (1.01, 1), (1.01, 1), (1.01, 1),
                   (-0.03, 1.45), (1.01, 1), (1.01, 1), (1.01, 1)]

    legend_columns = [2, 1, 1, 1, 1, 1, 1, 1]

    for i, (plot_loc, dim, title, class_column, legend, dataset, axis_name, class_name, visualized_area, legend_loc, legend_ncol) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets, axis_names, class_names, visualized_areas, legend_locs, legend_columns)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=legend_loc, title=title,
                                visualized_area=visualized_area, frame_on=True,
                                class_names=class_name, axis_names=axis_name, ncol_legend=legend_ncol)

    #plt.show()
    plt.savefig("../Plots/Paper_Grafiken/big_graphic.pdf", bbox_inches='tight')


def main() -> None:
    """
    just a test function
    """
    set_09_1 = dc.Data.load(
        r"D:\Gernot\Programmieren\Bachelor\Data\Parameters2\SoccerDataSet\NN\004")
    matrix_ = vs.get_change_matrix(set_09_1.data, ("org_pred_classes_QSM", "pred_with_ps_Laufweite_shifted_by_0.05"),
                                   class_names=set_09_1.class_names)
    matrix_.to_csv(r"C:\Users\gerno\OneDrive\Desktop\matrix.csv")
    print(matrix_)
    # visualize_3d(df, ("dim_01", "dim_02", "dim_03"))
    # for i in range(36):
    #    visualize_3d(df, ("dim_01", "dim_02", "dim_03"), class_column="classes", azim=10*i, elev=-150)


def soccerDataSet_figure_paper():
    plt.clf()
    plt.figure(0, figsize=(8, 9))
    sds = dc.SoccerDataSet()
    subplot_locs = [
        (100, 100, 0, 0, 44, 43),
        (100, 100, 0, 57, 44, 43),
        (100, 100, 56, 0, 44, 43),
        (100, 100, 56, 57, 44, 43)
    ]
    titles = ["A", "B", "C", "D"]
    dims = [
        ("ps_Pass", "Passprozente"),
        ("Zweikampfprozente", "ps_Laufweite"),
        ("ps_Fouls", "ps_Gefoult"),
        ("ps_Zweikampf", "ps_Abseits")
    ]
    axis_namess = [("ps_Pass", "Passprozente / %"),
                   ("ps_Zweikampf", "Zweikampfprozente / %"),
                   ("ps_Fouls", "ps_Gefoult"),
                   ("ps_Laufweite / km", "ps_Abseits")]

    for i, (plot_loc, dim, title, axis_names) in enumerate(zip(subplot_locs, dims, titles, axis_namess)):
        if i == 1:
            vs.visualize_2d_subplot(df=sds.data, dims=dim, subplot_location=plot_loc, class_column="classes",
                                    class_names=sds.class_names, show_legend=True, bbox_to_anchor=(1.01, 1),
                                    title=title,
                                    axis_names=axis_names)
        else:
            vs.visualize_2d_subplot(df=sds.data, dims=dim, subplot_location=plot_loc, class_column="classes",
                                    class_names=sds.class_names, title=title, axis_names=axis_names)

    #plt.show()
    plt.savefig("../Plots/Paper_Grafiken/Soccer_intro.png", bbox_inches='tight')


def soccer_QSM_comparison_figure_paper():
    plt.clf()
    plt.figure(0, figsize=(8, 9))
    qsm_NN_set = dc.Data.load(r"..\Data\Parameters2\SoccerDataSet\NN\004")
    improved_NN_set = dc.Data.load(
        r"..\Data\Parameters2\SoccerDataSet\NN\004\Splits\ps_Laufweite_005")

    visualized_area = vs.find_common_area(improved_NN_set.data["Zweikampfprozente"].values,
                                          improved_NN_set.data["ps_Laufweite_org"].values,
                                          improved_NN_set.data["Zweikampfprozente"].values,
                                          improved_NN_set.data["ps_Laufweite"].values)

    subplot_locs = [
        (100, 100, 0, 0, 28, 43),
        (100, 100, 0, 57, 28, 43),
        (100, 100, 36, 0, 28, 43),
        (100, 100, 36, 57, 28, 43)
    ]
    titles = ["A", "B", "C", "D", "E", "F"]
    dims = [
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite_org"),
        ("Zweikampfprozente", "ps_Laufweite_shifted_by_0.05"),
        ("Zweikampfprozente", "ps_Laufweite")
    ]
    class_columns = ["classes",
                     "org_pred",
                     "pred_with_ps_Laufweite_shifted_by_0.05",
                     "pred_classes"]
    show_legend = [False, True, False, False]
    datasets = [improved_NN_set, improved_NN_set, qsm_NN_set, improved_NN_set]
    axis_names = ("Zweikampfprozente / %", "ps_Laufweite / km")

    class_names = improved_NN_set.class_names

    for i, (plot_loc, dim, title, class_column, legend, dataset) in enumerate(
            zip(subplot_locs, dims, titles, class_columns, show_legend, datasets)):
        vs.visualize_2d_subplot(df=dataset.data, dims=dim, subplot_location=plot_loc, class_column=class_column,
                                show_legend=legend, bbox_to_anchor=(1.01, 1), title=title,
                                visualized_area=visualized_area,
                                class_names=class_names, axis_names=axis_names)

    # plt.show()
    plt.savefig("../Plots/Paper_Grafiken/Soccer_QSM_comparison.png", bbox_inches='tight')


if __name__ == "__main__":
    # main()
    # Iris_QSM_comparison_figure_quer()
    soccer_QSM_comparison_figure_paper()
