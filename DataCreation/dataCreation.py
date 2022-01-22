from typing import List, Tuple, Dict
import DataCreation.visualization as visualization

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from abc import ABC, abstractmethod
import numpy as np


def add_gaussian_noise(data: pd.Series, sd: float) -> pd.Series:
    length = len(data)
    noise = np.random.normal(0, sd, length)
    return data + noise


def add_random_dims(data: pd.DataFrame, dim_names: List[str]) -> None:
    length = len(data)
    for name in dim_names:
        range_ = np.random.uniform(-1, 1, 2)*1000
        data[name] = [val for val in np.random.uniform(range_[0], range_[1], length)]


class Data(ABC):
    data: pd.DataFrame
    members: List[int]


class NaturalData(Data):
    def __init__(self, members: List[int]):
        np.random.seed(42)
        if len(members) != 5:
            raise Exception("members needs to be initialized with a List of length 5!")
        self.members = members
        class_0 = self.create_class_0()
        class_1 = self.create_class_1()
        class_2 = self.create_class_2()
        class_3 = self.create_class_3()
        class_4 = self.create_class_4()
        self.data = pd.concat([class_0, class_1, class_2, class_3, class_4])
        add_random_dims(self.data, ["rand_00", "rand_01", "rand_02"])

    def create_class_0(self):
        new_df = pd.DataFrame()
        new_df["classes"] = [0 for _ in range(self.members[0])]

        dim_00 = np.random.normal(-47, 10, (self.members[0],))
        new_df["dim_00"] = dim_00

        dim_01 = np.random.normal(30, 10, (self.members[0],))
        new_df["dim_01"] = dim_01

        dim_02 = np.random.normal(0, 20, (self.members[0],))
        new_df["dim_02"] = dim_02
        return new_df

    def create_class_1(self):
        new_df = pd.DataFrame()
        new_df["classes"] = [1 for _ in range(self.members[1])]

        dim_00 = np.random.normal(140, 50, (self.members[1],))
        new_df["dim_00"] = dim_00

        dim_01 = np.random.normal(80, 50, (self.members[1],))
        new_df["dim_01"] = dim_01

        dim_02 = np.random.normal(50, 20, (self.members[1],))
        new_df["dim_02"] = dim_02

        return new_df

    def create_class_2(self):
        new_df = pd.DataFrame()
        new_df["classes"] = [2 for _ in range(self.members[2])]

        dim_00 = np.random.normal(-50, 20, (self.members[2],))
        new_df["dim_00"] = dim_00

        dim_01 = np.random.normal(150, 40, (self.members[2],))
        new_df["dim_01"] = dim_01

        dim_02 = np.random.normal(0, 20, (self.members[2],))
        new_df["dim_02"] = dim_02
        return new_df

    def create_class_3(self):
        new_df = pd.DataFrame()
        new_df["classes"] = [3 for _ in range(self.members[3])]

        dim_00 = np.random.normal(20, 15, (self.members[3],))
        new_df["dim_00"] = dim_00

        dim_01 = np.random.normal(80, 25, (self.members[3],))
        new_df["dim_01"] = dim_01

        dim_02 = np.random.normal(30, 20, (self.members[3],))
        new_df["dim_02"] = dim_02
        return new_df

    def create_class_4(self):
        new_df = pd.DataFrame()
        new_df["classes"] = [4 for _ in range(self.members[4])]

        dim_01 = np.random.normal(0, 25, (self.members[4],))
        new_df["dim_01"] = dim_01

        dim_00 = np.random.normal(30, 20, (self.members[4],))
        new_df["dim_00"] = dim_00

        dim_02 = np.random.normal(30, 20, (self.members[4],))
        new_df["dim_02"] = dim_02

        return new_df


class GeometricUniformData(Data):
    def __init__(self, members: List[int]):
        np.random.seed(42)
        self.members = members
        self.create_data()
        add_random_dims(self.data, ["rand_00", "rand_01", "rand_02"])

    #todo: different density of data in overlap of classes, probably needs to be removed -.-
    @staticmethod
    def get_low_values(length: int):
        return np.random.uniform(-0.1, 1.1, length)

    @staticmethod
    def get_high_values(length: int):
        return np.random.uniform(0.9, 2.1, length)

    def create_class_data(self, dist: Tuple[int, int, int], class_number: int):
        df = pd.DataFrame()
        x_high, y_high, z_high = dist
        try:
            members = self.members[class_number]
        except IndexError:
            members = 0

        df["classes"] = [class_number for _ in range(members)]
        df["dim_00"] = self.get_high_values(members) if x_high else self.get_low_values(members)
        df["dim_01"] = self.get_high_values(members) if y_high else self.get_low_values(members)
        df["dim_02"] = self.get_high_values(members) if z_high else self.get_low_values(members)
        return df

    def create_data(self):
        class_list = []
        class_number = 0
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    class_list.append(self.create_class_data((x, y, z), class_number))
                    class_number += 1
        self.data = pd.concat(class_list)


#todo: maybe rework the system, using one (or 2) "ground truth" dims with no noise, to calculate from; dim1 and 2 are then noisy version of ground truth
class SimpleCorrelationData(Data):
    def __init__(self, members: List[int], squared_classes=False):
        np.random.seed(42)
        self.members = members
        self.create_data(squared_classes)
        add_random_dims(self.data, ["rand_00", "rand_01", "rand_02"])
       
    @staticmethod    
    def fill_square(x_list: List[int],
                    y_list: List[int],
                    x_start: int,
                    x_end: int,
                    y_start: int,
                    y_end: int,
                    number_of_points: int):
        x_list = np.append(x_list, np.random.uniform(x_start, x_end, number_of_points))
        y_list = np.append(y_list, np.random.uniform(y_start, y_end, number_of_points))
        return x_list, y_list

    #if corner is false: number is calculated for edge
    @staticmethod
    def get_members(number_of_points: int, ratio: float, corner: bool):
        if corner:
            return int(number_of_points / 4 * ratio) + np.random.randint(0, 2)
        return int(number_of_points / 4 * (1 - ratio)) + np.random.randint(0, 2)

    @staticmethod
    def plot_progress(dim_00, dim_01):
        df = pd.DataFrame()
        df["1"] = dim_01
        df["0"] = dim_00
        print("progress:\n", df.describe())
        visualization.visualize_2d(df, ("0", "1"))

    #area is the area of the outer square
    def fill_square_shell(self, number_of_points: int, area: int, dim_00, dim_01):
        if area == 1:
            dim_00 = np.append(dim_00, np.random.uniform(-.5, .5, number_of_points))
            dim_01 = np.append(dim_01, np.random.uniform(-.5, .5, number_of_points))
        else:
            outer_bound = np.sqrt(area) / 2
            inner_bound = np.sqrt(area - 1) / 2
            diff = outer_bound - inner_bound
            #ratio of corner compared to corner + side
            ratio = diff / (2 * outer_bound)
            #fill corners
            corner_points = number_of_points * ratio
            points_in_corner_1 = self.get_members(number_of_points, ratio, True)
            points_in_corner_2 = self.get_members(number_of_points, ratio, True)
            points_in_corner_3 = self.get_members(number_of_points, ratio, True)
            points_in_corner_4 = int(corner_points - points_in_corner_1 - points_in_corner_2 - points_in_corner_3)
            dim_00, dim_01 = self.fill_square(dim_00, dim_01, inner_bound, outer_bound, inner_bound, outer_bound, points_in_corner_1)
            dim_00, dim_01 = self.fill_square(dim_00, dim_01, inner_bound, outer_bound, -1 * outer_bound, -1 * inner_bound, points_in_corner_2)
            dim_00, dim_01 = self.fill_square(dim_00, dim_01, -1 * outer_bound, -1 * inner_bound, inner_bound, outer_bound, points_in_corner_3)
            dim_00, dim_01 = self.fill_square(dim_00, dim_01, -1 * outer_bound, -1 * inner_bound, -1 * outer_bound, -1 * inner_bound, points_in_corner_4)

            #fill edges
            points_in_edge_1 = self.get_members(number_of_points, ratio, False)
            points_in_edge_2 = self.get_members(number_of_points, ratio, False)
            points_in_edge_3 = self.get_members(number_of_points, ratio, False)
            points_in_edge_4 = int(number_of_points - points_in_corner_1 - points_in_corner_2 - points_in_corner_3 -
                                   points_in_corner_4 - points_in_edge_1 - points_in_edge_2 - points_in_edge_3)
            dim_00, dim_01 = self.fill_square(dim_00, dim_01, -1 * inner_bound, inner_bound, inner_bound, outer_bound, points_in_edge_1)
            dim_00, dim_01 = self.fill_square(dim_00, dim_01, -1 * inner_bound, inner_bound, -1 * outer_bound, -1 * inner_bound, points_in_edge_2)
            dim_00, dim_01 = self.fill_square(dim_00, dim_01, -1 * outer_bound, -1 * inner_bound, -1 * inner_bound, inner_bound, points_in_edge_3)
            dim_00, dim_01 = self.fill_square(dim_00, dim_01, inner_bound, outer_bound, -1 * inner_bound, inner_bound, points_in_edge_4)

        return dim_00, dim_01

    #todo: check if polar coordinates make distributions seem more similar for dim_00 and dim_01
    @staticmethod
    def fill_circle_shell(number_of_points: int, area: int, dim_00, dim_01):
        outer_radius = np.sqrt(area / np.pi)
        inner_radius = np.sqrt((area - 1) / np.pi)
        for _ in range(number_of_points):
            x = np.random.uniform(-1 * outer_radius, outer_radius)
            y_high = np.sqrt((outer_radius**2) - (x**2))
            if np.abs(x) >= inner_radius:
                y_low = 0
            else:
                y_low = np.sqrt((inner_radius**2) - (x**2))
            #up_down determines if y will be positive or negative
            up_down = np.random.randint(0, 2)
            if up_down:
                y = np.random.uniform(y_low, y_high)
            else:
                y = np.random.uniform(-1 * y_high, -1 * y_low)
            dim_00 = np.append(dim_00, x)
            dim_01 = np.append(dim_01, y)

        return dim_00, dim_01

    def create_dim_0_1(self, squared_classes: bool):
        data = self.data
        dim_00 = []
        dim_01 = []
        classes = []
        for i, value in enumerate(self.members):
            if squared_classes:
                dim_00, dim_01 = self.fill_square_shell(value, i + 1, dim_00, dim_01)
            else:
                dim_00, dim_01 = self.fill_circle_shell(value, i + 1, dim_00, dim_01)
            #classes are square shells of area 1 in dim00/dim01
            classes.extend([i for _ in range(value)])
        data["dim_00"] = add_gaussian_noise(dim_00, .03)
        data["dim_01"] = add_gaussian_noise(dim_01, .03)
        data["classes"] = classes
        self.data = data

    def create_dim_2_3_4(self):
        data = self.data
        fuzzy_00 = add_gaussian_noise(data["dim_00"], 0.1)
        fuzzy_01 = add_gaussian_noise(data["dim_01"], 0.1)
        data["dim_02"] = fuzzy_01 * fuzzy_01 + fuzzy_00 * fuzzy_00
        fuzzy_00 = add_gaussian_noise(data["dim_00"], 0.1)
        fuzzy_01 = add_gaussian_noise(data["dim_01"], 0.1)
        data["dim_03"] = fuzzy_01 * fuzzy_00
        fuzzy_00 = add_gaussian_noise(data["dim_00"], 0.1)
        fuzzy_01 = add_gaussian_noise(data["dim_01"], 0.1)
        data["dim_04"] = fuzzy_01 + fuzzy_00


    def create_dim_3(self):
        pass

    def create_data(self, squared_classes: bool) -> None:
        data = pd.DataFrame()
        self.data = data
        self.create_dim_0_1(squared_classes)
        self.create_dim_2_3_4()


class MaybeActualDataSet(Data):
    class_00_param: Dict = {"dim_00": (0, 0.8),   "dim_01": (0, 0.8),   "dim_02": (0, 0.8),   "dim_03": (0, 0.8),   "dim_04": (0, 1, 2)}
    class_01_param: Dict = {"dim_00": (1.5, 0.8), "dim_01": (1.5, 0.8), "dim_02": (0, 0.8),   "dim_03": (0, 0.8),   "dim_04": (1, 2, 3)}
    class_02_param: Dict = {"dim_00": (1.5, 0.8), "dim_01": (1.5, 0.8), "dim_02": (0, 0.8),   "dim_03": (0, 0.8),   "dim_04": (5, 6, 7)}
    class_03_param: Dict = {"dim_00": (0, 0.8),   "dim_01": (0, 0.8),   "dim_02": (1.5, 0.8), "dim_03": (1.5, 0.8), "dim_04": (5.5, 6.5, 7.5)}
    class_04_param: Dict = {"dim_00": (0, 0.8),   "dim_01": (0, 0.8),   "dim_02": (1.5, 0.8), "dim_03": (1.5, 0.8), "dim_04": (6.5, 7.5, 8.5)}

    parameters: Dict

    def __init__(self, members: List[int]):
        np.random.seed(42)
        self.parameters = {"class_00": self.class_00_param,
                           "class_01": self.class_01_param,
                           "class_02": self.class_02_param,
                           "class_03": self.class_03_param,
                           "class_04": self.class_04_param}
        self.members = members
        self.create_data()
        add_random_dims(self.data, ["rand_00", "rand_01", "rand_02"])


    @staticmethod
    def create_class(class_number: int, members: int, class_parameters: Dict):
        new_df = pd.DataFrame()
        new_df["classes"] = [class_number for _ in range(members)]

        for i in range(4):
            center, standard_deviation = class_parameters[f"dim_{str(i).zfill(2)}"]
            dim = np.random.normal(center, standard_deviation, (members,))
            new_df[f"dim_{str(i).zfill(2)}"] = dim
        return new_df

    def add_dim_04(self):
        dim = []
        for i in range(5):
            low, middle, high = self.parameters[f"class_{str(i).zfill(2)}"]["dim_04"]
            try:
                dim.extend(np.random.triangular(low, middle, high, (self.members[i], )))
            except IndexError:
                pass
        self.data["dim_04"] = dim

    def create_data(self):
        data = pd.DataFrame()
        for i in range(5):
            try:
                class_members = self.members[i]
                class_parameters = self.parameters[f"class_{str(i).zfill(2)}"]
                data = data.append(self.create_class(i, class_members, class_parameters))
            except IndexError:
                pass
        self.data = data
        self.add_dim_04()





if __name__ == "__main__":
    members_ = [200 for _ in range(5)]
    df = MaybeActualDataSet(members_).data
    #visualization.visualize_2d(df, ("dim_00", "dim_01"), class_column="classes")
    #visualization.visualize_2d(df, ("dim_00", "dim_02"), class_column="classes")
    #visualization.visualize_2d(df, ("dim_00", "dim_03"), class_column="classes")
    #visualization.visualize_2d(df, ("dim_00", "dim_04"), class_column="classes")
    #visualization.create_cumulative_plot(df["dim_00"].values)
    #visualization.create_cumulative_plot(df["dim_04"].values)
    #visualization.create_hist(df["dim_00"])
    #visualization.create_hist(df["dim_04"])
    #visualization.visualize_2d(df, ("dim_00", "dim_01"), class_column="classes")
    #visualization.visualize_2d(df, ("dim_00", "dim_04"), class_column="classes")
    #visualization.visualize_3d(df, ("dim_00", "dim_01", "dim_02"), class_column="classes")
    #visualization.create_3d_gif(df=df, dims=("dim_01", "dim_03", "dim_04"), name="maybe_actual_data", class_column="classes", steps=120, duration=33)


