from typing import List, Tuple
import visualization

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from abc import ABC, abstractmethod
import numpy as np


"""
3 – 5 Classes
5 – 20 Variables with Information
~5 random Variables
Create Dependencies that require Binning
1.000 – 10.000 Datenpunkte
MVTNorm oä Pakete für synthetische Datensätze
"""


def add_gaussian_noise(data: pd.Series, sd: int):
    length = len(data)
    noise = np.random.normal(0, sd, length)
    return data + noise


def add_random_dims(data: pd.DataFrame, dim_names: List[str]):
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


if __name__ == "__main__":
    members = [200 for _ in range(5)]
    df = GeometricUniformData(members).data
    visualization.visualize_2d(df, ("dim_00", "dim_01"), class_column="classes")
    visualization.visualize_2d(df, ("dim_01", "dim_02"), class_column="classes")
    visualization.visualize_2d(df, ("dim_02", "dim_00"), class_column="classes")
    #visualization.visualize_3d(df, ("dim_00", "dim_01", "rand3"), class_column="classes")
    #visualization.create_3d_gif(df=df, dims=("dim_00", "dim_01", "dim_02"), name="cubes_6_classes", class_column="classes", steps=72, duration=50)


