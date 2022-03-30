from __future__ import annotations
from typing import List, Tuple, Dict

import sklearn.tree as tree

import Python.visualization as vs
import HiCS

import datetime
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import os
import pickle


path_to_data = "/Python/Experiments/Data"


class CustomError(Exception):
    pass


def add_gaussian_noise(data: pd.Series, sd: float) -> pd.Series:
    """
    Add gaussian noise to data
    :param data: pd.Series to add noise to
    :param sd: standard deviation for the noise
    :return: a new pd.Series object, with the original data blurred by the noise
    """
    length = len(data)
    noise = np.random.normal(0, sd, length)
    return data + noise


def add_random_dims(data: pd.DataFrame, dim_names: List[str]) -> None:
    """
    Adds new columns to the data containing uniformly distributed random values in a random range between -1000 and 1000.
    :param data: The dataframe the new columns will be appended to
    :param dim_names: The names of the columns to be generated. Length of dim_names determines how many new columns will be created
    """
    length = len(data)
    for name in dim_names:
        range_ = np.random.uniform(-1, 1, 2)*1000
        data[name] = [val for val in np.random.uniform(range_[0], range_[1], length)]


def get_date_from_string(created_line: str):
    """
    converts a string in the format of saved data classes and returns a datetime object of that time.
    :param created_line: string describing the time
    :return: datetime object of the time
    """
    date = created_line.split(" ")[1]
    time = created_line.split(" ")[2]
    day, month, year = date.split(".")
    hours, minutes, seconds = time.split(":")
    year = int(year)
    day = int(day)
    month = int(month)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    now = datetime.datetime(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds,
                            microsecond=0)
    return now


class Data(ABC):
    data: pd.DataFrame

    #names of the columns, that contain data (not classes or predicted classes etc)
    data_columns: List[str]
    members: List[int]
    path: str
    now: datetime.datetime
    HiCS_dims: List[str]
    notes: str or None
    note_buffer: List[str]

    def __init__(self, path: str = ""):
        self.HiCS_dims = []
        self.notes = None
        self.note_buffer = []
        class_name = type(self).__name__
        now = datetime.datetime.now()
        self.now = now
        now_string = now.strftime("%y%m%d_%H%M%S")
        if not path:
            path = os.path.join(os.path.dirname(__file__), "..", "Data", f"{now_string}_{class_name}")
        self.path = path

    def load_tree(self) -> tree.DecisionTreeClassifier:
        tree_path = os.path.join(self.path, "tree_classifier.pkl")
        if os.path.isfile(tree_path):
            with open(tree_path, "rb") as f:
                decision_tree = pickle.load(f)
            return decision_tree
        else:
            raise CustomError("No Tree was saved, so no tree can be loaded!")

    @staticmethod
    @abstractmethod
    def load(path: str) -> Data:
        """
        loads a data set from a given path
        :param path: path to the data set
        :return: the loaded dataset object
        """
        pass

    @abstractmethod
    def clone_meta_data(self, path: str = "") -> Data:
        pass

    @abstractmethod
    def take_new_data(self, data: pd.DataFrame) -> None:
        pass

    def buffer_note(self, note: str) -> None:
        """
        appends a note to the note_buffer
        :param note: note to be appended
        """
        self.note_buffer.append(note)

    def create_buffered_notes(self) -> None:
        """
        appends all notes in the note_buffer as a paragraph to the notes of the dataset.
        """
        for note in self.note_buffer:
            self.extend_notes_by_one_line(note)
        self.end_paragraph_in_notes()
        self.note_buffer = []

    def end_paragraph_in_notes(self):
        """
        appends a line of minuses to the notes, to signal the end of a paragraph
        """
        self.extend_notes_by_one_line("-------------------")

    def extend_notes_by_one_line(self, notes: str):
        """
        extends the notes string by the given notes
        :param notes: notes to be appended
        """
        if self.notes:
            if notes:
                self.notes += "\n" + (11 * " ") + notes
        else:
            self.notes = notes

    def write_description(self, file) -> None:
        """
        writes the attributes and date for the object in the description file.
        :param file: file thats written in
        """
        proper_date_time_string = self.now.strftime("%d.%m.%Y %H:%M:%S")
        file.write(f"CREATED: {proper_date_time_string}\nATTRIBUTES: \n")
        for k, v in vars(self).items():
            if str(k) == "data":
                continue
            file.write(f"    {str(k)}: {str(v)} \n")
        file.write("\n")

    def update_saved_info(self) -> None:
        """
        overwrites old saved data for the object and replaces it with the current information of the object
        """
        with open(os.path.join(self.path, "description.txt"), "r+") as f:
            _ = f.readline()
            rest = f.read()

            rest = rest.strip("\n")
            paragraphs = rest.split("\n\n")
            f.truncate(0)
            f.seek(0)
            self.write_description(f)
            for i, par in enumerate(paragraphs):
                if i != 0:
                    f.write(par)
                    f.write("\n\n")

    def save(self, save_path: str = "", notes: str = "") -> None:
        """
        creates a folder with a "description.txt" file, which contains the attributes of the object and possibly notes.
        Also, the creation date is saved.
        :param notes: notes to be saved
        :param save_path: name of folder, where the information will be saved
        """
        #update notes
        self.extend_notes_by_one_line(notes)

        if save_path:
            self.path = save_path
        path = self.path

        folder_exists = False
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            if not save_path:
                folder_exists = True
            else:
                print("warning, writing into existing folder!")
        if not folder_exists:
            # create new info for class
            with open(os.path.join(path, "description.txt"), "w") as f:
                self.write_description(f)
        else:
            #only update info
            self.update_saved_info()
        #data has to be updated/saved either way
        self.data.to_csv(os.path.join(path, "data.csv"), index=False)

    def save_data_for_hics(self, path: str = "") -> str:
        """
        Saves the "data" attribute without the "classes" column as a csv file in the folder for the description of the
        object or at a given path.
        :param path: Path to a location where the csv is supposed to be saved, if not in the folder describing the object
        :return: the path to the file
        """
        if not path:
            self.save()
            path = self.path
        #no reason to overwrite old data for HiCS
        if "HiCS_Input_Data.csv" not in os.listdir(path):
            data_for_hics = self.data[self.data_columns]
            file_path = os.path.join(path, "HiCS_Input_Data.csv")
            #HiCS program expects a csv file with ";" as delimiter. index column is not useful
            data_for_hics.to_csv(file_path, sep=";", index=False)
            return file_path
        return os.path.join(path, "HiCS_Input_Data.csv")

    def add_notes_for_HiCS(self, notes: List[str], params: List[str]) -> None:
        """
        adds notes for HiCS to a dataset object, depending on the parameters given for HiCS as well as possible additional
        notes
        :param notes: additional notes to be saved
        :param params: parameters for HiCS
        """
        self.extend_notes_by_one_line("Notes for HiCS:")
        for note in notes:
            self.extend_notes_by_one_line(note)
        self.extend_notes_by_one_line("Parameters:")
        string: str = ""
        for i, param in enumerate(params):
            if i % 2 == 0:
                string = param
            else:
                string += f" {param}"
                self.extend_notes_by_one_line(string)
        self.end_paragraph_in_notes()
        self.save()

    @staticmethod
    def create_csv_out(csv_in: str) -> str:
        """
        creates a path to save the output of HiCS, depending on the location of the input for HiCS
        :param csv_in: location of input file for HiCS
        :return: Path to a output file for HiCS
        """
        last_slash = csv_in.rfind("/")
        last_backslash = csv_in.rfind("\\")
        if last_slash > last_backslash:
            path = csv_in[0:last_slash]
        else:
            path = csv_in[0:last_backslash]
        csv_out = os.path.join(path, "HiCS_output.csv")
        return csv_out


    def run_hics(self, csv_out: str = "", further_params: List[str] = None,
                 notes: List[str] = "", silent:bool = True, csv_in: str = "") -> None:
        """
        runs HiCS for the Data object. First creates Info for the object using "create_class_info" and saves the input
        for HiCS into that folder. Output of Hics is also written into that folder, unless specified otherwise.
        :param csv_out: location and file name for output
        :param further_params: parameters for HiCS
        :param notes: notes to be saved in the description of the Data set
        :param silent: determines whether HiCS generates progress information, which will be printed to the console
        :param csv_in: possible path to a file with input data for HiCS. If not given the Data will be generated
        """
        if further_params is None:
            further_params = []
        if not csv_in:
            csv_in = self.save_data_for_hics()
        if "HiCS_output.csv" not in os.listdir(self.path):
            if not csv_out:
                csv_out = self.create_csv_out(csv_in)
            arguments = ["--csvIn", f"{csv_in}", "--csvOut", f"{csv_out}", "--hasHeader", "true"]
            if silent:
                arguments.extend(["-s", "true"])
            params = arguments + further_params
            HiCS.run_HiCS(params)
            self.add_notes_for_HiCS(notes=notes, params=params)


class NaturalData(Data):
    def __init__(self, members: List[int], notes="") -> None:
        np.random.seed(42)
        if len(members) < 5:
            raise Exception("members needs to be initialized with a List of min length 5!")
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


#todo: maybe rework the system, using one (or 2) "ground truth" dims with no noise, to calculate from; dim1 and 2 are
# then noisy version of ground truth
class SimpleCorrelationData(Data):
    def __init__(self, members: List[int], squared_classes=False):
        super().__init__()
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
        vs.visualize_2d(df, ("0", "1"))

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

    """
    Parameters for the distribution of the classes in the different dimensions. for "dim_00" to "dim_03" the first
    value is the mean und the second value is the stadard deviation of a random Gauss distribution.
    For "dim_04" the values represent the low, middle and high values for the triangular random distribution
    """
    class_params = [
        {"dim_00": (0,   0.8), "dim_01": (0,   0.8), "dim_02": (0,  0.8), "dim_03": (0,   0.8), "dim_04": (0,   1,   2)},
        {"dim_00": (1.5, 0.8), "dim_01": (1.5, 0.8), "dim_02": (0,  0.8), "dim_03": (0,   0.8), "dim_04": (1,   2,   3)},
        {"dim_00": (1.5, 0.8), "dim_01": (1.5, 0.8), "dim_02": (0,  0.8), "dim_03": (0,   0.8), "dim_04": (5,   6,   7)},
        {"dim_00": (.5,  0.8), "dim_01": (0,   0.8), "dim_02": (2,  0.8), "dim_03": (0,   0.8), "dim_04": (5,   6,   7)},
        {"dim_00": (-.5, 0.8), "dim_01": (1.5, 0.8), "dim_02": (1,  0.8), "dim_03": (1.5, 0.8), "dim_04": (5,   6,   7)},
        {"dim_00": (-2,  0.8), "dim_01": (-2,  0.8), "dim_02": (-2, 0.8), "dim_03": (-2,  0.8), "dim_04": (3,   4,   5)}
    ]

    parameters: Dict

    def __init__(self, members: List[int], path: str = "", notes: str = "", save: bool = True):
        """
        initializes the data class
        :param members: entries determine the number of data points per class
        :param notes: notes to be put in the description.txt file for the class
        """
        super().__init__(path=path)
        np.random.seed(42)
        self.parameters = {}
        for i, class_param in enumerate(self.class_params):
            self.parameters[f"class_{str(i).zfill(2)}"] = class_param
        self.members = members
        self.notes = notes
        self.create_data()
        add_random_dims(self.data, ["rand_00", "rand_01", "rand_02"])
        self.data_columns = [value for value in self.data.columns.values if value != "classes"]
        if save:
            self.save()

    @staticmethod
    def parse_class_dict(class_dict: str) -> Dict:
        res_dict = {}
        pair_index = 0
        while pair_index > -1:
            pair_index = class_dict.find(")")
            if pair_index > -1:
                pair = class_dict[:pair_index+1]
                class_dict = class_dict[pair_index+3:]
            else:
                break
            dim_name = pair.split(": ")[0].strip("\'")
            tuple_values = pair.split(": ")[1].strip("()").split(", ")
            tuple_ = tuple([float(value) for value in tuple_values])
            res_dict[dim_name] = tuple_
        return res_dict

    @staticmethod
    def parse_parameters(value: str) -> Dict:
        value = value[1:-1]
        result_dict = {}
        class_index = 0
        while class_index > -1:
            class_index = value.find("}")
            if class_index > -1:
                curr_class = value[:class_index+1]
                value = value[class_index+3:]
            else:
                curr_class = value
            name_index = curr_class.find(": ")
            class_name = curr_class[:name_index].strip("\'")
            class_dict = curr_class[name_index+2:].strip("{}")
            result_dict[class_name] = MaybeActualDataSet.parse_class_dict(class_dict)
        return result_dict

    @staticmethod
    def read_list(list_string) -> List[str]:
        list_string = list_string.strip("[] ")
        list_string = list_string.replace(" ", "").replace("\'", "")
        return [member for member in list_string.split(",") if member != ""]

    @staticmethod
    def set_attributes(attr_string: str, result: MaybeActualDataSet) -> None:
        """
        takes part of the description string and set the attributes found in that string for the process of loading a
        MaybeActualDataSet object from disc
        :param attr_string: part of the description file
        :param result: the MaybeActualDataSet object, that is created
        """
        attributes = attr_string.split(" \n")
        for attr in attributes:
            attr = attr.lstrip()
            index = attr.find(": ")
            if index != -1:
                name, value = attr[:index], attr[index + 2:]
                if name == "notes":
                    result.notes = value
                if name == "members":
                    value = value.strip("[] ")
                    result.members = [int(member) for member in value.split(",")]
                if name == "parameters":
                    dict_ = MaybeActualDataSet.parse_parameters(value)
                    result.parameters = dict_
                if name == "data_columns":
                    result.data_columns = MaybeActualDataSet.read_list(value)
                if name == "HiCS_dims":
                    result.HiCS_dims = MaybeActualDataSet.read_list(value)

    @staticmethod
    def load(path: str, ignore_validity_date: bool = False) -> MaybeActualDataSet:
        """
        loads a MaybeActualDataSet object from a saved location
        :param path: path to the save
        :param ignore_validity_date: flag to ignore, if the data was created with an older version of the code
        :return: a new MaybeActualDataSet object with the attributes set as described in the saved version
        """
        validity_date = datetime.datetime(2022, 2, 26, 13, 50, 0, 0)

        result = MaybeActualDataSet([1], save=False)
        result.data = pd.read_csv(os.path.join(path, "data.csv"))
        with open(os.path.join(path, "description.txt"), "r+") as f:
            created_line = f.readline()
            created_line = created_line.strip("\n")
            content = f.read()

        paragraphs = content.split("\n\n")
        if not created_line.startswith("CREATED: ") or not paragraphs[0].startswith("ATTRIBUTES: \n"):
            raise CustomError("file not in expected format!")

        result.path = path
        now = get_date_from_string(created_line)
        # makes sure, that no "old" data is loaded without a warning or accidentally
        if now < validity_date:
            if ignore_validity_date:
                print("WARNING: DATA OLDER THAN VALIDITY DATE!!")
            else:
                raise CustomError("Data older than validity date!")
        result.now = now
        MaybeActualDataSet.set_attributes(paragraphs[0], result)
        return result

    @staticmethod
    def create_class(class_number: int, members: int, class_parameters: Dict) -> pd.DataFrame:
        """
        creates dimensions 0 - 3 for a class, fills Dimensions with gauss distributed random data
        :param class_number: identifying number for the class
        :param members: the number of data points the class is supposed to contain
        :param class_parameters: Dictionary, that contains the mean as well as standard deviation for dimensions 0 - 3
        :return: A Dataframe containing the data of the class with columns ("classes" and "dim_00" to "dim_03")
        """
        new_df = pd.DataFrame()
        new_df["classes"] = [class_number for _ in range(members)]

        for i in range(4):
            center, standard_deviation = class_parameters[f"dim_{str(i).zfill(2)}"]     # zfill pads string with zeros
            dim = np.random.normal(center, standard_deviation, (members,))
            new_df[f"dim_{str(i).zfill(2)}"] = dim
        return new_df

    def add_dim_04(self) -> None:
        """
        Adds the fifth dimension to the data attribute. Each class contains triangular distributed random data
        """
        dim = []
        for i, members in enumerate(self.members):
            try:
                # will fail, if members isnt long enough
                # parameters for triangle distribution
                low, middle, high = self.parameters[f"class_{str(i).zfill(2)}"]["dim_04"]
                dim.extend(np.random.triangular(low, middle, high, (members, )))
            except IndexError:
                pass
        self.data["dim_04"] = dim

    def create_data(self) -> None:
        """
        creates and fills Dataframe for the data attribute.
        """
        data = pd.DataFrame()
        for i, class_members in enumerate(self.members):
            try:
                class_parameters = self.parameters[f"class_{str(i).zfill(2)}"]
                #data = data.append(self.create_class(i, class_members, class_parameters))
                data = pd.concat([data, self.create_class(i, class_members, class_parameters)])
            except IndexError:
                pass
        self.data = data
        self.add_dim_04()

    def clone_meta_data(self, path: str = "") -> MaybeActualDataSet:
        """
        creates a new MaybeActualDataSet object with same metadata.
        :param path: will be the path to create the new data at.
        :return: new MaybeActualDataSet object without meaningful Data
        """
        result = MaybeActualDataSet([1], save=False, path=path)
        result.extend_notes_by_one_line("this Dataset was created by duplicating metadata from another Dataset.")
        result.extend_notes_by_one_line("parameters refer to the original Dataset")
        result.end_paragraph_in_notes()
        result.now = self.now
        result.parameters = self.parameters
        return result

    def take_new_data(self, data: pd.DataFrame) -> None:
        """
        takes new Data for a MaybeActualDataSet object and adjusts members attribute
        :param data: new Data to take
        """
        self.data = data.copy(deep=True)
        class_counts = data["classes"].value_counts()
        members = []
        for i in range(len(self.class_params)):
            try:
                members.append(class_counts.at[i])
            except KeyError:
                members.append(0)
        self.members = members


if __name__ == "__main__":
    #MaybeActualDataSet.load("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220131_125348_MaybeActualDataSet")
    members_ = [10 for _ in range(6)]
    data1 = MaybeActualDataSet(members_)
    data2 = data1.clone_meta_data()
    data2.take_new_data(data1.data)
    #data.run_hics()
    #data = MaybeActualDataSet.load(data.path)
    #data.save()
    #data = MaybeActualDataSet.load(data.path)
    #data.save()

    #df = data1.data
    #data.run_hics()

    #vs.visualize_2d(df, ("dim_00", "dim_01"), class_column="classes")
    #vs.visualize_2d(df, ("dim_00", "dim_02"), class_column="classes")
    #vs.visualize_2d(df, ("dim_00", "dim_03"), class_column="classes")
    #vs.visualize_2d(df, ("dim_00", "dim_04"), class_column="classes")
    #vs.visualize_2d(df, ("dim_01", "dim_04"), class_column="classes")
    #vs.create_cumulative_plot(df["dim_00"].values)
    #vs.create_cumulative_plot(df, dim="dim_04", title="no constraints")
    #vs.create_cumulative_plot(df, dim="dim_04",
    #                                     constraints={"dim_00": [(False, -.5), (True, 3.5)], "dim_01": [(False, -.5), (True, 3.5)], "dim_02": [(False, -2), (True, 2)], "dim_03": [(False, -2), (True, 2)]},
    #                                     title="with constraints")
    #vs.create_hist(df["dim_00"])
    #vs.create_hist(df["dim_04"])
    #vs.visualize_2d(df, ("dim_00", "dim_01"), class_column="classes")
    #vs.visualize_2d(df, ("dim_00", "dim_04"), class_column="classes")
    #vs.visualize_3d(df, ("dim_00", "dim_01", "dim_04"), class_column="classes")
    #vs.create_3d_gif(df=df, dims=("dim_00", "dim_01", "dim_04"), name="maybe_actual_data_updated", class_column="classes", steps=120, duration=33)

