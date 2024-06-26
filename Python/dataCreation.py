from __future__ import annotations

import subprocess
from typing import List, Dict, Tuple

import sklearn.tree as tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
import torch

import HiCS
#import visualization as vs
import datetime
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import os
import pickle
import random


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
    Adds new columns to the data containing uniformly distributed random values in a random range between -1000 and
    1000.
    :param data: The dataframe the new columns will be appended to
    :param dim_names: The names of the columns to be generated. Length of dim_names determines how many new columns will
    be created
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


class Data(ABC, Dataset):
    # attributes, that will have to be assigned manually
    data: pd.DataFrame

    #names of the columns, that contain data (not classes or predicted classes etc)
    data_columns: List[str]
    class_names: List[str]
    #List with the number of datapoints per class. order of the numbers should be the same as in class_names
    members: List[int]

    #attributes, that will be filled automatically
    path: str
    now: datetime.datetime
    notes: str or None
    note_buffer: List[str]
    target_tensor: torch.Tensor or None
    data_tensor: torch.Tensor or None

    def __init__(self, path: str, members: List[int] = None, notes: str = ""):
        self.note_buffer = []
        self.notes = notes
        self.members = members
        class_name = type(self).__name__
        now = datetime.datetime.now()
        self.now = now
        now_string = now.strftime("%y%m%d_%H%M%S")
        self.target_tensor = None
        self.data_tensor = None
        if not path:
            path = os.path.join(os.path.dirname(__file__), "..", "Data", f"{now_string}_{class_name}")
        self.path = path

    @staticmethod
    @abstractmethod
    def load(path: str, **kwargs) -> Data:
        """
        loads a data set from a given path
        :param path: path to the data set
        :return: the loaded dataset object
        """
        with open(os.path.join(path, "description.txt"), "r+") as f:
            class_type = f.readline()
            class_type = class_type.strip("\n").split(": ")[-1]

        if class_type == "MaybeActualDataSet":
            return MaybeActualDataSet.load(path, **kwargs)
        elif class_type == "IrisDataSet":
            return IrisDataSet.load(path)
        elif class_type == "SoccerDataSet":
            return SoccerDataSet.load(path)
        elif class_type == "BinningDataSet":
            return BinningDataSet.load(path)
        else:
            raise CustomError("No method found for loading this Dataset!")

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
        file.write(f"CLASS: {type(self).__name__}\n")
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

    def save(self, save_path: str = "", notes: str = "", force_write: bool = False) -> None:
        """
        creates a folder with a "description.txt" file, which contains the attributes of the object and possibly notes.
        Also, the creation date is saved.
        :param notes: notes to be saved
        :param save_path: name of folder, where the information will be saved
        :param force_write: overwrite existing files in an already existing folder, when specifying the save_path
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
                files = os.listdir(path)
                if "data.csv" in files or "description.txt" in files:
                    if force_write:
                        print("Warning: overwriting existing files!")
                    else:
                        raise CustomError("warning, overwriting existing files!")
        if not folder_exists:
            # create new info for class
            with open(os.path.join(path, "description.txt"), "w") as f:
                self.write_description(f)
        else:
            #only update info
            self.update_saved_info()
        #data has to be updated/saved either way
        self.data.to_csv(os.path.join(path, "data.csv"), index=False)

    @staticmethod
    def set_attributes(attr_string: str, result: Data) -> None:
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
                    dict_ = Data.parse_parameters(value)
                    result.parameters = dict_
                if name == "data_columns":
                    result.data_columns = Data.read_list(value)
                if name == "class_names":
                    result.class_names = Data.read_list(value)

    @staticmethod
    def read_list(list_string) -> List[str]:
        list_string = list_string.strip("[] ")
        list_string = list_string.replace(", ", ",").replace("\'", "")
        return [member for member in list_string.split(",") if member != ""]

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
            result_dict[class_name] = Data.parse_class_dict(class_dict)
        return result_dict

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

    def save_data_for_hics(self, path: str = "") -> str:
        """
        Saves the "data" attribute without the "classes" column as a csv file in the folder for the description of the
        object or at a given path.
        :param path: Path to a location where the csv is supposed to be saved, if not in the folder describing the
        object
        :return: the path to the file
        """
        if not path:
            if not os.path.isdir(self.path):
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
        adds notes for HiCS to a dataset object, depending on the parameters given for HiCS as well as possible
        additional notes
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
                string = ""
        if string:
            self.extend_notes_by_one_line(string)
        self.end_paragraph_in_notes()
        self.save()

    def run_hics(self, csv_out: str = "", further_params: List[str] = None, args_as_string: str = "",
                 notes: List[str] = "", silent: bool = True, csv_in: str = "") -> None:
        """
        runs HiCS for the Data object. First creates Info for the object using "create_class_info" and saves the input
        for HiCS into that folder. Output of Hics is also written into that folder, unless specified otherwise.
        :param csv_out: location and file name for output
        :param further_params: parameters for HiCS
        :param args_as_string: parameters for HiCS, but as a single string instead of a list of strings (will be split
        at spaces (" ")).
        :param notes: notes to be saved in the description of the Data set
        :param silent: determines whether HiCS generates progress information, which will be printed to the console
        :param csv_in: possible path to a file with input data for HiCS. If not given the Data will be generated
        """
        if further_params is None:
            further_params = []
        if not csv_in:
            csv_in = self.save_data_for_hics()
        if not csv_out:
            csv_out = self.create_csv_out(csv_in)
        if not os.path.isfile(csv_out):
            arguments = ["--csvOut", f"{csv_out}", "--hasHeader", "true", "--numCandidates", "5000",
                         "--maxOutputSpaces", "5000", "--csvIn", f"{csv_in}"]
            params = arguments + further_params
            args_list = []
            if args_as_string:
                args_list = args_as_string.split(" ")
            if silent:
                params.extend(["-s"])
            HiCS.run_HiCS(params + args_list)
            self.add_notes_for_HiCS(notes=notes, params=params)

    def get_contrast(self, dimensions: List[str], further_params: List[str] = None,
                     silent: bool = True, csv_in: str = "") -> float:
        if further_params is None:
            further_params = []
        if not csv_in:
            csv_in = self.save_data_for_hics()
        arguments = ["--csvIn", f"{csv_in}", "--hasHeader", "true"]
        if silent:
            arguments.extend(["-s"])
        params = arguments + further_params
        return HiCS.get_contrast(dimensions=dimensions, params=params, silent=silent)

    def parse_params(self, paragraphs: List[str], path: str, created_line: str = "") -> None:
        """
        parses List of strings from loading a dataset into the attribute they represent
        :param paragraphs: List of strings representing attributes of the Dataset
        :param path: path for the Data Object
        :param created_line: string that represents the time the dataset was created. Is used for the "now" attribute
        """
        self.data = pd.read_csv(os.path.join(path, "data.csv"))
        self.path = path
        if created_line:
            now = get_date_from_string(created_line)
            self.now = now
        Data.set_attributes(paragraphs[0], self)

    def get_test_training_split(self, test_fraction: float = 0.2) -> Tuple[Data, Data]:
        """
        splits the data from the dataset into two new datasets, that contain random datapoints.
        :param test_fraction: Fraction of values, that will be assigned to the new test_set. The remainder of the data
        will be in the training_set
        :return: the two new datasets
        """
        training_set = self.clone_meta_data()
        test_set = self.clone_meta_data()
        training, test_ = train_test_split(self.data, test_size=test_fraction)
        training_set.take_new_data(training)
        test_set.take_new_data(test_)
        return training_set, test_set

    def create_tensors(self) -> None:
        """
        creates tensors, that are necessary for using a neural Network classification
        """
        frame = self.data.copy()
        self.data_tensor = torch.Tensor(frame[self.data_columns].values)
        try:
            self.target_tensor = torch.Tensor([self.class_names.index(clas) for clas in frame["classes"]])
        except ValueError:
            self.target_tensor = torch.Tensor([int(clas) for clas in frame["classes"]])
        #should only occur during prediction as opposed to training. therefore targets are not relevant here
        except KeyError:
            self.target_tensor = torch.zeros(len(self.data_tensor))

    def __len__(self) -> int:
        """
        returns the length of the tensor representing the data
        :return: length of the tensor
        """
        if self.target_tensor is None or self.data_tensor is None:
            self.create_tensors()
        return len(self.target_tensor)

    def __getitem__(self, idx: int):
        """
        returns an item from the tensor at the given index
        :param idx: index
        :return: Tuple of data values and target value
        """
        if self.target_tensor is None:
            self.create_tensors()
        return self.data_tensor[idx], self.target_tensor[idx]


class MaybeActualDataSet(Data):
    """
    Parameters for the distribution of the classes in the different dimensions. for "dim_00" to "dim_03" the first
    value is the mean und the second value is the stadard deviation of a random Gauss distribution.
    For "dim_04" the values represent the low, middle and high values for the triangular random distribution
    """
    class_params = [
        {"dim_00": (0,   0.8), "dim_01": (0,   0.8), "dim_02": (0,  0.8), "dim_03": (0,   0.8), "dim_04": (0, 1, 2)},
        {"dim_00": (1.5, 0.8), "dim_01": (1.5, 0.8), "dim_02": (0,  0.8), "dim_03": (0,   0.8), "dim_04": (1, 2, 3)},
        {"dim_00": (1.5, 0.8), "dim_01": (1.5, 0.8), "dim_02": (0,  0.8), "dim_03": (0,   0.8), "dim_04": (5, 6, 7)},
        {"dim_00": (.5,  0.8), "dim_01": (0,   0.8), "dim_02": (2,  0.8), "dim_03": (0,   0.8), "dim_04": (5, 6, 7)},
        {"dim_00": (-.5, 0.8), "dim_01": (1.5, 0.8), "dim_02": (1,  0.8), "dim_03": (1.5, 0.8), "dim_04": (5, 6, 7)},
        {"dim_00": (-3,  0.8), "dim_01": (-3,  0.8), "dim_02": (-3, 0.8), "dim_03": (-3,  0.8), "dim_04": (3, 4, 5)}
    ]

    parameters: Dict

    def __init__(self, members: List[int], path: str = "", notes: str = "", save: bool = True):
        """
        initializes the data class
        :param members: entries determine the number of data points per class
        :param path: indicates where the files for this dataset should be saved
        :param notes: notes to be put in the description.txt file for the class
        :param save: dataset will be saved after creation, when this argument is True. Won't be saved otherwise.
        """
        super().__init__(path=path, members=members, notes=notes)
        np.random.seed(42)
        self.parameters = {}
        for i, class_param in enumerate(self.class_params):
            self.parameters[f"class_{str(i).zfill(2)}"] = class_param
        self.create_data()
        add_random_dims(self.data, ["rand_00", "rand_01", "rand_02"])
        self.data_columns = [value for value in self.data.columns.values if value != "classes"]
        self.class_names = ["class_00", "class_01", "class_02", "class_03", "class_04", "class_05",
                            "rand_00", "rand_01", "rand_02"]
        if save:
            self.save()

    @staticmethod
    def load(path: str, ignore_validity_date: bool = False, use_legacy_load: bool = False) -> MaybeActualDataSet:
        """
        loads a MaybeActualDataSet object from a saved location
        :param path: path to the save
        :param ignore_validity_date: flag to ignore, if the data was created with an older version of the code
        :param use_legacy_load: indicates, whether the old method for loading should be used
        :return: a new MaybeActualDataSet object with the attributes set as described in the saved version
        """
        validity_date = datetime.datetime(2022, 4, 27, 16, 30, 0, 0)

        result = MaybeActualDataSet([1], save=False)
        with open(os.path.join(path, "description.txt"), "r+") as f:
            if not use_legacy_load:
                first_line = f.readline()
            created_line = f.readline()
            created_line = created_line.strip("\n")
            content = f.read()

        paragraphs = content.split("\n\n")
        if not created_line.startswith("CREATED: ") or not paragraphs[0].startswith("ATTRIBUTES: \n"):
            raise CustomError("file not in expected format!")

        now = get_date_from_string(created_line)
        # makes sure, that no "old" data is loaded without a warning or accidentally
        if now < validity_date:
            if ignore_validity_date:
                print("WARNING: DATA OLDER THAN VALIDITY DATE!!")
            else:
                raise CustomError("Data older than validity date!")
        #only checking for the first line after the check for the validity date, because old saves will definitely
        # not have the correct first line, so the warning would be confusing
        if not use_legacy_load:
            if first_line != "CLASS: MaybeActualDataSet\n":
                raise CustomError("Wrong method for loading this dataset!")

        result.now = now
        result.parse_params(paragraphs=paragraphs, path=path)
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
        result.class_names = self.class_names
        return result

    def take_new_data(self, data: pd.DataFrame) -> None:
        """
        takes new Data for a MaybeActualDataSet object and adjusts members attribute
        :param data: new Data to take
        """
        self.data = data.copy(deep=True)
        members = []
        try:
            class_counts = data["classes"].value_counts()
        except KeyError:
            class_counts = None
        if class_counts is None:
            members = [0 for i in self.class_names]
        else:
            for i in range(len(self.class_names)):
                try:
                    members.append(class_counts.at[i])
                except KeyError:
                    members.append(0)
        self.members = members


class IrisDataSet(Data):

    def __init__(self, path: str = "", notes: str = "", save: bool = True, create_data: bool = True):
        """
        initializes the data class
        :param path: indicates where the files for this dataset should be saved
        :param notes: notes to be put in the description.txt file for the class
        :param save: dataset will be saved after creation, when this argument is True. Won't be saved otherwise.
        :param create_data: determines whether the dataset will contain the data from the iris dataset, or an empty
        Dataframe
        """
        super().__init__(path=path, members=[50, 50, 50], notes=notes)
        np.random.seed(42)
        if create_data:
            self.create_data()
        else:
            self.data = pd.DataFrame()
        self.data_columns = [value for value in self.data.columns.values if value != "classes"]
        if save:
            self.save()

    def create_data(self) -> None:
        """
        loads data from the iris dataset and saves it as the data attribute of the class. Also sets class_names.
        """
        bunch = load_iris(as_frame=True)
        frame = bunch["frame"]
        frame.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "classes"]
        self.data = frame
        self.class_names = list(bunch["target_names"])

    @staticmethod
    def load(path: str, **kwargs) -> IrisDataSet:
        """
        loads a IrisDataSet object from a saved location
        :param path: path to the save
        :return: a new IrisDataSet object with the attributes set as described in the saved version
        """
        result = IrisDataSet(save=False)
        with open(os.path.join(path, "description.txt"), "r+") as f:
            first_line = f.readline()
            created_line = f.readline()
            created_line = created_line.strip("\n")
            content = f.read()

        if first_line != "CLASS: IrisDataSet\n":
            raise CustomError("Wrong method for loading this dataset!")

        paragraphs = content.split("\n\n")
        if not created_line.startswith("CREATED: ") or not paragraphs[0].startswith("ATTRIBUTES: \n"):
            raise CustomError("file not in expected format!")
        result.parse_params(created_line=created_line, paragraphs=paragraphs, path=path)
        return result

    def clone_meta_data(self, path: str = "") -> IrisDataSet:
        """
        creates a new IrisDataSet object with same metadata.
        :param path: will be the path to create the new data at.
        :return: new IrisDataSet object without meaningful Data
        """
        result = IrisDataSet(save=False, path=path, create_data=False)
        result.extend_notes_by_one_line("this Dataset was created by duplicating metadata from another Dataset.")
        result.end_paragraph_in_notes()
        result.now = self.now
        result.class_names = self.class_names
        result.data_columns = self.data_columns.copy()
        return result

    def take_new_data(self, data: pd.DataFrame) -> None:
        """
        takes new Data for an IrisDataSet object and adjusts members attribute
        :param data: new Data to take
        """
        self.data = data.copy(deep=True)
        members = []
        try:
            class_counts = data["classes"].value_counts()
        except KeyError:
            class_counts = None
        if class_counts is None:
            members = [0 for i in self.class_names]
        else:
            for i in range(len(self.class_names)):
                try:
                    members.append(class_counts.at[i])
                except KeyError:
                    members.append(0)
        self.members = members


class SoccerDataSet(Data):

    def __init__(self, path: str = "", notes: str = "", save: bool = True, create_data: bool = True,
                 small: bool = False):
        """
        initializes the data class
        :param path: indicates where the files for this dataset should be saved
        :param notes: notes to be put in the description.txt file for the class
        :param save: dataset will be saved after creation, when this argument is True. Won't be saved otherwise.
        :param create_data: determines whether the dataset will contain the data from the iris dataset, or an empty
        Dataframe
        """
        super().__init__(path=path, notes=notes)
        np.random.seed(42)
        self.class_names = ['Torwart', 'Innenverteidiger', 'Aussenverteidiger', 'Defensives Mittelfeld',
                            'Zentrales Mittelfeld', 'Mittelfeld Aussen', 'Offensives Mittelfeld', 'Mittelstuermer',
                            'Fluegelspieler']

        if create_data:
            self.create_data(small)
            self.update_members()
        else:
            self.data = pd.DataFrame()
        self.data_columns = [value for value in self.data.columns.values if value != "classes"]
        if save:
            self.save()

    def create_data(self, small: bool = False) -> None:
        """
        creates the data for the SoccerDataSet, by reading it in from a file
        """
        frame = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data_per_game_excel.csv"),
                            encoding="utf-8", sep=";")
        #drop unnecessary columns
        frame = frame[["ps_Pass", "Passprozente", "ps_Zweikampf", "Zweikampfprozente", "ps_Fouls", "ps_Gefoult",
                       "ps_Laufweite", "ps_Abseits", "ps_Assists", "ps_Fusstore", "ps_Kopftore", "Hauptposition_adj"]]
        frame.rename(columns={"Hauptposition_adj": "classes"}, inplace=True)

        #remove the libero lines (only 2 lines in the dataset) as well as rows with NaNs
        frame = frame.loc[frame["classes"] != "Libero"]
        frame.dropna(inplace=True)

        #rename values, that were not loaded properly
        pairs = {'Mittelfeld Au�en': "Mittelfeld Aussen",
                 'Fl�gelspieler': 'Fluegelspieler',
                 'Au�enverteidiger': 'Aussenverteidiger',
                 'Mittelstürmer': 'Mittelstuermer'}
        for old, new in pairs.items():
            frame.replace(old, new, inplace=True)
        if small:
            frame, _ = train_test_split(frame, test_size=0.5)
        self.data = frame

    def clone_meta_data(self, path: str = "") -> SoccerDataSet:
        """
        creates a new SoccerDataSet object with same metadata.
        :param path: will be the path to create the new data at.
        :return: new IrisDataSet object without meaningful Data
        """
        result = SoccerDataSet(save=False, path=path, create_data=False)
        result.extend_notes_by_one_line("this Dataset was created by duplicating metadata from another Dataset.")
        result.end_paragraph_in_notes()
        result.now = self.now
        result.data_columns = self.data_columns.copy()
        return result

    def take_new_data(self, data: pd.DataFrame) -> None:
        """
        takes new Data for an SoccerDataSet object and adjusts members attribute
        :param data: new Data to take
        """
        self.data = data.copy(deep=True)
        self.update_members()

    def update_members(self) -> None:
        """
        updates the members attribute of self
        """
        members = []
        try:
            class_counts = self.data["classes"].value_counts()
        except KeyError:
            class_counts = None

        if class_counts is None:
            members = [0 for i in self.class_names]
        else:
            #class_names is just a list of the possible values of "classes" in this case
            for i in self.class_names:
                try:
                    members.append(class_counts.at[i])
                except KeyError:
                    members.append(0)
        self.members = members

    @staticmethod
    def load(path: str, **kwargs) -> SoccerDataSet:
        """
        loads a SoccerDataSet object from a saved location
        :param path: path to the save
        :return: a new SoccerDataSet object with the attributes set as described in the saved version
        """
        result = SoccerDataSet(save=False)
        with open(os.path.join(path, "description.txt"), "r+") as f:
            first_line = f.readline()
            created_line = f.readline()
            created_line = created_line.strip("\n")
            content = f.read()

        #check if the dataset to load fits the class
        if first_line != "CLASS: SoccerDataSet\n":
            raise CustomError("Wrong method for loading this dataset!")

        paragraphs = content.split("\n\n")
        if not created_line.startswith("CREATED: ") or not paragraphs[0].startswith("ATTRIBUTES: \n"):
            raise CustomError("file not in expected format!")

        result.parse_params(paragraphs=paragraphs, created_line=created_line, path=path)
        return result


class BinningDataSet(Data):

    def __init__(self, path: str = "", notes: str = "", save: bool = True):
        """
        initializes the data class
        :param members: entries determine the number of data points per class
        :param path: indicates where the files for this dataset should be saved
        :param notes: notes to be put in the description.txt file for the class
        :param save: dataset will be saved after creation, when this argument is True. Won't be saved otherwise.
        """
        super().__init__(path=path, members=[100, 100, 100], notes=notes)
        np.random.seed(46)
        self.create_data()
        self.data_columns = [value for value in self.data.columns.values if value != "classes"]
        self.class_names = ["class_00", "class_01", "class_02"]
        if save:
            self.save()

    @staticmethod
    def load(path: str, ignore_validity_date: bool = False, use_legacy_load: bool = False) -> BinningDataSet:
        """
        loads a MaybeActualDataSet object from a saved location
        :param path: path to the save
        :param ignore_validity_date: flag to ignore, if the data was created with an older version of the code
        :param use_legacy_load: indicates, whether the old method for loading should be used
        :return: a new MaybeActualDataSet object with the attributes set as described in the saved version
        """

        result = BinningDataSet(save=False)
        with open(os.path.join(path, "description.txt"), "r+") as f:
            first_line = f.readline()
            created_line = f.readline()
            created_line = created_line.strip("\n")
            content = f.read()

        paragraphs = content.split("\n\n")
        if not created_line.startswith("CREATED: ") or not paragraphs[0].startswith("ATTRIBUTES: \n"):
            raise CustomError("file not in expected format!")

        now = get_date_from_string(created_line)
        #only checking for the first line after the check for the validity date, because old saves will definitely
        # not have the correct first line, so the warning would be confusing
        if first_line != "CLASS: BinningDataSet\n":
            raise CustomError("Wrong method for loading this dataset!")

        result.now = now
        result.parse_params(paragraphs=paragraphs, path=path)
        return result

    def create_data(self) -> None:
        """
        creates and fills Dataframe for the data attribute.
        """
        data1 = pd.DataFrame()
        data1["X"] = [random.uniform(0, 0.98) for _ in range(200)]
        data1["Y"] = [random.uniform(0.02, 2) for _ in range(200)]
        data1["classes"] = 0
        data2 = pd.DataFrame()
        data2["X"] = [random.uniform(1.02, 1.98) for _ in range(200)]
        data2["Y"] = [random.uniform(-1, -0.02) for _ in range(200)]
        data2["classes"] = 1
        data3 = pd.DataFrame()
        data3["X"] = [random.uniform(2.02, 3) for _ in range(200)]
        data3["Y"] = [random.uniform(0.02, 2) for _ in range(200)]
        data3["classes"] = 2

        self.data = pd.concat([data1, data2, data3])

    def clone_meta_data(self, path: str = "") -> BinningDataSet:
        """
        creates a new MaybeActualDataSet object with same metadata.
        :param path: will be the path to create the new data at.
        :return: new MaybeActualDataSet object without meaningful Data
        """
        result = BinningDataSet(save=False, path=path)
        result.extend_notes_by_one_line("this Dataset was created by duplicating metadata from another Dataset.")
        result.extend_notes_by_one_line("parameters refer to the original Dataset")
        result.end_paragraph_in_notes()
        result.now = self.now
        result.class_names = self.class_names
        return result

    def take_new_data(self, data: pd.DataFrame) -> None:
        """
        takes new Data for a MaybeActualDataSet object and adjusts members attribute
        :param data: new Data to take
        """
        self.data = data.copy(deep=True)
        members = []
        try:
            class_counts = data["classes"].value_counts()
        except KeyError:
            class_counts = None
        if class_counts is None:
            members = [0 for _ in self.class_names]
        else:
            for i in range(len(self.class_names)):
                try:
                    members.append(class_counts.at[i])
                except KeyError:
                    members.append(0)
        self.members = members



def test():
    set1 = BinningDataSet()
    set1.run_hics(further_params=["--maxOutputSpaces", "5000"], silent=False)


def soccer_visualization_modification():
    if not os.path.exists(r"..\Data\soccer_qsm_NN_graphics") or not os.path.exists(r"..\Data\soccer_improved_NN_graphics"):
        qsm_NN_set_soccer = Data.load(r"..\Data\Parameters2\SoccerDataSet\NN\004")
        improved_NN_set_soccer = Data.load(
            r"..\Data\Parameters2\SoccerDataSet\NN\004\Splits\ps_Laufweite_005")
        qsm_NN_set_soccer.save(r"..\Data\soccer_qsm_NN_graphics")
        improved_NN_set_soccer.save(r"..\Data\soccer_improved_NN_graphics")

    qsm_NN_set_soccer = Data.load(r"..\Data\soccer_qsm_NN_graphics")
    improved_NN_set_soccer = Data.load(r"..\Data\soccer_improved_NN_graphics")
    datasets = [improved_NN_set_soccer, improved_NN_set_soccer, qsm_NN_set_soccer, improved_NN_set_soccer]
    class_columns = ["classes", "org_pred", "pred_with_ps_Laufweite_shifted_by_0.05", "pred_classes"]
    for dataset_, col in zip(datasets, class_columns):
        #create new column with the name of col and the postfix "_simple"
        #content of the new column is "Goalkeeper" if the content of the original column is "Torwart",
        # "Field player" otherwise
        dataset_.data[col + "_simple"] = dataset_.data[col].apply(lambda x: "Goalkeeper" if x == "Torwart" else "Field player")
    qsm_NN_set_soccer.save()
    improved_NN_set_soccer.save()




if __name__ == "__main__":
    #test()
    soccer_visualization_modification()
    #MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Python\
    #Experiments\Data\220131_125348_MaybeActualDataSet")
    #members_ = [10 for _ in range(6)]
    #data1 = SoccerDataSet(save=False)
    #data1 = Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\220427_173646_IrisDataSet")
    #data1.save()
    #data1.run_hics(silent=False, args_as_string="-s")
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
    #                          constraints={"dim_00": [(False, -.5), (True, 3.5)],
    #                                       "dim_01": [(False, -.5), (True, 3.5)],
    #                                       "dim_02": [(False, -2), (True, 2)],
    #                                       "dim_03": [(False, -2), (True, 2)]},
    #                          title="with constraints")
    #vs.create_hist(df["dim_00"])
    #vs.create_hist(df["dim_04"])
    #vs.visualize_2d(df, ("dim_00", "dim_01"), class_column="classes")
    #vs.visualize_2d(df, ("dim_00", "dim_04"), class_column="classes")
    #vs.visualize_3d(df, ("dim_00", "dim_01", "dim_04"), class_column="classes")
    """vs.create_3d_gif(df=df, dims=("ps_Pass", "Passprozente", "ps_Zweikampf"), name="soccer1",
                     class_column="classes", steps=30)
    vs.create_3d_gif(df=df, dims=("Zweikampfprozente", "ps_Fouls", "ps_Gefoult"), name="soccer2",
                     class_column="classes", steps=30)
    vs.create_3d_gif(df=df, dims=("ps_Laufweite", "ps_Abseits", "ps_Assists"), name="soccer3",
                     class_column="classes", steps=30)
    vs.create_3d_gif(df=df, dims=("ps_Pass", "ps_Fusstore", "ps_Kopftore"), name="soccer4",
                     class_column="classes", steps=30)"""

