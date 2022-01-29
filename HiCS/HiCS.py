import os.path
import subprocess
from typing import List


def adjust_description(directory: str, arguments: List[str]) -> None:
    """
    If a description.txt file is in the folder (as expected for HiCS_data.csv files from data classes), the parameters
    used for HiCS will be appended to that file
    :param directory: Directory from which the input csv file was supplied
    :param arguments: Arguments used in HiCS, that will be appended to description.txt
    """
    files = os.listdir(directory)
    if "description.txt" in files:
        with open(os.path.join(directory, "description.txt"), "r+") as f:
            content = f.read()
            index = content.find("\n\nHiCS PARAMETERS")
            # empty the file after the beginning of the HiCS parameters, set the file pointer to the correct location to
            # start writing at the new end of the file
            if index != -1:
                f.truncate(index)
                f.seek(index)
            f.write("\n\nHiCS PARAMETERS:\n")
            for i, arg in enumerate(arguments):
                #entries should come in pairs with the first being "--command" and the second being the value
                # --> saving arguments as pairs
                if i % 2:
                    f.write(f"{arg}\n")
                else:
                    f.write(f"    {arg} ")


def run_HiCS(csv_in: str, csv_out: str = "", further_params: List[str] = None) -> None:
    """
    Calls HiCS with the data given in csv_in and saves it to csv_out or as out.csv in the same folder as csv_in.
    :param csv_in: path to the csv file to be used as input for HiCS
    :param csv_out: path to be used to save the output to (optional, will be saved in the same folder as csv_in as
    "out.csv" otherwise)
    :param further_params: further parameters to be used by HiCS implementation
    """
    path_here = os.path.dirname(__file__)
    command = f"{path_here}/HiCS_x64.exe"
    if not csv_out:
        last_slash = csv_in.rfind("/")
        last_backslash = csv_in.rfind("\\")
        if last_slash > last_backslash:
            path = csv_in[0:last_slash]
        else:
            path = csv_in[0:last_backslash]
        csv_out = os.path.join(path, "out.csv")
    arguments = ["--csvIn", f"{csv_in}", "--csvOut", f"{csv_out}", "--hasHeader", "true"]
    if further_params:
        arguments += further_params
    subprocess.run([command] + arguments)
    adjust_description(path, arguments)


if __name__ == "__main__":
    run_HiCS(csv_in="D:/Gernot/Programmieren/Bachelor/Python/Experiments/Data/220129_133016_MaybeActualDataSet/HiCS_Data.csv")

