import os.path
import subprocess
from typing import List


def adjust_description(directory: str, additional_arguments: List[str]):
    files = os.listdir(directory)
    if "description.txt" in files:
        with open(os.path.join(directory, "description.txt"), "r+") as f:
            content = f.read()
            index = content.find("\n\nHiCS PARAMETERS")
            if index != -1:
                f.truncate(index)
                f.seek(index)
            f.write("\n\nHiCS PARAMETERS:\n")
            for i, arg in enumerate(additional_arguments):
                if i % 2:
                    f.write(f"{arg}\n")
                else:
                    f.write(f"    {arg} ")


def run_HiCS(csv_in: str, csv_out: str = "", further_params: List[str] = None):
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
    additional_arguments = ["--csvIn", f"{csv_in}", "--csvOut", f"{csv_out}", "--hasHeader", "true"]
    if further_params:
        additional_arguments += further_params
    subprocess.run([command] + additional_arguments)
    adjust_description(path, additional_arguments)


if __name__ == "__main__":
    run_HiCS(csv_in="D:/Gernot/Programmieren/Bachelor/Python/Experiments/Data/220129_133016_MaybeActualDataSet/HiCS_Data.csv")

