from typing import List
import subprocess
import dataCreation as dc
import os


def run_R_script(additional_arguments: List,
                 path_to_script: str = ""):
    if not path_to_script:
        path_to_script = os.path.join("..", "R", "Binning", "run_binning.R")
    command = "C:/Program Files/R/R-4.1.2/bin/Rscript.exe"
    x = subprocess.check_output([command, path_to_script] + additional_arguments)
    print(x)


def create_folder_for_splits(dataset: dc.Data, HiCS_dims: List[str], dim_to_shift: str) -> str:
    splits_folder = os.path.join(dataset.path, "Data_splits")
    if not os.path.isdir(splits_folder):
        os.mkdir(splits_folder)
    hiCS_folder = "_".join(HiCS_dims)
    hiCS_folder = os.path.join(splits_folder, hiCS_folder)
    if not os.path.isdir(hiCS_folder):
        os.mkdir(hiCS_folder)
    dim_folder = os.path.join(hiCS_folder, dim_to_shift)
    if not os.path.isdir(dim_folder):
        os.mkdir(dim_folder)
    return dim_folder


def main(path: str, dim_to_shift: str):
    dataset = dc.MaybeActualDataSet.load(path)
    if "HiCS_output.csv" not in os.listdir(path):
        dataset.run_hics()
        print("running HiCS")
    with open(os.path.join(path, "HiCS_output.csv"), "r") as f:
        first_line = f.readline()
        dimensions = [dataset.data_columns[int(cand)] for cand in first_line.split(";")[1:]]
    if dim_to_shift in dimensions:
        folder_path = create_folder_for_splits(dataset=dataset, HiCS_dims=dimensions, dim_to_shift=dim_to_shift)
        split_input_file = os.path.join(folder_path, "Data_for_Splitting.csv")
        dataset.data[dimensions].to_csv(split_input_file, index=False)
        run_R_script(additional_arguments=[split_input_file, "dim_02", dim_to_shift])
    else:
        raise dc.CustomError("Dimension to shift was not in best HiCS")



def test():
    dataset = dc.MaybeActualDataSet([1000 for _ in range(6)])
    columns = dataset.data.columns.values
    test_dim = columns[2]
    print(columns)
    print(f"test_dim: {test_dim}")
    print(f"cand: {columns[1]}")
    columns[2] = columns[1]
    columns[1] = test_dim
    dataset.data.columns = columns
    dataset.save_data_for_hics()
    dataset.save()


if __name__ == "__main__":
    main(r"C:\Users\gerno\Programmieren\Bachelor\Data\220303_200505_MaybeActualDataSet", dim_to_shift="dim_04")
    #test()