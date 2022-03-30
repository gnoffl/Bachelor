import os.path
import re

import pandas as pd
import dataCreation as dc


def main(test_frame: pd.DataFrame):
    test_frame = pd.DataFrame()
    test_frame["0"] = [0]





if __name__ == "__main__":
    matrix = pd.read_csv("../Data/matrix.csv", index_col=0)
    print(matrix)
    summand = pd.DataFrame()
    """summand["0"] = [1, 2, 3, 4, 5, 6]
    summand["1"] = [1, 2, 3, 4, 5, 6]
    summand["2"] = [1, 2, 3, 4, 5, 6]
    summand["3"] = [1, 2, 3, 4, 5, 6]
    summand["4"] = [1, 2, 3, 4, 5, 6]
    summand["5"] = [1, 2, 3, 4, 5, 6]"""
    print(summand)
    print(summand.add(other=matrix, fill_value=0))
