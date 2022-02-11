import numpy as np
import sys
import datetime

import pandas as pd

import Python.DataCreation.dataCreation as dc
import os
import HiCS.HiCS as hics


def main():
    test = pd.DataFrame()
    test["a"] = [1, 2, 3]
    test["b"] = [2, 3, 4]
    test["b"] = [3, 4, 5]
    test["c"] = [4, 5, 6]
    test["d"] = [5, 6, 7]
    test["e"] = [6, 7, 8]
    print(test)
    print(test.columns)





if __name__ == "__main__":
    #main()
    test = [1, 2, 3, 4, 5]
    to_add = [0.1 for _ in test]
    test += to_add
    print(test)
