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
    print(test[["a", "b", "c"]])





if __name__ == "__main__":
    main()
    #hics.adjust_description("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220129_134113_MaybeActualDataSet", ["test1", "newnew", "test2", "newnew2"])
