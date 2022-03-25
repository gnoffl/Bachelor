import os.path
import re

import pandas as pd
import dataCreation as dc

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
    dataset = dc.MaybeActualDataSet.load(r"D:\Gernot\Programmieren\Bachelor\Data\220325_181838_MaybeActualDataSet\1\1_0\1_0_0\1_0_0_1")
    description = dataset.data.describe()
    print(description["dim_02"])
    print(description["dim_01"])
