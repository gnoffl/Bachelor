import numpy as np
import sys
import datetime
import Python.DataCreation.dataCreation as dc
import os
import HiCS.HiCS as hics


def main():
    with open("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220202_161849_MaybeActualDataSet\\description.txt", "r") as f:
        content = f.read()

    print(repr(content))





if __name__ == "__main__":
    main()
    #hics.adjust_description("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220129_134113_MaybeActualDataSet", ["test1", "newnew", "test2", "newnew2"])
