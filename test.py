import numpy as np
import sys
import datetime
import Python.DataCreation.dataCreation as dc
import os
import HiCS.HiCS as hics


def main():
    csv_in = "asdfadsf1/adsfasdf2/asdfasdf3\\asdfasdf4/asdfadsf5\\adsfadfadf6/test.csv"
    index = csv_in.find("test")
    print(csv_in[:-1])





if __name__ == "__main__":
    #main()
    hics.adjust_description("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220129_134113_MaybeActualDataSet", ["test1", "newnew", "test2", "newnew2"])
