import os.path

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
    test = [10 for _ in range(20)]
    print(len(test))
    print(len(test[:9]))