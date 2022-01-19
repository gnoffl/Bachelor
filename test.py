import numpy as np
import sys

sys.path.append("DataCreation/")
import dataCreation as dc


def main():
    members = [1000 for _ in range(5)]
    df = dc.MaybeActualDataSet(members).data
    test_val = df.iloc[[i * 1000 for i in range(5)]]
    print(test_val.columns)
    print(test_val.values)




if __name__ == "__main__":
    main()

