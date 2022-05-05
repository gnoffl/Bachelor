import os.path
import subprocess
from typing import List


def run_HiCS(params: List[str] = None) -> None:
    """
    Calls HiCS with the given parameters
    :param params: parameters to be used by HiCS implementation
    """
    path_here = os.path.dirname(__file__)
    command = os.path.join(path_here, "..", "HiCS", "HiCS_x64.exe")
    subprocess.run([command] + params)


def get_contrast(dimensions: List[int], params: List[str] = None, silent: bool = True) -> float:
    path_here = os.path.dirname(__file__)
    dimensions = [str(dim) for dim in dimensions]
    dims = ",".join(dimensions)
    params.append("--onlySubspace")
    params.append(dims)
    command = os.path.join(path_here, "..", "HiCS", "HiCS_x64.exe")
    process = subprocess.Popen([command] + params, stdout=subprocess.PIPE)
    output = process.communicate()[0].decode()
    if not silent:
        print(output)
    return float(output.split("\r\n")[-2])


def test():
    print(get_contrast(dimensions=[0, 1, 2, 3]))


if __name__ == "__main__":
    test()


