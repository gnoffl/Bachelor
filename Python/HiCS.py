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


