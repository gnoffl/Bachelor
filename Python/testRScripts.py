import subprocess


def main():
    command = "C:/Program Files/R/R-4.1.2/bin/Rscript.exe"
    path2script = 'D:/Gernot/Programmieren/Bachelor/R/R_tests/hellowrld.r'
    additional_arguments = ["arg1", "arg2"]
    x = subprocess.check_output([command, path2script] + additional_arguments)
    print(x)


if __name__ == "__main__":
    main()

