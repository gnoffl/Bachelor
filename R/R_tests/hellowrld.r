source("D:/Gernot/Programmieren/Bachelor/R/R_tests/test3.r")
args = commandArgs(trailingOnly=TRUE)

test = function(){
  test2()
  print(length(args))
  for (arg in args){
    print(arg)
  }
}
test()
