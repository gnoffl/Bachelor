## Hier nochmal alles schön in eine Funktion gepackt.
## Bei name_split den Namen der Variable eintragen, nach der gesplittet werden
## soll und bei name_shift den Namen der Variable eintragen, nach der später
## verschoben werden soll



data = read.csv("D:\\Gernot\\Programmieren\\Bachelor\\Data\\220311_221633_MaybeActualDataSet\\data.csv", sep=",")
splitted = split_datasets(data=data, name_split="dim_01", name_shift = "dim_04")
test = data[1:300, "dim_04"]



split_datasets <- function(data, name_split, name_shift){
  
  split_id <- which(names(data) == name_split)
  shift_id <- which(names(data) == name_shift)
  
  goodness_split <- data.frame()
  
  for(i in 100:(nrow(data)-100)){
    
    goodness_split <- rbind(
      goodness_split,
      data.frame(
        ks_statistic = ks.test(data[1:i, shift_id], data[(i+1):nrow(data), shift_id])$statistic,
        splitpoint = mean(data[i:(i+1), split_id])
      )
    )
    
  }
  
  split <- goodness_split$splitpoint[which.max(goodness_split$ks_statistic)]
  
  print(split)
  
  list(data[data[, split_id] < split,],
       data[data[, split_id] > split,])
  
}

args = commandArgs(trailingOnly=TRUE)
#test_arg1 = "D:\\Gernot\\Programmieren\\Bachelor\\Data\\220302_174106_MaybeActualDataSet\\Data_splits\\dim_02_dim_04\\dim_04\\Data_for_Splitting.csv"
#test_arg2 = "dim_02"
#test_arg3 = "dim_04"
#data = read.csv(test_arg1, sep=",")
#splitted = split_datasets(data=data, name_split=test_arg2, name_shift = test_arg3)
#plot(data)
#plot(splitted[[1]])
#plot(splitted[[2]])
#print(splitted[[1]])
#print(splitted[[2]])

for (arg in args){
  print(arg)
}




#running the function
#args = commandArgs(trailingOnly=TRUE)
#data = read.csv(args[1], sep=",")
#print(args[1])
#splitted = split_datasets(data=data, name_split=args[2], name_shift = args[3])
#
#lit = strsplit(args[1], "\\\\")[[1]]
#lit_new = lit[-length(lit)]


#write.csv(splitted[[1]], paste(append(lit_new, "split_file1.csv"), collapse = "\\\\"))
#write.csv(splitted[[2]], paste(append(lit_new, "split_file2.csv"), collapse = "\\\\"))
