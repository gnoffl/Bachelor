library("partykit")

ls <- data.frame(y = gl(3, 50, labels = c("A", "B", "C")),
                 x1 = rnorm(150) + rep(c(5, 0), c(100, 50)),
                 x2 = rnorm(150) + rep(c(1, 6), c(50, 100)))
ct <- ctree(y ~ x1 - x2, data = ls)

predict(ct, newdata = ls)

plot(ct)

data = read.csv("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\MaybeActualDataSet\\HiCS_Input_Data.csv", sep=";")
data
str(data)

ct_data = ctree(dim_04 ~ ., data = data)
plot(ct_data)
print(ct_data)


help("ctree_control")
help("ctree")
