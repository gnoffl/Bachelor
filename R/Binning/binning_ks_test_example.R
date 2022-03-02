## Hier werden nur Daten fuer das Beispiel generiert

set.seed(01032022)
norm1 <- rnorm(1000)
norm2 <- rnorm(1000, mean = 1)

## y ist die Variable, nach der wir verschieben wollen, weshalb wir Ueberpruefen 
## moechten, ob sich davon beim splitten nach x die Verteilung unterscheidet
df <- data.frame(y = c(norm1, norm2), x = 1:2000)

## Im gemeinsamen Histogramm sieht man nicht unbedingt, dass es sich um 2 
## unterschiedlich liegende Gruppen handeln koennte
hist(df$y, breaks = 100)

## Schaut man sich beide Dichten separat an, schon. Die Splitvariable sollte
## also hoffentlich diese beiden Gruppen trennen
hist(norm1, breaks = 100, xlim = c(-5,5))
hist(norm2, breaks = 100, xlim = c(-5,5))

####################################



## hier erstelle ich iterativ eine finale Tabelle, in der ich als Info die
## berechnete Statistik des Kolmogorov-Smirnov Tests und den möglichen Splitpunkt notiere
goodness_split <- data.frame()

## Wir haben insgesamt 2000 Datenpunkte. Tests, bei denen man nur 1 Datenpunkt 
## absplittet sind nicht so sinnvoll, deswegen lasse ich die for-Schleife von 
## 100 bis 1900 laufen (kann man auch dynamischer mit 100:(nrow(df)-100) machen,
## wenn man moechte)
for(i in 100:1900){
  
  goodness_split <- rbind(
    goodness_split,
    data.frame(
      ks_statistic = ks.test(df$y[1:i], df$y[(i+1):nrow(df)])$statistic,
      splitpoint = mean(df$x[i:(i+1)])
      )
  )
  
}
# dauert etwa 10 Sekunden

## Dort, wo die Statistik am hoechsten ist, unterscheidet sich die Verteilung 
## zwischen den beiden getrennten Datensaetzen am staerksten
which.max(goodness_split$ks_statistic)

## Hier koennen wir dann in die Tabelle schauen, welcher der optimale Splitpunkt
## waere
goodness_split[which.max(goodness_split$ks_statistic),]

## Hier koennen wir uns einfach nur den optimalen Splitpunkt ausgeben, wo wir 
## unsere Datens�tze am besten bezueglich x trennen sollten
split <- goodness_split$splitpoint[which.max(goodness_split$ks_statistic)]

## Hier verschwimmen die Verteilungen natuerlich noch ein wenig, wenn also die 
## ersten generierten Datenpunkte aus der zweiten Verteilung sehr niedrig sind,
## werden sie, wie in diesem Fall, noch zur ersten Gruppe hinzugeordnet. Da 
## wir ja aber eindeutigere Unterscheidungen in unserem Beispiel haben, werden 
## die berechneten Statistiken deutlicher und es sollte (hoffentlich) 
## gut fuer uns funktionieren :-)

## hier dann die beiden Datensaetze (abspeicherbar mit bspw. write.csv2)
df1 <- df[df$x < split,]
df2 <- df[df$x > split,]

## Hier nochmal alles schoen in eine Funktion gepackt.
## Bei name_split den Namen der Variable eintragen, nach der gesplittet werden
## soll und bei name_shift den Namen der Variable eintragen, nach der spaeter
## verschoben werden soll
split_datasets <- function(data, name_split, name_shift){
  
  split_id <- which(names(data) == name_split)
  shift_id <- which(names(data) == name_shift)
  
  goodness_split <- data.frame()
  
  for(i in 100:(nrow(data)-100)){
    
    goodness_split <- rbind(
      goodness_split,
      data.frame(
        ks_statistic = ks.test(data[1:i, shift_id], df[(i+1):nrow(df), shift_id])$statistic,
        splitpoint = mean(df[i:(i+1), split_id])
      )
    )
    
  }
  
  split <- goodness_split$splitpoint[which.max(goodness_split$ks_statistic)]
  
  list(data[data[, split_id] < split,],
       data[data[, split_id] > split,])
  
}


## Beispiel:
splitted <- split_datasets(data = df, name_split = "x", name_shift = "y")

## So kommst du aus der Liste an die beiden Datensätze
splitted[[1]]
splitted[[2]]