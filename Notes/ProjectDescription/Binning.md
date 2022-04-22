# Binning

## Todos

### Important Todos

#### Inhaltliche Änderungen

* [ ] Vernünftigen Tie-Breaker für ks-Werte mit gleichem p-Wert (spez. p=0) finden

#### Dokumentation

* [x] Dokumentation für code erstellen
* [x] Typing von funktionen einhalten
* [x] Code kommentieren
* [x] Notizen erstellen
* [x] Änderungen hier notieren
* [ ] Typing allgemein einhalten
* [x] verbliebene todos in allgemeiner todo-liste vermerken

#### Done

* [x] HiCS starten
* [x] ergebnisse von HiCS interpretieren
* [x] add notes for split criteria in datasets
* [x] save pictures from visualization of binning in better spots (pics folder)
* [x] in _find_optimal_split_index_ Kriterium festlegen, ob überhaupt gesplittet werden soll
* [x] min_split_size an korrekter Stelle berechnen, und nicht als Parameter nehmen
* [x] Testen ob Rekursion korrekt abbricht
* [x] testen was passiert, wenn _create_binning_splits_ auf Datensatz erneut aufgerufen wird
* [x] testen, ob abbruchkriterium in _find_optimal_split_index_ korrekt funktioniert
* [x] test if datasets actually only have the data they are supposed to
* [x] Code beschleunigen (dauert bei 1000 punkten pro Klasse viel zu lange, fühlt sich nicht
nach linearem wachstum an)
* [x] min_split_size muss immer aufgerundet werden

### Optional Todos
* [ ] Suche nach bestem HiCS optimieren, indem nicht max-funktion auf dem sortierten HiCS-output
angewandt wird
* [ ] Warnung von typing bei KstestResult beheben

## Notizen

Binning wird in der Datei "createDataSplits.py" umgesetzt. Die Daten sollen dabei rekursiv
immer weiter in kleinere Datensätze zerlegt werden. Dabei soll jeder Split dazu führen, dass
sich die Verteilungen einer gegebenen Dimension (_dim_to_shift_) in den beiden entstehenden
Datensätzen maximal unterscheidet. Zur Berechnung eines Splits wird zunächst HiCS durchgeführt,
um zu schauen, in welchem Subspace sich am ehesten interessante Zusammenhänge in den Daten
befinden. Unter den Dimensionen des Subspaces mit dem höchsten Kontrast wird dann diejenige
ausgewählt, die mit der _dim_to_split_ im Paar den höchsten Kontrast aufweist. Der Datensatz
wird nach dieser Dimension sortiert, und anschließend werden alle möglichen splits auf ihren
Unterschied bezüglich der _dim_to_shift_ untersucht. Der beste split wird gewählt, und die
entstandenen Datensätze werden erneut gesplittet.\
Dieses Vorgehen kann durch verschiedene Bedingungen unterbrochen werden. Zunächst wird am
Anfang eine maximale Anzahl von splits vorgegeben. Ist diese erreicht, werden keine weiteren
splits mehr durchgeführt. Weiterhin dürfen entstehende Datensätze eine Mindestanzahl an
Datenpunkten nicht unterschreiten. Diese Mindestanzahl wird berechnet als (Länge des
ursprünglichen Datensatzes * Quantil, um das der Datensatz später verschoben werden soll).
Hat ein Datensatz weniger als das Doppelte der Mindestanzahl an Datenpunkten, kann er nicht
weiter gesplittet werden, da alle resultierenden Splits mindestens einen Datensatz enthalten
würden, der die Mindestgrenze unterschreitet.\
Die dritte Art, auf die die Rekursion abbrechen kann ist, dass keiner der möglichen Splits
eines Datensatzes zu signifikant unterschiedlichen Verteilungen in der _dim_to_shift_ führt.

### Effizienz

Die Berechnung der Splits verlangt, dass für jeden möglichen Split ein statistischer Test
(Kolmogorov-Smirnov-Test (ks-test)) durchgeführt wird. Diese Berechnungen dauern für größere
Datensätze äußerst lang, weshalb der Code zur Berechnung der Tests parallelisiert wurde. 



## Änderungen

