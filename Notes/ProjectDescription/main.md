# Main

## Todos

### Important Todos
nothing atm

### Optional Todos
* [ ] write script to copy all relevant information into one folder --> less searching for results etc

### Done
* [x] Typing für Methoden einhalten
* [x] Code kommentieren
* [x] Dokumentation erstellen
* [ ] Notizen erstellen
* [x] Todos hier Füllen!
* [x] Änderungsbereich einfügen
* [x] run QSM on binned data
* [x] properly propagate results through recursion
* [x] visualisierung so überarbeiten, dass Bilder gespeichert und nicht gezeigt werden
* [x] visualisierung von allen QSM ansätzen korrekt speichern (visualisierung innerhalb
von QSM)
* [x] visualisierung von gesamtem Datensatz nach QSM nach Binning
* [x] visualisierung von allen splits korrekt speichern (Bilder für splits werden in 
  org Ordner gespeichert, nicht im root verzeichnis für den entsprechenden Shift)
  * wird erstmal so gelassen, evtl script zum verschieben von Bildern schreiben
* [x] convert all matrices to integers instead of floats
* [x] save change matrices
* [x] save "original" prediction of classifier to compare
* [x] color of datapoints should not be depending on what classes are present
* [x] Mehr meldungen über den Fortschritt der Methode (allg. sinnvollere ausgaben)

## Notizen
main.py ist das Script, das zur Ausführung des gesamten Prozesses gedacht ist.
Aktuell wird viel von der Abwicklung der Rekursion von QSM über die gesplitteten Datensätze
sowie die Visualisierung der Ergebnis in dieser Datei geregelt. In Zukunft wird dies
vielleicht noch in eine andere Datei refactored. Entscheidende Funktion ist
_compare_vanilla_split_, die einen Datensatz annimmt, einen DecisionTree darauf trainiert,
die Daten splittet und dann QSM auf dem gesamten Datensatz sowie auf den Splits laufen
lässt. Die Ergebnisse werden alle Visualisiert, und die Ergebnisse gespeichert.


## Änderungen
