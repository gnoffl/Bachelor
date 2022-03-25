# Classifier

## Todos

### Important Todos

#### Inhaltliche Änderungen

* [ ] get_HiCS nochmal testen
* [ ] Vernünftigen Tie-Breaker für ks-Werte mit gleichem p-Wert (spez. p=0) finden
* [ ] Warnung von typing bei KstestResult beheben
* [ ] Wenn große Subspaces bevorzugt werden sollen: aktuell muss der längste Subspace 70% des
Werts vom besten HiCS haben. Diese Zahl ist abstrakt gewählt und sollte überdacht werden.
* [ ] Code beschleunigen (dauert bei 1000 punkten pro Klasse viel zu lange, fühlt sich nicht
nach linearem wachstum an)

#### Dokumentation

* [ ] Dokumentation für code erstellen
* [ ] Code kommentieren
* [ ] Typing einhalten
* [ ] Notizen erstellen
* [ ] Änderungen hier notieren

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

### Optional Todos
* [ ] Suche nach bestem HiCS optimieren, indem nicht max-funktion auf dem sortierten HiCS-output
angewandt wird

## Notizen



## Änderungen

