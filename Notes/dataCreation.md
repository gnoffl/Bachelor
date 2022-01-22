# DataCreation

## Todos

### Important Todos
* [x] Code Kommentieren
* [x] 1d Verteilung in dim_04 sollte kein Loch enthalten --> eine Klasse "neben" das Loch
zwischen Klasse 1 und 2 legen
* [ ] Gap zwischen 1 und 2 verringern
* [ ] 2, 3, 4 besser unterscheidbar machen ohne dim_04, dafür in dim_04 vielleicht näher aneinander
rücken
* [ ] 5 auf "andere Seite" (--> niedrige Werte) rücken
* [ ] Typing überall einhalten

### Optional Todos
* [ ] Code für sekundäre Klassen kommentieren
* [ ] **GeometricUniformData:** Dichte in Überschneidungsbereichen ist nicht konstant
* [ ] **SimpleCorrelationData:** nicht die unverrauschten Originaldaten als Dimension benutzen,
    sondern mit Originaldaten rechnen, aber nur verrauschte Dimensionen in Datensatz verwenden
* [ ] **SimpleCorrelationData:** überlegen, ob die Dimensionen so gleichverteilt sind, oder ob auf polarkoordinaten
gewechselt werden sollte (bei der Kreisvariante)


## Notizen

### Ziel für Klasse
* 3 – 5 Classes
* 5 – 20 Variables with Information
* ~5 random Variables
* Create Dependencies that require Binning
* 1.000 – 10.000 Datenpunkte

### Genereller Aufbau
Script besteht aus allgemein nützlichen Funktionen (add_gaussian_noise und add_random_dims),
sowie mehreren Datenklassen. Datenklassen erben alle von "Data", die vorgibt, dass
Datenklassen Daten in Form eines Dataframe besitzen, sowie eine Liste von ints
("members"), in der angegeben wird, wie viele Punkte in den jeweiligen Klassen
der erbenden Datensätze enthalten sein sollen.\
Von den Datenklassen selbst ist "MaybeActualDataSet" die wichtigste, da sie
tatsächlich erstmal den Datensatz darstellt, mit dem Experimente durchgeführt werden.

### MaybeActualDataSet
Der Datensatz hat zur Zeit 8 Dimensionen, von denen 3 ("rand_00", "rand_01", "rand_02")
nur mit zufälligen Werten gefüllt sind. Die weiteren Dimensionen ("dim_00" - "dim_04")
sind mit relevanter Information gefüllt. Die Daten sind in den Dimensionen 00 - 03
Normalverteilt, während in dim_04 symmetrische Dreiecksverteilungen genutzt werden,
um die maximale Ausdehnung der Daten genauer kontrollieren zu können. Dabei haben
Klasse 01 und 02 in Dimensionen 00 - 03 die exakt gleichen Verteilungen (00/01: hoch,
02/03: niedrig). Genauso haben Klasse 03 und 04 die selben Verteilungen, nur 
mit entgegengesetzten Werten (00/01: niedrig, 02/03: hoch). Klasse 00 hat in
Dimensionen 00-03 jeweils den niedrigen Wert.\
Alle Klassen unterscheiden sich in ihren Verteilungen in der letzten Dimension
(04). Die Mittelwerte steigen mit der Klassennummer. Es besteht immer eine Überschneidung
zwischen den Bereichen aufeinander folgender Klassen, wobei die einzige Ausnahme ein Spalt
zwischen Klasse 01 (1-3) und klasse 02 (5-7) besteht.\
Durch die beisher beschriebenen Klassen entsteht in ein Loch zwischen Klasse 1 und 2, das 
am besten in Dimension 4 gesehen werden kann. In diesen Bereich sollen keine Daten geschoben
werden, wenn improved QSM angewendet wird. Damit beim regulären QSM Daten in diesen Bereich
geschoben werden, darf das Loch in der 1d Verteilung auf der dim_04-Achse nicht sichtbar sein.
Dazu wird eine weitere Klasse (class_05) in den Datensatz eingeführt, die in den Dimensionen
0 - 3 immer den Wert 5 hat, was deutlich über den Werten der anderen Klassen liegt (max. 1,5).
In Dimension 4 liegt sie genau auf Höhe des Lochs zwischen Klasse 01 und 02. 05 hat keine
wirkliche Nachbarschaft mit den anderen Klassen, sollte aber eine herausforderung für QSM
darstellen.