# DataCreation

## Todos

### Important Todos
* [ ] Typing überall einhalten

### Optional Todos
* [ ] Laden und Speichern mit pickle oder json oder so implementieren
* [ ] load_tree doesnt work when used on split of a dataset, because tree is saved in "root" dataset

### Done
* [x] Code Kommentieren
* [x] 1d Verteilung in dim_04 sollte kein Loch enthalten --> eine Klasse "neben" das Loch
zwischen Klasse 1 und 2 legen
* [x] Gap zwischen 1 und 2 verringern
* [x] 2, 3, 4 besser unterscheidbar machen ohne dim_04, dafür in dim_04 vielleicht näher aneinander
rücken
* [x] 5 auf "andere Seite" (--> niedrige Werte) rücken
* [x] Speichern von Objekt (auch in "Änderungen" vermerken)
* [x] Laden von Objekt (Kommentare fehlen, Methode übersichtlicher machen)
* [x] HiCS darf nur die korrekten Spalten sehen (keine die Klassen enthalten, nur daten)
* [x] Iris Daten kommentieren
* [x] Iris Daten dokumentieren
* [x] Namen von classen für Daten korrekt darstellen
* [x] Soccer datensatz implementieren
  * [x] initial members
  * [x] klassennamen (aus classes) "säubern"
  * [x] Libero entfernen
  * [x] mit NaNs dealen
* [x] parse_params soll nicht static sein --> sollte warning beheben
* [x] Dokumentation SoccerDataSet
* [x] __load_tree__ sollte zu __load_model__ werden --> (load ist jetzt in Classifier implementiert, nicht
mehr in dataCreation)


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
der erbenden Datensätze enthalten sein sollen. Weiterhin werden einige Methoden definiert,
die für alle Datenklassen praktisch sind.\
Von den Datenklassen selbst ist "MaybeActualDataSet" die wichtigste, da sie
tatsächlich erstmal den Datensatz darstellt, mit dem Experimente durchgeführt werden.

### Data
Wichtige Methoden aus Data, die allen Klassen zur Verfügung gestellt werden sind
folgende:

* load
* save
* run_hics

_load_ ist dabei nur eine abstrakte Methode, die in jeder Klasse einzeln implementiert
werden muss. _save_ erlaubt das Speichern von Objekten in einem Ordner. _run_hics_ 
speichert die Daten zunächst in einer .csv Datei, die den Anforderungen für HiCS 
entspricht. Anschließend wird HiCS auf dieser Datei aufgerufen, und die Ergebnisse in
einer eigenen Datei gespeichert. Wenn nicht anders angegeben, werden sowohl die input-
Daten für HiCS, als auch der output, im selben Ordner gespeichert, der zum Speichern 
vom Objekt verwendet wird.

### MaybeActualDataSet
Der Datensatz hat zur Zeit 8 Dimensionen, von denen 3 ("rand_00", "rand_01", "rand_02")
nur mit zufälligen Werten gefüllt sind. Die weiteren Dimensionen ("dim_00" - "dim_04")
sind mit relevanter Information gefüllt. Die Daten sind in den Dimensionen 00 - 03
Normalverteilt, während in dim_04 symmetrische Dreiecksverteilungen genutzt werden,
um die maximale Ausdehnung der Daten genauer kontrollieren zu können. Dabei haben
Klasse 01 und 02 in Dimensionen 00 - 03 die exakt gleichen Verteilungen (00/01: mean=1.5,
02/03: mean=0).\
Alle Klassen unterscheiden sich in ihren Verteilungen in der letzten Dimension
(04). Die Mittelwerte steigen mit der Klassennummer. Es besteht immer eine Überschneidung
zwischen den Bereichen aufeinander folgender Klassen, wobei die einzige Ausnahme ein Spalt
zwischen Klasse 01 (1-3) und klasse 02 (4-6) besteht.\
Durch die bisher beschriebenen Klassen entsteht in ein Loch zwischen Klasse 1 und 2 in
Dimension 4. In diesen Bereich sollen keine Daten geschoben werden, wenn improved QSM
angewendet wird. Damit beim regulären QSM Daten in diesen Bereich geschoben werden, darf
das Loch in der 1d Verteilung auf der dim_04-Achse nicht sichtbar sein. Dazu wird eine
weitere Klasse (class_05) in den Datensatz eingeführt, die in den Dimensionen 0 - 3 immer
den Wert -3 hat, was deutlich unter den Werten der anderen Klassen liegt. In Dimension 4
liegt sie genau auf Höhe des Lochs zwischen Klasse 01 und 02. 05 hat keine wirkliche
Nachbarschaft mit den anderen Klassen, sollte aber eine herausforderung für QSM darstellen.

### IrisDataSet
Diese Klasse ist eine "Data" entsprechende Implementierung, die die Daten des bekannten
Iris-Datensatzes enthält. Die Daten werden über _sklearn.datasets_ in einem DataFrame
importiert. Weitere erforderliche Metadaten werden beim Initialisieren der Objekte
erstellt. Alle abstrakten Methoden der Data Klasse sind implementiert.

### SoccerDataSet
Diese Klasse ist eine "Data" entsprechende Implementierung, die Leistungsdaten von
Fußballspieler Fußballspielern in verschiedenen Saisons enthält, sowie deren Position
auf dem Spielfeld. Der Datensatz wird genutzt, um aus den Leistungsdaten die
Position vorher zu sagen.

## Änderungen
### Neue Klasse eingeführt (24.01.22)
In bisherigen Daten war ein Loch zwischen class_01 und class_02. Das hat dazu geführt, dass
die 1d Verteilung in dim_04 in diesem Bereich geflatlined ist. Um bei QSM aber wirklich Werte
in das Loch zu schieben, darf die 1d-Dichte im Bereich des Lochs nicht 0 sein. Dazu
wurde eine weitere Klasse (class_05) eingeführt, die zwar in dim_00 bis dim_03 deutlich
neben dem Loch liegt, aber in dim_04 genau die Höhe des Lochs hat, und somit die 1d-
Dichte in dim_04 korrigiert.

### Update für abstrakte Klasse Data (29.01.22)
Neue Methoden:
* create_class_info
* save_data_for_hics
* run_hics

Funktionalität um HiCS durchführen zu können.

### Update für Data Klasse und MaybeActualDataSet (05.02.22)
Funktionalität für HiCS komplett überarbeitet, und Save / Load implementiert.\
Data hat neues Attribut "data_columns", eine Liste, in der die Namen der Spalten gespeichert sind,
die reine Daten enthalten. Dies schließt also hauptsächlich Spalten aus, in denen Klassen gespeichert
sind, oder Vorhersagen von Klassen. _save_data_for_hics_ nimmt nun nur noch die Spalten, die in
"data_columns" gespeichert sind, sodass keine vorhergesagten Klassen die Suche nach Unterräumen
verfälschen kann.\
Speichern ist als _save_ in der Überklasse "Data" implementiert. Aus dem Klassennamen der erbenden
Klasse sowie dem Datum der Erstellung der Klasse wird ein eindeutiger Pfad generiert, unter dem die
Informationen für das zu speichernde Objekt abgelegt werden. Beim Speichern wird eine Datei "data.csv"
anegelegt, die die tatsächlichen Daten des Objekts enthalten. Alle Metadaten sowie Notizen und
Informationen zu durchgeführten Experimenten mit diesem Datensatz werden in einer 2. Datei
"description.txt" gespeichert. Der Pfad / Ordner kann auch für andere Informationen verwendet werden,
die für den entsprechenden Datensatz relevant sind (z.B. HiCS input und output werden auch hier
gespeichert).\
Load wird von "Data" nur als abstrakte Methode vorgegeben und muss von den erbenden Klassen einzeln
einzeln implementiert werden. Dies wurde für "MaybeActualDataSet" durchgeführt.\
Funktionalität von Datei HiCS.py nahezu vollständing in diese Datei übernommen. Notizen und Parameter
für HiCS werden jetzt korrekt in den Notizen des Daten-Objekts gespeichert. Abschnitte in den 
Notizen lassen sich jetzt mit Methode _end_paragraph_in_notes_ von einander abtrennen.

### load_tree (11.02.22)

Data gibt jetzt Method zum Laden eines trainierten DecisionTrees vor. Diese Funktion
sollte in Zukunft so überarbeitet werden, dass ein allgemeines Model geladen werden
kann.

### Änderungen der Lage der Klassen (26.02.22)

Die Lage der Klassen in dim_04 wurde so verändert, dass class_05 in dim_04 nun mit
kleiner Klasse mehr eine Überschneidung hat. Dazu wurde die Dreiecksverteilung von
class_05 in dim_04 von (2.5, 3.5, 4.5) auf (3, 4, 5) geändert. Die darüberliegenden
Klassen (02, 03 und 04) wurden auch angehoben. Statt (4, 5, 6) haben sind nun die
Verteilung (5, 6, 7). Durch die Änderung soll bewirkt werden, dass alle Datenpunkte,
die in dim_04 zwischen 3 und 5 liegen, von einem DecisionTree automatisch als 
class_05 klassifiziert werden. Dies sollte den Unterschied zwischen QSM und der
verbesserten QSM besser verdeutlichen.

### clone_meta_data und take_new_data hinzugefügt (12.03.22)

Zwei neue abstrakte Methode zu Data Superklasse hinzugefügt. _clone_meta_data_ erlaubt es,
ein neues Data-Objekt zu erzeugen, das dieselben Metainformationen in seinen Attributen 
trägt.\
_take_new_data_ erlaubt es, einen neuen Dataframe als Daten anzunehmen und relevante Attribute
entsprechend anzupassen. Für MaybeActualDataSet ist dies vor allem das _members_ Attribut.\
Beide Methoden wurden bisher nur für MaybeActualDataSet implementiert.

### IrisDataSet zugefügt, refactoring (28.04.22)

IrisDataSet wurde entsprechend der Vorgaben von Data implementiert. Daten aus sklearn.datasets.\
Viele Methoden aus MaybeActualDataSet wurden nach Data verschoben (vor allem Methoden zum einlesen
von gespeicherten Datensätzen).\
Load Methode ist nun auch in Data implementiert. Die description.txt datei wurde um eine Zeile
ergänzt. In der ersten Zeile wird nun die Klasse angegeben, zu der der Datensatz gehört. Wird
load aus Data aufgerufen, wird zunächst die description datei geladen, und die Klasse des
Datensatzes überprüft. Je nach Klasse wird dann die korrekte Implementierung von load aus einer
der Tochterklassen aufgerufen.