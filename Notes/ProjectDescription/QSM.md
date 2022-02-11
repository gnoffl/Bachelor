# Visualization

## Todos

### Important Todos
* [ ] Todos hier Füllen!
* [x] Änderungsbereich einfügen
* [ ] Notizen erstellen
* [ ] Binary Search für Methoden "get_value_at_quantile" und "get_shifted_value" implementieren
* [ ] Überprüfen, dass methode stabil auch bei Randwerte ist und so
* [ ] Sicher stellen, dass lineare Interpolation eine valide Wahl ist

### Optional Todos
* [ ] Überlegen, wie Ergebnisse am besten präsentiert und gespeichert werden (vermutlich
gar nicht explizit speichern, sondern aus gespeicherten Daten nach Bedarf berechnen)

## Notizen

Diese Datei implementiert die _Quantile Shift Method_ (**QSM**). Der Kern ist die Methode __qsm__.
Dieser Methode muss ein Model übergeben werden, das in der Lage ist, auf einem Datensatz
Klassen vorherzusagen. Da diese Vorhersage nicht bei allen Models gleich abläuft, wird
auch eine Funktion mit übergeben, die verwendet werden kann, um Predictions mit dem Model
auf dem Datensatz zu machen, und die Ergebnisse im gewünschten Format zu erhalten. Weiterhin
muss natürlich der Datensatz selbst noch übergeben werden. Für QSM ist es außerdem noch 
nötig, dass eine Liste übergeben wird, in der Für die gewünschten Dimensionen angegeben
wird, wie weit sie verschoben werden sollen (Werte von 0 bis 1, da es sich um die Quantile
handelt).\
Mit Hilfe von Hilfsfunktionen können dann die verschobenen Daten berechnet werden, und 
auf diesen verschobenen Daten dann Vorhersagen mit dem Modell gemacht werden. Sowohl die
verschobenen Dimensionen, als auch die Vorhersagen auf diesen, werden in dem Datenobjekt
gespeichert, sodass später wieder auf sie zugegriffen werden kann.

## Änderungen
