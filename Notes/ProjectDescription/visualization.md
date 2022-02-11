# Visualization

## Todos

### Important Todos
* [x] Todos hier Füllen!
* [x] Änderungsbereich einfügen
* [x] Notizen erstellen
* [x] Code kommentieren

### Optional Todos
* [ ] Typing einhalten
* [ ] statt .gif ein richtiges Film format benutzen
* [ ] Bilder beim erstellen von Bildfolgen nicht zuerst auf Festplatte speichern
* [ ] Seaborn lib auschecken (kurz erwähnt [hier](https://youtu.be/lnfGvdCqGYs?t=147))
* [ ] bei einzelnen Funktionen die Möglichkeit zum abspeichern geben


## Notizen
visualization.py enthält die entscheidenden Funtionen, um die Visualisierung von Daten
zu ermöglichen. Wichtige Funktionen sind dabei:

* visualize_2d
* visualize_3d
* create_3d_gif
* create_hist
* create_cumulative_plot
* get_change_matrix

Während __visualize_2d__ einen Datensatz in 2d abbildet, kann __visualize_3d__
Datensaätze in 3 Dimensionen abbilden, und hat dabei zusätzlich Freiheiten, was die
Rotation angeht. __create_3d_gif__ nutzt diese Freiheiten aus und erstellt eine Serie von 3d Bilder, die 
dann in einer .gif Datei zusammen abgespeichert werden.\
__create_hist__ und __create_cumulative_plot__ sind Funktionen, die die 1d Verteilung von Daten
untersuchen. Es wird dabei entweder ein Histogramm mit gleich breiten "bins" erstellt,
oder ein Graph der kumulierten Daten.\
Die letzte Funktion __get_change_matrix__ erlaubt die Klassifizierung von Daten mit einer
Referenz zu vergleichen. Es wird eine Migrationsmatrix zurück gegeben, in der angegeben
ist, wie viele Datenpunkte von jeder Klasse in die anderen Klassen migrieren.

## Änderungen
_Noch keine Änderungen passiert_