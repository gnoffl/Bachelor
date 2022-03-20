# Fragen
1. benutzen wir nur KS-test auf der "shift"-dimension? sollten wir nicht die
unterschiede der Verteilungen in allen Dimensionen des HiCS mit einem Test checken?

2. HiCS werden gerade für nichts benutzt. Es wird nur paarweise die Korrelation zwischen
Dimensionen untersucht. Die Information von dem "gewählten" besten Subspace aus HiCS macht
weder beim split der Daten, noch bei der auswahl weiterer Dimensionen einen Unterschied.

3. ist HiCS geeignet, um korrelationen zwischen 2 dimensionen zu untersuchen?
gäbe es nicht andere tests?

4. sicher, dass wir, nachdem in einem HiCS gesplittet wurde, wieder alle
Dimensionen nehmen, um neuen HiCS zu suchen? sollte das nicht schon vorher
gefunden worden sein, wenn es abhängigkeiten in anderen Dimensionen gibt?

5. warum ist der pvalue beim ks-test manchmal flat 0, obwohl der D-Wert nicht
besonders ist? (D-Werte die größer und kleiner sind haben beide pvalues != 0)
 --> keine korrektur von a nötig --> verschiedene niveaus von a testen (0,05, 0,01)

6. kann ich als tiebreaker für p = 0 die Formeln --> erstmal größtes D nutzen, evtl gucken,
wie sich das mit n und m verhält
[hier](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) nutzen?\
![erste Formel](Formel01.png) ![zweite Formel](Formel02.png)

7. Laufzeit: parallelisieren?
