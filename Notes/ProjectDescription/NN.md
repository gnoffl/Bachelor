# Classifier

## Todos

### Important Todos
* [ ] Änderungen hier notieren

### Optional Todos
* [ ] Improve Net
* [ ] Speichern
* [ ] Laden(von Classifier.load_classifier aus)
* [ ] think about normalization (right now "normalizing" each dataset (training/test) individually) --> removed 
  normalization entirely

### Done

* [x] Todos hier Füllen!
* [x] Überklasse für classifier erstellen
* [x] predict_fn für tree aus qsm in klasse überführen, dataset statt dataset.data verwenden
* [x] fix QSM
* [x] predict funktion erstellen  
* [x] Visualisieren
* [x] Bug bei vergabe von Farben in bildern?? (Farbe von klasse wechselt zwischen org und pred) (--> probably fixed,
error was in loading / saving the datasets class_names, where all spaces were deleted --> didnt find the correct index
of the classname, to determine the correct color)
* [x] baseline prediction von vanilla qsm und binned qsm sind nicht die selben?? (--> wegen normalisierung)
* [x] Evaluate Performance in comparison with decisionTree (--> NN nur leicht besser (67% vs 61%)
* [x] Notizen erstellen

## Notizen
Neural Network with 1 hidden layers with 50 nodes.not the greatest performance so far.

## Änderungen