# TODOs

## General todos

* [ ] __comment code__
  * [ ] QSM
* [ ] make __proper notes__ of things that were already done
  * [ ] QSM
* [ ] make __typing__ in code consistent
  * [ ] visualization
  * [ ] dataCreation
  * [ ] Classifier
  * [ ] QSM


## Logical progress

* [ ] NN classifier for Soccer                                    5 days  (-15.05.)
* [ ] optimize parameters                                         3 days  
* [ ] think about HiCS Parameters (read HiCS paper again)         1 day
* [ ] check all (optional) todos                                  2 days  (-22.05.)
* [ ] Experiments                                                 ?
* [ ] write thesis                                                5,5 - ? weeks (-01.07.)



# Done

* [x] Correct Data creation (--> todos in dataCreation.md)
* [x] validate results from decision tree (how close classes actually are)
* [x] make HiCS implementation work
* [x] load and save data from data classes properly
* [x] save tree and corresponding visualization
* [x] make sure HiCS doesnt get to see more columns than its supposed to
* [x] official paperwork for thesis (need to wait for Daniel to hear about
approval of request)
* [x] refactor training / prediction process of decision trees (prediction not
needed)
* [x] change dataset to show weaknesses of vanilla QSM better (no overlap of
class_05 with any other class in dim_04)
* [x] add some outputs, so a sense of progression is given
* [x] implement execution of QSM on subsets of a given dataset
* [x] finish implementation QSM
* [x] extract all tuning parameters into one file
* [x] plan progress for whole thesis
* [x] add missing Documentation and notes
* [x] remaining todos from Binning.md
* [x] iris Dataset
  * [x] create Iris Dataset in Datacreation
  * [x] add notes for IrisDataSet in datacreation.md
  * [x] check for problems in the pipeline
  * [x] fix problems in the pipeline
* [x] make sure that qsm works (cumulated dists dont look the same)
* [x] fix splitting of data (dont split between 2 points with the same value)* [x] Soccer Dataset
* [x] create Soccer Dataset in Datacreation
  * [x] add notes for SoccerDataSet in datacreation.md
  * [x] check for problems in the pipeline
  * [x] fix problems in the pipeline
    * [x] HiCS doesnt calculate all 2d Subspaces --> calculate them directly
      * [x] direct calculation somehow gives different results from normal calculation in list
  * [x] __optional:__ make colors in visualization consistent