#!/bin/bash

sg_vecs='/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt'
wordnet_sensevecs='/Users/allysonettinger/Desktop/SenseRetrofit-master/data/mik.sampont.sense'
genformat='text'
MENtest='/Users/allysonettinger/Desktop/similarity-datasets/MEN/MEN_dataset_lemma_form.test'
ws353_combined='/Users/allysonettinger/Desktop/similarity-datasets/wordsim353/ws353.combined.tab.formatted'

python evaluateSim.py $sg_vecs $genformat $ws353_combined
