#!/bin/bash

sg_vecs='/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt'
wordnet_sensevecs='/Users/allysonettinger/Desktop/SenseRetrofit-master/data/mik.origscript.sense'
genformat='text'
MENtest='/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_lemma_form.test'
MENdev='/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_lemma_form.dev'
ws353_combined='/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/wordsim353/ws353.combined.tab.formatted'
MEN=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_natural_form_full
MC30=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MC-30.txt
RG65=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/RG-65.txt

python evaluateSim.py $sg_vecs $genformat $MC30 $RG65
