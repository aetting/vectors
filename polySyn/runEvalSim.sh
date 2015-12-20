#!/bin/bash

sg_vecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt
sg_vecs_cut=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80-cut.txt
wordnet_sensevecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/data/mik.origscript.sense
retroA_PMI=/Users/allysonettinger/Desktop/polysyn/ontVecs/11-5-1-1-12.sense

genformat=text

MENtest=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_lemma_form.test
MENdev=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_lemma_form.dev
ws353_combined=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/wordsim353/ws353.combined.tab.formatted
MEN=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_natural_form_full
MC30=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MC-30.txt
RG65=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/RG-65.txt

#python evaluateSim.py -v $sg_vecs -g text -v $retroA_PMI -g text -s $MC30 
python evaluateSim.py -s $MC30 -v $sg_vecs -g text -c $sg_vecs -c $wordnet_sensevecs -m max
