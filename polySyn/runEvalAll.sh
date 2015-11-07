#!/bin/bash

sg_vecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt
wordnet_sensevecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/data/mik.origscript.sense
retroA_PMI=/Users/allysonettinger/Desktop/polysyn/ontVecs/11-5-1-1-12.sense

testvecs=$wordnet_sensevecs
genformat=text

TOEFL=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/toefl/TOEFL-80
ESL=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/ESL-50
RD=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/RD-300

MENtest=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_lemma_form.test
MENdev=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_lemma_form.dev
ws353_combined=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/wordsim353/ws353.combined.tab.formatted
MEN=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MEN/MEN_dataset_natural_form_full
MC30=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/MC-30.txt
RG65=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/RG-65.txt

synout=/Users/allysonettinger/Desktop/vectors/polySyn/synResults/summary
simout=/Users/allysonettinger/Desktop/vectors/polySyn/simResults/summary

python evaluateSim.py $testvecs $genformat $ws353_combined $RG65 $MC30 $MENtest > $simout
python evaluateSyn.py $testvecs $genformat $ESL $RD $TOEFL > $synout
