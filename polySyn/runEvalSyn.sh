#!/bin/bash

sg_vecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt
wordnet_sensevecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/data/mik.origscript.sense
retroA_PMI=/Users/allysonettinger/Desktop/polysyn/ontVecs/11-5-1-1-12.sense
genformat=text
TOEFL=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/toefl/TOEFL-80
ESL=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/ESL-50
RD=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/RD-300

ESL=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/ESL-50-single-word
RD=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/RD-300-single-word


python evaluateSyn.py $sg_vecs $genformat $ESL $TOEFL
