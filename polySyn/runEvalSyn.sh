#!/bin/bash

sg_vecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt
sg_vecs_cut=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80-cut.txt
wordnet_sensevecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/data/mik.origscript.sense
retroA_PMI=/Users/allysonettinger/Desktop/polysyn/ontVecs/11-5-1-1-12.sense
modB=/Users/allysonettinger/Desktop/engw2vModel/engModelB

genformat=text

TOEFL=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/toefl/TOEFL-80
ESL=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/ESL-50
RD=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/RD-300

ESL_sw=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/ESL-50-single-word
RD_sw=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets/esl-rd/RD-300-single-word


python evaluateSyn.py -v $retroA_PMI -g text -s $ESL $TOEFL
#python evaluateSyn.py -v $sg_vecs -g text -v $retroA_PMI -g text -s $TOEFL -y 1 
