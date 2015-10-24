#!/bin/bash
#PBS -N makeOnt
#PBS -l walltime=3:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=10gb

ontdir=/Users/allysonettinger/Desktop/polysyn/ontVecs
simdir=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets

aligndir=/Users/allysonettinger/Desktop/alignments/300k_align
pivotlang=zh
pthresh=8
cthresh=3
ceil=1
k=1
mid=8

sg_vecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt
ont=$ontdir/ontology-$pthresh-$cthresh-$ceil-$k-$mid
sensevecs=$ontdir/$pthresh-$cthresh-$ceil-$k-$mid.sense

genformat=text
MENtest=$simdir/MEN/MEN_dataset_lemma_form.test
ws353_combined=$simdir/wordsim353/ws353.combined.tab.formatted
MENdev=$simdir/MEN/MEN_dataset_lemma_form.dev

python makeOntology.py $aligndir $pivotlang $ontdir $pthresh $cthresh $ceil $k $mid
python senseretrofit.py -v $sg_vecs -q $ont -o $sensevecs
python evaluateSim.py $sensevecs $genformat $MENdev 

