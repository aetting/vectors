#!/bin/bash
#PBS -N makeOnt
#PBS -l walltime=1:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=10gb

ontdir=/Users/allysonettinger/Desktop/polysyn/ontVecs
#aligndir=/fs/clip-xling/projects/polysemy/alignments/5m-onto-unfilt
aligndir=/Users/allysonettinger/Desktop/alignments/300k_align
pivotlang=zh
pthresh=8
cthresh=3
ceil=1
k=1
mid=8
gthresh=30

#python makeOntology.py $aligndir $pivotlang $ontdir $pthresh $cthresh $ceil $k $mid
python makeOntology.py $aligndir $pivotlang $ontdir $gthresh $ceil $k $mid

