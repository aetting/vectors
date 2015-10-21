#!/bin/bash
#PBS -N makeOnt
#PBS -l walltime=1:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=10gb

#aligndir='/fs/clip-xling/projects/polysemy/alignments/5m-onto-unfilt'
aligndir='/Users/allysonettinger/Desktop/alignments/300k_align'
pivotlang='zh'

python makeOntology.py $aligndir $pivotlang
