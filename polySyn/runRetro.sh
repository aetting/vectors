#!/bin/bash
#PBS -N makeOnt
#PBS -l walltime=1:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=10gb

vecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt
gvecs=/Users/allysonettinger/Desktop/engw2vModel/enModel
ont=/Users/allysonettinger/Desktop/SenseRetrofit-master/data/sampleonto.txt.gz
out=/Users/allysonettinger/Desktop/polysyn/ontVecs/testing-gformat.sense

python senseretrofit.py -v $gvecs -q $ont -o $out -g 1
