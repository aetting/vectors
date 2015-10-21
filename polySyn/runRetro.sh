#!/bin/bash
#PBS -N makeOnt
#PBS -l walltime=1:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=10gb

vecs='/Users/allysonettinger/Desktop/vectors/polySyn/SenseRetrofit-master/mik-sg-80.txt'
ont='/Users/allysonettinger/Desktop/vectors/polySyn/ontology'
out='/Users/allysonettinger/Desktop/vectors/polySyn/test-alignont.sense'

python senseretrofit.py -v $vecs -q $ont -o $out
