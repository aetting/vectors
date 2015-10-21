#!/bin/bash
#PBS -N makeOnt
#PBS -l walltime=1:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=10gb

vecs=''
ont=''
out=''

python senseretrofit.py -v $vecs -q $ont -o $out
