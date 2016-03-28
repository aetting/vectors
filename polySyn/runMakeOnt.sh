#!/bin/bash
#PBS -N makeOnt
#PBS -l walltime=1:00:00
#PBS -M aetting@umd.edu
#PBS -m ae
#PBS -l pmem=10gb

stat=G
statthresh=100
ceil=1
k=1
mid=110
lem=1
log=1
symm=1

notes='count thresh unchanged'

#aligndir=/fs/clip-xling/projects/polysemy/alignments/5m-onto-unfilt
aligndir=/Users/allysonettinger/Desktop/alignments/300k_align
pivotlang=zh


sg_vecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt
sg_lemvecs=/Users/allysonettinger/Desktop/SenseRetrofit-master/mik-sg-80.txt

if [ $lem == 1 ]; then 
    ontdir=/Users/allysonettinger/Desktop/polysyn/ontVecs-lem
    inputvecs=$sg_lemvecs
    retro_gform=0
else 
    ontdir=/Users/allysonettinger/Desktop/polysyn/ontVecs
    inputvecs=$sg_vecs
    retro_gform=0
fi

ont=$ontdir/ontology-$stat-$statthresh-$ceil-$k-$mid-$symm
outputvecs=$ontdir/$stat-$statthresh-$ceil-$k-$mid-$symm.sense
iters=500


simdir=/Users/allysonettinger/Desktop/vectors/polySyn/similarity-datasets
genformat=text
MENtest=$simdir/MEN/MEN_dataset_lemma_form.test
ws353_combined=$simdir/wordsim353/ws353.combined.tab.formatted
MENdev=$simdir/MEN/MEN_dataset_lemma_form.dev
testvecs=$outputvecs


python makeOntologyTesting.py -a $aligndir -p $pivotlang -o $ontdir -s $stat -h $statthresh -t $ceil -k $k -m $mid -l $lem -g $log -y $symm
#python senseretrofit.py -v $inputvecs -q $ont -o $outputvecs -g $retro_gform
#python evaluateSim.py $testvecs $genformat $MENdev

echo $notes
echo $stat,$statthresh,$ceil,$k,$mid
echo lem: $lem
echo log: $log
