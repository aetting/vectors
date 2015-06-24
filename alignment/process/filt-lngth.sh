#!/bin/bash

date=150430
exp_dir=/cliphomes/aetting/vectors/alignment/process

lang_in=zh
lang_out=en
lang_pair=$lang_in-$lang_out

## experiment files
train_dir=/fs/clip-scratch/aetting/onto/parl
filtered=/fs/clip-scratch/aetting/onto/parl-proc

#tar xvfJ Zh-En.P3.tar.xz

# cp Zh-En.P3/data/train/* zh-en/ 

#rm zh-en/*UN*

#for filename in *src.*; do mv $filename ${filename//src./}; done
#for filename in *tgt.*; do mv $filename ${filename//tgt./}; done
#for filename in *unseg.*; do mv $filename ${filename//unseg./}; done

cd $train_dir

gunzip *

for f in *.ne.*

do

echo $f

pre=$(echo $f | awk -F '\\.ne\\.' '{print $1}')
post=$(echo $f | awk -F '\\.ne\\.' '{print $2}')  

output=$exp_dir/train.both
# if [ ! -f $output ]; then
    paste -d\| $pre.ne.$post \
        $pre.$post        \
        | perl -pe 's/\|/ \|\|\| /'         \
        > $output
# fi

max_sentence_length=80
input_both=$output
output2=$input_both.filt
# if [ ! -f $output ]; then
    $exp_dir/filter-length.pl  \
        -$max_sentence_length           \
        $input_both             \
        > $output2
# fi

$exp_dir/splitfilt.pl $input_both.filt $filtered/$pre.ne.$post $filtered/$pre.$post

if [ ! -s $filtered/$pre.ne.$post ]; then
    cp $train_dir/$pre.ne.$post $filtered/
    cp $train_dir/$pre.$post $filtered/
fi

done

gzip *
