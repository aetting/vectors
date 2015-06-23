#!/bin/bash

date=150430
exp_dir=/Users/allysonettinger/Desktop/filterout

lang_in=zh
lang_out=en
lang_pair=$lang_in-$lang_out

## experiment files
train_dir=/Users/allysonettinger/Desktop/filtertest
filtered=/Users/allysonettinger/Desktop/filtered

cd $train_dir

gunzip *

for f in *.ne.*

do

pre=$(echo $f | awk -F '.ne.' '{print $1}')
post=$(echo $f | awk -F '.ne.' '{print $2}')  

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

done

gzip *