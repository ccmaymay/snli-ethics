#!/bin/bash

snli_dir=snli_1.0
output_dir=snli_stats
big_python=python
little_python=python

if [ -f snli_cooccur_loop_include.bash ]
then
    source snli_cooccur_loop_include.bash
fi

mkdir -p $output_dir

for model in between-prem-hypo within-premise within-hypothesis
do
    for inference_type in entailment contradiction neutral none
    do
        for filter_hypo_by_prem in true false
        do
            for max_ngram in 1 2
            do
                suffix=''
                args=''

                suffix="${suffix}_max-ngram-$max_ngram"
                args="$args --max-ngram $max_ngram"
                if [ $max_ngram -eq 1 ]
                then
                    python="$little_python"
                else
                    python="$big_python"
                    args="$args --num-proc 7"
                fi

                suffix="${suffix}_inference-type-$inference_type"
                if [ "$inference_type" != none ]
                then
                    args="$args --inference-type $inference_type"
                fi

                suffix="${suffix}_filter-hypo-by-prem-$filter_hypo_by_prem"
                if $filter_hypo_by_prem
                then
                    args="$args --filter-hypo-by-prem"
                fi

                input_path=$snli_dir/snli_1.0_train.jsonl
                output_path=$output_dir/${model}${suffix}.pkl
                echo "$input_path -> $output_path"
                $python snli_cooccur.py $model $args \
                    $input_path $output_path
            done
        done
    done
done
