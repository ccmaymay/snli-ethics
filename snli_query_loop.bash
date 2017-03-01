#!/bin/bash

min_count=5
output_dir=snli_stats
python=python
extra_args='-k 5 --filter-to-unigrams --top-y-score-func pmi'
query_type=top-y
query_path=top-y.yaml
output_ext=.txt
input_dir=snli_stats
input_paths=`find "$input_dir" -type f -name '*.pkl'`

if [ -f snli_query_loop_include.bash ]
then
    source snli_query_loop_include.bash
fi

mkdir -p "$output_dir"

for input_path in $input_paths
do
    input_bn=`basename $input_path`
    stem="${input_bn%.pkl}"
    output_bn="${stem}${output_ext}"
    output_path="$output_dir/$output_bn"
    echo "$input_path -> $output_path"
    $python snli_query.py "$input_path" $query_type "$query_path" \
        "$output_path" --min-count $min_count $extra_args
done
