#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Usage: $0 input-path" >&2
    exit 1
fi

set -e

input_path="$1"

while true
do
    for i in {1..80}
    do
        echo -n '-'
    done
    echo
    echo -n 'premise> '
    read premise
    echo -n 'hypothesis> '
    read hypothesis
    echo
    python snli_grep.py $input_path --premise-token "$premise" --hypothesis-token "$hypothesis"
    echo
done
