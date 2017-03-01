#!/bin/bash

set -e

wget http://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip -n snli_1.0.zip
sed -i '1001,$d' snli_1.0/snli_1.0_train.jsonl

virtualenv integration-tests
source integration-tests/bin/activate

pip install -r requirements.txt
python snli_cooccur.py \
    between-prem-hypo \
    --max-ngram 2 \
    --num-proc 7 \
    --filter-hypo-by-prem \
    snli_1.0/snli_1.0_train.jsonl snli_stats/counts.pkl
python snli_query.py \
    -k 5 \
    --filter-to-unigrams \
    --top-y-score-func pmi \
    --min-count 5 \
    snli_stats/counts.pkl top-y top-y.yaml snli_stats/pmi.txt
cat snli_stats/pmi.txt

deactivate
