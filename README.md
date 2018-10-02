# Co-occurrence computation for SNLI

[![Build Status](https://travis-ci.org/cjmay/snli-ethics.svg?branch=master)](https://travis-ci.org/cjmay/snli-ethics)
   
This repository contains the code for the 2017 paper
"[Social Bias in Elicited Natural Language Inferences](http://www.ethicsinnlp.org/workshop/pdf/EthNLP09.pdf)"
by Rachel Rudinger, Chandler May, and Benjamin Van Durme.
Rachel Rudinger and Chandler May contributed to this code, which is
released under the two-clause BSD license.

## Prerequisites

Install dependencies with:

```bash
pip install -r requirements.txt
```

Download and unzip [the SNLI data](http://nlp.stanford.edu/projects/snli/snli_1.0.zip):

```bash
wget http://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
```

## Computing counts

Compute counts for unigrams and bigrams, across all inference types,
using 7 subprocesses (in addition to the main process), and filtering
out hypothesis words that occur in the premise.  Read SNLI pairs from
`snli_1.0/snli_1.0_train.jsonl` and write counts to
`snli_stats/counts.pkl`:

```bash
python snli_cooccur.py \
    between-prem-hypo \
    --max-ngram 2 \
    --num-proc 7 \
    --filter-hypo-by-prem \
    snli_1.0/snli_1.0_train.jsonl snli_stats/counts.pkl
```

Run `python snli_cooccur.py --help` for more options.

### Looping over all configurations

Alternatively, compute counts for all parameter configurations, in a
loop:

```bash
bash snli_cooccur_loop.bash
```

To change the default input and output directories, or change the
Python interpreter used to run `snli_cooccur.py`, create a file named
`snli_cooccur_loop_include.bash` with the following contents and
modify them as desired (and then run `snli_cooccur_loop.bash`):

```
snli_dir=snli_1.0
output_dir=snli_stats
big_python=python
little_python=python
```

The `little_python` and `big_python` variables are the Python
commands used for unigram and unigram-and-bigram models, respectively.
The latter have higher memory requirements.
(Note the `little_python` and `big_python` variables can be set to job
submission scripts invoking a python interpreter to parallelize the
computation on a grid.)

## Querying co-occurrences from counts

Query top-five co-occurrence lists, ranked by PMI, filtering candidates
to unigrams (filtering out bigrams), and filtering out co-occurrence
candidates with count less than five.  Run queries from the YAML
specification in `top-y.yaml`, using counts from
`snli_stats/counts.pkl`, and write output to `snli_stats/pmi.txt`:

```bash
python snli_query.py \
    -k 5 \
    --filter-to-unigrams \
    --top-y-score-func pmi \
    --min-count 5 \
    snli_stats/counts.pkl top-y top-y.yaml snli_stats/pmi.txt
```

Run `python snli_query.py --help` for more options.

### Looping over all configurations

Alternatively, run queries for all parameter configurations, in a
loop:

```bash
bash snli_query_loop.bash
```

To change the default input paths and output directory, or change the
Python interpreter used to run `snli_query.py` or other settings,
create a file named `snli_query_loop_include.bash` with the following
contents and modify them as desired (and then run
`snli_query_loop.bash`):

```
min_count=5
output_dir=snli_stats
python=python
extra_args='-k 5 --filter-to-unigrams --top-y-score-func pmi'
query_type=top-y
query_path=top-y.yaml
output_ext=.txt
input_dir=snli_stats
input_paths=`find "$input_dir" -type f -name '*.pkl'`
```

## Errata

In the definition of the likelihood ratio Λ(C') in the paper (last equation on the second page, or page 75 in the proceedings), the summations [should be products](https://en.wikipedia.org/wiki/G-test#Derivation).  The code and results use the correct definition.

The implementation of the Holm-Bonferroni method has a bug:  Suppose m hypotheses are tested and ordered by their p-values, from smallest to largest.  In the implementation all hypotheses with p-values satisfying p ≤ α/(m + 1 - k) are rejected (where α is the significance level and k is the rank of the original p-value).  However, only the first K - 1 hypotheses satisfying that inequality, such that hypothesis K does not satisfy it, [should be rejected](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method#Formulation).  Our implementation may falsely reject hypotheses after hypothesis K if the p-values are sufficiently uniform.
