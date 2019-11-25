#!/bin/bash

# Run NER on FiNER news data

# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -euo pipefail

datadir="$SCRIPTDIR/../data/finer-news"
train_data="$datadir/train.tsv"
test_data="$datadir/test.tsv"

modeldir="$SCRIPTDIR/../models/bert-base-finnish-cased-v1"
model="$modeldir/bert_model.ckpt"
vocab="$modeldir/vocab.txt"
config="$modeldir/bert_config.json"

batch_size=8
learning_rate=5e-5
max_seq_length=128
epochs=2

if [ ! -e "$datadir" ]; then
    echo "Data not found (run scripts/get-finer.sh?)" >&2
    exit 1
fi

if [ ! -e "$modeldir" ]; then
    echo "Model not found (run scripts/get-models.sh?)" >&2
    exit 1
fi

python "$SCRIPTDIR/../ner.py" \
    --vocab_file "$vocab" \
    --bert_config_file "$config" \
    --init_checkpoint "$model" \
    --learning_rate $learning_rate \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --batch_size $batch_size \
    --train_data "$train_data" \
    --test_data "$test_data"
