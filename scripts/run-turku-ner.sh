#!/bin/bash

# Run on Turku NER corpus data

# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -euo pipefail

datadir="$SCRIPTDIR/../data/turku-ner"
train_data="$datadir/train.tsv"
test_data="$datadir/dev.tsv"
ner_model_dir="$SCRIPTDIR/../ner-models/turku-ner-model"

modeldir="$SCRIPTDIR/../models/bert-base-finnish-cased-v1"
model="$modeldir/bert_model.ckpt"
vocab="$modeldir/vocab.txt"
config="$modeldir/bert_config.json"

batch_size=8
learning_rate=5e-5
max_seq_length=128
epochs=2

if [ ! -e "$datadir" ]; then
    echo "Data not found (run scripts/get-turku-ner.sh?)" >&2
    exit 1
fi

if [ ! -e "$modeldir" ]; then
    echo "Model not found (run scripts/get-models.sh?)" >&2
    exit 1
fi

rm -rf "$ner_model_dir"
mkdir -p "$ner_model_dir"

python "$SCRIPTDIR/../ner.py" \
    --vocab_file "$vocab" \
    --bert_config_file "$config" \
    --init_checkpoint "$model" \
    --learning_rate $learning_rate \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --batch_size $batch_size \
    --train_data "$train_data" \
    --test_data "$test_data" \
    --ner_model_dir "$ner_model_dir" \
