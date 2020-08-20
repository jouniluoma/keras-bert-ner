#!/bin/bash

# Predict using model trained on Turku NER corpus data

# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -euo pipefail

datadir="$SCRIPTDIR/../data/turku-ner"
test_data="$datadir/test.tsv"

ner_model_dir="$SCRIPTDIR/../ner-models/turku-ner-model"
output_file="$SCRIPTDIR/../turku-ner-predictions.tsv"

python "$SCRIPTDIR/../predict.py" \
    --ner_model_dir "$ner_model_dir" \
    --test_data "$test_data" \
    --output_file "$output_file"
