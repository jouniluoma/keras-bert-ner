#!/bin/bash

# Get Turku NER data (https://github.com/TurkuNLP/turku-ner-corpus/)

# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -euo pipefail

DATADIR="$SCRIPTDIR/../data"

mkdir -p "$DATADIR"

turkunerdir="$DATADIR/turku-ner-corpus"
if [ -e "$turkunerdir" ]; then
    echo "$turkunerdir exists, not cloning again"
else
    cd "$DATADIR"
    git clone "https://github.com/TurkuNLP/turku-ner-corpus.git"
    cd "$turkunerdir"
    git checkout fde14ff 2>/dev/null    # Make sure we have the right version
fi

# Symlink .tsv data for more convenient access
targetdir="$DATADIR/turku-ner"
if [ -e "$targetdir" ]; then
    echo "$targetdir exists, not recreating"
else
    mkdir -p "$targetdir"
    for s in train dev test; do
	ln -s "../turku-ner-corpus/data/conll/$s.tsv" "$targetdir/$s.tsv"
    done
fi
