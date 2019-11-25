#!/bin/bash

# Get FiNER data (https://github.com/mpsilfve/finer-data)

# https://stackoverflow.com/a/246128
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -euo pipefail

DATADIR="$SCRIPTDIR/../data"

mkdir -p "$DATADIR"

finerdir="$DATADIR/finer-data"
if [ -e "$DATADIR/finer-data" ]; then
    echo "$finerdir exists, not cloning again"
else
    cd "$DATADIR"
    git clone "https://github.com/mpsilfve/finer-data.git"
    cd "$finerdir"
    git checkout 7006a18 2>/dev/null    # Make sure we have the right version
fi

# Create version with news train/dev/test top-level entities
newsdir="$DATADIR/finer-news"
if [ -e "$newsdir" ]; then
    echo "$newsdir exists, not recreating"
else
    mkdir -p "$newsdir"
    for s in train dev test; do
	# Cut token and top-level tag, remove section tags, and eliminate
	# consecutive and file-initial empty lines
	cut -f 1,2 "$finerdir/data/"digitoday.*.$s.csv \
	    | egrep -v '^<(HEADLINE|INGRESS|BODY)>' \
	    | perl -pe 's/^\s*$/\n/' | cat -s | sed '1{/^$/d}' \
	    > "$newsdir/$s.tsv"
    done
fi

# Create version with news train/dev and wiki test top-level entities
wikidir="$DATADIR/finer-wiki"
if [ -e "$wikidir" ]; then
    echo "$wikidir exists, not recreating"
else
    mkdir -p "$wikidir"
    # Train and dev are shared with news, just symlink 
    for s in train dev; do
	ln -s "../finer-news/$s.tsv" "$wikidir/$s.tsv"
    done
    # Same as above 
    cut -f 1,2 "$finerdir/data/wikipedia.test.csv" \
	| egrep -v '^<(HEADLINE|INGRESS|BODY)>' \
	| perl -pe 's/^\s*$/\n/' | cat -s | sed '1{/^$/d}' \
	> "$wikidir/test.tsv"
fi
