#!/usr/bin/env python3

# Compare two CoNLL-style files using conlleval.

import sys

import conlleval


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('gold')
    ap.add_argument('pred')
    return ap


def read_conll(input_file):
    # TODO avoid duplication
    words, labels = [], []
    curr_words, curr_labels = [], []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                fields = line.split('\t')
                if len(fields) > 1:
                    curr_words.append(fields[0])
                    curr_labels.append(fields[1])
                else:
                    print('ignoring line: {}'.format(line))
                    pass
            elif curr_words:
                words.append(curr_words)
                labels.append(curr_labels)
                curr_words, curr_labels = [], []
    if curr_words:
        words.append(curr_words)
        labels.append(curr_labels)
    return words, labels


def compare(gold_toks, gold_tags, pred_toks, pred_tags):
    if len(gold_toks) != len(pred_toks):
        raise ValueError('sentence count mismatch: {} in gold, {} in pred'.\
                         format(len(gold_toks), len(pred_toks)))
    lines = []
    for g_toks, g_tags, p_toks, p_tags in zip(
            gold_toks, gold_tags, pred_toks, pred_tags):
        if g_toks != p_toks:
            raise ValueError('text mismatch: gold "{}", pred "{}"'.\
                             format(g_toks, p_toks))
        for (g_tok, g_tag, p_tag) in zip(g_toks, g_tags, p_tags):
            lines.append('{}\t{}\t{}'.format(g_tok, g_tag, p_tag))

    return conlleval.report(conlleval.evaluate(lines))


def main(argv):
    args = argparser().parse_args(argv[1:])
    gold_toks, gold_tags = read_conll(args.gold)
    pred_toks, pred_tags = read_conll(args.pred)
    result = compare(gold_toks, gold_tags, pred_toks, pred_tags)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
