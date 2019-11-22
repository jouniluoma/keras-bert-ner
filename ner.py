import os

# TF_KERAS must be added to environment variables in order to use TPU
# os.environ['TF_KERAS'] = '1'

import sys

import numpy as np
import conlleval

from common import encode, label_encode, read_tags, read_data, write_result
from common import combine_lines, read_sentences, load_pretrained
from common import create_ner_model, create_optimizer, argument_parser


def main(argv):
    argparser = argument_parser()
    args = argparser.parse_args(argv[1:])
    seq_len = args.max_seq_length    # abbreviation

    pretrained_model, tokenizer = load_pretrained(args)

    tag_map = read_tags(args.train_data)
    tag_map['[SEP]']=len(tag_map)    # add special tags
    tag_map['[PAD]']=len(tag_map)

    train_lines, train_tags, train_lengths = read_data(
        args.train_data, tokenizer, seq_len-1)
    test_lines, test_tags, test_lengths = read_data(
        args.test_data, tokenizer, seq_len-1)

    tr_lines, tr_tags, tr_numbers = combine_lines(
        train_lines, train_tags, train_lengths, seq_len)
    te_lines,te_tags,te_numbers = combine_lines(
        test_lines, test_tags, test_lengths, seq_len)

    train_x = encode(tr_lines, tokenizer, seq_len)
    test_x = encode(te_lines, tokenizer, seq_len)

    train_y, train_weights = label_encode(tr_tags, tag_map, seq_len)
    test_y, test_weights = label_encode(te_tags, tag_map, seq_len)

    ner_model = create_ner_model(pretrained_model, len(tag_map))
    optimizer = create_optimizer(len(train_x[0]), args)

    ner_model.compile(
        optimizer,
        loss='sparse_categorical_crossentropy',
        sample_weight_mode='temporal',
        metrics=['sparse_categorical_accuracy']
    )

    ner_model.fit(
        train_x,
        train_y,
        sample_weight=train_weights,
        epochs=args.num_train_epochs,
        batch_size=args.batch_size
    )

    probs = ner_model.predict(test_x, batch_size=args.batch_size)

    tag_dict_inv = { v: k for k, v in tag_map.items() }

    preds = np.argmax(probs, axis=-1)
    pred_tags = []
    for i, pred in enumerate(preds):
        pred_tags.append([tag_dict_inv[t] for t in pred[1:len(test_lines[i])+1]])

    sent = read_sentences(args.test_data)

    lines = write_result(
        args.result_file, sent, test_lengths, test_lines, test_tags, pred_tags)

    c = conlleval.evaluate(lines)
    conlleval.report(c)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
