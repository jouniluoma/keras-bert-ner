import os

import sys

import numpy as np
import conlleval

from common import encode, label_encode, write_result
from common import load_pretrained
from common import create_ner_model, create_optimizer, argument_parser
from common import read_conll, process_sentences, get_labels
from common import save_ner_model


def main(argv):
    argparser = argument_parser()
    args = argparser.parse_args(argv[1:])
    seq_len = args.max_seq_length    # abbreviation

    pretrained_model, tokenizer = load_pretrained(args)

    train_words, train_tags = read_conll(args.train_data)
    test_words, test_tags = read_conll(args.test_data)
    train_data = process_sentences(train_words, train_tags, tokenizer, seq_len)
    test_data = process_sentences(test_words, test_tags, tokenizer, seq_len)

    label_list = get_labels(train_data.labels)
    tag_map = { l: i for i, l in enumerate(label_list) }
    inv_tag_map = { v: k for k, v in tag_map.items() }

    train_x = encode(train_data.combined_tokens, tokenizer, seq_len)
    test_x = encode(test_data.combined_tokens, tokenizer, seq_len)

    train_y, train_weights = label_encode(
        train_data.combined_labels, tag_map, seq_len)
    test_y, test_weights = label_encode(
        test_data.combined_labels, tag_map, seq_len)

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

    if args.ner_model_dir is not None:
        label_list = [v for k, v in sorted(list(inv_tag_map.items()))]
        save_ner_model(ner_model, tokenizer, label_list, args)

    probs = ner_model.predict(test_x, batch_size=args.batch_size)
    preds = np.argmax(probs, axis=-1)

    pred_tags = []
    for i, pred in enumerate(preds):
        pred_tags.append([inv_tag_map[t] 
                          for t in pred[1:len(test_data.tokens[i])+1]])

    lines = write_result(
        args.output_file, test_data.words, test_data.lengths,
        test_data.tokens, test_data.labels, pred_tags
    )

    c = conlleval.evaluate(lines)
    conlleval.report(c)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
