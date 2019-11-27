import os
import sys
import unicodedata

import numpy as np

from common import process_sentences, load_ner_model
from common import encode, write_result
from common import argument_parser


punct_chars = set([
    chr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(chr(i)).startswith('P') or
        ((i >= 33 and i <= 47) or (i >= 58 and i <= 64) or
         (i >= 91 and i <= 96) or (i >= 123 and i <= 126)))
])

translation_table = str.maketrans({ c: ' '+c+' ' for c in punct_chars })


def tokenize(text):
    return text.translate(translation_table).split()


def main(argv):
    argparser = argument_parser('serve')
    args = argparser.parse_args(argv[1:])

    ner_model, tokenizer, labels, config = load_ner_model(args.ner_model_dir)
    max_seq_len = config['max_seq_length']

    label_map = { t: i for i, t in enumerate(labels) }
    inv_label_map = { v: k for k, v in label_map.items() }

    for line in sys.stdin:
        line = line.strip()
        words = [tokenize(line)]
        dummy_labels = [['O'] * len(words[0])]

        test_data = process_sentences(words, dummy_labels, tokenizer,
                                      max_seq_len)

        test_x = encode(test_data.combined_tokens, tokenizer, max_seq_len)
        probs = ner_model.predict(test_x, batch_size=args.batch_size)

        preds = np.argmax(probs, axis=-1)
        pred_labels = []
        for i, pred in enumerate(preds):
            pred_labels.append([inv_label_map[t] for t in 
                                pred[1:len(test_data.tokens[i])+1]])

        lines = write_result(
            args.output_file, test_data.words, test_data.lengths,
            test_data.tokens, test_data.labels, pred_labels, mode='predict'
        )
        for line in lines:
            print(line, end='')
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
