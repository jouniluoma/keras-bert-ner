import os
import numpy as np
import conlleval

from collections import deque
from argparse import ArgumentParser

os.environ['TF_KERAS'] = '1'

from tensorflow import keras
from bert import tokenization
from keras_bert import load_trained_model_from_checkpoint, AdamWarmup
from keras_bert import calc_train_steps

from config import DEFAULT_SEQ_LEN, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from config import DEFAULT_LR, DEFAULT_WARMUP_PROPORTION


def argument_parser():
    argparser = ArgumentParser()
    argparser.add_argument(
        '--train_data', required=True,
        help='Training data'
    )
    argparser.add_argument(
        '--dev_data', default=None,
        help='Training data'
    )
    argparser.add_argument(
        '--test_data', required=True,
        help='Test data'
    )
    argparser.add_argument(
        '--vocab_file', required=True,
        help='Vocabulary file that BERT model was trained on'
    )
    argparser.add_argument(
        '--bert_config_file', required=True,
        help='Configuration for pre-trained BERT model'
    )
    argparser.add_argument(
        '--init_checkpoint', required=True,
        help='Initial checkpoint for pre-trained BERT model'
    )
    argparser.add_argument(
        '--max_seq_length', type=int, default=DEFAULT_SEQ_LEN,
        help='Maximum input sequence length in WordPieces'
    )
    argparser.add_argument(
        '--do_lower_case', default=False, action='store_true',
        help='Lower case input text (for uncased models)'
    )
    argparser.add_argument(
        '--learning_rate', type=float, default=DEFAULT_LR,
        help='Initial learning rate'
    )
    argparser.add_argument(
        '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    argparser.add_argument(
        '--num_train_epochs', type=int, default=DEFAULT_EPOCHS,
        help='Number of training epochs'
    )
    argparser.add_argument(
        '--warmup_proportion', type=float, default=DEFAULT_WARMUP_PROPORTION,
        help='Proportion of training to perform LR warmup for'
    )
    argparser.add_argument(
        '--repeats', type=int, default=5,
        help='How many times the test is repeated for mean of F1 score'
    )
    argparser.add_argument(
        '--result_dir', default="./output/",
        help='Default directory to write the results'
    )
    argparser.add_argument(
        '--result_file', default="results.txt",
        help='Default file to write the results'
    )
    argparser.add_argument(
        '--result_prefix', default="result",
        help='Default file to write the results'
    )
    return argparser


def load_pretrained(options):
    model = load_trained_model_from_checkpoint(
        options.bert_config_file,
        options.init_checkpoint,
        training=False,
        trainable=True,
        seq_len=options.max_seq_length,
    )
    tokenizer = tokenization.FullTokenizer(
        vocab_file=options.vocab_file,
        do_lower_case=options.do_lower_case
    )
    return model, tokenizer


def create_ner_model(pretrained_model, num_labels):
    ner_inputs = pretrained_model.inputs[:2]
    ner_output = keras.layers.Dense(
        num_labels,
        activation='softmax'
    )(pretrained_model.output)
    ner_model = keras.models.Model(inputs=ner_inputs, outputs=ner_output)
    return ner_model


def create_optimizer(num_example, options):
    total_steps, warmup_steps = calc_train_steps(
        num_example=num_example,
        batch_size=options.batch_size,
        epochs=options.num_train_epochs,
        warmup_proportion=options.warmup_proportion,
    )
    optimizer = AdamWarmup(
        total_steps,
        warmup_steps,
        lr=options.learning_rate,
        epsilon=1e-6,
        weight_decay=0.01,
        weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo']
    )
    return optimizer



def encode(lines, tokenizer, max_len):
    tids = []
    sids = []
    for line in lines:
        tokens = ["[CLS]"]+line
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        if len(token_ids) < max_len:
            pad_len = max_len - len(token_ids)
            token_ids += tokenizer.convert_tokens_to_ids(["[PAD]"]) * pad_len
            segment_ids += [0] * pad_len
        tids.append(token_ids)
        sids.append(segment_ids)
    return [np.array(tids), np.array(sids)]


def label_encode(labels, tag_dict, max_len):
    encoded = []
    sample_weights = []
    for sentence in labels:
        enc = [tag_dict[i] for i in sentence]
        enc.insert(0, tag_dict['O'])
        weight = [0 if i=='[SEP]' else 1 for i in sentence]
        weight.insert(0,0)
        if len(enc) < max_len:
            weight.extend([0]*(max_len-len(enc)))
            enc.extend([tag_dict['[PAD]']]*(max_len-len(enc)))
        encoded.append(np.array(enc))
        sample_weights.append(np.array(weight))
    lab_enc = np.expand_dims(np.stack(encoded, axis=0), axis=-1)
    weights = np.stack(sample_weights, axis=0)
    return lab_enc, weights


def read_tags(path):
    f = open(path, 'r')
    tags = set(l.split()[1] for l in f if l.strip() != '')
    return {tag: index for index, tag in enumerate(tags)}


def tokenize_and_split(words, word_labels, tokenizer, max_length):
    # Tokenize each word in sentence, propagate labels
    tokens, labels, lengths = [], [], []
    for word, label in zip(words, word_labels):
        tokenized = tokenizer.tokenize(word)
        tokens.extend(tokenized)
        lengths.append(len(tokenized))
        for i, token in enumerate(tokenized):
            if i == 0:
                labels.append(label)
            else:
                if label.startswith('B'):
                    labels.append('I'+label[1:])
                else:
                    labels.append(label)

    # Split into multiple sentences if too long
    split_tokens, split_labels = [], []
    start, end = 0, max_length
    while end < len(tokens):
        # Avoid splitting inside tokenized word
        while end > start and tokens[end].startswith('##'):
            end -= 1
        if end == start:
            end = start + max_length    # only continuations
        split_tokens.append(tokens[start:end])
        split_labels.append(labels[start:end])
        start = end
        end += max_length
    split_tokens.append(tokens[start:])
    split_labels.append(labels[start:])

    return split_tokens, split_labels, lengths


def read_sentences(input_file):
    sentences, words = [], []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(line.split('\t')[0])
            elif words:
                sentences.append(words)
                words = []
    if words:
        sentences.append(words)
    return sentences
    

def read_data(input_file, tokenizer, max_seq_length):
    lines, tags, lengths = [], [], []

    def add_sentence(words, labels):
        split_tokens, split_labels, lens = tokenize_and_split(
            words, labels, tokenizer, max_seq_length-1)
        lines.extend(split_tokens)
        tags.extend(split_labels)
        lengths.extend(lens)

    curr_words, curr_labels = [], []
    with open(input_file) as rf:
        for line in rf:
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
                # empty lines separate sentences
                add_sentence(curr_words, curr_labels)
                curr_words, curr_labels = [], []

        # Process last sentence also when there's no empty line after
        if curr_words:
            add_sentence(curr_words, curr_labels)
    return lines, tags, lengths


def write_result(fname, original, token_lengths, tokens, labels, predictions):
    lines=[]
    with open(fname,'w+') as f:
        toks = deque([val for sublist in tokens for val in sublist])
        labs = deque([val for sublist in labels for val in sublist])
        pred = deque([val for sublist in predictions for val in sublist])
        lengths = deque(token_lengths)
        print(len(toks),len(labs),len(pred))
        for sentence in original:
            for word in sentence:
                label = labs.popleft()
                predicted = pred.popleft()
                for i in range(int(lengths.popleft())-1):
                    labs.popleft()
                    pred.popleft()                           
                line = "{}\t{}\t{}\n".format(word,label,predicted)
                f.write(line)
                lines.append(line)
            f.write("\n")
    f.close()
    return lines


# Include maximum number of consecutive sentences to each sample
def combine_lines(lines, tags, lengths, max_seq):
    lines_in_sample = []
    new_lines = []
    new_tags = []
    
    for i, line in enumerate(lines):
        line_numbers = [i]
        new_line = []
        new_line.extend(line)
        new_tag = []
        new_tag.extend(tags[i])
        j = 1
        linelen = len(lines[(i+j)%len(lines)])
        while (len(new_line) + linelen) < max_seq-2:
            new_line.append('[SEP]')
            new_tag.append('[SEP]')
            new_line.extend(lines[(i+j)%len(lines)])
            new_tag.extend(tags[(i+j)%len(tags)])
            line_numbers.append((i+j)%len(lines))
            j += 1
            linelen = len(lines[(i+j)%len(lines)])
        new_lines.append(new_line)
        new_tags.append(new_tag)
        lines_in_sample.append(line_numbers)
    return new_lines, new_tags, lines_in_sample
