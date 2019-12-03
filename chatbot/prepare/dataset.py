"""
Prepares the dataset from which you will train your model on.
"""
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from tf.keras import preprocessing


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # create a space between a word and the punctuation
    # following it
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence = re.sub(r"[" "]+", " ", sentence)
    # replace everything with spaces except (a-Z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # add a start and end token to the sentence
    sentence = "<START> " + sentence + " <END>"

    return sentence


def load_conversations(hyper_params, lines_filename, conversations_filename):
    # dictioinary of line id to text
    id2line = {}
    with open(lines_filename, errors='ignore') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    questions, answers = [], []

    with open(conversations_filename, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '') as file:
        # retrieve conversation in a list of line IDs
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            questions.append(preprocess_sentence(id2line[conversation[i]]))
            answers.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(questions) >= hyper_params.max_samples:
                return questions, answers
    return questions, answers


def tokenize_and_filter(hyper_params, tokenizer, questions, answers):
    tokenized_questions, tokenized_answers = [], []

    for (question, answer) in zip(questions, answers):
        # tokenize sentence
        sentence1 = hyper_params.start_token + tokenizer.encode(
            question) + hyper_params.end_token
        sentence2 = hyper_params.start_token + tokenizer.encode(
            answer) + hyper_params.end_token

        # check tokenize sentence length
        if len(sentence1) <= hyper_params.max_length and
         len(sentence2) <= hyper_params.max_length:
             tokenized_questions.append(sentence1)
             tokenized_questions.append(sentence2)

    # pad tokenized sentences
    tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_questions, maxlen=hyper_params.max_length, padding='post')
    tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=hyper_params.max_length, padding='post')

    return tokenized_questions, tokenized_answers


def get_dataset(hyper_params):
    # download corpus
    path_to_zip = tf.keras.utils.get_file(
        'cornell_movie_dialogs.zip',
        origin=
        'http://www.cs.cornell.edu/~cristian/data/cornell/cornell_movie_dialogs_corpus.zip',
        extract=True)

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

    # get movie_lines.txt and movie_conversations.txt
    lines_filename = os.path.join(path_to_dataset, 'movie_liens.txt')
    conversations_filename = os.path.join(path_to_dataset, 'movie_conversations.txt')

    questions, answers = load_conversations(hyper_params, lines_filename,
         conversations_filename)

    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**14)

    hyper_params.start_token = [tokenizer.vocab_size]
    hyper_params.end_token = [tokenizer.vocab_size + 1]
    hyper_params.vocab_size = tokenizer.vocab_size + 2

    questions, answers = tokenize_and_filter(hyper_params=, tokenizer, questions,
                                                                        answers)

    dataset = tf.data.Dataset.from_tensor_slices(({
        'inputs' : questions,
        'dec_inputs' : answers[:, :-1]
        }, answers[:, 1:]))
    dataset = dataset.cache()
    dataset = dataset.shuffle(len(questions))
    dataset = dataset.batch(hyper_params.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
