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
