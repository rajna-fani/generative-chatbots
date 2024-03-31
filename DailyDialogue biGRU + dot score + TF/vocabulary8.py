import itertools
import json
import os, sys
import random
import re
import unicodedata

import torch
import tqdm
import logging

from pip._internal.utils._log import init_logging

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus = "/Users/rajnu/Desktop/BA Abgabe/Generative Chatbots BA/data/ijcnlp_dailydialog/dialogues_text.txt"
corpus_name = "ijcnlp_dailydialog"
datafile = "/Users/rajnu/Desktop/BA Abgabe/Generative Chatbots BA/data/ijcnlp_dailydialog/dialogues_text.txt"


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


'''
This class will keep mapping from words to indexes, 
a reverse mapping of indexes to words, a count of each word 
and a total word count. The class provides methods for
- adding a word to the vocabulary "addWord", 
- adding all words in a sentence "addSentence"
- trimming infrequently seen word "trim"
'''
# Padding short sentences
PAD_token = 0
# Start of sentence token
SOS_token = 1
# End of sentence token
EOS_token = 2


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index),
                                                   len(keep_words) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


# PREPROCESSING AND CLEANING THE DATA

# Maximum length to consider
MAX_LENGTH = 10


# Turn a Unicode string to ASCII

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8'). \
        read().strip().split('__eou__\n')
    # Split every line into pairs and normalize
    test = [l.split("__eou__") for l in lines]
    test = [[list(i) for i in zip(a, a[1:])] for a in test]
    test = [item for sublist in test for item in sublist]
    pairs = [[normalizeString(s) for s in l] for l in test]
    voc = Voc(corpus_name)
    return voc, pairs


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    # pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# PREPARE DATA FOR MODELS


# Converting words to their indexes

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


'''
Padding is needed, since every sentence in the text has not the same number of words, 
we can also define maximum number of words for each sentence, if a sentence is longer then we can drop some words
padding=” post”: add the zeros at the end of the sequence to make the samples in the same size
/// Sentences shorter than max_length are zero padded after an EOS_token
we need to be able to index our batch along time, and across all sequences in the batch. 
Therefore, we transpose our input batch shape to (max_length, batch_size), 
so that indexing across the first dimension returns a time step across all sentences in the batch in the zeroPadding function
'''


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
'''The inputVar function handles the process of converting sentences to tensor, 
ultimately creating a correctly shaped zero-padded tensor. 
It also returns a tensor of lengths for each of the sequences in the batch which 
will be passed to our decoder later.
'''


def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
'''The outputVar function performs a similar function to inputVar, but instead of 
returning a lengths tensor, it returns a binary mask tensor and a maximum target sentence length. 
The binary mask tensor has the same shape as the output target tensor, but every element that is a PAD_token is 0 
and all others are 1.
'''


def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs (batch: sample words)
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)
