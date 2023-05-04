'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not
use other non-standard modules (including nltk). Some modules that might be helpful are
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here


def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    output = []

    count_word_tag = defaultdict(lambda: defaultdict(lambda: 0))
    count_tag = defaultdict(lambda: 0)
    for sentence in train:
        for pair in sentence:
            word, tag = pair
            count_word_tag[word][tag] += 1
            count_tag[tag] += 1

    unseen_tag = max(count_tag, key=count_tag.get)
    word_tag = {word: max(
        count_word_tag[word], key=count_word_tag[word].get) for word in count_word_tag}

    for sentence in test:
        output_sentence = []
        for word in sentence:
            if word in word_tag:
                output_sentence.append((word, word_tag[word]))
            else:
                output_sentence.append((word, unseen_tag))
        output.append(output_sentence)

    return output


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    output = []

    valid_tags = set()
    trans_data = defaultdict(lambda: defaultdict(lambda: 0))
    emit_data = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in train:
        next = 1
        for pair in sentence:
            word, tag = pair
            valid_tags.add(tag)
            emit_data[tag][word] += 1
            if next != len(sentence):
                next_word = sentence[next]
                trans_data[tag][next_word[1]] += 1
            next += 1

    laplace = 1e-3

    trans_probability = defaultdict()

    for tag in trans_data:
        denominator = sum(trans_data[tag].values()) + \
            laplace * (len(trans_data[tag]) + 1)
        trans_probability[tag] = defaultdict(lambda: laplace / denominator)
        for nextTag in trans_data[tag]:
            trans_probability[tag][nextTag] = (
                trans_data[tag][nextTag] + laplace) / denominator

    emit_probability = defaultdict()

    for tag in emit_data:
        denominator = sum(emit_data[tag].values()) + \
            laplace * (len(emit_data[tag]) + 1)
        emit_probability[tag] = defaultdict(lambda: laplace / denominator)
        for nextTag in emit_data[tag]:
            emit_probability[tag][nextTag] = (
                emit_data[tag][nextTag] + laplace) / denominator
        emit_probability[tag]['OOV'] = laplace / denominator

    iProb = (laplace + len(test)) / (len(test) + laplace * (len(test) + 1))
    NotStartProb = 1 - iProb

    # Viterbi algorithm from https: // en.wikipedia.org/wiki/Viterbi_algorithm
    for sentence in test:
        obs = sentence
        V = [{}]
        valid_tags = list(valid_tags)

        for st in valid_tags:
            if obs[0] in emit_probability[st]:
                em_prob = emit_probability[st][obs[0]]
            else:
                em_prob = emit_probability[st]['OOV']
            if st == 'START':
                V[0][st] = {"prob": log(
                    iProb) + log(em_prob), "prev": None}
            else:
                V[0][st] = {"prob": log(
                    NotStartProb) + log(em_prob), "prev": None}

        for t in range(1, len(obs)):
            V.append({})
            for st in valid_tags:
                if obs[t] in emit_probability[st]:
                    em_prob = emit_probability[st][obs[t]]
                else:
                    em_prob = emit_probability[st]['OOV']

                max_tr_prob = V[t - 1][valid_tags[0]]["prob"] + log(
                    trans_probability[valid_tags[0]][st]) + log(em_prob)
                prev_st_selected = valid_tags[0]

                for prev_st in valid_tags[1:]:
                    if prev_st == 'END':
                        continue

                    try_prob = V[t - 1][prev_st]["prob"] + \
                        log(trans_probability[prev_st][st]) + \
                        log(em_prob)
                    if try_prob > max_tr_prob:
                        max_tr_prob = try_prob
                        prev_st_selected = prev_st

                max_prob = max_tr_prob
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
        opt = []
        max_prob = float("-inf")
        best_st = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]
        output.append([])
        for word, tag in zip(obs, opt):
            output[-1].append((word, tag))

    return output


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")
