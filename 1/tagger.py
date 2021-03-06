"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torch.nn as nn
import torch.optim as optim
from math import log, isfinite
from collections import Counter, defaultdict, OrderedDict
import numpy as np

import sys
import random

from bilstm import BiLSTM, device


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=1512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    #torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i():     # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id', 'email']
        Make sure you return your own info!
    """
    # if work is submitted by a pair of students, add the following keys: name2, id2, email2
    return {'name1': 'Asaf Fried', 'id1': '314078676', 'email1': 'friedas@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None

    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()

    return sentence


def load_annotated_corpus(filename):
    sentences = []

    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)

        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)

    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"
UNIVERSAL_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
                  "SCONJ", "SYM", "VERB", "X"]
PSEUDO_TAGS = {START, END}
EPSILON = 1e-5


class AUX:
    allTagCounts = Counter()

    # use Counters inside these
    perWordTagCounts = {}
    transitionCounts = {}
    emissionCounts = {}

    # log probability distributions: do NOT use Counters inside these because
    # missing Counter entries default to 0, not log(0)
    A = {}  # transitions probabilities
    B = {}  # emissions probabilities
    vocabulary = set()


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and emmisionCounts

    Args:
    tagged_sentences: a list of tagged sentences, each tagged sentence is a
     list of pairs (w,t), as returned by load_annotated_corpus().

    Return:
    [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B] (a list)
    """
    # init allTagCounts
    AUX.allTagCounts = Counter([tag for sentence in tagged_sentences for _, tag in sentence if tag not in PSEUDO_TAGS])

    # init perWordTagCounts and emissionCounts
    AUX.perWordTagCounts = defaultdict(Counter)
    AUX.emissionCounts = defaultdict(Counter)
    for sentence in tagged_sentences:
        for word, tag in sentence:
            AUX.emissionCounts[tag].update([word])

            if tag not in PSEUDO_TAGS:
                AUX.perWordTagCounts[word].update([tag])

    # init transitionCounts
    AUX.transitionCounts = defaultdict(Counter)
    for sentence in tagged_sentences:
        for i in range(len(sentence) - 1):
            _, tag = sentence[i]
            _, next_tag = sentence[i + 1]
            AUX.transitionCounts[tag].update([next_tag])

    # A = C(ti, ti-1) / C(ti-1) -> use allTagCounts and transitionCounts
    AUX.A = defaultdict(dict)
    for t_prev in UNIVERSAL_TAGS + list(PSEUDO_TAGS):
        for t_curr in UNIVERSAL_TAGS + list(PSEUDO_TAGS):
            prob = AUX.transitionCounts[t_prev][t_curr] / AUX.allTagCounts[t_prev] if AUX.allTagCounts[t_prev] > 0 else 0
            log_prob = np.log(prob) if prob > 0 else np.log(EPSILON)

            AUX.A[t_prev][t_curr] = log_prob

    # B = C(ti, wi) / C(ti) -> use allTagCounts and emissionCounts
    AUX.B = defaultdict(dict)
    for tag in UNIVERSAL_TAGS + list(PSEUDO_TAGS):
        for word, count in AUX.emissionCounts[tag].items():
            prob = count / AUX.allTagCounts[tag] if AUX.allTagCounts[tag] > 0 else 0
            log_prob = np.log(prob) if prob > 0 else np.log(EPSILON)

            AUX.B[tag][word] = log_prob

    AUX.vocabulary = set(AUX.perWordTagCounts.keys())

    return [AUX.allTagCounts, AUX.perWordTagCounts, AUX.transitionCounts, AUX.emissionCounts, AUX.A, AUX.B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """
    tagged_sentence = []
    tags = list(allTagCounts.keys())
    values = np.array(list(allTagCounts.values()))
    probs_values_tag = values / sum(values)

    for token in sentence:
        is_OOV = token not in AUX.vocabulary
        if is_OOV:
            tag = np.random.choice(tags, p=probs_values_tag)
        else:
            tag = max(perWordTagCounts[token], key=perWordTagCounts[token].get)

        tagged_sentence.append((token, tag))

    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================

class ViterbiCell:
    def __init__(self, t, r, p):
        self.t = t
        self.r = r
        self.p = p


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emission probabilities.

    Return:
        list: list of pairs
    """

    v_last = viterbi(sentence, A, B)
    tags = retrace(v_last)[1:-1]    # remove start and end tags
    tagged_sentence = list(zip(sentence, tags))

    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tuple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): The HMM emission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtracking.

        """
        # Hint 1: For efficiency reasons - for words seen in training there is no
        #      need to consider all tags in the tagset, but only tags seen with that
        #      word. For OOV you have to consider all tags.
        # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
        #         current list = [ the dummy item ]
        # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END

    viterbi_matrix = [[ViterbiCell(t=START, r=None, p=0)]]    # list of lists

    for i in range(len(sentence)):
        word = sentence[i]
        is_OVV = word not in AUX.vocabulary
        tags = UNIVERSAL_TAGS if is_OVV else list(AUX.perWordTagCounts[word].keys())
        predecessor_list = viterbi_matrix[i]
        viterbi_matrix.append([])

        for tag in tags:
            max_log_prob = -sys.maxsize
            best_previous_cell = None

            for cell in predecessor_list:
                log_prob_prior = cell.p
                log_prob_transition = A[cell.t][tag]
                log_prob_emission = B[tag][word] if word in B[tag] else log(EPSILON)

                likelihood = log_prob_transition + log_prob_emission
                log_prob = likelihood + log_prob_prior

                if log_prob > max_log_prob:
                    assert isfinite(log_prob) and log_prob < 0
                    max_log_prob = log_prob
                    best_previous_cell = cell

            viterbi_matrix[i + 1].append(ViterbiCell(t=tag, r=best_previous_cell, p=max_log_prob))

    max_log_prob = -sys.maxsize
    best_previous_cell = None
    for cell in viterbi_matrix[-1]:
        if cell.p > max_log_prob:
            max_log_prob = cell.p
            best_previous_cell = cell

    v_last = ViterbiCell(t=END, r=best_previous_cell, p=max_log_prob)
    viterbi_matrix.append([v_last])

    return v_last


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    tags = []

    while end_item is not None:
        tags.append(end_item.t)
        end_item = end_item.r

    return tags[::-1]


# a suggestion for a helper function. Not an API requirement
# def predict_next_best(word, tag, predecessor_list):
#     """Returns a new item (tupple)
#     """
#     pass


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): The HMM emission probabilities.
     """
    p = 0   # joint log prob. of words and tags
    prev_tag = None

    for word, tag in sentence:
        log_prob_transition = A[prev_tag][tag] if prev_tag else 0
        log_prob_emission = B[tag][word] if word in B[tag] else log(EPSILON)

        likelihood = log_prob_transition + log_prob_emission
        p += likelihood

        prev_tag = tag

    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p

# ===========================================
#       POS tagging with BiLSTM
# ===========================================


""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""

# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU, LeRU)


def _process_vocab(data_train, max_vocab_size=-1, min_frequency=1, lower=True):
    """process vocabulary learned from data set

    Args:
        data_train (list): list of (w, t) tuples
        max_vocab_size (int): max vocabulary size
        min_frequency (int): the occurrence threshold to consider
        lower (bool): whether to lower case word in vocabulary or not

    Return
        vocab (list)
    """
    counter = Counter([word.lower() for sentence in data_train for word, tag in sentence]) if lower\
        else Counter([word for sentence in data_train for word, tag in sentence])

    counter = Counter(dict([(x, y) for x, y in counter.items() if y >= min_frequency]))

    if max_vocab_size > -1:
        counter = Counter(dict(counter.most_common(max_vocab_size)))

    vocab = list(set(counter.keys()))

    return vocab


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurrence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimension.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """
    data_train = load_annotated_corpus(params_d["data_fn"])
    max_vocab_size = params_d["max_vocab_size"]
    min_frequency = params_d["min_frequency"]
    embedding_dimension = params_d["embedding_dimension"]

    vocab = _process_vocab(data_train, max_vocab_size=max_vocab_size, min_frequency=min_frequency, lower=True)

    embedding = load_pretrained_embeddings(params_d["pretrained_embeddings_fn"], vocab=vocab)

    vectors = torch.FloatTensor(list(embedding.values()))

    word_to_idx = dict(zip(embedding.keys(), range(len(embedding))))

    hidden_dim = params_d["hidden_dim"] if "hidden_dim" in params_d else 10

    return {"lstm": BiLSTM(
        embedding_dimension=embedding_dimension,
        num_of_layers=params_d["num_of_layers"],
        output_dimension=params_d["output_dimension"],
        vectors=vectors,
        word_to_idx=word_to_idx,
        input_rep=params_d["input_rep"],
        hidden_dim=hidden_dim),
        "input_rep": params_d["input_rep"]
    }


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    embedding = OrderedDict()

    vocab_set = set(vocab) if vocab is not None else None

    with open(path, "r") as f:
        for i, line in enumerate(f):
            entry = line.split()
            word = entry[0]
            vec = np.array(entry[1:]).astype(np.float32)

            if vocab_set is not None and word in vocab_set:
                embedding[word] = vec

            if vocab_set is None:
                embedding[word] = vec

    # unknown embedding is mean of all other vectors
    embedding[UNK] = np.array([x for x in embedding.values()]).mean(axis=0)

    return embedding


def _process_sentence_rnn(sentence, vocab, lower=True):
    """process given sentence for rnn training by lower-casing words and replacing OOV words with default token

    Args:
        sentence (list): list of string tokens
        vocab (set): vocabulary

    Return
        processed_sentence (list)
    """
    processed_sentence = []

    for i in range(len(sentence)):
        word = sentence[i].lower() if lower else sentence[i]

        if word not in vocab:
            word = UNK

        processed_sentence.append(word)

    return processed_sentence


def _prepare_sequence(items, to_ix):
    """helper function for training rnn

    Args:
        items (list): list of items
        to_ix (dict): map from item to index

    Return
        indices (tensor)
    """
    idxs = [to_ix[w] for w in items]

    return torch.tensor(idxs, dtype=torch.long)


def _create_case_features(sentence):
    """create case features for BiLSTM case according to https://arxiv.org/pdf/1510.06168.pdf

    Args:
        sentence (list): list of words

    Return
        sentence_features (list)
    """
    sentence_features = []

    for word in sentence:
        if word == word.lower():
            word_features = [1, 0, 0]
        elif word == word.upper():
            word_features = [0, 1, 0]
        elif word[0].isupper():
            word_features = [0, 0, 1]
        else:
            word_features = [0, 0, 0]

        sentence_features.append(word_features)

    return sentence_features


def train_rnn(model, train_data, val_data=None, epochs=25):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
    """
    #Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)

    lstm = model["lstm"]
    input_rep = model["input_rep"]

    criterion = nn.CrossEntropyLoss()   # you can set the parameters as you like
    optimizer = optim.Adam(lstm.parameters(), lr=0.01)

    tag_to_ix = dict(zip(UNIVERSAL_TAGS, range(len(UNIVERSAL_TAGS))))
    vocab = set(lstm.word_to_idx.keys())

    lstm = lstm.to(device)
    criterion = criterion.to(device)

    processed_sentences = []
    tags = []

    for sentence in train_data:
        processed_sentences.append(_process_sentence_rnn([x[0] for x in sentence], vocab))
        tags.append([x[1] for x in sentence])

    losses = []

    for _ in range(epochs):
        epoch_loss = []

        for i in range(len(train_data)):
            processed_sentence, tag = processed_sentences[i], tags[i]

            lstm.init_hidden()
            lstm.zero_grad()

            input = _prepare_sequence(processed_sentence, lstm.word_to_idx).to(device)
            targets = _prepare_sequence(tag, tag_to_ix).to(device)
            if len(targets) < 2:
                continue

            input_case = None
            if input_rep == 1:
                raw_sentence = [x[0] for x in train_data[i]]
                input_case = torch.tensor(_create_case_features(raw_sentence)).to(device)

            probs = lstm(input, input_case)

            loss = criterion(torch.squeeze(probs), targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        losses.append(np.mean(epoch_loss))

    return losses


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.
    Return:
        list: list of pairs
    """

    lstm = model["lstm"]
    input_rep = model["input_rep"]

    vocab = set(lstm.word_to_idx.keys())
    processed_sentence = _process_sentence_rnn(sentence, vocab)
    input = _prepare_sequence(processed_sentence, lstm.word_to_idx).to(device)

    input_case = None
    if input_rep == 1:
        input_case = torch.tensor(_create_case_features(sentence)).to(device)

    with torch.no_grad():
        logits = lstm(input, input_case)

    tags = [UNIVERSAL_TAGS[x] for x in torch.argmax(torch.squeeze(logits), dim=1)] if len(processed_sentence) > 1 \
        else [UNIVERSAL_TAGS[x] for x in torch.argmax(torch.squeeze(logits).view(1, -1), dim=1)]

    return list(zip(sentence, tags))


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
         BiLSTM model.
         IMPORTANT: this is a *hard coded* dictionary that will be used to create
         a model and train a model by calling
                initialize_rnn_model() and train_lstm()
     """
    return {
        "max_vocab_size": -1,
        "min_frequency": 1,
        "embedding_dimension": 100,
        "num_of_layers": 1,
        "output_dimension": len(UNIVERSAL_TAGS),
        "pretrained_embeddings_fn": "glove.6B.100d.txt",
        "data_fn": "en-ud-train.upos.tsv",
        "hidden_dim": 32,
        "input_rep": 1
    }


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    key = list(model.keys())[0]
    values = model[key]

    if key == 'baseline':
        return baseline_tag_sentence(sentence, *values)
    if key == 'hmm':
        return hmm_tag_sentence(sentence, *values)
    if key == 'blstm':
        return rnn_tag_sentence(sentence, *values)
    if key == 'cblstm':
        return rnn_tag_sentence(sentence, *values)


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags, the total number of
    correctly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)
    correct = correctOOV = OOV = 0

    for i in range(len(gold_sentence)):
        word = gold_sentence[i][0]
        gold_tag = gold_sentence[i][1]
        pred_tag = pred_sentence[i][1]
        is_OOV = word not in AUX.perWordTagCounts

        if is_OOV:
            OOV += 1
            if gold_tag == pred_tag:
                correctOOV += 1

        if gold_tag == pred_tag:
            correct += 1

    return correct, correctOOV, OOV
