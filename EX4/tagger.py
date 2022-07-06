"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torch.nn as nn
import torchtext
from torchtext import data
from math import log, isfinite
from collections import Counter
import numpy as np
import random
import copy

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed = 1512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    #torch.backends.cudnn.deterministic = True
# utility functions to read the corpus
def who_am_i(): #this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Dor Zazon', 'id': '312237803', 'email': 'zazond@post.bgu.ac.il'}
def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
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

START = "<s>"
END = "</s>"
UNK = "<unk>"
epsilon = 1e-6  # can change to 1 for add 1 laplace smoothing

# train_path = r'C:\Users\dell\PycharmProjects\NLP\EX4\en-ud-train.upos.tsv'
# dev_path = r'C:\Users\dell\PycharmProjects\NLP\EX4\en-ud-dev.upos.tsv'
# train_annotated_corpus = load_annotated_corpus(train_path)
# new_sentences = train_annotated_corpus[-100]
# train_annotated_corpus = train_annotated_corpus[:-100]
# dev_annotated_corpus = load_annotated_corpus(dev_path)

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {} #transisions probabilities
B = {} #emmissions probabilities
voc = []


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
    tagged_sentences: a list of tagged sentences, each tagged sentence is a
     list of pairs (w,t), as retunred by load_annotated_corpus().

    Return:
    [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)"""
    global allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B, voc
    voc = list(set([w[0].lower() for s in tagged_sentences for w in s]))
    allTagCounts = dict(Counter([w[1] for s in tagged_sentences for w in s]))
    # use Counters inside these
    perWordTagCounts = dict(Counter([(w[0].lower(), w[1]) for s in tagged_sentences for w in s]))
    tagged_sentences_temp = copy.deepcopy(tagged_sentences)
    for s in tagged_sentences_temp:
        s.insert(0, (START, 'START'))
        s.insert(-1, (END, 'END'))
    allTagCountsDum = dict(Counter([w[1] for s in tagged_sentences_temp for w in s]))
    # use Counters inside these
    transitionCounts = dict(Counter([s[i][1] + " " + s[i+1][1] for s in tagged_sentences_temp for i in range(len(s) - 1)]))
    emissionCounts = dict(Counter([(w[0].lower(), w[1]) for s in tagged_sentences_temp for w in s]))
    # log probability distributions: do NOT use Counters inside these because
    # missing Counter entries default to 0, not log(0)
    tag_num = len(allTagCounts)

    # transisions probabilities
    A = copy.deepcopy(transitionCounts)
    allTagsTuples = [t1 + ' ' + t2 for t1 in list(allTagCounts.keys())+['START'] for t2 in list(allTagCounts.keys())+['END']]
    allWordTagsTuples = [(w, t) for w in voc + [UNK] for t in allTagCounts.keys()]
    for t in allTagsTuples: A[t] = np.log((A.get(t, 0) + epsilon) / (allTagCountsDum.get(t.split()[0]) + tag_num*epsilon))
    # emmissions probabilities
    B = copy.deepcopy(emissionCounts)
    for e in allWordTagsTuples: B[e] = np.log((B.get(e, 0) + epsilon) / (allTagCountsDum.get(e[1]) + tag_num*epsilon))
    B[(START, 'START')] = 0; B[(END, 'END')] = 0
    return [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts, A, B]
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
    for word in sentence:
        if word.lower() not in voc:# [key[0] for key, value in perWordTagCounts.items()]: # OOV
            tag = random.choices(list(allTagCounts.keys()), weights=allTagCounts.values(), k=1)
            tagged_sentence.append((word, tag[0]))
        else:  # not OOV
            TagCounts = dict((key[1], value) for key, value in perWordTagCounts.items() if word.lower() == key[0])
            tagged_sentence.append((word, max(TagCounts, key=TagCounts.get)))
    return tagged_sentence
#===========================================
#       POS tagging with HMM
#===========================================
def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emmission probabilities.

    Return:
        list: list of pairs
    """
    # set unknown words with the UNK tag
    new_sent = []
    for token in sentence:
        if token.lower() not in voc:
            new_sent.append(UNK)
        else:
            new_sent.append(token.lower())
    # i added only end token not start
    new_sent.append(END)
    v_last = viterbi(new_sent, A, B)
    tags_list = retrace(v_last)
    tagged_sentence = [(w, t) for w, t in zip(sentence, tags_list)]
    return tagged_sentence
def viterbi(sentence, A ,B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tuple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtracking.

        """
        # Hint 1: For efficiency reasons - for words seen in training there is no
        #      need to consider all tags in the tagset, but only tags seen with that
        #      word. For OOV you have to consider all tags.
        # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
        #         current list = [ the dummy item ]
        # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END
    viterbi_matrix = [[('START', None, 0)]]
    for word in sentence:
        col = []
        if word == UNK:
            for tag in allTagCounts:
                PossibleTransitions = [(tag, PrevTag, PrevTag[2] + A[PrevTag[0] + ' ' + tag] + B[(word, tag)])
                                       for PrevTag in viterbi_matrix[-1]]
                col.append(max(PossibleTransitions, key=lambda x: x[2]))
        else:
            # find all tags that appear with the word
            PossibleTags = [key[1] for key, value in B.items() if word.lower() == key[0]]
            for tag in PossibleTags:
                # find all possible items, have to find the item with the max value
                PossibleTransitions = [(tag, PrevTag, PrevTag[2] + A[PrevTag[0] + ' ' + tag] + B[(word, tag)])
                                         for PrevTag in viterbi_matrix[-1]]
                # add the best item to col list
                col.append(max(PossibleTransitions, key=lambda x: x[2]))
        viterbi_matrix.append(col)
    v_last = max(viterbi_matrix[-1], key=lambda x: x[2])
    return v_last
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    prev_item = end_item
    tag_list = [end_item[0]]
    while tag_list[0] != 'START':
        prev_item = prev_item[1]
        tag_list.insert(0, prev_item[0])
    return tag_list[1:]

def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): tthe HMM emmission probabilities.
     """
    # new_sent = []
    # for token, tag in sentence:
    #     if token not in voc:
    #         new_sent.append((UNK, tag))
    #     else:
    #         new_sent.append((token, tag))
    # i added only end token not start
    sentenceN = copy.deepcopy(sentence)
    sentenceN.append((END, 'END'))
    p = 0   # joint log prob. of words and tags
    prev_tag = 'START'
    for word, tag in sentenceN:
        p += A['{} {}'.format(prev_tag, tag)] + B[(word, tag)] if word in voc else log(epsilon)
    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p



# learn_params(train_annotated_corpus)
# acc_base = []
# acc_hmm = []
# for s in dev_annotated_corpus[:10]:
#     sentence = [x[0].lower() for x in s]
#     tags = [x[1] for x in s]
#     baseline = baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts)
#     tagged_sent = hmm_tag_sentence(sentence, A, B)
#     x = count_correct(s, baseline)
#     acc_base.append(x[0]/len(s))
#     x = count_correct(s, tagged_sent)
#     acc_hmm.append(x[0] / len(s))
# print('HMM ACC: {}\t Base ACC: {}'.format(np.mean(acc_hmm), np.mean(acc_base)))
# x=1
# print('baseline summary: correct - {}\t correctOOV - {}\t OOV - {}'.format(x[0], x[1], x[2]))
#
# print('HMM tag:', tagged_sent)
# print('HMM prob: ', joint_prob(tagged_sent, A, B))
# x = count_correct(tagged_sent, tagged_sent)
# print('HMM summary: correct - {}\t correctOOV - {}\t OOV - {}'.format(x[0], x[1], x[2]))


#===========================================
#       POS tagging with BiLSTM
#===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanila biLSTM in which the input layer is based on simple word embeddings
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
#  6. Consider using different unit types (LSTM, GRU,LeRU)

class BiLSTM_Tagger(nn.Module):
    def __init__(self, embedding_dimension, vectors, num_of_layers=2, hidden_size=64, output_dimension=17):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_of_layers = num_of_layers
        self.hidden_size = hidden_size
        self.output_dimension = output_dimension
        self.embedding = nn.Embedding.from_pretrained(vectors)
        self.lstm = nn.LSTM(input_size=embedding_dimension, hidden_size=hidden_size, num_layers=num_of_layers,
                            batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_size*2, output_dimension + 1)
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        tag_pred = self.hidden2tag(lstm_out)
        return torch.transpose(tag_pred, 2, 1)
    def set_embeddings(self, embedding_weights):
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        # optional
        # self.embedding.weight.requires_grad = False
    def add_stoi(self, stoi):
        self.stoi = stoi
    def add_tag_itos(self, tag_itos):
        self.tag_itos = tag_itos
class CBiLSTM_Tagger(nn.Module):
    def __init__(self, embedding_dimension, vectors, num_of_layers=1, hidden_size=128, output_dimension=17):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_of_layers = num_of_layers
        self.hidden_size = hidden_size
        self.output_dimension = output_dimension
        self.embedding = nn.Embedding.from_pretrained(vectors)
        self.lstm = nn.LSTM(input_size=embedding_dimension + 3, hidden_size=hidden_size, num_layers=num_of_layers,
                            batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_size*2, output_dimension + 1)
    def forward(self, text, binary):
        embedded = self.embedding(text)
        embedded = torch.cat([embedded, binary], dim=-1)
        lstm_out, _ = self.lstm(embedded)
        tag_pred = self.hidden2tag(lstm_out)
        return torch.transpose(tag_pred, 2, 1)
    def set_embeddings(self, embedding_weights):
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
    def add_stoi(self, stoi):
        self.stoi = stoi
    def add_tag_itos(self, tag_itos):
        self.tag_itos = tag_itos


#================== API ====================================
# i assume valid input - so the parameters should be valid with the model you want to initialize
def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
           the lstm model. The LSTM is initialized based on the specified parameters.
           thr returned dict is may have other or additional fields.

        Args:
            params_d (dict): a dictionary of parameters specifying the model. The dict
                            should include (at least) the following keys:
                            {'max_vocab_size': max vocabulary size (int),
                            'min_frequency': the occurence threshold to consider (int),
                            'input_rep': 0 for the vanilla and 1 for the case-base (int),
                            'embedding_dimension': embedding vectors size (int),
                            'num_of_layers': number of layers (int),
                            'output_dimension': number of tags in tagset (int),
                            'pretrained_embeddings_fn': str,
                            'data_fn': str
                            }
                            max_vocab_size sets a constraints on the vocab dimention.
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
    train_annotated_corpus = load_annotated_corpus(params_d['data_fn'])
    # preprocess train data
    train_sentences, train_tags = preprocess(train_annotated_corpus)
    # build vocabulary
    Sentences = data.Field(batch_first=True, unk_token=UNK)# , pad_first=True
    Sentences.build_vocab(train_sentences, min_freq=params_d['min_frequency'] if params_d['min_frequency'] != -1 else 1,
                          max_size=params_d['max_vocab_size'] if params_d['max_vocab_size'] != -1 else None)
    # load pretrained embeddings
    vocabulary = Sentences.vocab.itos
    vectors = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'], vocabulary)
    if params_d['input_rep'] == 0:
        model = BiLSTM_Tagger(embedding_dimension=params_d['embedding_dimension'], output_dimension=params_d['output_dimension'],
                              vectors=vectors, num_of_layers=params_d['num_of_layers'])
    else:
        model = CBiLSTM_Tagger(embedding_dimension=params_d['embedding_dimension'], output_dimension=params_d['output_dimension'],
                               vectors=vectors, num_of_layers=params_d['num_of_layers'])
    model.add_stoi(Sentences.vocab.stoi)
    model.min_frequency = params_d['min_frequency']; model.max_vocab_size = params_d['max_vocab_size']
    return {'lstm': model, 'input_rep': params_d['input_rep'], 'stoi': Sentences.vocab.stoi, 'vectors': vectors,
            'field': Sentences, 'sent_vocab': Sentences.vocab}
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

    vectors = torchtext.vocab.Vectors(path)
    if vocab != None:
        return vectors.get_vecs_by_tokens(vocab)
    else:
        return vectors.vectors
def train_rnn(model, train_data, val_data = None):
    """Trains the BiLSTM model on the specified data.

       Args:
           model (dict): the model dict as returned by initialize_rnn_model()
           train_data (list): a list of annotated sentences in the format returned
                               by load_annotated_corpus()
           val_data (list): a list of annotated sentences in the format returned
                               by load_annotated_corpus() to be used for validation.
                               Defaults to None
           input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                            1: case-base; <other int>: other models, if you are playful
       """
    global voc
    voc = list(set([w[0].lower() for s in train_data for w in s]))
    if model['input_rep'] == 0:
        train_sentences, train_tags = preprocess(train_data)
        if val_data == None:
            train_dataset = build_dataset(train_sentences, train_tags, model)
        else:
            dev_sentences, dev_tags = preprocess(val_data)
            train_dataset, dev_dataset = build_dataset(train_sentences, train_tags, model, dev_sentences, dev_tags)
            dev_loader = data.BucketIterator(dev_dataset, batch_size=64, sort_key=lambda x: len(x), shuffle=False)

    else:  # input rep = 1
        train_sentences, train_cases, train_tags = preprocess_case(train_data)
        if val_data == None:
            train_dataset = build_case_dataset(train_sentences, train_cases, train_tags, model)
        else:
            dev_sentences, dev_cases, dev_tags = preprocess_case(val_data)
            train_dataset, dev_dataset = build_case_dataset(train_sentences, train_cases, train_tags, model, dev_sentences,
                                                       dev_cases, dev_tags)
            dev_loader = data.BucketIterator(dev_dataset, batch_size=64, sort_key=lambda x: len(x), shuffle=False)
    lstm = model['lstm']
    train_loader = data.BucketIterator(train_dataset, batch_size=64, sort_key=lambda x: len(x), shuffle=True,
                                       train=True)
    lstm.add_tag_itos(train_dataset.fields['Tags'].vocab.itos)
    lstm.add_stoi(train_dataset.fields['Sentences'].vocab.stoi)
    TAG_PAD_IDX = train_dataset.fields['Tags'].vocab.stoi['<pad>']
    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-2)
    print('Training model for default 5 epochs!')
    for epoch in range(5):
        if model['input_rep'] == 0:
            train_epoch(train_loader, lstm, optimizer, epoch + 1, TAG_PAD_IDX)
        else:
            train_case_epoch(train_loader, lstm, optimizer, epoch + 1, TAG_PAD_IDX)
    print('Training completed - model ready!')
    if val_data != None:
        print('Evaluating model ! ')
        if model['input_rep'] == 0:
            _ = evaluate_model(dev_loader, lstm, optimizer, TAG_PAD_IDX)
        else:
            _ = evaluate_case_model(dev_loader, lstm, optimizer, TAG_PAD_IDX)
#=================================================================================
# The users should add embeddings and data path to the returned dictionary!!!!!!
#=================================================================================
def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    return {'max_vocab_size': -1, 'min_frequency': 3, 'input_rep': 1, 'embedding_dimension': 100, 'num_of_layers': 2,
            'output_dimension': 17}
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
    sentence = [w.lower() for w in sentence]
    dataset = build_test_dataset(sentence, model['stoi'], model['input_rep'])
    loader = data.BucketIterator(dataset, batch_size=1)
    for s in loader:
        if model['input_rep'] == 0:
            predictions = model['lstm'](s.Sentences)
        else:
            predictions = model['lstm'](s.Sentences, torch.tensor(s.Cases))
    labels_idx = np.argmax(predictions.detach().numpy(), axis=1)
    tagged_sentence = build_tags(labels_idx[0], sentence, model['lstm'].tag_itos)
    return tagged_sentence
#==========================================================


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================
# check this one works
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
        return rnn_tag_sentence(sentence, values[0]['lstm'])
    if key == 'cblstm':
        return rnn_tag_sentence(sentence, values[0]['lstm'], input_rep=1)
def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    OOV = sum([w[0].lower() not in voc for w in gold_sentence])
    correct = sum([g[1] == p[1] for g, p in zip(gold_sentence, pred_sentence)])
    correctOOV = sum([g[1] == p[1] for g, p in zip(gold_sentence, pred_sentence) if g[0].lower() not in voc])
    # print('ACCURACY: ', correct/len(gold_sentence))
    assert len(gold_sentence) == len(pred_sentence)
    return correct, correctOOV, OOV

#================ Helpres ===========================
#===================================================
def numericalize(sentences, stoi):
    num_sent = []
    if type(sentences[0]) == list:  # multi sentences
        for s in sentences:
            temp = []
            for w in s:
                temp.append(stoi.get(w, 0))  # if the word is not in the vocabulary add UNK token
            num_sent.append(temp)
    else:  # have 1 sentence - test phase
        for w in sentences:
            num_sent.append(stoi.get(w, 0))  # if the word is not in the vocabulary add UNK token
    return num_sent
def build_dataset(train_sentences, train_tags, model, dev_sentences=None, dev_tags=None):
    # Sentences.build_vocab(train_sentences, vectors=vectors, min_freq=3)
    Sentences = data.Field(batch_first=True, unk_token=UNK)#, pad_first=True)
    Tags = data.Field(batch_first=True, unk_token=None, is_target=True)#, pad_first=True)
    fields = [('Sentences', Sentences), ('Tags', Tags)]
    Sentences.vocab = model['sent_vocab']
    train_examples = []
    for text, label in zip(train_sentences, train_tags):
        train_examples.append(data.Example.fromlist([text, label], fields))
    train_dataset = data.Dataset(train_examples, fields=fields)

    Tags.build_vocab(train_dataset)
    if dev_sentences != None:
        dev_examples = []
        for text, label in zip(dev_sentences, dev_tags):
            dev_examples.append(data.Example.fromlist([text, label], fields))
        dev_dataset = data.Dataset(dev_examples, fields=fields)
        return train_dataset, dev_dataset
    else:
        return train_dataset
def build_case_dataset(train_sentences, train_cases, train_tags, model, dev_sentences=None, dev_cases=None, dev_tags=None):
    Sentences = data.Field(batch_first=True, unk_token=UNK) #, pad_first=True
    Cases = data.RawField()
    Tags = data.Field(batch_first=True, unk_token=None, is_target=True)#, pad_first=True)
    fields = [('Sentences', Sentences), ('Cases', Cases), ('Tags', Tags)]
    Sentences.vocab = model['sent_vocab']
    # train_sentences = numericalize(train_sentences, model['stoi'])
    train_examples = []
    for text, cases, label in zip(train_sentences, train_cases, train_tags):
        train_examples.append(data.Example.fromlist([text, cases, label], fields))
    train_dataset = data.Dataset(train_examples, fields=fields)
    Tags.build_vocab(train_dataset)
    if dev_sentences != None:
        # dev_sentences = numericalize(dev_sentences, model['stoi'])
        dev_examples = []
        for text, cases, label in zip(dev_sentences, dev_cases, dev_tags):
            dev_examples.append(data.Example.fromlist([text, cases, label], fields))
        dev_dataset = data.Dataset(dev_examples, fields=fields)
        return train_dataset, dev_dataset
    else:
        return train_dataset
def build_test_dataset(test_sentence, stoi, input_rep):
    Sentences = data.Field(use_vocab=False, batch_first=True, unk_token=UNK)# , pad_first=True
    test_example = []
    if input_rep == 1:  # caseLSTM
        Cases = data.RawField()
        fields = [('Sentences', Sentences), ('Cases', Cases)]
        words_cases = []
        for w in test_sentence:
            words_cases.append(case(w.lower()))
        test_sentence = numericalize(test_sentence, stoi)
        test_example.append(data.Example.fromlist([test_sentence, words_cases], fields))

    else:  # vanilla
        test_sentence = numericalize(test_sentence, stoi)
        fields = [('Sentences', Sentences)]
        test_example.append(data.Example.fromlist([test_sentence], fields))
    test_dataset = data.Dataset(test_example, fields=fields)
    return test_dataset
def unk_dev(dev_sentences, s_stoi):
    unk_dev_sentences = []
    for s in dev_sentences:
        temp_s = []
        for w in s:
            if s_stoi.get(w, 0) == 0:
                temp_s.append(UNK)
            else:
                temp_s.append(w)
        unk_dev_sentences.append(temp_s)
    return unk_dev_sentences
def unk_test(sentence, s_stoi):
    unk_dev_sentences = []
    for w in sentence:
        if s_stoi.get(w, 0) == 0:
            unk_dev_sentences.append(UNK)
        else:
            unk_dev_sentences.append(w)
    return unk_dev_sentences
def preprocess(text):
    sentences = []
    tags = []
    for s in text:
        temp_s = []
        temp_an = []
        for w in s:
            temp_s.append(w[0].lower())
            temp_an.append(w[1])
        sentences.append(temp_s)
        tags.append(temp_an)
    return sentences, tags
def case(word):
    if word.isupper(): return [1, 0, 0]
    elif word[0].isupper(): return [0, 1, 0]
    else: return [0, 0, 1]
def preprocess_case(text):
    sentences = []
    word_case = []
    tags = []
    for s in text:
        temp_s = []
        temp_c = []
        temp_an = []
        for w in s:
            temp_s.append(w[0].lower())
            temp_an.append(w[1])
            temp_c.append(case(w[0]))
        sentences.append(temp_s)
        word_case.append(temp_c)
        tags.append(temp_an)
    return sentences, word_case, tags
def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    return np.mean(preds[non_pad_elements] == y[non_pad_elements])
def train_epoch(train_loader, model, optimizer, epoch, TAG_PAD_IDX=0, verbose=True):
    """
    complete one epoch of training for the non LSTM combined models
    :param train_loader: batch generator for the train data
    :param model: initialized model
    :param optimizer: optimizer instances
    :param epoch: the epoch number
    :param verbose: False if no console output is wanted
    """
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)
    model = model.to(device)
    criterion = criterion.to(device)
    epoch_losses = list()
    epoch_acc = list()
    model.train()
    for text_batch, labels_batch in train_loader:
        prediction = model(text_batch)  # compute model output
        loss = criterion(prediction, labels_batch)  # calculate loss
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute gradients of all variables wrt loss
        optimizer.step()  # perform updates using calculated gradients
        epoch_losses.append(loss.item())
        batch_acc = categorical_accuracy(np.argmax(prediction.detach().numpy(), axis=1), labels_batch.numpy(), TAG_PAD_IDX)
        epoch_acc.append(batch_acc)
    # epoch_acc = sum(train_predictions == train_labels) / len(train_labels)
    if verbose:
        print('epoch {}\t train loss: {:.3f}\t train accuracy: {:.3f}'.format(epoch, np.mean(epoch_losses), np.mean(epoch_acc)))
def evaluate_model(test_loader, model, optimizer, TAG_PAD_IDX):
    """
    evaluate the non LSTM combined models on the validation data at every fold end
    :param test_loader: batch generator for the test data
    :param model: initialized model
    :param optimizer: optimizer instances
    :param fold: the fold number (for console printing)
    """
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)  # you can set the parameters as you like
    model = model.to(device)
    criterion = criterion.to(device)
    test_losses = list()
    test_acc = list()
    model.eval()
    for text_batch, labels_batch in test_loader:
        with torch.no_grad():
            optimizer.zero_grad()
            prediction = model(text_batch)
            loss = criterion(prediction, labels_batch)
            test_losses.append(loss.item())
            batch_acc = categorical_accuracy(np.argmax(prediction.detach().numpy(), axis=1), labels_batch.numpy(),
                                             TAG_PAD_IDX)
            test_acc.append(batch_acc)
    print('Test results - test loss: {:.3f}\t test accuracy: {:.3f}\n'.format(np.mean(test_losses), np.mean(test_acc)))
    return np.mean(test_acc)
def train_case_epoch(train_loader, model, optimizer, epoch, TAG_PAD_IDX=0, verbose=True):
    """
    complete one epoch of training for the non LSTM combined models
    :param train_loader: batch generator for the train data
    :param model: initialized model
    :param optimizer: optimizer instances
    :param epoch: the epoch number
    :param verbose: False if no console output is wanted
    """
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)  # you can set the parameters as you like ignore_index=TAG_PAD_IDX
    model = model.to(device)
    criterion = criterion.to(device)
    epoch_losses = list()
    epoch_acc = list()
    model.train()
    for text_batch, labels_batch in train_loader:
        case_batch = []
        maximum = len(max(text_batch[1], key=lambda x: len(x)))
        for s in text_batch[1]:
            x = maximum - len(s)
            case_batch.append(([[0, 0, 0]]*x) + s)
        case_batch = torch.tensor(case_batch)
        prediction = model(text_batch[0], case_batch)  # compute model output
        loss = criterion(prediction, labels_batch)  # calculate loss
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute gradients of all variables wrt loss
        optimizer.step()  # perform updates using calculated gradients
        epoch_losses.append(loss.item())
        batch_acc = categorical_accuracy(np.argmax(prediction.detach().numpy(), axis=1), labels_batch.numpy(), TAG_PAD_IDX)
        epoch_acc.append(batch_acc)
    # epoch_acc = sum(train_predictions == train_labels) / len(train_labels)
    if verbose:
        print('epoch {}\t train loss: {:.3f}\t train accuracy: {:.3f}'.format(epoch, np.mean(epoch_losses), np.mean(epoch_acc)))
def evaluate_case_model(test_loader, model, optimizer, TAG_PAD_IDX):
    """
    evaluate the non LSTM combined models on the validation data at every fold end
    :param test_loader: batch generator for the test data
    :param model: initialized model
    :param optimizer: optimizer instances
    :param fold: the fold number (for console printing)
    """
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)  # you can set the parameters as you like
    model = model.to(device)
    criterion = criterion.to(device)
    test_losses = list()
    test_acc = list()
    model.eval()
    for text_batch, labels_batch in test_loader:
        case_batch = []
        maximum = len(max(text_batch[1], key=lambda x: len(x)))
        for s in text_batch[1]:
            x = maximum - len(s)
            case_batch.append(([[0, 0, 0]] * x) + s)
        case_batch = torch.tensor(case_batch)
        with torch.no_grad():
            optimizer.zero_grad()
            prediction = model(text_batch[0], case_batch)
            loss = criterion(prediction, labels_batch)
            test_losses.append(loss.item())
            batch_acc = categorical_accuracy(np.argmax(prediction.detach().numpy(), axis=1), labels_batch.numpy(),
                                             TAG_PAD_IDX)
            test_acc.append(batch_acc)
    print('Test results - test loss: {:.3f}\t test accuracy: {:.3f}\n'.format(np.mean(test_losses), np.mean(test_acc)))
    return np.mean(test_acc)
def build_tags(labels_idx, sentence, labels_itos):
    tagged_sent = []
    for w, t in zip(sentence, labels_idx):
        tagged_sent.append((w, labels_itos[t]))
    return tagged_sent
#=========================================================
#=========================================================



