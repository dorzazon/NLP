from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from torchtext import data
import pandas as pd
import nltk
import re
import random
from sklearn.metrics import roc_auc_score
from datetime import datetime
from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def load_best_model():
    """returning your best performing model that was saved as part of the submission bundle"""
    with open('best_model.pkl', 'rb') as path:
        state = pickle.load(path)
    with open('embeddings.pkl', 'rb') as path:
        embed = pickle.load(path)
    m = lstm_combined(embed)
    m.load_state_dict(state)
    m.eval()
    return m
def train_best_model():
   """ training a classifier from scratch (should be the same classifier and parameters returned by load_best_model().
    Of course, the final model could be slightly different than the one returned by  load_best_model(),
    due to randomization issues. This function call training on the data file you received.
    You could assume it is in the current directory. It should trigger the preprocessing and the whole pipeline."""
   text_path = 'trump_train.tsv'
   twitter_corpus = pd.read_csv(text_path, sep='\\t', names=['twit_id', 'account', 'twit_text', 'time', 'device'])
   twit = twitter_classification(twitter_corpus)
   twit.preprocess_twits_text()
   twit.prepare_embeddings()
   return twit.TrainBestModel()
def predict(m, fn):  # todo: pay attention to <unk> words transfer
    """this function does expect parameters. m is the trained model and fn is the full path to a file in the same
    format as the test set (see above). predict(m, fn) returns a list of 0s and 1s, corresponding to the lines
    in the specified file."""
    with open('stoi.pkl', 'rb') as path:
        stoi = pickle.load(path)
    twitter_corpus = pd.read_csv(fn, sep='\\t', names=['account', 'twit_text', 'time'])
    twit = twitter_classification(twitter_corpus)
    twit.preprocess_new_twits_text(stoi)
    twit.prepare_test_embeddings(stoi)
    predictions = m(twit.IntTextFFNN, torch.tensor(twit.twit_features, dtype=torch.float32))
    predictions = (predictions.detach().numpy() > 0.5).astype(np.int)
    predictions = np.array2string(predictions.reshape(-1), separator=' ')[1:-1]
    return predictions.replace('\n', '')


# Function to remove Stopwords
def remove_stopwords(tokenized_list):
    """
    this function removes stops words from a list of tokens and return a list
    :param tokenized_list: list of tokens include stopwords
    :return: list of tokens without stopwords
    """
    stopword = nltk.corpus.stopwords.words('english')  # All English Stopwords
    text = [word for word in tokenized_list if word not in stopword]  # To remove all stopwords
    return text
def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts. this is hekper function for lemmatizing
    :param word - a word to find its POS tag
    :return - the POS tag of the input word
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
def lemmatizing(tokenized_text):
    """
    this function lemmatizing each word token in the list input.
    :param tokenized_text: list of tokens
    :return: lemmatize tokens list
    """
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word, get_wordnet_pos(word)) for word in tokenized_text]
    return text
class twitter_classification:

    def __init__(self, text, stoi=None):
        if 'device' in text.columns:
            labels = text.loc[:, 'device']
            labels.where(labels == 'android', 1, inplace=True)
            labels.where(labels != 'android', 0, inplace=True)
            self.labels = labels.to_numpy().astype('int')
        self.twits = list(text.loc[:, 'twit_text'])
        self.twits_att = text.loc[:, ['account', 'time']]
        self.stoi = stoi
    # todo: can be adjusted to run the processing phase
    def preprocess_twits_text(self):
        """
        clean the twits and extract meta-features
        :return:
        """
        # todo: if you wish to run the processing undo comment from this line
        # clean_text = []
        # features_df = []
        # for sent in self.twits:
        #     features_df.append(self.__extract_features(sent))
        #     text_no_urls = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', sent, flags=re.MULTILINE)
        #     text_no_punctuation = re.sub(r'[^\w\s]', '', text_no_urls)
        #     text_no_digits = text_no_punctuation.replace('\d+', '<NUMBER>')
        #     lower_text = text_no_digits.lower()
        #     tw_tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        #     tw_token_list = tw_tknzr.tokenize(lower_text)
        #     tw_token_list_no_stop = remove_stopwords(tw_token_list)
        #     tw_token_list_no_stop_lemm = lemmatizing(tw_token_list_no_stop)
        #     clean_text.append(tw_token_list_no_stop_lemm)
        # features_df = pd.DataFrame(features_df, columns=['hashtag_num', 'mentions_num', 'capital_words_num',
        #                                                  'exclamation_mark_num'])
        # todo: until this line
        # save to directory
        # with open('NLP_clean_text.pkl', 'wb') as path:
        #     pickle.dump(clean_text, path)
        #     path.close()
        # with open('NLP_features_df.pkl', 'wb') as path:
        #     pickle.dump(features_df, path)
        #     path.close()
        # todo: and comment this section
        # load from directory
        path = open(r'NLP_clean_text.pkl', 'rb')
        clean_text = pickle.load(path)
        path.close()
        path2 = open(r'NLP_features_df.pkl', 'rb')
        features_df = pickle.load(path2)
        path2.close()
        # todo: until here

        ss = StandardScaler(copy=False)
        features_df = ss.fit_transform(features_df)
        time_f = self.__part_of_day()
        time_f = pd.get_dummies(time_f, prefix='quarter')
        features_df = np.concatenate((features_df, time_f.to_numpy()), axis=1)
        self.clean_twits = clean_text
        self.twit_features = features_df
    def preprocess_new_twits_text(self, stoi):
        """
        clean the twits and extract meta-features
        :return:
        """
        # todo: if you wish to run the processing undo comment from this line
        clean_text = []
        features_df = []
        for sent in self.twits:
            features_df.append(self.__extract_features(sent))
            text_no_urls = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', sent, flags=re.MULTILINE)
            text_no_punctuation = re.sub(r'[^\w\s]', '', text_no_urls)
            text_no_digits = text_no_punctuation.replace('\d+', '<NUMBER>')
            lower_text = text_no_digits.lower()
            tw_tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
            tw_token_list = tw_tknzr.tokenize(lower_text)
            tw_token_list_no_stop = remove_stopwords(tw_token_list)
            tw_token_list_no_stop_lemm = lemmatizing(tw_token_list_no_stop)
            # convert OOV work to <UNK>
            tw_token_list_no_stop_lemm = ['<unk>' if x not in stoi.keys() else x for x in tw_token_list_no_stop_lemm]
            clean_text.append(tw_token_list_no_stop_lemm)
        features_df = pd.DataFrame(features_df, columns=['hashtag_num', 'mentions_num', 'capital_words_num',
                                                         'exclamation_mark_num'])
        # todo: until this line
        # save to directory
        # with open('NLP_clean_text.pkl', 'wb') as path:
        #     pickle.dump(clean_text, path)
        #     path.close()
        # with open('NLP_features_df.pkl', 'wb') as path:
        #     pickle.dump(features_df, path)
        #     path.close()
        # todo: and comment this section
        # load from directory
        # path = open(r'NLP_clean_text.pkl', 'rb')
        # clean_text = pickle.load(path)
        # path.close()
        # path2 = open(r'NLP_features_df.pkl', 'rb')
        # features_df = pickle.load(path2)
        # path2.close()
        # todo: until here

        ss = StandardScaler(copy=False)
        features_df = ss.fit_transform(features_df)
        time_f = self.__part_of_day()
        time_f = pd.get_dummies(time_f, prefix='quarter')
        features_df = np.concatenate((features_df, time_f.to_numpy()), axis=1)
        self.clean_twits = clean_text
        self.twit_features = features_df

    def __extract_features(self, text):
        """
        extracts 4 meta features from twits text
        :param text: twits text
        :return: list of 3 meta-features extracted from text
        """
        hashtag_num = len(re.findall(r'#(\w+)', text))
        mentions_num = len(re.findall(r'@(\w+)', text))
        tokens = text.split()
        capital_words_num = sum([x.isupper() for x in tokens])
        exclamation_mark_num = sum([ch == '!' for ch in text])
        return [hashtag_num, mentions_num, capital_words_num, exclamation_mark_num]
    def __part_of_day(self):
        """
        extracts the quarter of the day in which the twit where posted
        0 - 0:6 ; 1 - 6:12 ; 2 - 12-18 ; 3 - 18-24
        :return: series of part of day category
        """
        part_of_day = []
        time_arr = list(self.twits_att.loc[:, 'time'])
        for time in time_arr:
            if pd.isna(time):
                part_of_day.append(random.randint(0, 3))
            else:
                time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                time_hour = int(time.strftime("%H"))
                part_of_day.append(0 if time_hour <= 6 else 1 if time_hour <= 12 else 2 if time_hour <= 18 else 3)
        return pd.Series(part_of_day, name='part_of_day')
    def __convert_text_to_num(self, vocab, list_text):
        """
        converts a twit from list of tokens to list of integers for the embedding layer
        :param vocab: dict holds int value for each word in the vocabolary
        :param list_text: list of token words
        :return: numpy array of int tokens
        """
        int_text = []
        for sent in list_text:
            int_sent = []
            for word in sent:
                int_sent.append(vocab[word])
            int_text.append(int_sent)
        return np.array(int_text)
    def __pad_and_EOS_text(self):
        """
        add <EOS> and padding to each twit text to make them same length
        :return: the length of the longest twit.
        """
        max_len = max([len(x) for x in self.clean_twits])
        for sent in self.clean_twits:
            sent.extend(['<EOS>'] + (['<pad>']*(max_len-len(sent))))
        return max_len + 1
    def prepare_embeddings(self):
        """
        builds the embeddings vectors from pre-trained Glove veectors
        """
        # convert string to int for model using embeddings
        # TEXT = data.Field(sequential=True, batch_first=True, include_lengths=True)
        # TEXT.build_vocab(self.clean_twits, vectors='glove.6B.100d')
        # self.IntText = self.__convert_text_to_num(dict(TEXT.vocab.stoi), self.clean_twits)
        # self.lstm_embeddings = TEXT.vocab.vectors
        # build numerical rep for twits to FFNN with padding and EOS tokens
        self.max_in_len = self.__pad_and_EOS_text()
        TEXT = data.Field(sequential=True, batch_first=True, include_lengths=True)
        TEXT.build_vocab(self.clean_twits, vectors='glove.6B.100d')
        # with open('stoi.pkl', 'wb') as path:
        #     pickle.dump(dict(TEXT.vocab.stoi), path)
        self.IntTextFFNN = np.array(self.__convert_text_to_num(dict(TEXT.vocab.stoi), self.clean_twits))
        self.ffnn_embeddings = TEXT.vocab.vectors
    def prepare_test_embeddings(self, stoi):
        self.max_in_len = self.__pad_and_EOS_text()
        self.IntTextFFNN = torch.tensor(self.__convert_text_to_num(stoi, self.clean_twits), dtype=torch.long)

    def TrainLRModel(self):
        """
        trains Logistic Regression model with the meta-features (no embeddings used) and tune the C parameter
        :return:
        """
        parameters = {'C': np.logspace(-4, 2, 10)}
        LR = GridSearchCV(LogisticRegression(random_state=42), parameters, 'roc_auc')
        self.LRModel = LR.fit(self.twit_features, self.labels)
        self.__generate_report_LR(LR.cv_results_)
        print('LR best C param: {}\t auc score: {}'.format(LR.best_params_['C'], LR.best_score_))
    def TrainSVMModel(self):
        """
        trains SVM model with the meta-features (no embeddings used) and tune the C and kernel parameter
        this function also saves a results figure to the working directory
        :return:
        """
        parameters = {'C': np.logspace(-4, 1, 8), 'kernel': ['linear', 'rbf', 'sigmoid']}  #, 'poly']}
        SVM = GridSearchCV(SVC(random_state=42), parameters, 'roc_auc')
        self.SVMModel = SVM.fit(self.twit_features, self.labels)
        self.__generate_report_SVM(SVM.cv_results_)
        print('SVM best C param: {}\t best kernel: {}\t auc score: {}'.
              format(SVM.best_params_['C'], SVM.best_params_['kernel'], SVM.best_score_))
    def TrainFFNNModel(self):
        """
        trains FFNN model based on concatenation of embeddings vectors with fixed length and tune the learning rate
        parameter. also generates results graph
        """
        p = {'epochs': 10, 'batch_size': 64, 'neurons_each_layer': [64, 32, 1],
             'LearningRates': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]}
        mean_metric_results = {'learning_rate': [], 'score': []}
        # tune learning rate
        for lr in p['LearningRates']:
            # use cross-validation for parameter tuning
            skf = StratifiedKFold(n_splits=10, random_state=2021)
            fold = 0
            folds_results = []
            for train_index, test_index in skf.split(self.IntTextFFNN, self.labels):
                x_train, x_test = self.IntTextFFNN[train_index],  self.IntTextFFNN[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.int64), torch.tensor(y_train, dtype=torch.float32))
                test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.int64), torch.tensor(y_test, dtype=torch.float32))
                train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)
                model = ffnn(self.ffnn_embeddings, self.max_in_len, p['neurons_each_layer'])
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                # train loop
                fold += 1
                print('Fold {} start!\n'.format(fold))
                for epoch in range(1, p['epochs'] + 1):
                    self.__train_epoch(train_loader, model, optimizer, epoch)
                test_auc, test_acc = self.__evaluate_model(test_loader, model, optimizer, fold)
                folds_results.append(test_auc)
            mean_metric_results['learning_rate'].append(lr)
            mean_metric_results['score'].append(np.mean(folds_results))
        best_lr = mean_metric_results['learning_rate'][mean_metric_results['learning_rate'].
            index(max(mean_metric_results['learning_rate']))]
        self.__generate_report(mean_metric_results, 'FFNN')
        self.FFNNModel = self.__refit_best_model(best_lr, p, 'ffnn')
    def TrainLSTMModel(self):
        """
        trains LSTM model based on sequence of embeddings vectors with fixed length and tune the learning rate
        parameter. also generates results graph
        """
        p = {'epochs': 20, 'batch_size': 64, 'hidden_state_size': 64,
             'LearningRates': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]}
        mean_metric_results = {'learning_rate': [], 'score': []}        # tune learning rate
        for lr in p['LearningRates']:
            # use cross-validation for parameter tuning
            skf = StratifiedKFold(n_splits=10, random_state=2021)
            fold = 0
            folds_results = []
            for train_index, test_index in skf.split(self.IntTextFFNN, self.labels):
                x_train, x_test = self.IntTextFFNN[train_index], self.IntTextFFNN[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                train_dataset = trump_dataset(x=x_train, y=y_train)
                test_dataset = trump_dataset(x=x_test, y=y_test)
                train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)
                model = lstm(self.ffnn_embeddings, p['hidden_state_size'])
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                # train loop
                fold += 1
                print('Fold {} start!\n'.format(fold))
                for epoch in range(1, p['epochs'] + 1):
                    self.__train_epoch(train_loader, model, optimizer, epoch)
                test_auc, test_acc = self.__evaluate_model(test_loader, model, optimizer, fold)
                folds_results.append(test_auc)
            mean_metric_results['learning_rate'].append(lr)
            mean_metric_results['score'].append(np.mean(folds_results))
        best_lr = mean_metric_results['learning_rate'][mean_metric_results['learning_rate'].
            index(max(mean_metric_results['learning_rate']))]
        self.__generate_report(mean_metric_results, 'LSTM')
        self.LSTMModel = self.__refit_best_model(best_lr, p, 'lstm')
    def TrainCombinedLSTMModel(self):
        """
        trains LSTM model based on LSTM of embeddings vectors with fixed length and concatenation of meta-features
        at the linear layer.
        tune the learning rate parameter. also generates results graph
        """
        p = {'epochs': 20, 'batch_size': 50, 'hidden_state_size': 64, 'add_features_size': self.twit_features.shape[1],
             'LearningRates': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]}
        mean_metric_results = {}
        # tune learning rate
        for lr in p['LearningRates']:
            # use cross-validation for parameter tuning
            skf = StratifiedKFold(n_splits=10, random_state=2021)
            fold = 0
            folds_results = []
            for train_index, test_index in skf.split(self.IntTextFFNN, self.labels):
                x_train, x_test = self.IntTextFFNN[train_index], self.IntTextFFNN[test_index]
                x_train_features, x_test_features = self.twit_features[train_index], self.twit_features[test_index]
                y_train, y_test = self.labels[train_index], self.labels[test_index]
                train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.int64),
                                              torch.tensor(x_train_features, dtype=torch.int64),
                                              torch.tensor(y_train, dtype=torch.float32))
                test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.int64),
                                             torch.tensor(x_test_features, dtype=torch.int64),
                                             torch.tensor(y_test, dtype=torch.float32))
                train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)
                model = lstm_combined(self.ffnn_embeddings, p['hidden_state_size'], p['add_features_size'])
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                # train loop
                fold += 1
                print('Fold {} start!\n'.format(fold))
                for epoch in range(1, p['epochs'] + 1):
                    self.__train_epoch_lstm_combined(train_loader, model, optimizer, epoch)
                test_auc, test_acc = self.__evaluate_model_lstm_combined(test_loader, model, optimizer, fold)
                folds_results.append(test_auc)
            mean_metric_results['learning_rate'].append(lr)
            mean_metric_results['score'].append(np.mean(folds_results))
        best_lr = mean_metric_results['learning_rate'][mean_metric_results['learning_rate'].
            index(max(mean_metric_results['learning_rate']))]
        self.__generate_report(mean_metric_results, 'LSTM_combined')
        self.LSTMCombinedModel = self.__refit_best_lstm_combined(best_lr, p)
    def TrainBestModel(self):
        """
        trains LSTMCombined model based on embeddings vectors with fixed length and concatenation of meta-features
        at the linear layer.
        """
        p = {'epochs': 20, 'batch_size': 50, 'hidden_state_size': 64, 'add_features_size': self.twit_features.shape[1],
             'LearningRate': 5e-3}
        dataset = TensorDataset(torch.tensor(self.IntTextFFNN, dtype=torch.int64),
                                      torch.tensor(self.twit_features, dtype=torch.int64),
                                      torch.tensor(self.labels, dtype=torch.float32))
        data_loader = DataLoader(dataset, batch_size=p['batch_size'], shuffle=True)
        model = lstm_combined(self.ffnn_embeddings, p['hidden_state_size'], p['add_features_size'])
        optimizer = torch.optim.Adam(model.parameters(), lr=p['LearningRate'])
        # train loop
        for epoch in range(1, p['epochs'] + 1):
            self.__train_epoch_lstm_combined(data_loader, model, optimizer, epoch, verbose=False)
        print('best model training finished !')
        return model

    def __train_epoch(self, train_loader, model, optimizer, epoch, verbose=True):
        """
        complete one epoch of training for the non LSTM combined models
        :param train_loader: batch generator for the train data
        :param model: initialized model
        :param optimizer: optimizer instances
        :param epoch: the epoch number
        :param verbose: False if no console output is wanted
        """
        loss_fn = nn.BCELoss()
        epoch_losses = list()
        train_predictions_prob = list()
        train_labels = list()
        model.train()
        for text_batch, labels_batch in train_loader:
            prediction = model(text_batch)  # compute model output
            loss = loss_fn(prediction, torch.reshape(labels_batch, (-1, 1)))  # calculate loss
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()  # compute gradients of all variables wrt loss
            optimizer.step()  # perform updates using calculated gradients
            epoch_losses.append(loss.item())
            train_predictions_prob.extend(prediction)
            train_labels.extend(labels_batch)
        train_predictions = (np.array(train_predictions_prob) > 0.5).astype(np.int)
        epoch_auc = roc_auc_score(train_labels, train_predictions_prob)
        epoch_acc = sum(train_predictions == train_labels) / len(train_labels)
        if verbose:
            print('epoch {}\t train loss: {:.3f}\t train AUC: {:.3f}\t train accuracy: {:.3f}'
                  .format(epoch, np.mean(epoch_losses), epoch_auc, epoch_acc))
    def __evaluate_model(self, test_loader, model, optimizer, fold):
        """
        evaluate the non LSTM combined models on the validation data at every fold end
        :param test_loader: batch generator for the test data
        :param model: initialized model
        :param optimizer: optimizer instances
        :param fold: the fold number (for console printing)
        """
        loss_fn = nn.BCELoss()
        test_losses = list()
        test_predictions_prob = list()
        test_labels = list()
        model.eval()
        for text_batch, labels_batch in test_loader:
            with torch.no_grad():
                optimizer.zero_grad()
                prediction = model(text_batch)
                loss = loss_fn(prediction, torch.reshape(labels_batch, (-1, 1)))
                test_predictions_prob.extend(prediction)
                test_losses.append(loss.item())
                test_labels.extend(labels_batch)
        test_predictions = (np.array(test_predictions_prob) > 0.5).astype(np.int)
        test_auc = roc_auc_score(test_labels, test_predictions_prob)
        test_acc = sum(test_predictions == test_labels) / len(test_labels)
        print('Fold {} test results - test loss: {:.3f}\t test AUC: {:.3f}\t test accuracy: {:.3f}\n'
              .format(fold, np.mean(test_losses), test_auc, test_acc))
        return test_auc, test_acc
    def __train_epoch_lstm_combined(self, train_loader, model, optimizer, epoch, verbose=True):
        """
        complete one epoch of training for the LSTM combined model
        :param train_loader: batch generator for the train data
        :param model: initialized model
        :param optimizer: optimizer instances
        :param epoch: the epoch number
        :param verbose: False if no console output is wanted
        """
        loss_fn = nn.BCELoss()
        epoch_losses = list()
        train_predictions_prob = list()
        train_labels = list()
        model.train()
        for text_batch, add_features_batch, labels_batch in train_loader:
            prediction = model(text_batch, add_features_batch)  # compute model output
            loss = loss_fn(prediction, torch.reshape(labels_batch, (-1, 1)))  # calculate loss
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()  # compute gradients of all variables wrt loss
            optimizer.step()  # perform updates using calculated gradients
            epoch_losses.append(loss.item())
            train_predictions_prob.extend(prediction)
            train_labels.extend(labels_batch)
        train_predictions = (np.array(train_predictions_prob) > 0.5).astype(np.int)
        epoch_auc = roc_auc_score(train_labels, train_predictions_prob)
        epoch_acc = sum(train_predictions == train_labels) / len(train_labels)
        if verbose:
            print('epoch {}\t train loss: {:.3f}\t train AUC: {:.3f}\t train accuracy: {:.3f}'
                  .format(epoch, np.mean(epoch_losses), epoch_auc, epoch_acc))
    def __evaluate_model_lstm_combined(self, test_loader, model, optimizer, fold):
        """
           evaluate the LSTM combined models on the validation data at every fold end
           :param test_loader: batch generator for the test data
           :param model: initialized model
           :param optimizer: optimizer instances
           :param fold: the fold number (for console printing)
       """
        loss_fn = nn.BCELoss()
        test_losses = list()
        test_predictions_prob = list()
        test_labels = list()
        model.eval()
        for text_batch, add_features_batch, labels_batch in test_loader:
            with torch.no_grad():
                optimizer.zero_grad()
                prediction = model(text_batch, add_features_batch)
                loss = loss_fn(prediction, torch.reshape(labels_batch, (-1, 1)))
                test_predictions_prob.extend(prediction)
                test_losses.append(loss.item())
                test_labels.extend(labels_batch)
        test_predictions = (np.array(test_predictions_prob) > 0.5).astype(np.int)
        test_auc = roc_auc_score(test_labels, test_predictions_prob)
        test_acc = sum(test_predictions == test_labels) / len(test_labels)
        print('Fold {} test results - test loss: {:.3f}\t test AUC: {:.3f}\t test accuracy: {:.3f}\n'
              .format(fold, np.mean(test_losses), test_auc, test_acc))
        return test_auc, test_acc
    def __refit_best_model(self, best_lr, p, model_name):
        """
        refitting best non LSTM combined model with the whole data and the best lr parameter
        :param best_lr: the best lr found at the tuning process
        :param p: dictionary holds the model defulat parameters
        :param model_name: (str) the model name
        :return: trained model
        """
        if model_name == 'ffnn':
            print('refitting ffnn model using best learn rate: {}'.format(best_lr))
            model = ffnn(self.ffnn_embeddings, self.max_in_len, p['neurons_each_layer'])
        else:
            print('refitting lstm model using best learn rate: {}'.format(best_lr))
            model = lstm(self.ffnn_embeddings, p['hidden_state_size'])
        train_dataset = TensorDataset(torch.tensor(self.IntTextFFNN, dtype=torch.int64),
                                      torch.tensor(self.labels, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
        for epoch in range(1, p['epochs'] + 1):
            self.__train_epoch(train_loader, model, optimizer, epoch, verbose=False)
        print('best {} model saved!'.format(model_name))
        return model
    def __refit_best_lstm_combined(self, best_lr, p):
        """
           refitting best LSTM combined model with the whole data and the best lr parameter
           :param best_lr: the best lr found at the tuning process
           :param p: dictionary holds the model defulat parameters
           :return: trained model
       """
        print('refitting lstm combined model using best learn rate: {}'.format(best_lr))
        model = lstm_combined(self.ffnn_embeddings, p['hidden_state_size'], p['add_features_size'])
        train_dataset = TensorDataset(torch.tensor(self.IntTextFFNN, dtype=torch.int64),
                                     torch.tensor(self.twit_features, dtype=torch.int64),
                                     torch.tensor(self.labels, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
        for epoch in range(1, p['epochs'] + 1):
            self.__train_epoch_lstm_combined(train_loader, model, optimizer, epoch, verbose=False)
        print('best lstm_combined model saved!')
        return model


    def __generate_report(self, mean_metric_results, model_name):
        """
        generates graph of the AUC score by the learning rate of the model
        :param mean_metric_results: dictionary holds two lists (lr's and AUC scores)
        :param model_name: (str) the model name
        """
        plt.semilogx(mean_metric_results['learning_rate'], mean_metric_results['score'])
        plt.title("model AUC by learning_rate parameter")
        plt.ylabel("AUC Score")
        plt.xlabel("learning_rate")
        plt.savefig('{}_figure.png'.format(model_name))
        print('figure saved to directory!')
        # plt.show()
        plt.clf()
    def __generate_report_LR(self, mean_metric_results):
        """
        generates graph of the AUC score by the C parameter of the model
        :param mean_metric_results: dictionary holds two lists (C's and AUC scores)
        """
        plt.plot(mean_metric_results['param_C'], mean_metric_results['mean_test_score'])
        plt.title("model AUC by C parameter")
        plt.ylabel("AUC Score")
        plt.xlabel("C")
        plt.xscale('log')
        plt.savefig('LR_figure.png')
        print('LR figure saved to directory!')
        # plt.show()
        plt.clf()
    def __generate_report_SVM(self, mean_metric_results):
        """
            generates graph of the AUC score by the C and kernel parameters of the model
            :param mean_metric_results: dictionary holds three lists (C's, kernels and AUC scores)
        """
        kernels = np.unique(mean_metric_results['param_kernel'])
        for kernel in kernels:
            kernel_indices = [i for i, x in enumerate(mean_metric_results['param_kernel']) if x == str(kernel)]
            kernel_cs = [mean_metric_results['param_C'][i] for i in kernel_indices]
            kernel_scores = [mean_metric_results['mean_test_score'][i] for i in kernel_indices]
            plt.plot(kernel_cs, kernel_scores, label='{} kernel'.format(kernel))
        plt.title("SVM model AUC by C and kernel parameter")
        plt.ylabel("AUC Score")
        plt.xlabel("C")
        plt.xscale('log')
        plt.legend()
        plt.savefig('SVM_figure.png')
        print('SVM figure saved to directory!')
        # plt.show()
        plt.clf()


class ffnn(nn.Module):

    def __init__(self, embedding_weights, input_len, neurons_each_layer):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(input_len * embedding_weights.shape[1], neurons_each_layer[0])
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(neurons_each_layer[0], neurons_each_layer[1])
        self.lin3 = nn.Linear(neurons_each_layer[1], neurons_each_layer[2])
        self.sigmoid = nn.Sigmoid()
    def forward(self, text):
        embedded = self.embedding(text)
        x = self.flatten(embedded)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return self.sigmoid(x)
class lstm(nn.Module):
    def __init__(self, embedding_weights, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        hidden_state = torch.randn(1, 50, hidden_size)
        cell_state = torch.randn(1, 50, hidden_size)
        self.hidden = (hidden_state, cell_state)
        self.lstm = nn.LSTM(embedding_weights.shape[1], hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        # _, self.hidden = self.lstm(embedded, self.hidden)
        x = self.lin(lstm_out[:, -1, :])
        return self.sigmoid(x)
class lstm_combined(nn.Module):
    def __init__(self, embedding_weights, hidden_size=64, add_feature_size=8):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        self.lstm = nn.LSTM(embedding_weights.shape[1], hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size + add_feature_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, text, add_features):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        x = torch.cat((lstm_out[:, -1, :], add_features), dim=1)
        x = self.lin(x)
        return self.sigmoid(x)

class trump_dataset(Dataset):
    """
    this dataset used just once to test it. will be implamented in future work
    """
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_idx = torch.tensor(self.x[idx], dtype=torch.long)
        if self.y is not None:
            y_idx = torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            y_idx = None
        return x_idx, y_idx


# text_path = 'trump_train.tsv'
# twitter_corpus = pd.read_csv(text_path, sep='\\t', names=['twit_id', 'account', 'twit_text', 'time', 'device'])
# twit = twitter_classification(twitter_corpus)
# twit.preprocess_twits_text()
# twit.prepare_embeddings()
# twit.TrainLRModel()
# twit.TrainSVMModel()
# twit.TrainFFNNModel()
# twit.TrainLSTMModel()
# twit.TrainCombinedLSTMModel()

# with open('twit.pkl', 'wb') as path:
#     pickle.dump(twit, path)






