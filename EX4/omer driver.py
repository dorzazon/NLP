from EX4 import tagger
from tqdm import tqdm
TRAIN_DATA_PATH = r'C:\Users\dell\PycharmProjects\NLP\EX4\en-ud-train.upos.tsv'
TEST_DATA_PATH = r'C:\Users\dell\PycharmProjects\NLP\EX4\en-ud-dev.upos.tsv'
EMBEDDING_PATH = r'C:\Users\dell\PycharmProjects\NLP\.vector_cache\glove.6B.100d.txt'
train_tagged_sentences = tagger.load_annotated_corpus('en-ud-train.upos.tsv')
test_tagged_sentences = tagger.load_annotated_corpus(TEST_DATA_PATH)
model_params = tagger.get_best_performing_model_params()
# model_params['input_rep'] = 0
model_params['pretrained_embeddings_fn'] = EMBEDDING_PATH
model_params['data_fn'] = TRAIN_DATA_PATH
# model = tagger.initialize_rnn_model(model_params)
# tagger.train_rnn(model, train_tagged_sentences)
# correctS = 0; correctOOVS = 0; OOVS = 0
# for tagged_sentence in test_tagged_sentences:
#     sentence = [t[0] for t in tagged_sentence]
#     pred = tagger.rnn_tag_sentence(sentence, model)
#     correct, correctOOV, OOV = tagger.count_correct(tagged_sentence, pred)
#     correctS+= correct; correctOOVS += correctOOV; OOVS += OOV
# print('BiLSTM\ncorrect: {}\tcorrectOOV: {}\tOOV: {}'.format(correctS, correctOOVS, OOVS))

model_params['input_rep'] = 1
model = tagger.initialize_rnn_model(model_params)
tagger.train_rnn(model, train_tagged_sentences, test_tagged_sentences)
correctS = 0; correctOOVS = 0; OOVS = 0
for tagged_sentence in test_tagged_sentences:
    sentence = [t[0] for t in tagged_sentence]
    pred = tagger.rnn_tag_sentence(sentence, model)
    correct, correctOOV, OOV = tagger.count_correct(tagged_sentence, pred)
    correctS+= correct; correctOOVS += correctOOV; OOVS += OOV
print('CBLSTM\ncorrect: {}\tcorrectOOV: {}\tOOV: {}'.format(correctS, correctOOVS, OOVS))
# 22716
# tagger.learn_params(train_tagged_sentences)
# correctS = 0; correctOOVS = 0; OOVS = 0; i=0
# for tagged_sentence in test_tagged_sentences:
#     sentence = [t[0] for t in tagged_sentence]
#     pred = tagger.hmm_tag_sentence(sentence, tagger.A, tagger.B)
#     correct, correctOOV, OOV = tagger.count_correct(tagged_sentence, pred)
#     correctS+= correct; correctOOVS += correctOOV; OOVS += OOV
#     i+=len(tagged_sentence)
#     print('HMM\ncorrect: {}\tcorrectOOV: {}\tOOV: {}\tACC: {}'.format(correctS, correctOOVS, OOVS, correctS/i))
# 25098 88% ACC
# correctS = 0; correctOOVS = 0; OOVS = 0
# for tagged_sentence in tqdm(test_tagged_sentences):
#     sentence = [t[0] for t in tagged_sentence]
#     pred = tagger.baseline_tag_sentence(sentence, tagger.perWordTagCounts, tagger.allTagCounts)
#     correct, correctOOV, OOV = tagger.count_correct(tagged_sentence, pred)
#     correctS+= correct; correctOOVS += correctOOV; OOVS += OOV
# print('Base\ncorrect: {}\tcorrectOOV: {}\tOOV: {}'.format(correctS, correctOOVS, OOVS))
# correct: 21144	correctOOV: 157	OOV: 1709


