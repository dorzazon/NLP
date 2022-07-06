
import tagger
from random import randrange


def calc_score(dev_data, model_dict):
    score_nom, score_denom = 0, 0
    for gold_sentence in dev_data:
        pred_sentence = [w[0] for w in gold_sentence]
        tagged_sentence = tagger.tag_sentence(pred_sentence, model_dict)
        correct, correctOOV, OOV = tagger.count_correct(gold_sentence, tagged_sentence)
        score_nom += correct
        score_denom += len(pred_sentence)
    print(f"{list(model_dict.keys())[0]} score is {score_nom / score_denom}")


def check_sampled_sentence(gold_sentence, model_dict):
    pred_sentence = [w[0] for w in gold_sentence]
    tagged_sentence = tagger.tag_sentence(pred_sentence, model_dict)
    correct, correctOOV, OOV = tagger.count_correct(gold_sentence, tagged_sentence)
    print(f"correct: {correct}, correctOOV: {correctOOV}, OOV: {OOV}\n")


train_path = r"C:\src\MastersCourses\NLP\Assign_4\data\en-ud-train.upos.tsv"
dev_path = r"C:\src\MastersCourses\NLP\Assign_4\data\en-ud-dev.upos.tsv"

train_data = tagger.load_annotated_corpus(train_path)
dev_data = tagger.load_annotated_corpus(dev_path)

[allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] = tagger.learn_params(train_data)


#draw random sentnece
gold_sentence = dev_data[randrange(len(dev_data))]
print(f"tested random sentence is {gold_sentence} of length {len(gold_sentence)}\n")


#test beseline
calc_score(dev_data, {'baseline': [perWordTagCounts, allTagCounts]})
check_sampled_sentence(gold_sentence, {'baseline': [perWordTagCounts, allTagCounts]})


#test hmm
calc_score(dev_data, {'hmm': [A,B]})
check_sampled_sentence(gold_sentence, {'hmm': [A,B]})


# #LSTM settings:
# model_dict = {'input_dimension': 0, 'embedding_dimension': 0, 'num_of_layers': 0, 'output_dimension': 0}
# data_fn = train_path
# pretrained_embeddings_fn = r"C:\src\MastersCourses\NLP\Assign_4\embed\glove.6B.100d.txt"
#
# #test Vanilla BiLSTM:
# print(f"provided model dict is {model_dict}")
# model = tagger.initialize_rnn_model(model_dict)
# saved_model_dict = tagger.initialize_rnn_model(model)
# print(f"saved model dict is {saved_model_dict}")
# tagger.train_rnn(model, data_fn, pretrained_embeddings_fn, input_rep=0)
# calc_score(dev_data, {'blstm': [model, 0]})
# check_sampled_sentence(gold_sentence, {'blstm': [model, 0]})
#
#
# #test BiLSTM + case:
# print(f"provided model dict is {model_dict}")
# model = tagger.initialize_rnn_model(model_dict)
# saved_model_dict = tagger.initialize_rnn_model(model)
# print(f"saved model dict is {saved_model_dict}")
# tagger.train_rnn(model, data_fn, pretrained_embeddings_fn, input_rep=1)
# calc_score(dev_data, {'blstm': [model, 0]})
# check_sampled_sentence(gold_sentence, {'blstm': [model, 0]})


