import EX4.tagger as tagger

##########################################
TRAIN_DATA_PATH = r'C:\Users\dell\PycharmProjects\NLP\EX4\en-ud-train.upos.tsv'
TEST_DATA_PATH = r'C:\Users\dell\PycharmProjects\NLP\EX4\en-ud-dev.upos.tsv'
EMBEDDING_PATH = r'C:\Users\dell\PycharmProjects\NLP\.vector_cache\glove.6B.100d.txt'

# BLSTM_PARAMS = {
#     "max_vocab_size": -1,
#     "min_frequency": 1,
#     "embedding_dimension": 100,
#     "num_of_layers": 1,
#     "output_dimension": len(tagger.UNIVERSAL_TAGS),
#     "pretrained_embeddings_fn": EMBEDDING_PATH,
#     "data_fn": TRAIN_DATA_PATH,
#     "hidden_dim": 32,
#     "input_rep": 0
# }
#
# CBLSTM_PARAMS = {
#     "max_vocab_size": -1,
#     "min_frequency": 1,
#     "embedding_dimension": 100,
#     "num_of_layers": 1,
#     "output_dimension": len(tagger.UNIVERSAL_TAGS),
#     "pretrained_embeddings_fn": EMBEDDING_PATH,
#     "data_fn": TRAIN_DATA_PATH,
#     "hidden_dim": 32,
#     "input_rep": 1
# }
##########################################

train_data = tagger.load_annotated_corpus(TRAIN_DATA_PATH)
dev_data = tagger.load_annotated_corpus(TEST_DATA_PATH)

tagger.learn_params(train_data)
# allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tuple(tagger.learn_params(train_data))
#
# vocabulary = set(perWordTagCounts.keys())
# total = sum([len(x) for x in dev_data])

tagger.use_seed()

print()
print("============ HMM CHECK ============")
print()

for sentence in dev_data:
    tagger.joint_prob(sentence, tagger.A, tagger.B)

print("Passed sanity check HMM")
print()

print()
print("============ TRAIN ============")
print()

print("Training blstm ...")
BLSTM_PARAMS = tagger.get_best_performing_model_params()
BLSTM_PARAMS['pretrained_embeddings_fn'] = EMBEDDING_PATH
BLSTM_PARAMS['data_fn'] = TRAIN_DATA_PATH

blstm_model = tagger.initialize_rnn_model(BLSTM_PARAMS)
tagger.train_rnn(blstm_model, train_data, dev_data)

print()
print("Training cblstm ...")
CBLSTM_PARAMS = tagger.get_best_performing_model_params()
CBLSTM_PARAMS['pretrained_embeddings_fn'] = EMBEDDING_PATH
CBLSTM_PARAMS['data_fn'] = TRAIN_DATA_PATH
CBLSTM_PARAMS['input_rep'] = 1
cblstm_model = tagger.initialize_rnn_model(CBLSTM_PARAMS)
tagger.train_rnn(cblstm_model, train_data, dev_data, input_rep=1)

# print()
# print("============ TEST ============")
# print()
# print(f"Total words: {total}")
# print(f"Total unique: {len(vocabulary)}")
# print(f"Total IV: {len([x for sentence in dev_data for x, _ in sentence if x in vocabulary])}")
# print(f"Total OOV: {len([x for sentence in dev_data for x, _ in sentence if x not in vocabulary])}")
# print()


MODELS = [
    {"baseline": [tagger.perWordTagCounts, tagger.allTagCounts]},
    {"hmm": [tagger.A, tagger.B]},
    {"blstm": [blstm_model]},
    {"cblstm": [cblstm_model]}]

total = len([w for s in dev_data for w in s])
for model in MODELS:
    sum_correct = sum_correct_OOV = sum_OOV = 0

    for gold_sentence in dev_data:
        sentence = [w[0] for w in gold_sentence]
        tagged_sentence = tagger.tag_sentence(sentence, model)
        correct, correctOOV, OOV = tagger.count_correct(gold_sentence, tagged_sentence)

        sum_correct += correct
        sum_correct_OOV += correctOOV
        sum_OOV += OOV

    print(f"{list(model.keys())[0]} - accuracy: {round(sum_correct / total, 2)}"
          f" - correct: {sum_correct}, correctOOV: {sum_correct_OOV}, OOV: {sum_OOV}")
