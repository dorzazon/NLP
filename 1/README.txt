A few notes about the implementation:

1) The rnn model params dictionary has the following structure:

{
    "max_vocab_size": -1,
    "min_frequency": 1,
    "input_dimension": 100,
    "embedding_dimension": 100,
    "num_of_layers": 1,
    "output_dimension": len(UNIVERSAL_TAGS),            # 17 - the universal tags defined in the project document
    "pretrained_embeddings_fn": "glove.6B.100d.txt",    # note that I assume the embeddings are in the root directory
    "data_fn": "en-ud-train.upos.tsv",                  # same assumption for training data
    "input_rep": 0
    "hidden_dim": 32                                    # extra param with default value (optional)
}


2) train_rnn function runs by default for 25 epochs. The expected running time per epoch for both bilstms is:
    a. 70 sec for GPU
    b. 90 sec for CPU

changing the number of epochs can be done in the train_rnn function


3) expected results for 25 epoch training and default randomization seed:

baseline - accuracy: 0.84 - correct: 21088, correctOOV: 172, OOV: 2090
hmm - accuracy: 0.88 - correct: 22136, correctOOV: 677, OOV: 2090
blstm - accuracy: 0.89 - correct: 22342, correctOOV: 993, OOV: 2090
cblstm - accuracy: 0.9 - correct: 22638, correctOOV: 1234, OOV: 2090


4) Note that a few minor changes were made to the tag_sentence function that should not change external usage


5) Adding the driver.py file that I used to create the results from #3, and for your usage if anything breaks in the code
