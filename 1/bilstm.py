import torch

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLSTM(torch.nn.Module):
    def __init__(self, embedding_dimension, num_of_layers, output_dimension, vectors, word_to_idx, input_rep, hidden_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        input_dimension = embedding_dimension if input_rep == 0 else embedding_dimension + 3

        self.embedding = torch.nn.Embedding.from_pretrained(vectors)

        self.word_to_idx = word_to_idx

        self.bi_lstm = torch.nn.LSTM(input_dimension, hidden_dim, bidirectional=True)

        self.h2t = torch.nn.Linear(hidden_dim * 2, output_dimension)

        self.multi_layer = num_of_layers > 1

        if self.multi_layer:
            self.linears = torch.nn.ModuleList([torch.nn.Linear(output_dimension, output_dimension)
                                               for _ in range(num_of_layers - 1)])
        self.hidden = None

    def init_hidden(self):
        self.hidden = (torch.zeros(2, 1, self.hidden_dim).to(device),
                       torch.zeros(2, 1, self.hidden_dim).to(device))

    def forward(self, sentence, input_case=None):
        embedding = self.embedding(sentence)

        if input_case is not None:
            embedding = torch.cat((embedding, input_case), dim=1).to(device)

        lstm_output, self.hidden = self.bi_lstm(embedding.view(len(sentence), 1, -1), self.hidden)
        logits = self.h2t(lstm_output.view(len(sentence), 1, -1))

        if self.multi_layer:
            for l in self.linears:
                logits = l(logits)

        return logits
