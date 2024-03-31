
import torch.nn as nn

''' ENCODER
*    This code is adapted from the Chatbot Tutorial code from Matthew Inkawhich and is available in the
*    pytorch official website: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
*    Changes are made.

The encoder RNN iterates through the input sentence one word(token)
at a time, outputting an "output" vector and a "hidden state" vector
The hidden state vector is then passed to the next time step
The output vector will be recorded
The encoder transforms the context from each point in the sequence into
a set of points in a high-dimensional space, which the decoder will use
to generate meaningful output for the given task
'''

'''
Computation Graph

1) Convert word indexes to embeddings
2) Pack padded batch of sequences for RNN module
3) Forward pass through GRU
4) Unpack padding
5) Sum bidirectional GRU outputs
6) Return output and final hidden state
'''


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # hidden_size == number of neurons in a GRU layer

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    '''
INPUT:
input_seq: batch of input sequences
input_lengths: list of sentence lengths corresponding to each sentence in the batch
hidden: hidden state

OUTPUT:
outputs: output features from the last hidden layer of the GRU
hidden: updated hidden state from GRU
'''

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return OUTPUT and final HIDDEN STATE
        return outputs, hidden
