
import torch
import torch.nn.functional as F
from torch import nn

import attention8
from vocabulary8 import device

""" DECODER

*    This code is adapted from the Chatbot Tutorial code from Matthew Inkawhich and is available in the
*    pytorch official website: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
*    Changes are made.

Implementing the decoder model after the Attention submodule
- Computation Graph
1) Get embedding of current input word
2) Forward through unindirectional GRU
3) Calculate attention weights from the current GRU output from 2
4) Multiply attention weights to encoder outputs to get new "weighted sum" context vector
5) Concatenate weighted context vector and GRU output usin Luong eq 5
6) without softmax
7) Return output and finl hidden state

"""


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = attention8.Attn(attn_model, hidden_size)

    '''
    INPUT:
    input_step: one word of input sequence batch
    las_hidden: final hidden layer of GRU
    encoder_outputs: encoder model's output

    OUTPUTS:
    output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence
    hidden: final hidden state of GRU
    '''

    def forward(self, input_step, last_hidden, encoder_outputs):
        # we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


# --------- Calculating the loss --------

'''
'maskNLLLoss' is defined to calculate our loss based on our decoder's output
tensor, the target tensor and a binary mask tensor describing the padding
of the target tensor. This loss calculates the average negative log likehood of the elements
that correspond to a *1* in the mask tensor
'''
'''
INPUT: 
inp: decoder's output
target: target tensor
'''


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
