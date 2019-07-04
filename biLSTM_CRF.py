# coding:utf-8
# James
# 20190705
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# helper functions to make the code more readable.
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm.
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class biLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2ix, embedding_dim, hidden_dim):
        super(biLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2ix = tag2ix
        self.tagset_size = len(tag2ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,
                            num_layers=1, bidirectional=True)

        # Maps the output pf the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # Matrix of transition parameters. Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag.
        self.transitions.data[tag2ix[START_TAG], :] = -10000
        self.transitions.data[:, tag2ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        pass

    def _forward_alg(self, feats):
        pass

    def _get_lstm_feature(self, feats, tags):
        pass
    
    def _score_sentence(self, feats, tags):
        pass

    def _viterbi_decode(self, feats):
        pass
    
    def neg_log_likelihood(self, sentence, tags):
        pass

    def forward(self, sentence):
        pass


