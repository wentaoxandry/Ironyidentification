import torch
import torch.nn.functional as F
from transformers import AutoModel, RobertaModel, AutoModelForSequenceClassification


class E2EBERTweetpretrained(torch.nn.Module):
    def __init__(self, cachedir):
        torch.nn.Module.__init__(self)
        self.bert = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-large", cache_dir=cachedir,
                                                               ignore_mismatched_sizes=True, output_hidden_states=True)
        #cache_dir = cachedir,

    def forward(self, nodes, mask):
        x = self.bert(nodes, mask)
        embedding = torch.cat(tuple([x.hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        x = x.logits
        return x, embedding

class E2EBERTweet(torch.nn.Module):
    def __init__(self, cachedir):
        torch.nn.Module.__init__(self)
        self.bert = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-large", #cache_dir=cachedir,
                                                               ignore_mismatched_sizes=True)
        #cache_dir = cachedir,

    def forward(self, nodes, mask):
        x = self.bert(nodes, mask)

        x = x.logits
        return x


class CONV(torch.nn.Module):
    def __init__(self, odim):
        torch.nn.Module.__init__(self)
        self.poolingemb = torch.nn.Linear(4096, 2048)
        #self.poolingprob = torch.nn.Linear(odim, 10)
        self.tanh = torch.nn.Tanh()
        self.convemb = torch.nn.Conv1d(2048, 64, 36, stride=1)
        #self.convprob = torch.nn.Conv1d(10, 5, 3, stride=1)
        self.relu = torch.nn.ReLU()
        self.maxpoolemb = torch.nn.MaxPool1d(8)
        #self.maxpoolprob = torch.nn.MaxPool1d(4)
        self.linearemb = torch.nn.Linear(64, odim)  # (d, c)
        #self.linearprob = torch.nn.Linear(5, odim)  # (d, c)
        self.linear = torch.nn.Linear(4, odim)  # (d, c)
        self.dropout = torch.nn.Dropout(0.5)


    def forward(self, predict, embed, mask, epoch):
        x_emb = self.tanh(self.poolingemb(embed))
        x_emb = self.dropout(torch.transpose(x_emb, 1, 2))
        x_emb = self.convemb(x_emb)
        x_emb = self.relu(x_emb)
        x_emb = self.maxpoolemb(x_emb)
        x_emb[x_emb != x_emb] = 0
        x_emb = torch.mean(x_emb, dim=-1)
        x_emb = self.linearemb(x_emb)
        if epoch < 2:
            x = x_emb
        else:
            x_prob = torch.sum(predict, dim=1) / (torch.FloatTensor(mask).unsqueeze(-1).to(predict.device))
            weight = torch.softmax(self.linear(torch.cat((x_emb, x_prob), dim=-1)), dim=-1)
            x = weight[:, 0].unsqueeze(-1) * x_emb + weight[:, 1].unsqueeze(-1) * x_prob

        return x


