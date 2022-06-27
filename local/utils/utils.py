import torch
import random
from utils.TweetNormalizer import normalizeTweet
from torch.utils.data import Dataset
import emoji
import os

def pad_bert_custom_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    node_sets_sequence = []
    mask_sequence = []
    label_sequence = []
    filename_sequence = []
    for node_sets, mask, label, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=1)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, label_sequence, filename_sequence

def pad_bert_eval_custom_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    node_sets_sequence = []
    mask_sequence = []
    filename_sequence = []
    for node_sets, mask, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=1)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    return node_sets_sequence, mask_sequence, filename_sequence

def pad_bert_cnn_custom_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    prob_sequence = []
    emb_sequence = []
    label_sequence = []
    mask_sequence = []
    filename_sequence = []
    for prob, emb, label, mask, filename in sequences:
        prob_sequence.append(prob)
        emb_sequence.append(emb)
        label_sequence.append(label)
        mask_sequence.append(mask)
        filename_sequence.append(filename)
    prob_sequence = torch.nn.utils.rnn.pad_sequence(prob_sequence, batch_first=True, padding_value=0)
    emb_sequence = torch.nn.utils.rnn.pad_sequence(emb_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return prob_sequence, emb_sequence, label_sequence, mask_sequence, filename_sequence

def pad_bert_cnn_eval_custom_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    prob_sequence = []
    emb_sequence = []
    mask_sequence = []
    filename_sequence = []
    for prob, emb, mask, filename in sequences:
        prob_sequence.append(prob)
        emb_sequence.append(emb)
        mask_sequence.append(mask)
        filename_sequence.append(filename)
    prob_sequence = torch.nn.utils.rnn.pad_sequence(prob_sequence, batch_first=True, padding_value=0)
    emb_sequence = torch.nn.utils.rnn.pad_sequence(emb_sequence, batch_first=True, padding_value=0)
    return prob_sequence, emb_sequence, mask_sequence, filename_sequence

def combine_text(dictionary, shuffle=False):
    authors = dictionary.keys()
    for author in authors:
        texts = dictionary[author]['text']
        random.shuffle(texts)
        if shuffle is True:
            random.shuffle(texts)
        texts = ' '.join(texts)
        dictionary[author]['text'] = texts
    return dictionary

def get_split(dictionary, max_len, overlap):
    outdict = {}
    window_shift = max_len - overlap
    authors = dictionary.keys()
    for author in authors:
        texts = dictionary[author]['text']
        l_total = []
        l_parcial = []
        if len(texts.split()) // window_shift > 0:
            n = len(texts.split()) // window_shift
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = texts.split()[:max_len]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = texts.split()[w * window_shift:w * window_shift + max_len]
                l_total.append(" ".join(l_parcial))
        for i in range(len(l_total)):
            textid = author + '_' + str(i)
            outdict.update({textid: {}})
            outdict[textid].update({'text': l_total[i]})
            outdict[textid].update({'label': dictionary[author]['label']})
    return outdict

def get_eval_split(dictionary, max_len, overlap, shulfe=False):
    outdict = {}
    window_shift = max_len - overlap
    authors = dictionary.keys()
    for author in authors:
        texts = dictionary[author]['text']
        l_total = []
        l_parcial = []
        if len(texts.split()) // window_shift > 0:
            n = len(texts.split()) // window_shift
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = texts.split()[:max_len]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = texts.split()[w * window_shift:w * window_shift + max_len]
                l_total.append(" ".join(l_parcial))
        for i in range(len(l_total)):
            if shulfe is True:
                textid = author + '_shulfe' + str(i)
            else:
                textid = author + '_' + str(i)
            outdict.update({textid: {}})
            outdict[textid].update({'text': l_total[i]})
            #outdict[textid].update({'label': dictionary[author]['label']})
    return outdict

class BERTweetdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, test_file, tokenizer, device, max_len):
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.train_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        [self.train_file[id].update(
            {'encode': self.tokenizer(normalizeTweet(self.train_file[id]['text']), return_tensors='pt', truncation=True, max_length=self.max_len).to(self.device)})
            for id in list(self.train_file.keys())]
        [self.test_file[id].update(
            {'encode': self.tokenizer(normalizeTweet(self.test_file[id]['text']), return_tensors='pt', truncation=True, max_length=self.max_len).to(self.device)})
            for id in list(self.test_file.keys())]
        train_dataset = BERTweetdatasetloader(self.train_file)
        test_dataset = BERTweetdatasetloader(self.test_file)
        return train_dataset, test_dataset
class BERTweetdatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict):
        super(BERTweetdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        id = self.datadict[self.datakeys[index]]['encode'].data['input_ids']
        mask = self.datadict[self.datakeys[index]]['encode'].data['attention_mask']

        filename = self.datakeys[index]
        label = self.datadict[self.datakeys[index]]['label']
        label = torch.LongTensor([label])


        return id, mask, label, filename  # twtfsingdata.squeeze(0), filename


class BERTweetrepredatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, trainlist, testlist, representationdir):
        self.trainlist = trainlist
        self.testlist = testlist
        self.representationdir = representationdir
        self.train_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):
        train_dataset = BERTweetcnndatasetloader(self.trainlist, self.representationdir)
        test_dataset = BERTweetcnndatasetloader(self.testlist, self.representationdir)
        return train_dataset, test_dataset
class BERTweetcnndatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datalist, representationdir):
        super(BERTweetdatasetloader).__init__()
        self.datakeys = datalist
        self.representationdir = representationdir

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        author = self.datakeys[index]
        data = torch.load(os.path.join(self.representationdir, author))
        prob = data['predict']
        emb = data['embedding']
        label = data['label']
        mask = prob.size()[0]


        return prob, emb, label, mask, author  # twtfsingdata.squeeze(0), filename


class BERTweetrepreevaldatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, authorlist, representationdir):
        self.authorlist = authorlist
        self.representationdir = representationdir
        self.dataset = self.prepare_dataset()

    def prepare_dataset(self):
        dataset = BERTweetcnnevaldatasetloader(self.authorlist, self.representationdir)
        return dataset
class BERTweetcnnevaldatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datalist, representationdir):
        super(BERTweetcnnevaldatasetloader).__init__()
        self.datakeys = datalist
        self.representationdir = representationdir

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        author = self.datakeys[index]
        data = torch.load(os.path.join(self.representationdir, author))
        prob = data['predict']
        emb = data['embedding']
        mask = prob.size()[0]


        return prob, emb, mask, author  # twtfsingdata.squeeze(0), filename

class BERTweetevaldatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, data_file, tokenizer, device, max_len):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        [self.data_file[id].update(
            {'encode': self.tokenizer(normalizeTweet(self.data_file[id]['text']), return_tensors='pt', truncation=True, max_length=self.max_len).to(self.device)})
            for id in list(self.data_file.keys())]
        dataset = BERTweetevaldatasetloader(self.data_file)
        return dataset
class BERTweetevaldatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict):
        super(BERTweetevaldatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        id = self.datadict[self.datakeys[index]]['encode'].data['input_ids']
        mask = self.datadict[self.datakeys[index]]['encode'].data['attention_mask']

        filename = self.datakeys[index]


        return id, mask, filename  # twtfsingdata.squeeze(0), filename