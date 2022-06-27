import os, sys
from utils.read_dataset import *
from eval_evalBERTweet_CONV import *
from eval_evalBERTweet_single import *


def run(sourcedir, savedir, modeltype):
    sourcedir = os.path.join(sourcedir, 'en')
    datadict = readevaldataset(sourcedir)

    if modeltype == 'BERT_CEWF' or modeltype == 'BERT_CE' or modeltype == 'BERT_Focal':
        evalBERT(datadict, savedir, modeltype)

    elif modeltype == 'BERT_CONV':
        evalBERTCONV(datadict, savedir, modeltype)



'''sourcedir = './../PAN22/pan22-author-profiling-training-2022-03-29/en/test'
savedir = './../save'

if __name__ == "__main__":
    run(sourcedir, savedir)'''
run(sys.argv[1], sys.argv[2], sys.argv[3])
