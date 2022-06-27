import os, sys
import json
from utils.read_dataset import *
from trainBERTewwt_external import *
from trainBERTweet_single import *
from trainBERT_CONV import *
'''#from trainCONV_single import * ### change max length in utils/utils for padding it will influence the results
from trainBERTweet_single_CEFL_overlap import *
from trainBERTweet_Focal_single import *
#from train_statistics import *
from trainBERT_BLSTM_single_CEFL import *
from trainBERTweet_Tempscal import *


#from trainBERT_GCAN_single import *
#from trainBERT_att_single import *
#from trainBERT_CONV_pretrain import *'''

import numpy


def run(sourcedir, datasetdir, savedir, modeltype, cachedir, ifdebug, ifgpu, start_stage=None, stop_stage=None):
    if start_stage == None or stop_stage == None:
        start_stage = 0
        stop_stage = 100
    else:
        start_stage = int(start_stage)
        stop_stage = int(stop_stage)

    if start_stage <= 0 and stop_stage >= 0:
        ## extract text from xml files and save in json file
        ## Here only replace #USER#, #URL# and #HASHTAG#. Convert Emoji to text
        if modeltype != 'External':
            readdataset(sourcedir, datasetdir, ifdebug, modeltype)
        else:
            readexternalbertweetdataset(sourcedir, datasetdir, ifdebug)

    if start_stage <= 1 and stop_stage >= 1:
        if modeltype == 'External':
            trainBERTextr(datasetdir, savedir, modeltype, cachedir, ifgpu)
        elif modeltype == 'BERT_CEWF' or modeltype == 'BERT_CE' or modeltype == 'BERT_Focal':
            trainBERT(datasetdir, savedir, modeltype, cachedir, ifgpu)
        elif modeltype == 'BERT_CONV':
            trainBERT_CONV(datasetdir, savedir, modeltype, cachedir, ifgpu)



'''sourcedir = './../PAN22/pan22-author-profiling-training-2022-03-29/en'
sourceexternaldir = './../external_dataset'
datasetdir = './../dataset'
datasetexternaldir = './../dataset/external_BERTweet'
savedir = './../save'
modeltype = 'BERT'
cachedir = './../CACHE'
ifdebug = 'true'
ifgpu = 'true'
start_stage = '0'
stop_stage = '1'
if __name__ == "__main__":
    run(sourcedir, datasetdir, savedir, modeltype, cachedir, ifdebug, ifgpu, start_stage, stop_stage)'''
run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
