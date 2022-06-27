import sys, os
import random
import re
import csv
import json
from xml.dom import minidom
import multiprocessing as mp
import emoji
import numpy as np
from utils.TweetNormalizer import *
SEED=1
random.seed(SEED)
def processtext(sourcedir, dset, name, modeltype):
    xmldir = os.path.join(sourcedir, dset, name + ".xml")
    mydoc = minidom.parse(xmldir)
    items = mydoc.getElementsByTagName('document')
    Utts = []
    for i in range(len(items)):
        rowutt = items[i].childNodes[0].data
        if 'BERT' in modeltype:
            rowutt = rowutt.replace('#USER#', '@USER')
            rowutt = rowutt.replace('#URL#', 'HTTPURL')
            rowutt = rowutt.replace('#HASHTAG#', '#HASHTAG')
        else:
            rowutt = rowutt.replace('\n', '')
            rowutt = rowutt.replace('RT #USER#: ', '')
            rowutt = rowutt.replace('#USER#', '')
            rowutt = rowutt.replace('#URL#', '')
            rowutt = rowutt.replace('#HASHTAG#', '')
            rowutt = re.sub(' +', ' ', rowutt)  # replace multiple spaces by single space
            rowutt = emoji.demojize(rowutt, delimiters=("", ""))
            rowutt = rowutt.lower()
        Utts.append(rowutt)
    templist = {name: Utts}
    return templist


def product_helper(args):
    return processtext(*args)

def processevaltext(sourcedir, name):
    xmldir = os.path.join(sourcedir, name + ".xml")
    mydoc = minidom.parse(xmldir)
    items = mydoc.getElementsByTagName('document')
    Utts = []
    for i in range(len(items)):
        rowutt = items[i].childNodes[0].data
        rowutt = rowutt.replace('#USER#', '@USER')
        rowutt = rowutt.replace('#URL#', 'HTTPURL')
        rowutt = rowutt.replace('#HASHTAG#', '#HASHTAG')
        Utts.append(rowutt)
    templist = {name: Utts}
    return templist


def product_eval_helper(args):
    return processevaltext(*args)


def readdataset(sourcedir, savedir, ifdebug, modeltype):
    """
    # sourcedir (str) The dir of the Dataset
    # savedir (str) Where save the extracted data
    # ifdebug (str) If ifdebug is 'true', split 10% from training set as intern test set
    """
    print('data processing')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dset = 'training'
    with open(os.path.join(sourcedir, dset, "truth.txt")) as f:
        linesorg = f.readlines()

    trainlines = []
    for i in linesorg:
        index = i.split(":::")[0]
        trainlines.append(i)
    dset = 'training'
    lines = trainlines
    labeldict = {}
    for i in range(len(lines)):
        index = lines[i].split(":::")[0]
        label = lines[i].split(":::")[1].strip('\n')
        if label == 'I':
            labelid = 1
        elif label == 'NI':
            labelid = 0
        labeldict.update({index: {}})
        labeldict[index].update({"label": labelid})
    idkey = labeldict.keys()
    results = []
    pool = mp.Pool()
    job_args = [(sourcedir, 'training', i, modeltype) for i in list(idkey)]
    results.extend(pool.map(product_helper, job_args))
    outdict = {}
    for i in range(len(results)):
        name = list(results[i].keys())[0]
        text = results[i][name]
        outdict.update({name: {}})
        outdict[name].update({"text": text})
        outdict[name].update({"label": labeldict[name]["label"]})
    if 'BERT' in modeltype:
        with open(os.path.join(savedir, dset + "_BERTweet.json"), 'w', encoding='utf-8') as f:
            json.dump(outdict, f, ensure_ascii=False, indent=4)
    else:
        with open(os.path.join(savedir, dset + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outdict, f, ensure_ascii=False, indent=4)


def readexternalbertweetdataset(sourcedir, savedir, ifdebug):
    """
    # sourcedir (str) The dir of the Dataset
    # savedir (str) Where save the extracted data
    # ifdebug (str) If ifdebug is 'true', split 10% from training set as intern test set
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    testdict = {}
    setsdir1 = os.path.join(sourcedir, 'test.txt')
    with open(setsdir1) as f:
        lines = f.readlines()
    for i in range(1, len(lines)):
        datasplit = lines[i].split('\t')
        id = 'strain' + datasplit[0]
        testdict.update({id: {}})
        testdict[id].update({"text": datasplit[2]})
        testdict[id].update({"label": int(datasplit[1])})
    trainingdict = {}
    setsdir1 = os.path.join(sourcedir, 'training_1.txt')
    with open(setsdir1) as f:
        lines = f.readlines()
    for i in range(1, len(lines)):
        datasplit = lines[i].split('\t')
        id = 'strain' + datasplit[0]
        trainingdict.update({id: {}})
        trainingdict[id].update({"text": datasplit[2]})
        trainingdict[id].update({"label": int(datasplit[1])})
    setsdir2 = os.path.join(sourcedir, 'training_2.csv')
    with open(setsdir2, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in csv_reader:
            data.append(row)
    for i in range(1, len(data)):
        if data[i][1] == '-1':
            data[i][1] = 0
        else:
            data[i][1] = 1
        id = 'ktrain' + str(i)
        trainingdict.update({id: {}})
        trainingdict[id].update({"text": data[i][0]})
        trainingdict[id].update({"label": data[i][1]})


    with open(os.path.join(savedir, "training.json"), 'w', encoding='utf-8') as f:
        json.dump(trainingdict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(savedir, "test.json"), 'w', encoding='utf-8') as f:
        json.dump(testdict, f, ensure_ascii=False, indent=4)

def readevaldataset(sourcedir):
    """
    # sourcedir (str) The dir of the Dataset
    """
    print('data processing')
    filelist = os.listdir(sourcedir)
    idkey = [i.strip('.xml') for i in filelist]

    results = []
    pool = mp.Pool()
    job_args = [(sourcedir, i) for i in list(idkey)]
    results.extend(pool.map(product_eval_helper, job_args))
    outdict = {}
    for i in range(len(results)):
        name = list(results[i].keys())[0]
        text = results[i][name]
        outdict.update({name: {}})
        outdict[name].update({"text": text})
    return outdict