#This code is used for test if the method works, 



import json
import collections
import os
import numpy as np
from sklearn.cluster import KMeans
from xml.etree import  ElementTree

def cread_xml(ensembleout, savedir):
    outkeys = ensembleout.keys()
    NIcount = 0
    Icount = 0
    for author in list(outkeys):
        # write_predictions_to_xmls
        author_id = author
        lang = "en"
        if ensembleout[author]['predict'] == 0:
            type = 'NI'
            NIcount = NIcount + 1
        else:
            type = 'I'
            Icount = Icount + 1
        root = ElementTree.Element('author', attrib={'id': author_id,
                                                     'lang': lang,
                                                     'type': type,
                                                     })
        tree = ElementTree.ElementTree(root)
        # Write the tree to an XML file
        tree.write(os.path.join(savedir, author + '.xml'), encoding="utf-8", xml_declaration=True)
    print('NI has: ' + str(NIcount))
    print('I has: ' + str(Icount))

def read_data(filedir):
    selectdir = filedir + 'train.json'
    savedict = {}
    with open(selectdir) as json_file:
        data = json.load(json_file)
    for author in list(data.keys()):
        filename = author
        savedict.update({filename: {}})
        savedict[filename].update({'prob': data[filename]['prob']})
    return savedict

def read_datasigmoid(filedir):
    scores = []
    label = []
    filename = []
    with open(filedir) as json_file:
        data = json.load(json_file)
    for author in list(data.keys()):
        for id in list(data[author].keys()):
            scores.append(data[author][id]['prob'])
            label.append(data[author][id]['label'])
            filename.append(author + '_' + id)
    scores = np.asarray(scores)
    label = np.asarray(label)
    return scores[:, 1], label, filename

def read_single_data(filedir):
    selectdir = filedir + 'train.json'
    savedict = {}
    with open(selectdir) as json_file:
        data = json.load(json_file)
    for author in list(data.keys()):
        filename = author
        savedict.update({filename: {}})
        savedict[filename].update({'prob': data[filename]['prob']})
    return savedict



def calresults(srcdir, savedir):
    models = os.listdir(srcdir)
    convmodel = 'BERT_CONV_prob'
    bertmodels = [i for i in  models if 'CONV' not in i]
    #bertmodels = ['BERT_CEWF']

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    foldpredict = {}
    for fold in range(4):
        bertprobdict = {}
        for bertmodel in bertmodels:
            modeldir = os.path.join(srcdir, bertmodel, str(fold))
            savedict = read_data(modeldir)
            filenames = list(savedict.keys())
            bertprobdict.update({bertmodel: savedict})
        modelmeanfoldprob = {}
        for filename in filenames:
            modelmeanfoldprob.update({filename: {}})
            #modelmeanfoldprob[filename].update({'prob': []})
            # bertnewprob[filename].update({'label': []})
            scores = []
            for bertmodel in bertmodels:
                #a = np.argmax(bertprobdict[bertmodel][filename]['prob'])
                scores.append(bertprobdict[bertmodel][filename]['prob'])
                #modelmeanfoldprob[filename]['prob'].append(bertprobdict[bertmodel][str(fold)][filename]['prob'])
            #scores = np.asarray(scores)
            meanscore = np.mean(scores, axis=0)
            modelmeanfoldprob[filename].update({'prob': meanscore})


        out = {}
        for documentid in modelmeanfoldprob.keys():
            author = documentid.split('_')[0]
            if out.get(author) is not None:
                pass
            else:
                out.update({author: {}})
            doc_id = documentid.split('_')[1]
            out[author].update({doc_id: {}})
            # out[author][doc_id].update({'label': float(outpre[documentid]['label'])})
            out[author][doc_id].update({'predict': modelmeanfoldprob[documentid]['prob']})
        authors = out.keys()
        # correct = 0
        outfinal = {}
        # total = 0
        for author in authors:
            scorelist = []
            for j in list(out[author].keys()):
                scorelist.append(out[author][j]['predict'])
            scorelist = np.mean(scorelist, axis=0)
            #final_decision = np.argmax(scorelist)
            outfinal.update({author: {}})
            outfinal[author].update({'predict': scorelist})
            # outfinal[author].update({'label': int(lebel)})
            # outfinal[author].update({'score': vote})

        # cread_xml(outfinal, savedir)
        # print('BERT NI has: ' + str(nohate) + ' I has: ' + str(hate))

        convmodeldir = os.path.join(srcdir, convmodel, str(fold))
        convdict = read_single_data(convmodeldir)


        ensembleout = {}
        for author in list(convdict.keys()):
            ensembleout.update({author: {}})
            convpred = convdict[author]['prob']
            bertpred = outfinal[author]['predict']
            finalpred = (convpred + bertpred) / 2.0
            '''if finalpred == 0:
                nohate = nohate + 1
            else:
                hate = hate + 1'''
            ensembleout[author].update({'predict': finalpred})
        foldpredict.update({str(fold): ensembleout})

    hate = 0
    nohate = 0
    outputdic = {}
    for filename in list(foldpredict['0'].keys()):
        outputdic.update({filename: {}})
        score = []
        for i in range(4):
            score.append(foldpredict[str(i)][filename]['predict'])
        score = np.mean(score, axis=0)
        score = np.argmax(score)
        if score == 0:
                nohate = nohate + 1
        else:
                hate = hate + 1
        outputdic[filename].update({'predict': int(score)})





    cread_xml(outputdic, savedir)
    #print('BERT NI has: ' + str(nohate) + ' I has: ' + str(hate))



calresults(sys.argv[1], sys.argv[2])



'''srcdir = './test_save'
savedir = './savedir'
calresults(srcdir, savedir)'''
