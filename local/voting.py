import sys, os
import random
import re
import json
from xml.dom import minidom
from xml.etree import ElementTree

import numpy as np
SEED=1
random.seed(SEED)

def read_xml(xmldir):
    with open(xmldir, 'r') as f:
        data = f.read()
    data = data.split(' ')
    for j in data:
        if 'type' in j:
            type = j.strip('type=').strip('"')
            if type == 'I':
                label = 1
            elif type == 'NI':
                label = 0
    return label


def run(sourcedir, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    resultsdict = {}
    subdir = os.path.join(sourcedir, '0')
    files = os.listdir(subdir)
    filename = [i.split('.')[0] for i in files]
    for file in filename:
        resultsdict.update({file: []})
    for i in range(4):
        for j in filename:
            xmldir = os.path.join(sourcedir, str(i), j + ".xml")
            resultsdict[j].append(read_xml(xmldir))

    for k in list(resultsdict.keys()):
        resultsdict[k] = round(sum(resultsdict[k]) / 4)

    outkeys = resultsdict.keys()
    NIcount = 0
    Icount = 0
    for author in list(outkeys):
        # write_predictions_to_xmls
        author_id = author
        lang = "en"
        if resultsdict[author] == 0:
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


'''sourcedir = './../eval/BERT_CE'
savedir = './../output'

run(sourcedir, savedir)'''

run(sys.argv[1], sys.argv[2])

