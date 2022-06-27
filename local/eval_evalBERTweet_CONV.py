import numpy as np
import collections
from model import *
import pickle
import copy, json
from utils.utils import *
from xml.etree import ElementTree
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def decoding(model, dataload, config):
    if not os.path.exists(config["representationdir"]):
            os.makedirs(config["representationdir"])
    with torch.no_grad():
        outpre = {}
        for count, data in enumerate(dataload, 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            filename = data[2]
            outputs, embedding = model(node_sets, mask)
            predicted = outputs

            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'predict': predicted[i].cpu().detach()})
                outpre[filename[i]].update({'embedding': embedding[i].cpu().detach()})
            print(count)
    out = {}
    for documentid in outpre.keys():
        author = documentid.split('_')[0]
        if out.get(author) is not None:
            pass
        else:
            out.update({author: {}})
        doc_id = documentid.split('_')[1]
        out[author].update({doc_id: {}})
        out[author][doc_id].update({'predict': outpre[documentid]['predict']})
        out[author][doc_id].update({'embedding': outpre[documentid]['embedding']})
    authors = out.keys()
    outfinal = {}
    for author in authors:
        outfinal.update({author: {}})
        predicts = []
        embeds = []
        for j in list(out[author].keys()):
            predicts.append(out[author][j]['predict'].unsqueeze(0))
            embeds.append(out[author][j]['embedding'])
        outfinal[author].update({'predict': torch.cat(predicts, dim=0)})
        outfinal[author].update({'embedding': torch.cat(embeds, dim=0)})
        torch.save(outfinal[author], os.path.join(config["representationdir"], author))



def embeddingextract(config, datadict):
    model = E2EBERTweetpretrained(config["cachedir"])
    dataset = BERTweetevaldatasetclass(data_file=datadict,
                               tokenizer=config["tokenizer"],
                               device=config["device"],
                               max_len=config["max_len"])


    BERTdir = config["pretrainedBERTdir"]
    netlist = os.listdir(BERTdir)
    best_model = 'acc'
    netlist.sort()
    net_dict = {}
    if best_model == 'loss':
        for m in range(len(netlist)):
            templist = netlist[m].split('_')
            net_dict[templist[1]] = netlist[m]
        net_dict = collections.OrderedDict(sorted(net_dict.items(), reverse=True))
    else:
        for m in range(len(netlist)):
            templist = netlist[m].split('_')
            net_dict[templist[4]] = netlist[m]
        net_dict = collections.OrderedDict(sorted(net_dict.items()))
    netname = net_dict.get(list(net_dict)[-1])
    print(netname)
    savedirnew = os.path.join(config["savedir"])
    if not os.path.exists(savedirnew):
        os.makedirs(savedirnew)

    #netname = net_dict.get(list(net_dict)[-1])
    #print(netname)
    loaddir = os.path.join(BERTdir, netname)
    self_state = model.state_dict()
    loaded_state = torch.load(loaddir, map_location=config["device"]).state_dict()
    loaded_state = {k.replace('module.', ''): v for k, v in loaded_state.items() if
                    k.replace('module.', '') in self_state}
    print(len(loaded_state.keys()))
    self_state.update(loaded_state)
    model.load_state_dict(self_state)
    model = model.to(config["device"])

    data_loader= torch.utils.data.DataLoader(dataset.dataset, shuffle=True, drop_last=True,
                                                    batch_size=8,
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_bert_eval_custom_sequence)

    model.eval()
    decoding(model, data_loader, config)

def evaluation(config):
    originaldataset = copy.deepcopy(config["testdict"])
    testdict = combine_text(originaldataset, shuffle=True)
    testdict = get_eval_split(testdict, config["max_len"], config["overlap"])

    if not os.path.exists(config["representationdir"]):
        embeddingextract(config, testdict)
    authorlist = list(set([i.split('_')[0] for i in list(testdict.keys())]))
    dataset = BERTweetrepreevaldatasetclass(authorlist=authorlist,
                                        representationdir=config["representationdir"])
    print(len(dataset.dataset))

    savedir = config["savedir"]
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    num_labels = 2
    model = CONV(num_labels)
    modeldir = os.path.join(config["modeldir"])
    netlist = os.listdir(modeldir)
    best_model = 'acc'
    netlist.sort()
    net_dict = {}
    if best_model == 'loss':
        for m in range(len(netlist)):
            templist = netlist[m].split('_')
            net_dict[templist[1]] = netlist[m]
        net_dict = collections.OrderedDict(sorted(net_dict.items(), reverse=True))
    else:
        for m in range(len(netlist)):
            templist = netlist[m].split('_')
            net_dict.update({templist[4]: {}})
        for m in range(len(netlist)):
            templist = netlist[m].split('_')
            net_dict[templist[4]].update({templist[1]: netlist[m]})
        net_dict = collections.OrderedDict(sorted(net_dict.items()))
    netsubdict = net_dict.get(list(net_dict)[-1])
    netsubdict = collections.OrderedDict(sorted(netsubdict.items(), reverse=True))
    netname = netsubdict.get(list(netsubdict)[-1])
    print(netname)

    savedirnew = os.path.join(config["savedir"])
    if not os.path.exists(savedirnew):
        os.makedirs(savedirnew)
    #self_state = torch.load(os.path.join(BERTdir, netname), map_location=config['device']).state_dict()
    self_state = model.state_dict()
    loaded_state = torch.load(os.path.join(modeldir, netname), map_location=config["device"]).state_dict()
    self_state.update(loaded_state)
    model.load_state_dict(self_state)
    model = model.to(config["device"])

    data_loader_dev = torch.utils.data.DataLoader(dataset.dataset, shuffle=True,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_bert_cnn_eval_custom_sequence)
    model.eval()
    outpre = {}
    for count, data in enumerate(data_loader_dev, 0):
        with torch.no_grad():
            prob = data[0].squeeze(0).to(config["device"])
            emb = data[1].squeeze(0).to(config["device"])
            mask = data[2]
            filename = data[3]
            outputs = model(prob, emb, mask, 4)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                outpre[filename[i]].update({'prob': prob[i]})
            #print(count)

    outkeys = outpre.keys()
    NIcount = 0
    Icount = 0
    for author in list(outkeys):
        # write_predictions_to_xmls
        author_id = author
        lang = "en"
        if outpre[author]['predict'] == 0:
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
        tree.write(os.path.join(savedirnew, author + '.xml'), encoding="utf-8", xml_declaration=True)
    print('NI has: ' + str(NIcount))
    print('I has: ' + str(Icount))
    with open(savedirnew + "train.json", 'w', encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)











def evalBERTCONV(datadict, mainsavedir, modeltype):
    cachedir = './CACHE'
    max_len = 500
    overlap = 64
    pretrained_BERTdir = os.path.join('save', 'BERT_CEWF', 'model')
    modeldir = os.path.join('save', modeltype, 'model')
    savedir = os.path.join(mainsavedir, modeltype)
    for makedir in [savedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    device = 'cuda'
    testdict = datadict
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", cache_dir=cachedir)
    testdict = datadict

    for fold in range(4):
        print(fold)
        representationdir = os.path.join(modeldir, str(fold) + '_representation_test')
        pretrained_BERTdirnew = os.path.join(pretrained_BERTdir, str(fold))
        modeldirnew = os.path.join(modeldir, str(fold))
        savedirnew = os.path.join(savedir, str(fold))
        for makedir in [savedirnew]:
            if not os.path.exists(makedir):
                os.makedirs(makedir)
        config = {
            "NWORKER": 0,
            "device": device,
            "batch_size": 4,
            "tokenizer": tokenizer,
            "modeldir": modeldirnew,
            "max_len": max_len,
            "overlap": overlap,
            "savedir": savedirnew,  # tune.choice([3, 5, 10, 15])
            "testdict": testdict,
            "cachedir": cachedir,
            "pretrainedBERTdir": pretrained_BERTdirnew,
            "representationdir": representationdir
        }
        evaluation(config)







