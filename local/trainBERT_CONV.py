import os
import json
import random
import copy
import numpy as np
import collections
import pickle
from model import *
from utils.utils import *
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def decoding(model, dataload, savedir):
    if not os.path.exists(savedir):
            os.makedirs(savedir)
    with torch.no_grad():
        outpre = {}
        for count, data in enumerate(dataload, 0):
            node_sets = data[0]
            mask = data[1]
            labels = data[2]
            filename = data[3]
            outputs, embedding = model(node_sets, mask)
            predicted = outputs

            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': labels[i].cpu().detach()})
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
        out[author][doc_id].update({'label': outpre[documentid]['label']})
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
        outfinal[author].update({'label': out[author][j]['label']})
        torch.save(outfinal[author], os.path.join(savedir, author))



def embeddingextract(config, traindict, testdict):
    model = E2EBERTweetpretrained(config["cachedir"])
    dataset = BERTweetdatasetclass(train_file=traindict,
                                   test_file=testdict,
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
    loaddir = os.path.join(BERTdir, netname)
    self_state = model.state_dict()
    loaded_state = torch.load(loaddir, map_location=config["device"]).state_dict()
    loaded_state = {k.replace('module.', ''): v for k, v in loaded_state.items() if
                    k.replace('module.', '') in self_state}
    print(len(loaded_state.keys()))
    self_state.update(loaded_state)
    model.load_state_dict(self_state)
    model = model.to(config["device"])

    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                                                    batch_size=8,
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_bert_custom_sequence)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True, batch_size=8,
                                                  num_workers=config["NWORKER"],
                                                    collate_fn=pad_bert_custom_sequence)
    model.eval()
    decoding(model, data_loader_train, config["representationdir"])
    decoding(model, data_loader_dev, config["representationdir"])



def training(config, dataset=None, checkpoint_dir=None, data_dir=None):
    alltraindict = copy.deepcopy(config["traindict"])
    traindict = {}
    testdict = {}
    for i in config["cv"]["train_subsampler"]:
        traindict.update({i: alltraindict[i]})
    for i in config["cv"]["dev_subsampler"]:
        testdict.update({i: alltraindict[i]})
    traindict = combine_text(traindict, shuffle=True)
    testdict = combine_text(testdict, shuffle=True)
    traindict = get_split(traindict, config["max_len"], config["overlap"])
    testdict = get_split(testdict, config["max_len"], config["overlap"])

    if not os.path.exists(config["representationdir"]):
        embeddingextract(config, traindict, testdict)
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    #representationdir = config["representationdir"]
    for makedir in [modeldir, resultsdir]: #, representationdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    evalacc_best = 0
    evalloss_best = np.Inf
    early_wait = 4
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    criterion = torch.nn.CrossEntropyLoss()
    num_labels = 2
    model = CONV(num_labels, config["weight"])
    model = model.to(config["device"])
    '''if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)'''
    optimizer = torch.optim.Adam(model.parameters(), \
                                 lr=config["lr"], \
                                 betas=[0.9, 0.999], \
                                 eps=config["eps"], weight_decay=config["weight_decay"], \
                                 amsgrad=False
                                 )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, factor=0.001)

    trainauthorlist = list(set([i.split('_')[0] for i in list(traindict.keys())]))
    testauthorlist = list(set([i.split('_')[0] for i in list(testdict.keys())]))
    dataset = BERTweetrepredatasetclass(trainlist=trainauthorlist,
                                        testlist=testauthorlist,
                                        representationdir=config["representationdir"])
    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_bert_cnn_custom_sequence)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True, drop_last=True,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                    collate_fn=pad_bert_cnn_custom_sequence)



    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(data_loader_train, 0):
            prob = data[0].squeeze(0).to(config["device"])
            emb = data[1].squeeze(0).to(config["device"])
            label = data[2].to(config["device"])
            label = label.squeeze(-1)
            mask = data[3]
            filename = data[4]
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(prob, emb, mask, epoch)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            print("\r%f" % loss, end='')

            # print statistics
            tr_loss += loss.item()
            nb_tr_steps += 1
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().data.numpy().tolist())
        trainallscore = np.sum(np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1), axis=-1) / len(
            trainlabel)

        # Validation loss
        torch.cuda.empty_cache()
        evallossvec = []
        evalacc = 0
        model.eval()
        correct = 0
        outpre = {}
        total = 0
        for i, data in enumerate(data_loader_dev, 0):
            with torch.no_grad():
                prob = data[0].squeeze(0).to(config["device"])
                emb = data[1].squeeze(0).to(config["device"])
                labels = data[2].to(config["device"])
                labels = labels.squeeze(-1)
                mask = data[3]
                filename = data[4]
                outputs = model(prob, emb, mask, epoch)
                dev_loss = criterion(outputs, labels)
                evallossvec.append(dev_loss.cpu().data.numpy())
                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(len(filename)):
                    outpre.update({filename[i]: {}})
                    outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                    outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                    outpre[filename[i]].update({'prob': prob[i]})


        allscore = correct / total
        scheduler.step(allscore)

        evallossmean = np.mean(np.array(evallossvec))
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,
                          str(epoch) + '_' + str(evallossmean) + '_' + str(
                              currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                              allscore)[:6] + '.pkl')
        torch.save(model, OUTPUT_DIR)
        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                                        currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                        allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)


        torch.cuda.empty_cache()
        if allscore <= evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = allscore
            continuescore = continuescore + 1

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break

    print("Finished Training")
    #return model


def trainBERT_CONV(datasetdir, savedir, modeltype, cachedir, ifgpu):
    max_len = 500
    overlap = 64
    #pretrained_BERTdir = os.path.join('/PAN22/local', savedir, 'BERT', 'model')
    pretrained_BERTdir = os.path.join(savedir, 'BERT_CEWF', 'model')
    modeldir = os.path.join(savedir, modeltype, 'model')
    resultsdir = os.path.join(savedir, modeltype, 'result')
    max_num_epochs = 70
    nsplit = 4

    for makedir in [modeldir, resultsdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    if ifgpu == 'true':
        device = "cuda"
    else:
        device = 'cpu'

    with open(os.path.join(datasetdir, "training_BERTweet.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", cache_dir=cachedir)

    authors = traindict.keys()
    with open('./local/cv', 'rb') as fp:
        author_cvlist = pickle.load(fp)
    for fold in range(len(author_cvlist)):
        dev_author = list(author_cvlist[fold])
        train_author = []
        for author in authors:
            if author not in dev_author:
                train_author.append(author)

        print(str(fold))
        modelnewdir = os.path.join(modeldir, str(fold))
        resultsnewdir = os.path.join(resultsdir, str(fold))
        pretrained_sub_BERTdir = os.path.join(pretrained_BERTdir, str(fold))
        representationdir = os.path.join(modeldir, str(fold) + '_representation')
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        # dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_ids)
        ## Configuration
        # Best trial config: {'NWORKER': 0, 'device': 'cuda', 'weight_decay': 0, 'eps': 1e-08, 'lr': 2e-05, 'batch_size': 8, 'epochs': 20}
        config = {
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "eps": 1e-8,
            "lr": 0.02,
            "nsplit": nsplit,
            "batch_size": 4,
            "weight": 0.4,
            "modeldir": modelnewdir,
            "resultsdir": resultsnewdir,
            "epochs": max_num_epochs,  # tune.choice([3, 5, 10, 15])
            "traindict": traindict,
            "tokenizer": tokenizer,
            "max_len": max_len,
            "overlap": overlap,
            "cachedir": cachedir,
            "representationdir": representationdir,
            "pretrainedBERTdir": pretrained_sub_BERTdir,
            "cv": {"train_subsampler": train_author,
                   "dev_subsampler": dev_author}
        }
        training(config)

