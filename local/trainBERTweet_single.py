import os
import json
import random
import numpy as np
import collections
from model import *
import pickle
import copy
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

class CEWF(torch.nn.Module):
    def __init__(self, t, gamma):
        super(CEWF, self).__init__()
        self.t = t
        self.gamma = gamma

    def forward(self, logits, targets):
        prob = torch.softmax(logits, dim=-1)
        p_k = torch.gather(prob, -1, targets.unsqueeze(-1)).squeeze(-1)
        p_k_inv = 1 - p_k
        exp_p_k = torch.exp(p_k * self.t)
        exp_p_k_inv = torch.exp(p_k_inv * self.t)

        loss1 = (- exp_p_k_inv / (exp_p_k + exp_p_k_inv)) * torch.log(p_k)
        loss2 = ((- exp_p_k * torch.pow(p_k_inv, self.gamma)) / (exp_p_k + exp_p_k_inv)) * torch.log(p_k)
        loss = torch.mean(loss1 + loss2)


        return loss

class Focal(torch.nn.Module):
    def __init__(self, gamma):
        super(Focal, self).__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        prob = torch.softmax(logits, dim=-1)
        p_k = torch.gather(prob, -1, targets.unsqueeze(-1)).squeeze(-1)
        p_k_inv = 1 - p_k
        loss = -torch.pow(p_k_inv, self.gamma) * torch.log(p_k)

        return torch.mean(loss)

def training(config, dataset=None, checkpoint_dir=None, data_dir=None):
    alltraindict = copy.deepcopy(config["traindict"])
    traindict = {}
    testdict = {}
    for i in config["cv"]["train_subsampler"]:
        traindict.update({i: alltraindict[i]})
    for i in config["cv"]["dev_subsampler"]:
        testdict.update({i: alltraindict[i]})
    if len(alltraindict[i]['text']) == 200:
        traindict = combine_text(traindict)
        testdict = combine_text(testdict)
    else:
        pass

    traindict = get_split(traindict, config["max_len"], config["overlap"])
    testdict = get_split(testdict, config["max_len"], config["overlap"])

    dataset = BERTweetdatasetclass(train_file=traindict,
                               test_file=testdict,
                               tokenizer=config["tokenizer"],
                               device=config["device"],
                               max_len=config["max_len"])
    print(len(dataset.train_dataset))
    print(len(dataset.test_dataset))

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    for makedir in [modeldir, resultsdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)
    evalacc_best = 0
    evalloss_best = np.Inf
    early_wait = 4
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    if config["losstype"] == 'CEWF':
        criterion = CEWF(4, 5)
    elif config["losstype"] == 'Focal':
        criterion = Focal(5)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    model = E2EBERTweet(config["cachedir"])
    loaddir = config["pretraineddir"]
    self_state = model.state_dict()
    loaded_state = torch.load(loaddir, map_location=config["device"]).state_dict()
    loaded_state = {k.replace('module.', ''): v for k, v in loaded_state.items() if
                    k.replace('module.', '') in self_state}

    self_state.update(loaded_state)
    model.load_state_dict(self_state)
    model = model.to(config["device"])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  eps=config["eps"], weight_decay=config["weight_decay"]
                                  )
    train_examples_len = len(dataset.train_dataset)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 5,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])

    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=pad_bert_custom_sequence)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=pad_bert_custom_sequence)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(data_loader_train, 0):
            node_sets = data[0]
            mask = data[1]
            label = data[2].to(config["device"])
            label = label.squeeze(-1)
            filename = data[3]
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(node_sets, mask)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            print("\r%f" % loss, end='')

            # print statistics
            tr_loss += loss.item()
            nb_tr_steps += 1
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().data.numpy().tolist())
        trainallscore = np.sum(np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1), axis=-1) / len(trainlabel)
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
                node_sets = data[0]
                mask = data[1]
                labels = data[2].to(config["device"])
                labels = labels.squeeze(-1)
                filename = data[3]
                outputs = model(node_sets, mask)
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
        out = {}
        for documentid in outpre.keys():
            author = documentid.split('_')[0]
            if out.get(author) is not None:
                pass
            else:
                out.update({author: {}})
            doc_id = documentid.split('_')[1]
            out[author].update({doc_id: {}})
            out[author][doc_id].update({'label': float(outpre[documentid]['label'])})
            out[author][doc_id].update({'predict': int(outpre[documentid]['predict'])})
            out[author][doc_id].update({'prob': outpre[documentid]['prob']})
        authors = out.keys()
        correct = 0
        outfinal = {}
        total = 0
        for author in authors:
            n_doc = len(out[author])
            lebel = int(out[author][list(out[author].keys())[0]]['label'])
            vote = 0
            for j in list(out[author].keys()):
                vote = vote + out[author][j]['predict']
            vote = vote / n_doc
            final_decision = np.round(vote)
            correct += (final_decision == lebel).sum()
            total = total + 1
            outfinal.update({author: {}})
            outfinal[author].update({'predict': int(final_decision)})
            outfinal[author].update({'label': int(lebel)})

        allscore = correct / total
        # evalacc = evalacc / len(evallabel)
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
            json.dump(outfinal, f, ensure_ascii=False, indent=4)
        with open(os.path.join(resultsdir, 'analyse_' + str(epoch) + '_' + str(evallossmean) + '_' + str(
                                        currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                        allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

        torch.cuda.empty_cache()
        if allscore < evalacc_best:
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



def trainBERT(datasetdir, savedir, modeltype, cachedir, ifgpu):
    losstype = modeltype.split('_')[1]
    modeldir = os.path.join(savedir, modeltype, 'model')
    resultsdir = os.path.join(savedir, modeltype, 'results')
    nsplit = 4
    max_len = 500
    overlap = 128
    max_num_epochs = 70
    #pretrainmodeldir = os.path.join('/PAN22/local', savedir, "External", 'model', 'BERTweet.pkl')
    pretrainmodeldir = os.path.join(savedir, "External", 'model', 'BERTweet.pkl')
    for makedir in [modeldir, resultsdir, cachedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    if ifgpu == 'true':
        device = "cuda"
    else:
        device = 'cpu'

    with open(os.path.join(datasetdir, "training_BERTweet.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)

    #traindict = mix_text(traindict)
    #testdict = mix_text(testdict)

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", cache_dir=cachedir)
    # split for cv
    authors = list(traindict.keys())
    random.shuffle(authors)
    # author_cvlist = np.array_split(authors, nsplit)
    # with open('cv', 'rb') as fp:
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
        config = {
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "eps": 1e-8,
            "lr": 2e-5,
            "nsplit": nsplit,
            "batch_size": 4,
            "modeldir": modelnewdir,
            "resultsdir": resultsnewdir,
            "tokenizer": tokenizer,
            "pretraineddir": pretrainmodeldir,
            "max_len": max_len,
            "overlap": overlap,
            "cachedir": cachedir,
            "losstype": losstype,
            "epochs": max_num_epochs,  # tune.choice([3, 5, 10, 15])
            "traindict": traindict,
            "cv": {"train_subsampler": train_author,
                   "dev_subsampler": dev_author}
        }
        training(config)







