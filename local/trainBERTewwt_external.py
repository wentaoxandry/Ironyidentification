import os
import json
import random
import numpy as np
from model import *
from utils.utils import *
import collections
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def training(config, dataset=None):
    criterion = torch.nn.CrossEntropyLoss()
    model = E2EBERTweet(config["cachedir"])
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    evalacc_best = 0
    early_wait = 4
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  eps=config["eps"], weight_decay=config["weight_decay"]
                                  )
    train_examples_len = len(dataset.train_dataset)
    print(len(dataset.train_dataset))
    print(len(dataset.test_dataset))
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 5,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])


    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_bert_custom_sequence)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True, drop_last=True, batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                    collate_fn=pad_bert_custom_sequence)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(data_loader_train, 0):
            node_sets = data[0]
            mask = data[1]
            label = data[2].to(config["device"])
            filename = data[3]
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(node_sets, mask)
            label = label.squeeze(-1)
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
            trainlabel.extend(label.cpu().detach().data.numpy().tolist())
            del loss, outputs, node_sets, mask, label
        trainallscore = np.sum(np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1), axis=-1) / len(
                trainlabel)
        # Validation loss
        print('evaluation')
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
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(len(filename)):
                    outpre.update({filename[i]: {}})
                    outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                    outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
            del dev_loss, outputs, node_sets, mask, labels

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
            json.dump(outpre, f, ensure_ascii=False, indent=4)

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
    netlist = os.listdir(os.path.join(modeldir))
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
    oldmodeldir = os.path.join(modeldir, netname)
    newmodeldir = os.path.join(modeldir, 'BERTweet.pkl')
    os.rename(oldmodeldir, newmodeldir)
    print("Finished Training")


def trainBERTextr(datasetdir, savedir, modeltype, cachedir, ifgpu):
    max_len = 500
    modeldir = os.path.join(savedir, modeltype, 'model')
    resultsdir = os.path.join(savedir, modeltype, 'results')
    max_num_epochs = 70
    for makedir in [modeldir, resultsdir, cachedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    if ifgpu == 'true':
        device = "cuda"
    else:
        device = 'cpu'

    with open(os.path.join(datasetdir, "training.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasetdir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", cache_dir=cachedir)
    dataset = BERTweetdatasetclass(train_file=traindict,
                                   test_file=testdict,
                                   tokenizer=tokenizer,
                                   device=device,
                                   max_len=max_len)

    config = {
        "NWORKER": 0,
        "device": device,
        "weight_decay": 0,
        "eps": 1e-8,
        "lr": 1e-5,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "batch_size": 16,
        "cachedir": cachedir,
        "epochs": max_num_epochs
    }
    training(config, dataset)




