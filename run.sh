#!/bin/bash -u


datasetdir=PAN22/pan22-author-profiling-training-2022-03-29/en		# The training data dir
evaldatasetdir=PAN22/pan22-author-profiling-test-2022-04-22-without_truth		# The test data dir
externaldatasetdir=external_dataset					# The external data source dir
datadir=dataset								# The dir where saves the processed data
externaldatadir=dataset/external_BERTweet				# The dir where saves the processed external data
savedir=save								# The dir saves the trained models and results
evalsavedir=eval							# The dir saves official test set results
finalsavedir=output							# The dir saves final official test set results	(after ensemble)
cachedir=CACHE								# The dir saves the downloaded BERT model 
ifgpu=true								# ifgpu is true, then using GPU for training
ifdebug=true								# ifdebug is true, when the test set haven't released. Split a part of training set as the test set	
start_stage=1								
stop_stage=100

if [ -d "$externaldatasetdir" ] ; then
    echo "external dataset is already exists"
else
    echo "Please download dataset from https://www.kaggle.com/datasets/rtatman/ironic-corpus?resource=download and https://github.com/omidrohanian/irony_detection/tree/master/datasets"
    # download https://github.com/omidrohanian/irony_detection/blob/master/datasets/train/SemEval2018-T3-train-taskA_emoji.txt, rename it as training_1.txt and save it under $externaldatasetdir
    # download https://github.com/omidrohanian/irony_detection/blob/master/datasets/test_TaskA/SemEval2018-T3_input_test_taskA_emoji.txt, rename it as test.txt and save it under $externaldatasetdir
    # download https://www.kaggle.com/datasets/rtatman/ironic-corpus?resource=download, rename it as training_2.txt and save it under $externaldatasetdir
fi

if [ -d "$datasetdir" ] ; then
    echo "PAN dataset is already exists"
else
    echo "Please download dataset from PAN dataset. Save it under $datasetdir"
fi

# train BERTweet model with external dataset
modeltype=External
python3 local/run_BERTweet.py $externaldatasetdir \
			$externaldatadir \
			$savedir $modeltype \
			$cachedir $ifdebug \
			$ifgpu $start_stage \
			$stop_stage


# train BERTweet model with different loss functions on PAN dataset
for modeltype in BERT_CE BERT_CEWF BERT_Focal; do
python3 local/run_BERTweet.py $datasetdir \
			$datadir \
			$savedir $modeltype \
			$cachedir $ifdebug \
			$ifgpu $start_stage \
			$stop_stage
# evaluate model on PAN official test set
python3 ./local/eval_BERTweet.py $evaldatasetdir $evalsavedir $modeltype 
done

# train BERTweet feature-based CNN model on PAN dataset
modeltype=BERT_CONV
python3 local/run_BERTweet.py $datasetdir \
			$datadir \
			$savedir $modeltype \
			$cachedir $ifdebug \
			$ifgpu $start_stage \
			$stop_stage
# evaluate model on PAN official test set
python3 ./local/eval_BERTweet.py $evaldatasetdir $evalsavedir $modeltype 



# if you want to ensemble single model across different fold:
modeltype=BERT_CEWF
python3 local/voting.py $evalsavedir/$modeltype $finalsavedir

# if you want to ensemble multi models across different fold:
python3 local/ensemble.py $evalsavedir $finalsavedir



