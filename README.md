# Ironyidentification
Implementation of our Profiling Irony and Stereotype Spreaders on Twitter (IROSTEREO) at PAN 2022 <a href="https://pan.webis.de/clef22/pan22-web/author-profiling.html#evaluation">challenge</a>.  We train the BERTweet and a combination of BERTweet with CNN models with segmentation strategy. Three different loss functions are applied to deal with the unbalanced produced by the segmenting strategy. Soft-voting methods help us obtain the best results. Our model achieves the best performance in the IROSTEREO challenge.

### How to run the code
* Please follow ./run.sh download all necessary dataset and run the code.

# Acknowledgement 
The work was supported by the PhD School ”SecHuman - Security for Humans in Cyberspace” by the federal state of NRW, and partially funded by the Deutsche Forschungsgemeinschaft (DFG–German Research Foundation) [Project-ID 429873205] and by the German Federal Ministry of Education and Research [Grant No: 16KIS1518K]. 

# Cite as 
@article{yu2022bert,
  title={BERT-based ironic authors profiling},
  author={Yu, Wentao and Boenninghoff, Benedikt and Kolossa, Dorothea},
  year={2022}
}



