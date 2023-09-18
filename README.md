# KAIST CCC-Project FM & Online FM for Recommendation Systems

In this project, we study factorization machine and its variations for recommendation systems.
Here, we consider the two problem settings for recommendation systems: 
(1) Offline setting that assumes that given sufficient amount of data, the model trains the given dataset and then 
conducts the recommendations tasks, (2) Online setting that assumes that as data points are given sequentially, the 
 model sequentially trains the data and recommends the items. 
 
What we have done for this project can be summarized as : <br/>
(1) We study the previous models and implement them based on pytorch for offline setting and online setting.  <br/>
(2) We try to propose Online Neural Factorization Machine. <br/>
(3) We propose Meta Embedding Factorization machine designed for the specific dataset which 
    requires the large number of feature embeddings for some of the features.
   
For more details, you can check it in models and jupyters directory. 


## Models for Offline Setting
* Factorization Machine
* Meta Embedding Factorization Machine (proposed)
* Deep Factorization Machine
* Neural Factorization Machine
* Attentional Factorization Machine


## Models for Online Setting 
* Online Factorization Machine by FTRL, Sketched FTRL
* Online Deep Factorization Machine by SGD 
* Online Neural Factorization Machine by SGD
* Online Deep Factorization Machine by Hedge Backpropagation (proposed)
* Online Neural Factorization Machine by Hedge Backpropagation (proposed)


## Dataset

#### Real Valued Dataset <br/>
[1] [movielens 100k](https://grouplens.org/datasets/movielens/)  <br/>
[2] [YearPredictionMSD](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd/) <br/>
#### Discrete Valued Dataset  <br/>
[3] [cod-rna](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)   <br/>
[4] [Frappe](https://github.com/hexiangnan/neural_factorization_machine/tree/master/data/frappe/) <br/>
[5] [Criteo](https://github.com/nzc/dnn_ctr/tree/master/data) <br/>
    

## Reference Code
[1] [PyTorch Implementations of Factorization Machines](https://github.com/nzc/dnn_ctr) <br/>
[2] [Factorization Machine Review Paper](https://github.com/rixwew/pytorch-fm) <br/>
[3] [A PyTorch Implementation of DeepFM for CTR Prediction Problem](https://github.com/chenxijun1029/DeepFM_with_PyTorch) <br/>
[4] [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://github.com/phquang/OnlineDeepLearning/tree/master/src) <br/>
[5] [Sketched Follow-The-Regularized-Leader for Online Factorization Machine](https://github.com/bmdy/SFTRL) <br/>
[6] [Large-scale Online Kernel Learning with Random Feature Reparameterization](https://github.com/tund/RRF ) 


## Funding
This research was supported by the Korean MSIT (Ministry of Science and ICT), under the National Program for Excellence in SW (2016-0-00018), supervised by the IITP (Institute for Information communication Technology Planning and Evaluation)


## Author
Yohan Jung (becre1776@kaist.ac.kr) <br/>
Yeji Han (yejihan@kaist.ac.kr)
