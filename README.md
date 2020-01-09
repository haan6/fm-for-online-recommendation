# KAIST CCC-Project FM & Online FM for Recommendation Systems

This project studies the Factorization Machine model and its variations for recommendation systems.
Here, we consider the two problem settings for recommendation systems: 
(1) Off-line setting assumes that given sufficient amount of data, the model trains the given dataset and then 
conducts the recommendations tasks, (2) On-line setting assumes that as data points are given sequentially, the 
 model sequentially trains the data and recommends the items. 
 
What we have done for this projects can be summarized as : <br/>
(1) We study the previous models and construct them based on pytorch for  Off-line setting and On-line setting.  <br/>
(2) We try to propose Online Neural Factorization Machine. <br/>
(3) We propose the Meta Embedding Factorization machine designed for the specific dataset which 
    requires the large scale feature embeddings for some of the features.
   
For model detail, you can check in models and jupyters directory. 


## Models for Off-line setting
* Factorization Machine
* Meta Embedding Factorization Machine (proposed)
* Deep Factorization Machine
* Neural Factorization Machine
* Attentional Factorization Machine


## Models for On-line setting 
* Online Factorization Machine by FTRL, Sketched FTRL
* Online Deep Factorization Machine by SGD 
* Online Neural Factorization Machine by SGD
* Online Deep Factorization Machine by Hedge Backpropagation (proposed)
* Online Neural Factorization Machine by Hedge Backpropagation (proposed)



## Dataset

#### Real Valued Dataset <br/>
[1] movielens 100k - https://grouplens.org/datasets/movielens/  <br/>
[2] YearPredictionMSD - https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd/ <br/>
#### Discrete Valued Dataset  <br/>
[3] cod-rna - https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ <br/>
[4] Frappe - https://github.com/hexiangnan/neural_factorization_machine/tree/master/data/frappe/ <br/>
    
    
## Reference Paper


[1] Factorization Machines -S Rendle et al. <br/>
[2] Online Compact Convexified Factorization Machine -Wenpeng Zhang et al. <br/>
[3] Sketched Follow-The-Regularized-Leader for Online Factorization Machine -Luo Luo et al. <br/>
[4] Ad Click Prediction: a View from the Trenches -H. Brendan McMahan et al. <br/>
[5] Large-scale Online Kernel Learning with Random Feature Reparameterization -Tu Dinh Nguyen et al. <br/>



## Reference Code
[1] [PyTorch Implementations of Factorization Machines](https://github.com/nzc/dnn_ctr) <br/>
[2] [Factorization Machine Review Paper](https://github.com/rixwew/pytorch-fm) <br/>
[3] [A PyTorch Implementation of DeepFM for CTR Prediction Problem](https://github.com/chenxijun1029/DeepFM_with_PyTorch) <br/>
[4] [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://github.com/phquang/OnlineDeepLearning/tree/master/src) <br/>
[5] [Sketched Follow-The-Regularized-Leader for Online Factorization Machine](https://github.com/bmdy/SFTRL) <br/>
[6] [Large-scale Online Kernel Learning with Random Feature Reparameterization](https://github.com/tund/RRF ) 


## Funding
This research was supported by the Korean MSIT (Ministry of Science and ICT), under the National Program for Excellence in SW (2016-0-00018), supervised by the IITP (Institute for Information & communications Technology Planning&Evaluation)

## Author
Yohan Jung, Yeji Han
