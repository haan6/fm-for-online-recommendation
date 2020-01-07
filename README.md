# SILAB-Online-Recommendation

This package has code for simulating and performancing inference in a variety of factorization machines.

## Online Models
Online models take single data as an input, make a prediction, and train with the data.
* Online Factorization Machine
* Online Deep Factorization Machine
* Online Neural Factorization Machine
* Online Deep Factorization Machine with Hedge Backpropagation
* Online Neural Factorization Machine with Hedge Backpropagation


## Naive Models
Naive models take a batch of data as an input and train with the data. After traversing all the batches, naive models 
* Factorization Machine
* Deep Factorization Machine
* Neural Factorization Machine
* Attentional Factorization Machine


## Reference
[1](https://github.com/nzc/dnn_ctr) PyTorch Implementations of Factorization Machines
[2](https://github.com/rixwew/pytorch-fm) Factorization Machine Review Paper
[3](https://github.com/chenxijun1029/DeepFM_with_PyTorch) A PyTorch implementation of DeepFM for CTR prediction problem.
[4](https://github.com/phquang/OnlineDeepLearning/tree/master/src) Online Deep Learning: Learning Deep Neural Networks on the Fly