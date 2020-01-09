import numpy as np
import torch
import time
import matplotlib.pyplot as plt


def regression_metric(pred,real):
    val = 0.0
    metric = [np.inf]
    for idx,(pred_idx,real_idx) in enumerate(zip(pred,real)):
        val += (pred_idx - real_idx)**2
        metric.append( (1/(idx + 1) )* val)
    metric = np.asarray(metric).reshape([-1,1])

    return metric


def classfication_metric(pred,real):
    val = 0.0
    metric = []
    metric_acc = []
    for idx,(pred_idx,real_idx) in enumerate(zip(pred,real)):
        val += 1 if pred_idx == real_idx else 0
        metric.append( 1/(idx + 1)* np.log( 1.0 + np.exp(-pred_idx*real_idx) )    )
        metric_acc.append( 1/(idx + 1)*val )
    metric = np.asarray(metric).reshape([-1,1])
    metric_acc = np.asarray(metric_acc).reshape([-1, 1])

    return metric,metric_acc
