# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:39:49 2019

@author: adm
"""
import random
import numpy as np

#random.seed(32)
#np.random.seed(32)

def weight_choice(unnormalized_probs): 
    if len(unnormalized_probs) > 0: #有符合条件的下一个点
        norm_const = sum(unnormalized_probs)
        normalized_probs = [ float(u_prob / norm_const) for u_prob in unnormalized_probs] #归一化
    
        J = alias_setup(normalized_probs)[0]
        q = alias_setup(normalized_probs)[1]
        idx = alias_draw(J, q)        
    return idx


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
