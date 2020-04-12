# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:10:58 2018

@author: adm
"""

import time
from gensim.models import Word2Vec
import random
import numpy as np

from utils import sigmoid, softmax,  tanh
from weight_choice import weight_choice

class tGraphNE(object):
    
    def __init__(self, tG,  time_biased_type, first_biased_type, amount_biased, alpha,  dimensions, num_walks, walk_length, output, output_pklG = False, window_size=10, workers=8, hs=1):
        self.G = tG.G
        self.min_time = tG.min_time
        self.max_time = tG.max_time
        
        self.time_biased_type = time_biased_type # choice = "unbiased", "amount-weighted" "linear", "exp"
        self.first_biased_type = first_biased_type    
        self.amount_biased = amount_biased
        self.alpha = alpha
        print("Walking...")
        t1 = time.time()
        walks = self.simulate_walks(num_walks, walk_length)#随机游走
        t2 = time.time()
        print("  Walking time:", t2-t1)
        
        print("Learn embeddings...")   
        
        #walks = [map(str, walk) for walk in walks]        
        word2vec_model = Word2Vec(sentences = walks, size= dimensions, window= window_size, min_count=0, sg=1, hs=1, workers= workers)
        t3 = time.time()
        print("Learn embeddings time:", t3-t2) 
        
        self.vectors = {}
        for word in list(self.G.nodes()):
            self.vectors[str(word)] = word2vec_model.wv[str(word)]            

        print("  Embeddings are saved in ", output)
        word2vec_model.wv.save_word2vec_format(output)
        
        del word2vec_model
        
        return

###################################################################

    
    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        对每个结点，根据num_walks得出其多条随机游走路径
        
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.temporal_walk( walk_length = walk_length, start_node = node ))      
        return walks
            
    def temporal_walk(self, walk_length, start_node):        
        """
        功能： 从一个初始结点计算一个随机游走
        输入：
        walk_length: 随机游走序列长度
        start_node: 初始结点
        返回：
        列表，随机游走序列
        """
       
        walk = [start_node] #类型：list
        walk_edge = []
        walk_time = [] ##类型：list, 大小比walk的小1
        # walk_key = []
        
        cur = start_node
        next_node, next_time, next_key = self.get_first_step(cur)
        if next_node != None:
            walk.append(next_node)
            walk_time.append(next_time)
            walk_edge.append(next_key)
        else:
            return walk
        
        while len(walk) < walk_length:
            prevtime = walk_time[-1]
            cur = walk[-1]  #名为walk的list的最后一个元素，当前游走到的结点
            next_node, next_time, next_key = self.get_next_step(cur, prevtime)
            if next_node != None:
                walk.append(next_node)
                walk_time.append(next_time)
                walk_edge.append(next_key)
            else:
                break
        return walk
    
    def get_first_step(self, cur):
        G = self.G
        tmp_key = []
        tmp_node = []
        tmp_time = []
        unnormalized_probs_t = []
        
        cur_nbrs = list(G.neighbors(cur))
        if self.time_biased_type == "simple_graph": #DeepWalk
            for nbr in cur_nbrs:
                tmp_node.append(nbr)
                unnormalized_probs_t.append(1)                
                
            if len(unnormalized_probs_t) > 0:
                idx = weight_choice(unnormalized_probs_t)
                next_node = tmp_node[idx]
                next_time = 0 
                next_key = 0
                return next_node, next_time, next_key
            else:
                return None, None, None  #没有符合条件的   

        else:
            for nbr in cur_nbrs:
                nbr_key = list(G.get_edge_data(cur,nbr))    #cur领边的key数组       
                for k in nbr_key:
                    t = k             
                    if self.first_biased_type == "time_uniform":
                        unnormalized_probs_t.append(1)
                    elif self.first_biased_type == "time_freq":
                        unnormalized_probs_t.append(self.max_time-t+1)
                    elif self.first_biased_type == "time_close_linear":
                        unnormalized_probs_t.append(self.max_time-t+1)
                    elif self.first_biased_type == "time_far":
                        unnormalized_probs_t.append(t-self.min_time+1) 
                    elif self.first_biased_type == "time_far_linear":
                        unnormalized_probs_t.append(t)
                    
                    tmp_node.append(nbr)
                    tmp_time.append(t)
                    tmp_key.append(k)
                
            if self.first_biased_type == "time_close_linear" :
                unnormalized_probs_t = linear_rank_mapping( unnormalized_probs_t, order='descending' )
            elif self.first_biased_type == "time_far_linear":
                unnormalized_probs_t = linear_rank_mapping( unnormalized_probs_t )
            
    
            if len(unnormalized_probs_t) > 0: #有符合条件的下一个点
                selected = weight_choice(unnormalized_probs_t)               
                next_node = tmp_node[selected]
                next_time = tmp_time[selected]        
                next_key = tmp_key[selected]   
                return next_node, next_time, next_key
            
            else:
                return None, None, None  #没有符合条件的
        
        
    def get_next_step(self, cur, prevtime=0):
        """
        功能：给定一个当前随机游走到的结点cur，这个两个相连的结点（可能有多条边），得出
        输出：
        #return J, q
        直接输出下一个节点，以及时间戳
        """
        G = self.G
        
        tmp_key = []
        tmp_node = []
        tmp_time = []
        unnormalized_probs_t = []
        unnormalized_probs_a = []

        cur_nbrs = list(G.neighbors(cur))
        if self.time_biased_type == "simple_graph": #DeepWalk
            for nbr in cur_nbrs:
                tmp_node.append(nbr)
                unnormalized_probs_t.append(1)                
                
            if len(unnormalized_probs_t) > 0:
                idx = weight_choice(unnormalized_probs_t)
                next_node = tmp_node[idx]
                next_time = 0 
                next_key = 0
                return next_node, next_time, next_key
            else:
                return None, None, None  #没有符合条件的
        else:    
            for nbr in cur_nbrs:
                nbr_key = list(G.get_edge_data(cur,nbr))    #cur领边的key数组        
                for k in nbr_key:
                    t = k
                    a = G[cur][nbr][k]['weight']
                    if self.time_biased_type == "no_time_limit":
                        unnormalized_probs_t.append(1)
                    
                    elif t >= prevtime:
                        unnormalized_probs_a.append(a)
                        
                        if self.time_biased_type == "time_uniform"  :
                            unnormalized_probs_t.append(1)
                        elif self.time_biased_type == "time_close_raw"  :
                            unnormalized_probs_t.append( self.max_time - t + 1 )
                        elif self.time_biased_type == "time_close_exp"  :
                            unnormalized_probs_t.append( t - prevtime )
                        else:
                            unnormalized_probs_t.append( t - prevtime + 1 )
                        tmp_time.append(t)
                        tmp_node.append(nbr)
                        tmp_key.append(k)
                        

            if self.time_biased_type == "time_close_linear" :
                unnormalized_probs_t = linear_rank_mapping( unnormalized_probs_t, order='descending' )
            elif self.time_biased_type == "time_far_linear" :
                unnormalized_probs_t = linear_rank_mapping( unnormalized_probs_t)   
            elif self.time_biased_type == "time_freq_tanh":
                unnormalized_probs_t = tanh(unnormalized_probs_t)
            elif self.time_biased_type == "time_close_exp":
                unnormalized_probs_t = softmax(unnormalized_probs_t)

            if self.amount_biased == "amount_linear":
                unnormalized_probs_a = linear_rank_mapping(unnormalized_probs_a)
            elif self.amount_biased == "amount_tanh":
                unnormalized_probs_a = tanh(unnormalized_probs_a)
            elif self.amount_biased == "amount_exp":
                unnormalized_probs_a = softmax(unnormalized_probs_a)

            
            if len(unnormalized_probs_t) > 0: #有符合条件的下一个点
                if self.amount_biased != "amount_uniform":
                    unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha)        
                else:
                    unnormalized_probs = unnormalized_probs_t
                    
                selected = weight_choice(unnormalized_probs)               
                next_node = tmp_node[selected]  
                next_time = tmp_time[selected]        
                next_key = tmp_key[selected]   
                return next_node, next_time, next_key  
            
            else:
                return None, None, None  #没有符合条件的

    
def linear_rank_mapping( original_array, order='ascending' ):
    x = np.array(original_array)
    if order == 'ascending':
        return (np.argsort(x) + 1)
    elif order == 'descending':  
        return (np.argsort(-x) + 1)  
        # return (x.argsort() + 1)
        
def normalized_probs(unnormalized_probs):
    if len(unnormalized_probs) > 0: #有符合条件的下一个点
        norm_const = sum(unnormalized_probs)
        normalized_probs = [ u_prob / norm_const for u_prob in unnormalized_probs] #归一化  

    return normalized_probs    
    
def combine_probs(p1, p2, alpha) :
    probs1 = normalized_probs(p1)
    probs2 = normalized_probs(p2)

    if len(probs1) != len(probs2):
        print("ERROR", "len(probs1) != len(probs2)" )
                          
    combine_probs = np.multiply( np.power(probs1, alpha), np.power(probs2, 1-alpha) )        
    
    return combine_probs

   
    








    
    