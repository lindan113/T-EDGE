# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 13:07:39 2019

@author: adm
"""

import numpy as np
from utils import sigmoid, softmax,  tanh

def get_edge_embeddings(edge_list, emb_matrix, edge_score_mode ):
    
    embs = []
    
    if edge_score_mode == "multiply":        
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = np.array(emb_matrix[str(node1)])
            emb2 = np.array(emb_matrix[str(node2)])
            edge_emb = np.multiply(emb1, emb2)
            embs.append(edge_emb)
    
    elif edge_score_mode == "subtract":   
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = np.array(emb_matrix[str(node1)])
            emb2 = np.array(emb_matrix[str(node2)])
            edge_emb = np.subtract(emb1, emb2)
            embs.append(edge_emb)
            
    elif edge_score_mode == "subtract_sigmoid":   
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = np.array(emb_matrix[str(node1)])
            emb2 = np.array(emb_matrix[str(node2)])
            edge_emb = sigmoid(np.subtract(emb1, emb2))
            embs.append(edge_emb)         
            
    elif edge_score_mode == "subtract_tanh":   
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = np.array(emb_matrix[str(node1)])
            emb2 = np.array(emb_matrix[str(node2)])
            edge_emb = tanh(np.subtract(emb1, emb2))
            embs.append(edge_emb)         
                  
    elif edge_score_mode == "append":   
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = np.array(emb_matrix[str(node1)])
            emb2 = np.array(emb_matrix[str(node2)])
            edge_emb = np.append(emb1, emb2)
            embs.append(edge_emb)
    else:
        print("ERROR!!! No suitable edge_score_mode")
        
    embs = np.array(embs)
    embs = np.array(embs)
    return embs   


   