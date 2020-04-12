# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:11:02 2019

@author: adm
"""

import sys
import networkx as nx

import pickle as pickle

class tGraph(object):
    def __init__( self, file_, filetype, output_Gpkl=False ):
        self.G = nx.MultiDiGraph() 
        self.min_time = sys.maxsize
        self.max_time = 0
        self.min_amount = sys.maxsize
        self.max_amount = 0
        
        print("Loading file", file_, "...")
        
        edge_key = 0
        
        if filetype == "txt_raw":
            with open(file_) as f:
                for l in f:                    
                    x, y = l.strip().split(' ')
                    self.G.add_edge(x,y,key=edge_key)                    
                    edge_key = edge_key + 1
        else:            
            if filetype == "txt_f":
                with open(file_) as f:
                    for l in f:                    
                        x, y, a, t = l.strip().split(',')
                        a = float(a)
                        t = int(t)                       
                        if self.G.has_edge(x,y,t):
                            if self.G[x][y][t]['weight'] != a:
                                self.G[x][y][t]['weight'] += a
                        else:    
                            self.G.add_edge(x,y,key=t, weight=a)     
                        edge_key = edge_key + 1 
                                              
                        if t < self.min_time:
                            self.min_time = t
                        elif t > self.max_time:
                            self.max_time = t                            

            elif filetype == "pkl_f":
                with open( file_,"rb") as f: 
                    df_in = pickle.load(f)            
                for i in df_in.index:
                    x = str(int(df_in.From[i]))
                    y = str(int(df_in.To[i]))
                    t = int(df_in.TimeStamp[i])
                    a = df_in.Value[i]
                    
                    if self.G.has_edge(x,y,t):
                        self.G[x][y][t]['weight'] += a                        
                    else:    
                        self.G.add_edge(x,y,key=t, weight=a)     
                        
                    edge_key = edge_key + 1                     
                    if t < self.min_time:
                        self.min_time = t
                    elif t > self.max_time:
                        self.max_time = t    

            if output_Gpkl == True:            
                pklfile_G = "tGraph.pickle"
                with open(pklfile_G, "wb") as f:
                    print("Writing", pklfile_G, "...")
                    pickle.dump( self.G, pklfile_G, pickle.HIGHEST_PROTOCOL )

        self.number_of_nodes = self.G.number_of_nodes()
        self.number_of_edges = self.G.number_of_edges()
        print("Summary of graph:")
        print("Number of nodes: ", self.number_of_nodes)
        print("Number of edges: ", self.number_of_edges)
        print("Number of edge_key: ", edge_key)
        print("Min time: ", self.min_time) 
        print("Max time: ", self.max_time) 
