# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:49:07 2019

@article{lin2020modeling,
   title={Modeling and Understanding Ethereum Transaction Records via A Complex Network Approach},
   author={Dan Lin, Jiajing Wu, Qi Yuan, Zibin Zheng},
   journal={IEEE Transactions on Circuits and Systems--II: Express Briefs },
   year={2020},
   month={to be published},
   publisher={IEEE},
   doi={10.1109/TCSII.2020.2968376}
}

For more datasets, please visit http://xblock.pro/ethereum/.

"""

from tGraphNE import tGraphNE
from tGraph import tGraph
from link_prediction import LP

# Graph construction
dataset = "LPsubG3"
file_ = dataset + "_df_train_0.5.pickle"
filetype = "pkl_f"

tG = tGraph(file_, filetype)
print(tG)

# Parameter setting
dimensions = 128
window_size =  4
workers = 8
num_walks = 1
walk_length = 20

#------------ Embedding -----------------#
first_biased_type = "time_uniform"  # time_uniform
time_biased_type = "time_uniform"
#  ["time_uniform", "time_close_linear", "time_close_raw", "time_far_linear", "time_close_exp", "time_freq_tanh", "time_close_exp"]
amount_biased = "amount_uniform"  #amount_linear, amount_uniform, amount_raw
alpha = 0.5
output = "../data/" + dataset + "/" + "vec_all_" + time_biased_type+ "_" + amount_biased + "_wl" + str(walk_length) + "_ws" + str(window_size) + "nw" + str(num_walks) + \
        "dim" + str(dimensions) +"_" + dataset + "_df_train.txt"
tGraphNE(tG, time_biased_type, first_biased_type, amount_biased, alpha, dimensions, num_walks, walk_length, output)

#------------ Link prediction -----------------#
edge_score_mode = "append"   #  [ "append", "multiply","subtract', "subtract_sigmoid"]
emb_file = output
LP(dataset, emb_file, edge_score_mode)
