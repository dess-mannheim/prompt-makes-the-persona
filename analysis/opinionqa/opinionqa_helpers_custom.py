import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import random



def get_probs(confs, separate_refusal = False):
    if separate_refusal:
        all_remaining_options = sorted(list(confs.keys()))[:-1]
        refusal = sorted(list(confs.keys()))[-1]
        ref_prob = np.exp(confs[refusal])
        confs = {k:confs[k] for k in all_remaining_options}
        a = list(confs.values())
        probs = np.exp(a) / (np.exp(a)).sum()
        return probs, ref_prob
    
    a = list(confs.values())
    probs = np.exp(a) / (np.exp(a)).sum()
    return probs

def compute_match(human_df_subset, model_df_subset = False, baseline =  False):
    alignment = []
    if not baseline:
        all_qkeys = set(model_df_subset['qa_key'].unique())
        all_qkeys = list(all_qkeys.intersection(human_df_subset['qkey'].unique()))
        for qkey in all_qkeys:
            a1 = human_df_subset[human_df_subset['qkey'] == qkey]['A_H'].values[0]
            a2 = model_df_subset[model_df_subset['qa_key'] == qkey]['A_M'].values[0]
            if a1 == a2:
                alignment.append(1)
            else:
                alignment.append(0)
    else:
        ## upper bound for majority match
        all_qkeys = human_df_subset['qkey'].unique()
        for qkey in all_qkeys:
            d1 = human_df_subset[human_df_subset['qkey'] == qkey]['D_H'].values[0]
            a1 = human_df_subset[human_df_subset['qkey'] == qkey]['A_H'].values[0]
            a2 = random.randint(0, len(d1))
            if a1 == a2:
                alignment.append(1)
            else:
                alignment.append(0)
        
    return(np.sum(alignment) / len(all_qkeys))

def compute_alignment(human_df_subset, model_df_subset = False, baseline =  False):
    alignment = []
    if not baseline:
        all_qkeys = set(model_df_subset['qa_key'].unique())
        all_qkeys = list(all_qkeys.intersection(human_df_subset['qkey'].unique()))
        for qkey in all_qkeys:
            d1 = model_df_subset[model_df_subset['qa_key'] == qkey]['D_M'].values[0]
            d2 = human_df_subset[human_df_subset['qkey'] == qkey]['D_H'].values[0]
            wd_m = 1 - (wasserstein_distance(d1, d2) / (len(d1)-1)) # this metric is a bit sketchy
            wd_m = wasserstein_distance(d1, d2) # used in https://arxiv.org/pdf/2502.16761
            alignment.append(wd_m)
    else:
        ## upper bound for wd
        all_qkeys = human_df_subset['qkey'].unique()
        for qkey in all_qkeys:
            d2 = human_df_subset[human_df_subset['qkey'] == qkey]['D_H'].values[0]
            d1 = np.random.dirichlet(np.ones(len(d2)-1)).tolist() # random
            wd_m = 1 - (wasserstein_distance(d1, d2) / (len(d1)-1)) # this metric is a bit sketchy
            wd_m = wasserstein_distance(d1, d2) # used in https://arxiv.org/pdf/2502.16761
            alignment.append(wd_m)
        
    
    return(np.mean(alignment))
    
