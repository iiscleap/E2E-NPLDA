#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:36:02 2020

@author: prashantk, shreyasr
"""


import numpy as np
import kaldi_io
import os
from pdb import set_trace as bp
from tqdm import tqdm
import sys


path_to_write = '/home/data2/shreyasr/voxceleb/v1/data/sre18_eval_combined_no_sil/arks'
scpfile = sys.argv[1]
# scpfile = '/home/data2/SRE2019/prashantk/voxceleb/v1/exp/train_16k_combined_no_sil/xvector_feats_train_16k_combined.1.scp'
num_files = 0
with open(scpfile,'r') as f:
    scp_list = f.readlines()
    scp_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
for key in tqdm(scp_dict.keys()):
    mat = kaldi_io.read_mat(scp_dict[key])
    if mat.shape[0] >= 2000:
        for i in range((mat.shape[0]//2000)):
            key_to_write = key + "_20s_" + str(i)
            strt_idx = i*2000
            fin_idx = (i+1)*2000
            nxt=i+1
            mat_to_write = mat[strt_idx:fin_idx,:]
            with open(os.path.join(path_to_write, key_to_write+".ark"),'wb') as f:
                kaldi_io.write_mat(f, mat_to_write, key=key_to_write)
            num_files+=1
        res = mat[fin_idx:,:]
    else:
        res=mat
        nxt=0
        
    # Write the remaining duration of mfcc if dur > 10 seconds
    if res.shape[0] > 1000:
        mat_to_write = res[:1000]
        key_to_write = key + "_10s_" + str(nxt)
        nxt = nxt+1
        with open(os.path.join(path_to_write, key_to_write+".ark"),'wb') as f:
            kaldi_io.write_mat(f, mat_to_write, key=key_to_write)
        num_files+=1
        res = res[1000:]
    if res.shape[0] > 500:
        mat_to_write = res[:500]
        key_to_write = key + "_5s_" + str(nxt)
        with open(os.path.join(path_to_write, key_to_write+".ark"),'wb') as f:
            kaldi_io.write_mat(f, mat_to_write, key=key_to_write)
        num_files+=1
            
        

print("Number of files : ", num_files)
print("DONE")