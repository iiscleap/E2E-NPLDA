#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:15:48 2020

@author: shreyasr
"""

import re
import numpy as np
import random
import sys
import subprocess
import pickle
import os
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
import kaldi_io
from pdb import set_trace as bp

def list_split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

class TrialSampler:
    def __init__(self, spk2utt_file, batch_size, num_spks_per_batch, mega_scp_dict, id_to_num_dict):
        with open(spk2utt_file) as f:
            self.spk2utt_list = f.readlines()
        random.shuffle(self.spk2utt_list)
        self.batch_size = batch_size
        self.num_spks_per_batch = num_spks_per_batch
        self.num_utts_per_spk = batch_size/num_spks_per_batch
        self.n_repeats_tgt = int(0.7*(self.num_utts_per_spk - 1))
        self. n_repeats_imp = int(0.7 * self.num_utts_per_spk * (self.num_spks_per_batch - 1))
        self.mega_scp_dict = mega_scp_dict
        self.id_to_num_dict = id_to_num_dict
        self.spk2utt_dict = {}
        
    def spk2utt_dict_from_list(self):
        for x in self.spk2utt_list:
            a = np.asarray(x.rstrip('\n').split(' ', 1)[1].split(' '))
            random.shuffle(a)
            self.spk2utt_dict[x.split(' ', 1)[0]] = np.array_split(a, np.ceil(a.shape[0] / self.num_utts_per_spk))
            
    def check_spk2utt_dict(self):
        keys_to_remove=[]
        for k, v in self.spk2utt_dict.items():
            if len(self.spk2utt_dict[k]) == 0:
                keys_to_remove.append(k)
        if len(keys_to_remove) > 0:
            for k in keys_to_remove:
                del self.spk2utt_dict[k]
        return len(self.spk2utt_dict.keys()) > self.num_spks_per_batch
    
    def get_batch(self):
        spk2utt_keys = list(self.spk2utt_dict.keys())
        random.shuffle(spk2utt_keys)
        keys_to_sample = spk2utt_keys[:self.num_spks_per_batch]
        sampled_keys_utts_per_spk = []
        diff_speaker_spk2utt_dict = {}
        for x in keys_to_sample:
            sampled_keys_utts_per_spk.append(self.spk2utt_dict[x][0])
            diff_speaker_spk2utt_dict[x] = self.spk2utt_dict[x][0]
            del self.spk2utt_dict[x][0]
        t1, t2 = same_speaker_list(sampled_keys_utts_per_spk, self.mega_scp_dict, self.id_to_num_dict, n_repeats=self.n_repeats_tgt)
        nt1, nt2 = diff_speaker_list(diff_speaker_spk2utt_dict, self.mega_scp_dict, self.id_to_num_dict, n_repeats=self.n_repeats_imp)
        targets = torch.ones(t1.size())
        non_targets = torch.zeros(nt1.size())
        d1, d2, labels = torch.cat((t1,nt1)).float(), torch.cat((t2,nt2)).float(), torch.cat((targets,non_targets)).float()
        return d1, d2, labels

        
    def load_epoch(self):
        self.spk2utt_dict_from_list()
        epoch_data = []
        while self.check_spk2utt_dict():
            epoch_data.append(self.get_batch())
        return epoch_data
    

    
class TrialSampler2:
    def __init__(self, spk2utt_file, tot_batch_dur, num_spks_per_batch, mega_scp_dict, id_to_num_dict, utt2dur):
        with open(spk2utt_file) as f:
            self.spk2utt_list = f.readlines()
        random.shuffle(self.spk2utt_list)
        tmp = np.genfromtxt(utt2dur, dtype='str')
        self.utt2dur = dict(zip(tmp[:,0], tmp[:,1].astype(int)))
        self.uniq_durs = np.unique(list(self.utt2dur.values()))
        self.tot_batch_dur = tot_batch_dur
        self.num_spks_per_batch = num_spks_per_batch
        self.tot_dur_per_spk = tot_batch_dur/num_spks_per_batch
        
        self.mega_scp_dict = mega_scp_dict
        self.id_to_num_dict = id_to_num_dict
        self.spk2utt_dict = {}
        

    def spk2utt_dict_from_list(self):
        for x in self.spk2utt_list:
            a = x.rstrip('\n').split(' ', 1)[1].split(' ')
            random.shuffle(a)
            self.spk2utt_dict[x.split(' ', 1)[0]] = a
            
    def check_spk2utt_dicts(self):
        keys_to_remove=[]
        for k, v in self.spk2utt_dict.items():
            if len(self.spk2utt_dict[k]) < self.tot_dur_per_spk/max(self.uniq_durs):
                keys_to_remove.append(k)
        if len(keys_to_remove) > 0:
            for k in keys_to_remove:
                del self.spk2utt_dict[k]
        return len(self.spk2utt_dict.keys()) > self.num_spks_per_batch
    
    def get_batch(self):
        spk2utt_keys = list(self.spk2utt_dict.keys())
        random.shuffle(spk2utt_keys)
        keys_to_sample = spk2utt_keys[:self.num_spks_per_batch]
        batch_spk2utt_enroll = {}
        batch_spk2utt_test = {}
        for x in keys_to_sample:
            utts = self.load_utts(x)
            batch_spk2utt_enroll[x], batch_spk2utt_test[x] = np.array_split(utts,2)
        d1, d2, labels = self.make_all_trials(batch_spk2utt_enroll, batch_spk2utt_test)
        return d1, d2, labels
    
    def load_utts(self, speaker):
        tot_dur = 0.
        utts = []
        while(True):
            if self.spk2utt_dict[speaker]:
                utt = self.spk2utt_dict[speaker][0]
                del self.spk2utt_dict[speaker][0]
            else:
                break
            if tot_dur + self.utt2dur[utt] > self.tot_dur_per_spk:
                break
            tot_dur += self.utt2dur[utt]
            utts.append(utt)
        return utts
    
    def make_all_trials(self, spk2utt_enroll, spk2utt_test):
        d1, d2, labels = [], [], []
        spks = list(spk2utt_enroll.keys())
        for spk in spks:
            for enr in spk2utt_enroll[spk]:
                for tst in spk2utt_test[spk]:
                    d1.append(self.id_to_num_dict[enr])
                    d2.append(self.id_to_num_dict[tst])
                    labels.append(1)
        for spk in spks:
            spks.remove(spk)
            for enr in spk2utt_enroll[spk]:
                for imp in spks:
                    for tst in spk2utt_test[imp]:
                        d1.append(self.id_to_num_dict[enr])
                        d2.append(self.id_to_num_dict[tst])
                        labels.append(0)
        d1, d2, labels = torch.tensor(d1).float(), torch.tensor(d2).float(), torch.tensor(labels).float()
        return d1, d2, labels

    def load_epoch(self):
        self.spk2utt_dict_from_list()
        epoch_data = []
        while self.check_spk2utt_dicts():
            epoch_data.append(self.get_batch())
        return epoch_data
    
class TrialSampler3:
    def __init__(self, spk2utt_file, tot_batch_dur, num_spks_per_batch, mega_scp_dict, id_to_num_dict, utt2dur, enroll_durs=[20,10]):
        with open(spk2utt_file) as f:
            self.spk2utt_list = f.readlines()
        random.shuffle(self.spk2utt_list)
        tmp = np.genfromtxt(utt2dur, dtype='str')
        self.utt2dur = dict(zip(tmp[:,0], tmp[:,1].astype(int)))
        self.tot_batch_dur = tot_batch_dur
        self.num_spks_per_batch = num_spks_per_batch
        self.tot_dur_per_spk = tot_batch_dur/num_spks_per_batch
        
        self.mega_scp_dict = mega_scp_dict
        self.id_to_num_dict = id_to_num_dict
        self.enroll_spk2utt_dict = {}
        self.spk2utt_dict = {}
        self.enroll_durs = enroll_durs
        
    
    def enroll_spk2utt_dict_from_list(self):
        for x in self.spk2utt_list:
            a = x.rstrip('\n').split(' ', 1)[1].split(' ')
            a = np.asarray([y for y in a if self.utt2dur[y] in self.enroll_durs])
            random.shuffle(a)
            self.enroll_spk2utt_dict[x.split(' ', 1)[0]] = a

    def spk2utt_dict_from_list(self):
        for x in self.spk2utt_list:
            a = np.asarray(x.rstrip('\n').split(' ', 1)[1].split(' '))
            random.shuffle(a)
            self.spk2utt_dict[x.split(' ', 1)[0]] = a
            
    def check_spk2utt_dicts(self):
        keys_to_remove=[]
        for k, v in self.spk2utt_dict.items():
            if len(self.spk2utt_dict[k]) < self.tot_dur_per_spk/max(self.enroll_durs):
                keys_to_remove.append(k)
        if len(keys_to_remove) > 0:
            for k in keys_to_remove:
                del self.spk2utt_dict[k]
        return len(self.spk2utt_dict.keys()) > self.num_spks_per_batch
    
    def get_batch(self):
        spk2utt_keys = list(self.spk2utt_dict.keys())
        random.shuffle(spk2utt_keys)
        keys_to_sample = spk2utt_keys[:self.num_spks_per_batch]
        batch_spk2utt_enroll = {}
        batch_spk2utt_test = {}
        for x in keys_to_sample:
            batch_spk2utt_enroll[x] = self.load_enroll_utts(self, x)
            batch_spk2utt_test[x] = self.load_test_utts(self, x)
        d1, d2, labels = self.make_all_trials(batch_spk2utt_enroll, batch_spk2utt_test)
        return d1, d2, labels
    
    def load_enroll_utts(self, speaker):
        pass
            
    
    def load_test_utts(self, speaker):
        pass
        
    def load_epoch(self):
        self.spk2utt_dict_from_list()
        epoch_data = []
        while self.check_spk2utt_dict():
            epoch_data.append(self.get_batch())
        return epoch_data


class kaldi_ark_reader_faster:
    def __init__(self):
        self.fp_dict = {}
        
    def get_fp(self, arkpointer):
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', arkpointer):
            (prefix,arkpointer) = arkpointer.split(':',1)
        if re.search(':[0-9]+$', arkpointer):
            (file,offset) = arkpointer.rsplit(':',1)
        else:
            return arkpointer
        if file not in self.fp_dict.keys():
            self.fp_dict[file] = open(file, 'rb')
        fd = self.fp_dict[file]
        fd.seek(int(offset))
        return fd
    
    def close_all(self):
        for fd in self.fp_dict.values():
            fd.close()


def custom_loader_e2e(data_spk2utt_list, mega_scp_dict, id_to_num_dict, batch_size=64, num_spks_per_batch=4):
    mega_loader = []
    if type(data_spk2utt_list) == str:
        data_spk2utt_list = np.genfromtxt(data_spk2utt_list, dtype='str')
    if data_spk2utt_list.ndim == 2:
        data_spk2utt_list = data_spk2utt_list[:, 0]
    else:
        raise("Something wrong here.")
    for spk2utt_file in data_spk2utt_list:
        ts = TrialSampler(spk2utt_file, batch_size, num_spks_per_batch, mega_scp_dict, id_to_num_dict)
        mega_loader.extend(ts.load_epoch())
    random.shuffle(mega_loader)
    return mega_loader

def custom_loader_e2e_v2(nc, mega_scp_dict, id_to_num_dict):
    mega_loader = []
    for spk2utt_file in nc.train_spk2utt_list:
        for num_spks_per_batch in range(nc.min_num_spks_per_batch, nc.max_num_spks_per_batch+1):
            ts = TrialSampler(spk2utt_file, nc.batch_size, num_spks_per_batch, mega_scp_dict, id_to_num_dict)
            mega_loader.extend(ts.load_epoch())
    random.shuffle(mega_loader)
    return mega_loader

def custom_loader_e2e_v3(nc, mega_scp_dict, id_to_num_dict, n_repeats=1, n_subepochs=20):
    mega_loaders = []
    for i in range(n_repeats):
        mega_loader = []
        for spk2utt_file in nc.train_spk2utt_list:
            for num_spks_per_batch in range(nc.min_num_spks_per_batch, nc.max_num_spks_per_batch+1):
                ts = TrialSampler2(spk2utt_file, nc.total_batch_duration, num_spks_per_batch, mega_scp_dict, id_to_num_dict, nc.train_utt2dur)
                mega_loader.extend(ts.load_epoch())
        random.shuffle(mega_loader)
        mega_loaders.extend(mega_loader)
    mega_loaders = list_split(mega_loaders, n_subepochs)
    return mega_loaders
       
def same_speaker_list(utts_per_spk, combined_scp_dict, id_to_num_dict, n_repeats=1):
    d1,d2 = [], []
    for repeats in range(n_repeats):
        for utts in utts_per_spk:
            utts_shuffled = list(utts.copy())
            random.shuffle(utts_shuffled)
            while len(utts_shuffled) >= 2:
                tmp1 = utts_shuffled.pop()
                if tmp1 not in combined_scp_dict:
                    continue
                tmp2 = utts_shuffled.pop()
                if tmp2 not in combined_scp_dict:
                    continue
                d1.append(id_to_num_dict[tmp1])
                d2.append(id_to_num_dict[tmp2])
    d1, d2 = torch.tensor(d1), torch.tensor(d2)
    return d1, d2


def diff_speaker_list(spk2utt_dict, combined_scp_dict, id_to_num_dict, n_repeats=1):
    spk2utt_keys = list(spk2utt_dict.keys())
    utt2spk = []
    for i in spk2utt_keys:
        for j in spk2utt_dict[i]:
            utt2spk.append([j, i])
    d1, d2 = [], []
    for repeats in range(n_repeats):
        utt2spk_list = list(utt2spk)
        random.shuffle(utt2spk_list)
        i = 0
        while len(utt2spk_list) >= 2:
            if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                tmp1 = utt2spk_list.pop()
                if list(tmp1)[0] not in combined_scp_dict:
                    continue
                tmp2 = utt2spk_list.pop()
                if list(tmp2)[0] not in combined_scp_dict:
                    continue
                d1.append(id_to_num_dict[list(tmp1)[0]])
                d2.append(id_to_num_dict[list(tmp2)[0]])
                i = 0
            else:
                i = i + 1
                random.shuffle(utt2spk_list)
                if i == 50:
                    break
    d1, d2 = torch.tensor(d1), torch.tensor(d2)
    return d1, d2
        

def make_same_speaker_list(spk2utt_file, combined_scp_dict, n_repeats=1, train_and_valid=False,train_ratio=0.95):
    # print("In same speaker list")
    assert train_ratio < 1, "train_ratio should be less than 1."
    with open(spk2utt_file) as f:
        spk2utt_list = f.readlines()
    random.seed(2)
    random.shuffle(spk2utt_list)
    uttsperspk = [(a.rstrip('\n').split(' ', 1)[1]).split(' ') for a in spk2utt_list]
    
    train_uttsperspk = uttsperspk[:int(train_ratio * len(uttsperspk))]
    train_same_speaker_list = []
    for repeats in range(n_repeats):
        for utts in train_uttsperspk:
            utts_shuffled = utts.copy()
            random.shuffle(utts_shuffled)
            while len(utts_shuffled) >= 2:
                tmp1 = utts_shuffled.pop()
                if tmp1 not in combined_scp_dict:
                    continue
                tmp2 = utts_shuffled.pop()
                if tmp2 not in combined_scp_dict:
                    continue
                train_same_speaker_list.append([tmp1, tmp2])
    train_same_speaker_list = np.asarray(train_same_speaker_list)
    
    valid_uttsperspk = uttsperspk[int((train_ratio) * len(uttsperspk)):]
    valid_same_speaker_list = []
    for repeats in range(n_repeats):
        for utts in valid_uttsperspk:
            utts_shuffled = utts.copy()
            random.shuffle(utts_shuffled)
            while len(utts_shuffled) >= 2:
                tmp1 = utts_shuffled.pop()
                if tmp1 not in combined_scp_dict:
                    continue
                tmp2 = utts_shuffled.pop()
                if tmp2 not in combined_scp_dict:
                    continue
                valid_same_speaker_list.append([tmp1, tmp2])
    valid_same_speaker_list = np.asarray(valid_same_speaker_list)

    return train_same_speaker_list, valid_same_speaker_list

    if train_and_valid:  # Returns two lists for training and validation
        return train_same_speaker_list, valid_same_speaker_list
    else:
        return train_same_speaker_list + valid_same_speaker_list


def make_diff_speaker_list(spk2utt_file, combined_scp_dict, n_repeats=1, train_and_valid=True, train_ratio=0.95):
    # print("In diff speaker list")
    assert train_ratio < 1, "train_ratio should be less than 1."
    with open(spk2utt_file) as f:
        spk2utt_list = f.readlines()
    random.seed(2)
    random.shuffle(spk2utt_list)
    spk2utt_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1].split(' ') for x in spk2utt_list}
    spk2utt_keys = list(spk2utt_dict.keys())
    train_keys = spk2utt_keys[:int(train_ratio * len(spk2utt_keys))]
    valid_keys = spk2utt_keys[int(train_ratio * len(spk2utt_keys)):]
    utt2spk_train = []
    utt2spk_valid = []
    for i in train_keys:
        for j in spk2utt_dict[i]:
            utt2spk_train.append([j, i])
    for i in valid_keys:
        for j in spk2utt_dict[i]:
            utt2spk_valid.append([j, i])

    train_diff_speaker_list = []
    for repeats in range(n_repeats):
        utt2spk_list = list(utt2spk_train)
        random.shuffle(utt2spk_list)
        i = 0
        while len(utt2spk_list) >= 2:
            if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                tmp1 = utt2spk_list.pop()
                if list(tmp1)[0] not in combined_scp_dict:
                    continue
                tmp2 = utt2spk_list.pop()
                if list(tmp2)[0] not in combined_scp_dict:
                    continue
                train_diff_speaker_list.append([list(tmp1)[0], list(tmp2)[0]])
                i = 0
            else:
                i = i + 1
                random.shuffle(utt2spk_list)
                if i == 50:
                    # bp()
                    break

    valid_diff_speaker_list = []
    for repeats in range(n_repeats):
        utt2spk_list = list(utt2spk_valid)
        random.shuffle(utt2spk_list)
        i = 0
        while len(utt2spk_list) >= 2:
            if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                tmp1 = utt2spk_list.pop()
                if list(tmp1)[0] not in combined_scp_dict:
                    continue
                tmp2 = utt2spk_list.pop()
                if list(tmp2)[0] not in combined_scp_dict:
                    continue
                valid_diff_speaker_list.append([list(tmp1)[0], list(tmp2)[0]])
                i = 0
            else:
                i = i + 1
                random.shuffle(utt2spk_list)
                if i == 50:
                    # bp()
                    break
    train_diff_speaker_list = np.asarray(train_diff_speaker_list)
    valid_diff_speaker_list = np.asarray(valid_diff_speaker_list)
    
    if train_and_valid: # Returns two lists for training and validation
        return train_diff_speaker_list, valid_diff_speaker_list
    else:
        return train_diff_speaker_list + valid_diff_speaker_list
    
    
def generate_train_trial_keys(data_spk2utt_list, xvector_scp_list, train_and_valid=True, train_ratio=0.95):

    #    Make sure that each spk2utt in data_spk2utt_list is of same gender, same source, same language, etc. More Matching Metadata --> Better the model training.

    #    Can also specify the num_repeats after the dir name followed with space/tab separation in 2 column format. If not specified, default num_repeats is set to 1.
    
    xvector_scp_combined = {}
    
    for fx in xvector_scp_list:
        with open(fx) as f:
            scp_list = f.readlines()
        scp_dict = {os.path.splitext(os.path.basename(x.split(' ', 1)[0]))[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
        xvector_scp_combined.update(scp_dict)

    if type(data_spk2utt_list) == str:
        data_spk2utt_list = np.genfromtxt(data_spk2utt_list, dtype='str')

    if data_spk2utt_list.ndim == 2:
        num_repeats_list = data_spk2utt_list[:, 1].astype(int)
        data_spk2utt_list = data_spk2utt_list[:, 0]
    elif data_spk2utt_list.ndim == 1:
        num_repeats_list = np.ones(len(data_spk2utt_list)).astype(int)
    else:
        raise("Something wrong here.")


    sampled_list_train = []
    sampled_list_valid = []

    for i, d in enumerate(data_spk2utt_list):
        # print("In for loop get train dataset")
        same_train_list, same_valid_list = make_same_speaker_list(d, xvector_scp_combined, xvector_scp_list, n_repeats = num_repeats_list[i], train_and_valid=True, train_ratio=0.95)
        diff_train_list, diff_valid_list = make_diff_speaker_list(d, xvector_scp_combined, n_repeats = num_repeats_list[i], train_and_valid=True, train_ratio=0.95)
        # bp()
        zeros = np.zeros((diff_train_list.shape[0], 1)).astype(int)
        ones = np.ones((same_train_list.shape[0], 1)).astype(int)
        same_list_with_label_train = np.concatenate((same_train_list, ones), axis=1)
        diff_list_with_label_train = np.concatenate((diff_train_list, zeros), axis=1)
        zeros = np.zeros((diff_valid_list.shape[0], 1)).astype(int)
        ones = np.ones((same_valid_list.shape[0], 1)).astype(int)
        same_list_with_label_valid = np.concatenate((same_valid_list, ones), axis=1)
        diff_list_with_label_valid = np.concatenate((diff_valid_list, zeros), axis=1)
        concat_pair_list_train = np.concatenate((same_list_with_label_train, diff_list_with_label_train))
        concat_pair_list_valid = np.concatenate((same_list_with_label_valid, diff_list_with_label_valid))

        np.random.shuffle(concat_pair_list_train)
        sampled_list_train.extend(concat_pair_list_train)

        np.random.shuffle(concat_pair_list_valid)
        sampled_list_valid.extend(concat_pair_list_valid)
    
    if train_and_valid:
        return sampled_list_train, sampled_list_valid
    else:
        return sampled_list_train + sampled_list_valid
    
def save_unique_train_valid_xvector_scps(data_spk2utt_list, xvector_scp_list, train_scp_path, valid_scp_path, train_ratio=0.95):
    if type(data_spk2utt_list) == str:
        data_spk2utt_list = np.genfromtxt(data_spk2utt_list, dtype='str')

    if data_spk2utt_list.ndim == 2:
        data_spk2utt_list = data_spk2utt_list[:, 0]
        
    xvector_scp_combined = {}
    
    for fx in xvector_scp_list:
        with open(fx) as f:
            scp_list = f.readlines()
        scp_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
        xvector_scp_combined.update(scp_dict)
    
    train_scp = []
    valid_scp = []
    # bp()
    for i, d in enumerate(data_spk2utt_list):
        with open(d) as f:
            spk2utt_list = f.readlines()
        random.seed(2)
        random.shuffle(spk2utt_list)
        spk2utt_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1].split(' ') for x in spk2utt_list}
        spks = list(spk2utt_dict.keys())
        train_keys = spks[:int(train_ratio * len(spks))]
        valid_keys = spks[int(train_ratio * len(spks)):]
        # bp()
        for i in train_keys:
            for j in spk2utt_dict[i]:
                if j in xvector_scp_combined:
                    train_scp.append([j, xvector_scp_combined[j]])
        for i in valid_keys:
            for j in spk2utt_dict[i]:
                if j in xvector_scp_combined:
                    valid_scp.append([j, xvector_scp_combined[j]])
    train_scp = np.asarray(train_scp)
    valid_scp = np.asarray(valid_scp)
    subprocess.call(['mkdir','-p',os.path.dirname(train_scp_path)])
    subprocess.call(['mkdir','-p',os.path.dirname(valid_scp_path)])
    np.savetxt(train_scp_path, train_scp, fmt='%s', delimiter=' ', comments='')
    np.savetxt(valid_scp_path, valid_scp, fmt='%s', delimiter=' ', comments='')

def combine_trials_and_get_loader(trials_key_files_list, id_to_num_dict, subsample_factors=None, batch_size=2048, subset=0):
    if subsample_factors is None:
        subsample_factors = [1 for w in trials_key_files_list]
    datasets = []
    for f, sf in zip(trials_key_files_list, subsample_factors):
        t = np.genfromtxt(f, dtype = 'str')
        x1, x2, l = [], [], []
        for tr in t:
            try:
                a, b, c = id_to_num_dict[tr[0]], id_to_num_dict[tr[1]], float(tr[2])
                x1.append(a); x2.append(b); l.append(c)
            except:
                pass
        tdset = TensorDataset(torch.tensor(x1),torch.tensor(x2),torch.tensor(l))
        inds = np.arange(len(tdset))[np.random.rand(len(tdset))<sf]
        datasets.append(Subset(tdset, inds))
    combined_dataset = ConcatDataset(datasets)
    if subset > 0:
        inds = np.arange(len(combined_dataset))[np.random.rand(len(combined_dataset))<subset]
        combined_dataset = Subset(combined_dataset, inds)
    trials_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return trials_loader

def get_trials_loaders_dict(trials_key_files_list, id_to_num_dict, subsample_factors=None, batch_size=2048, subset=0):
    trials_loaders_dict = {}
    if subsample_factors is None:
        subsample_factors = [1 for w in trials_key_files_list]
    for f, sf in zip(trials_key_files_list, subsample_factors):
        t = np.genfromtxt(f, dtype = 'str')
        x1, x2, l = [], [], []
        for tr in t:
            try:
                a, b, c = id_to_num_dict[tr[0]], id_to_num_dict[os.path.splitext(tr[1])[0]], float(tr[2])
                x1.append(a); x2.append(b); l.append(c)
            except:
                pass
        tdset = TensorDataset(torch.tensor(x1),torch.tensor(x2),torch.tensor(l))
        inds = np.arange(len(tdset))[np.random.rand(len(tdset))<sf]
        dataset = Subset(tdset, inds)
        if subset > 0:
            inds = np.arange(len(dataset))[np.random.rand(len(dataset))<subset]
            dataset = Subset(dataset, inds)
        trials_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trials_loaders_dict[os.path.splitext(os.path.basename(f))[0]] = trials_loader
    return trials_loaders_dict

def extract_till_plda_embeddings(model, mega_mfcc_dict, data_loader, num_to_id_dict, device):
    utts = []
    for x1,x2,l in data_loader:
        for x in x1:
            if num_to_id_dict[int(x)] not in utts:
                utts.append(num_to_id_dict[int(x)])
                # numframes.append(mega_utt2num_frames_dict[num_to_id_dict[int(x)]])
        for x in x2:
            if num_to_id_dict[int(x)] not in utts:
                utts.append(num_to_id_dict[int(x)])
                # numframes.append(mega_utt2num_frames_dict[num_to_id_dict[int(x)]])
    
    # Here we are forward passing each MFCC one by one, and this is very slow. We need to figure out a way to group the mfccs of similar durations, and extract embeddings together to improve speed and efficiently utilize GPUs.
    model.eval()
    extracted_plda_embeddings = {}
    with torch.no_grad():
        for utt in utts:
            if utt in extracted_plda_embeddings:
                continue
            mfcc = kaldi_io.read_mat(mega_mfcc_dict[utt]).T
            mfcc_t = torch.from_numpy(mfcc[np.newaxis,:,:]).to(device)
            data_extracted_plda_embeddings = model.extract_plda_embeddings(mfcc_t)
            extracted_plda_embeddings[utt] = np.asarray(data_extracted_plda_embeddings[0].cpu())
    return extracted_plda_embeddings

def extract_xvectors(model, mega_mfcc_dict, device):
    model.eval()
    extracted_xvectors = {}
    for utt in mega_mfcc_dict:
        mfcc = kaldi_io.read_mat(mega_mfcc_dict[utt]).T
        mfcc_t = torch.from_numpy(mfcc[np.newaxis,:,:]).to(device)
        data_extracted_xvectors = model.xvector_extractor.extract(mfcc_t)
        extracted_xvectors[utt] = data_extracted_xvectors[0].cpu().detach().numpy()
    return extracted_xvectors

def load_mfccs_from_numbatch(mega_dict, num_to_id_dict, data, device):
    data_mfcc = []
    durs = []
    kar = kaldi_ark_reader_faster()
    for i, d in enumerate(data):
        fd = kar.get_fp(mega_dict[num_to_id_dict[int(d)]])
        data_mfcc_temp = kaldi_io.read_mat(fd).T
        data_mfcc.append(data_mfcc_temp)
        durs.append(data_mfcc_temp.shape[1])
    kar.close_all()
    try:
        tmparr = np.asarray(data_mfcc)
        data_mfcc = tmparr
    except:
        tmparr = np.empty(len(data_mfcc), dtype=object)
        for i in range(len(data_mfcc)):
            tmparr[i] = data_mfcc[i]
        data_mfcc = tmparr
    if len(data_mfcc.shape)>1:
        tensor_X = torch.from_numpy(np.asarray(data_mfcc)).float().to(device)
        return tensor_X
    else:
        sorted_durs, sort_idx = torch.sort(torch.tensor(durs))
        data_mfcc = data_mfcc[sort_idx]
        _, unsort_idx = torch.sort(sort_idx)
        uniq_durs, uniq_counts = torch.unique(sorted_durs, return_counts=True)
        split_sections = tuple(np.cumsum(uniq_counts)[:-1])
        data_mfcc = np.split(data_mfcc, split_sections)
        tensor_mfcc_list = [torch.from_numpy(np.asarray(list(mfcc))).float().to(device) for mfcc in data_mfcc]
        return tensor_mfcc_list, sort_idx, unsort_idx

def load_mfccs_from_numbatch_old(mega_dict, num_to_id_dict, data, device):
    data_mfcc = []
    durs = []
    for i, d in enumerate(data):
        fd = mega_dict[num_to_id_dict[int(d)]]
        data_mfcc_temp = kaldi_io.read_mat(fd).T
        data_mfcc.append(data_mfcc_temp)
        durs.append(data_mfcc_temp.shape[1])
    try:
        tmparr = np.asarray(data_mfcc)
        data_mfcc = tmparr
    except:
        tmparr = np.empty(len(data_mfcc), dtype=object)
        for i in range(len(data_mfcc)):
            tmparr[i] = data_mfcc[i]
        data_mfcc = tmparr
    if len(data_mfcc.shape)>1:
        tensor_X = torch.from_numpy(np.asarray(data_mfcc)).float().to(device)
        return tensor_X
    else:
        sorted_durs, sort_idx = torch.sort(torch.tensor(durs))
        data_mfcc = data_mfcc[sort_idx]
        _, unsort_idx = torch.sort(sort_idx)
        uniq_durs, uniq_counts = torch.unique(sorted_durs, return_counts=True)
        split_sections = tuple(np.cumsum(uniq_counts)[:-1])
        data_mfcc = np.split(data_mfcc, split_sections)
        tensor_mfcc_list = [torch.from_numpy(np.asarray(list(mfcc))).float().to(device) for mfcc in data_mfcc]
        return tensor_mfcc_list, sort_idx, unsort_idx

def load_xvec_trials_from_numbatch(mega_dict, num_to_id_dict, data1, data2, device):
    data1_xvec, data2_xvec = [], []  # torch.tensor([[]]), torch.tensor([[]])
    for i, (d1, d2) in enumerate(zip(data1, data2)):
        data1_xvec_temp, data2_xvec_temp = mega_dict[num_to_id_dict[int(d1)]], mega_dict[num_to_id_dict[int(d2)]]
        data1_xvec.append(data1_xvec_temp)
        data2_xvec.append(data2_xvec_temp)
    tensor_X1 = torch.from_numpy(np.asarray(data1_xvec)).float().to(device)
    tensor_X2 = torch.from_numpy(np.asarray(data2_xvec)).float().to(device)
    return tensor_X1, tensor_X2


def load_xvec_trials_from_idbatch(mega_dict, trials, device):
    data1_xvec, data2_xvec = [], []  # torch.tensor([[]]), torch.tensor([[]])
    for i, (d1, d2) in enumerate(zip(trials[:,0], trials[:,1])):
        data1_xvec_temp, data2_xvec_temp = mega_dict[os.path.splitext(os.path.basename(d1))[0]], mega_dict[os.path.splitext(os.path.basename(d2))[0]]
        data1_xvec.append(data1_xvec_temp)
        data2_xvec.append(data2_xvec_temp)
    tensor_X1 = torch.from_numpy(np.asarray(data1_xvec)).float().to(device)
    tensor_X2 = torch.from_numpy(np.asarray(data2_xvec)).float().to(device)
    return tensor_X1, tensor_X2