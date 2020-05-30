#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:39:06 2020

@author: shreyasr
"""

import numpy as np
import random
import pickle
import subprocess
import re
import os
import sys
import kaldi_io

from pdb import set_trace as bp

from utils.sv_trials_loaders import generate_train_trial_keys, save_unique_train_valid_xvector_scps



if __name__=='__main__':
    
    base_path = '/home/data2/SRE2019/prashantk/voxceleb/v3'
    xvector_model_dir_relative = 'exp/xvector_nnet_2a'
    xvectors_base_path = os.path.join(base_path, xvector_model_dir_relative)
    
    stage = 3
    
    # %% Generate and save training trial keys using SRE SWBD and MX6 datasets
    if stage <= 1:
        data_spk2utt_list = np.asarray([['{}/data/sre18_eval_combined_20s_no_sil/male/spk2utt'.format(base_path), '1'],
                                    ['{}/data/sre18_eval_combined_20s_no_sil/female/spk2utt'.format(base_path), '1']])
    
        xvector_scp_list = np.asarray(
            ['{}/xvectors_sre18_eval_combined_20s_no_sil/xvector.scp'.format(xvectors_base_path)])
    
    
        train_trial_keys, val_trial_keys = generate_train_trial_keys(data_spk2utt_list, xvector_scp_list, train_and_valid=True, train_ratio=0.95)
        
        # Save the training and validation trials and keys for training NPLDA and other discriminative models
        np.savetxt('trials_and_keys/sre18eval_20s_train_trial_keys.tsv', train_trial_keys, fmt='%s', delimiter='\t', comments='none')
        np.savetxt('trials_and_keys/sre18eval_20s_validate_trial_keys.tsv', val_trial_keys, fmt='%s', delimiter='\t', comments='none')
        
        # Save the train and validation xvectors for training a Kaldi PLDA if required
        train_scp_path = '{}/xvectors_sre18_eval_combined_20s_no_sil/train_split/xvector.scp'.format(xvectors_base_path)
        valid_scp_path = '{}/xvectors_sre18_eval_combined_20s_no_sil/valid_split/xvector.scp'.format(xvectors_base_path)
        save_unique_train_valid_xvector_scps(data_spk2utt_list, xvector_scp_list, train_scp_path, valid_scp_path, train_ratio=0.95)

    
    # %% Make the mega xvector scp with all the xvectors, averaged enrollment xvectors, etc.
        
    if stage <= 2:
        xvector_scp_list = np.asarray(
            ['{}/xvectors_swbd_sre_mx6_combined/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre16_eval_enrollment/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre16_eval_test/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre16_eval_enrollment/spk_xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_dev_enrollment/spk_xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_dev_test/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_dev_enrollment/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_eval_test/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_eval_enrollment/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_eval_enrollment/spk_xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre19_eval_test/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre19_eval_enrollment/xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre19_eval_enrollment/spk_xvector.scp'.format(xvectors_base_path),
             '{}/xvectors_sre18_eval_combined_20s_no_sil/xvector.scp'.format(xvectors_base_path)])
        
        mega_scp_dict = {}
        mega_xvec_dict = {}
        for fx in xvector_scp_list:
            subprocess.call(['sed','-i', 's| {}| {}|g'.format(xvector_model_dir_relative, xvectors_base_path), fx])
            with open(fx) as f:
                scp_list = f.readlines()
            scp_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
            xvec_dict = {x.split(' ', 1)[0]: kaldi_io.read_vec_flt(x.rstrip('\n').split(' ', 1)[1]) for x in scp_list}
            mega_scp_dict.update(scp_dict)
            mega_xvec_dict.update(xvec_dict)
        
        mega_scp = np.c_[np.asarray(list(mega_scp_dict.keys()))[:,np.newaxis], np.asarray(list(mega_scp_dict.values()))]
        
        np.savetxt('xvectors/mega_xvector_megamodel_8k.scp', mega_scp, fmt='%s', delimiter=' ', comments='')
        
        pickle.dump(mega_xvec_dict, open('xvectors/mega_xvector_megamodel_8k.pkl', 'wb'))
        
    # %% Combine the 20s train feats and sre18 dev enrollment concatenated features to get the mega mfcc scp dictionary.
    
    if stage <= 3:
        mfcc_scp_list = np.asarray(
            ['{}/data/sre18_eval_combined_20s_no_sil/feats.scp'.format(base_path),
             '{}/data/sre18_dev_enrollment/feats_enrollment_cmvn_vad.scp'.format(base_path),
             '{}/data/sre18_dev_test/feats_cmvn_vad.scp'.format(base_path)])
        mega_scp_dict = {}
        for fx in mfcc_scp_list:
            with open(fx) as f:
                scp_list = f.readlines()
            scp_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
            mega_scp_dict.update(scp_dict)
        mega_scp = np.c_[np.asarray(list(mega_scp_dict.keys()))[:,np.newaxis], np.asarray(list(mega_scp_dict.values()))]
        
        np.savetxt('mfcc/mega_mfcc_megamodel_8k.scp', mega_scp, fmt='%s', delimiter=' ', comments='')
        
        pickle.dump(mega_scp_dict, open('mfcc/mega_mfcc_scp_megamodel_8k.pkl', 'wb'))