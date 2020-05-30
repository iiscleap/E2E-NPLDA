#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:02:04 2020

@author: shreyasr, prashantk
"""

# %% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import random
import pickle
import subprocess
from utils.NpldaConf import E2EConf
from pdb import set_trace as bp
import time
from utils.sv_trials_loaders import combine_trials_and_get_loader, get_trials_loaders_dict, load_xvec_trials_from_numbatch, load_xvec_trials_from_idbatch, extract_till_plda_embeddings, load_mfccs_from_numbatch, custom_loader_e2e, custom_loader_e2e_v2, custom_loader_e2e_v3

from datetime import datetime
import logging

from utils.models import Etdnn_Xvec_NeuralPlda

# %% Function Definitions
def train(nc, model, device, train_loader, mega_mfcc_dict, num_to_id_dict, optimizer, epoch, timestamp, valid_loaders=None, start_at_iter=0):
    model.train1()
    losses = []
    n_trials = sum([len(target) for data1, data2, target in train_loader])
    n_trials_processed = 0
    cooldown_timer_start = datetime.timestamp(datetime.now())
    
    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        if batch_idx < start_at_iter:
            continue
        optimizer.zero_grad()
        target = target.to(device)
        bs = len(data1)
        n_trials_processed += bs
        uniq, inds = torch.unique(torch.cat((data1,data2)), return_inverse=True)
        tmpvar = load_mfccs_from_numbatch(mega_mfcc_dict, num_to_id_dict, uniq, device)
        if type(tmpvar) is not tuple:
            data_xvec = model.extract_plda_embeddings(tmpvar)
            data1_xvec, data2_xvec = data_xvec[inds][:bs], data_xvec[inds][bs:]
        else:
            data_mfcc, sort_idx, unsort_idx = tmpvar
            data_xvec = torch.cat(tuple(model.extract_plda_embeddings(mfcc) for mfcc in data_mfcc))
            data_xvec = data_xvec[unsort_idx]
            data1_xvec, data2_xvec = data_xvec[inds][:bs], data_xvec[inds][bs:]
            
        output = model.forward_from_plda_embeddings(data1_xvec, data2_xvec)
        loss = model.loss(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        gpu_up_time = datetime.timestamp(datetime.now()) - cooldown_timer_start
        if gpu_up_time > 3600:
            model = model.to(torch.device('cpu'))
            del data_mfcc
            del data_xvec
            del data1_xvec
            del data2_xvec
            del output
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            print("\nWaiting for a minute now for the GPU to cool down...\n")
            logging.info("\nWaiting for a minute now for the GPU to cool down...\n")
            pickle.dump([model, epoch, batch_idx+1, nc.lr], open('models/model_progress_{}.pkl'.format(timestamp),'wb'))
            sys.stdout.flush()
            time.sleep(60)
            model = model.to(device)
            cooldown_timer_start = datetime.timestamp(datetime.now())
            
        if batch_idx % nc.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t {}: {:.6f}'.format(
                epoch, n_trials_processed, n_trials,
                       100. * batch_idx / len(train_loader), nc.loss, sum(losses)/len(losses)))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t {}: {:.6f}'.format(
                epoch, n_trials_processed, n_trials,
                       100. * batch_idx / len(train_loader), nc.loss, sum(losses)/len(losses)))
            losses = []



def validate(nc, model, device, mega_mfcc_dict, num_to_id_dict, data_loader, update_thresholds=False):
    model.eval()
    extracted_plda_embeddings = extract_till_plda_embeddings(model, mega_mfcc_dict, data_loader, num_to_id_dict, device)
    with torch.no_grad():
        targets, scores = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for data1, data2, target in data_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            data1_xvec, data2_xvec = load_xvec_trials_from_numbatch(extracted_plda_embeddings, num_to_id_dict, data1, data2, device)
            targets = torch.cat((targets, target))
            scores_batch = model.forward_from_plda_embeddings(data1_xvec, data2_xvec)
            scores = torch.cat((scores, scores_batch))
        soft_cdet_loss = model.softcdet(scores, targets)
        cdet_mdl = model.cdet(scores, targets)
        # minc, minc_threshold = model.minc(scores, targets, update_thresholds=update_thresholds, showplots=True)
        minc, minc_threshold = model.minc(scores, targets, update_thresholds=update_thresholds)
    
    logging.info('\n\nTest set: C_det (mdl): {:.4f}\n'.format(cdet_mdl))
    logging.info('Test set: soft C_det (mdl): {:.4f}\n'.format(soft_cdet_loss))
    logging.info('Test set: C_min: {:.4f}\n'.format(minc))
    for beta in nc.beta:
        logging.info('Test set: argmin threshold [{}]: {:.4f}\n'.format(beta, minc_threshold[beta]))
    
    print('\n\nTest set: C_det (mdl): {:.4f}\n'.format(cdet_mdl))
    print('Test set: soft C_det (mdl): {:.4f}\n'.format(soft_cdet_loss))
    print('Test set: C_min: {:.4f}\n'.format(minc))
    for beta in nc.beta:
        print('Test set: argmin threshold [{}]: {:.4f}\n'.format(beta, minc_threshold[beta]))
        
    return minc, minc_threshold

def initialize_model(nc, device, timestamp, mega_mfcc_dict, valid_loaders_dict, id_to_num_dict, num_to_id_dict):
    
    model = Etdnn_Xvec_NeuralPlda(nc).to(device)
    
    if nc.initialization == 'kaldi':
        model.LoadParamsFromKaldi(nc.xvec_model, nc.meanvec, nc.transformmat, nc.kaldiplda)
    else:
        model = pickle.load(open(nc.initialization,'rb'))
        

    print("Initializing the thresholds... Whatever numbers that get printed here are junk.\n")
    valloss, minC_threshold = validate(nc, model, device, mega_mfcc_dict, num_to_id_dict, valid_loaders_dict[nc.heldout_set_for_th_init], update_thresholds=True)

    print("\n\nEpoch 0: After Initialization\n")
    for val_set, valid_loader in valid_loaders_dict.items():
        print("Validating {}".format(val_set))
        logging.info("Validating {}".format(val_set))
        valloss, minC_threshold = validate(nc, model, device, mega_mfcc_dict, num_to_id_dict, valid_loader)
            
    return model


def main(timestamp=False):
    if not timestamp:
        timestamp = int(datetime.timestamp(datetime.now()))
    
    print(timestamp)
    logging.basicConfig(filename='logs/e2e_NPLDA_{}.log'.format(timestamp),
                        filemode='a',
                        format='%(levelname)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    # %% Configure Training
    configfile = 'conf/sre18_eval_20s_config_e2e.cfg'
    
    nc = E2EConf(configfile)
    
    torch.manual_seed(nc.seed)
    np.random.seed(nc.seed)
    random.seed(nc.seed)
    
    logging.info(" Running file {}\n\nStarted at {}.\n".format(sys.argv[0], datetime.now()))

    if not torch.cuda.is_available():
        nc.device='cpu'
    device = torch.device(nc.device)
    
    print("Running on {}...".format(nc.device))
    logging.info("Running on {} ...\n".format(nc.device))
    logging.info("\nConfiguration:\n\n{}\n\n".format(''.join(open(configfile,'r').readlines())))
          
    # %%Load the generated training data trials and make loaders here

    mega_mfcc_dict = pickle.load(open(nc.mega_mfcc_pkl, 'rb'))
    num_to_id_dict = {i: j for i, j in enumerate(list(mega_mfcc_dict))}
    id_to_num_dict = {v: k for k, v in num_to_id_dict.items()}
    
    
    # train_loader = combine_trials_and_get_loader(nc.training_data_trials_list, id_to_num_dict, subsample_factors=nc.train_subsample_factors ,batch_size=nc.batch_size)
    
    # train_loader_sampled = combine_trials_and_get_loader(nc.training_data_trials_list, id_to_num_dict, batch_size=nc.batch_size, subset=0.05)

    valid_loaders_dict = get_trials_loaders_dict(nc.validation_trials_list, id_to_num_dict, subsample_factors=nc.valid_subsample_factors, batch_size=5*nc.batch_size)
    
    # %% Initialize model and stuff
    if os.path.exists('models/model_progress_{}.pkl'.format(timestamp)):
        model, resume_epoch, resume_iter, nc.lr = pickle.load(open('models/model_progress_{}.pkl'.format(timestamp)),'rb')
        model = model.to(device)
    else:
        model = initialize_model(nc, device, timestamp, mega_mfcc_dict, valid_loaders_dict, id_to_num_dict, num_to_id_dict)
        resume_epoch, resume_iter = 1,0
    bp()
    params_dict = dict(model.named_parameters())
    updatable_params = []
    for param in params_dict.keys():
        # if ('xvector_extractor' in param):
        # if ('xvector_extractor.tdnn' in param) and ('10' not in param):
        if False:
            params_dict[param].requires_grad = False
        else:
            updatable_params.append(params_dict[param])
    optimizer = optim.Adam(updatable_params, lr=nc.lr, weight_decay=1e-5)
    
    if os.path.exists('trials_and_keys/train_loaders_{}.pkl'.format(timestamp)):
        train_loaders = pickle.load(open('trials_and_keys/train_loaders_{}.pkl'.format(timestamp),'rb'))
    else:
        train_loaders = custom_loader_e2e_v3(nc, mega_mfcc_dict, id_to_num_dict, n_subepochs=nc.n_epochs)
        pickle.dump(train_loaders, open('trials_and_keys/train_loaders_{}.pkl'.format(timestamp),'wb'))
    
    #%% Training
    all_losses = []

    for epoch, train_loader in enumerate(train_loaders, start=1):
        if epoch < resume_epoch:
            continue
        train(nc, model, device, train_loader , mega_mfcc_dict, num_to_id_dict, optimizer, epoch, timestamp, start_at_iter=resume_iter)
        
        resume_iter=0
        model.SaveModel("models/e2e_NPLDA_{}_{}.pt".format(epoch, timestamp))
        
        for val_set, valid_loader in valid_loaders_dict.items():
            print("Validating {}".format(val_set))
            logging.info("Validating {}".format(val_set))
            valloss, minC_threshold = validate(nc, model, device, mega_mfcc_dict, num_to_id_dict, valid_loader)
            if val_set==nc.heldout_set_for_lr_decay:
                all_losses.append(valloss)
        try:
            if (all_losses[-1] > all_losses[-2]) and (all_losses[-2] > all_losses[-3]):
                nc.lr = nc.lr / 2
                print("REDUCING LEARNING RATE to {} since loss trend looks like {}".format(nc.lr, all_losses[-3:]))
                logging.info("REDUCING LEARNING RATE to {} since loss trend looks like {}".format(nc.lr, all_losses[-3:]))
                optimizer = optim.Adam(updatable_params, lr=nc.lr, weight_decay=1e-5)
        except:
            pass
        
        # for trial_file in nc.test_trials_list:
        #     print("Generating scores for Epoch {} with trial file {}".format(epoch, trial_file))
        #     nc.generate_scorefile("scores/kaldipldanet_epoch{}_{}_{}.txt".format(epoch, os.path.splitext(os.path.basename(trial_file))[0], timestamp), trial_file, mega_mfcc_dict, model, device
                                  


# %% __main__

if __name__ == '__main__':
    main()
