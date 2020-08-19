import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time
from tensorboardX import SummaryWriter

import configs
import backbone
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file
from utils import Timer
from data.dataloader_vae import DATA_LOADER as dataloader
from methods.vae_proto import Model
from methods.am3_protonet import AM3_ProtoNet



def train_vae(base_dataset, val_dataset, gen_model, fsl_model, start_epoch, stop_epoch, miss_rate, params):
    loss_all = []
    max_acc = 0       
    timer = Timer()

    # summary
    writer = SummaryWriter()  # default ./runs/***

    print('train for reconstruction')

    for epoch in range(start_epoch,stop_epoch):
        # gen_model.train()
        start = time.time()
        losses = gen_model.train_vae(base_dataset, epoch)
        loss_all += losses
        end = time.time()
        print("train time = %.2f s" % (end-start))

#       validation 
        # gen_model.eval()
        start = time.time()
        syn_acc, raw_acc = gen_model.test_loop(val_dataset, fsl_model, miss_rate)
        end = time.time()
        print("test time = %.2f s" % (end-start))

        writer.add_scalar('val raw attribute acc', raw_acc, epoch)
        writer.add_scalar('val syn attribute acc', syn_acc, epoch)


        if syn_acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = syn_acc
            outfile = os.path.join(params.save_dir, 'best_model.tar')
            save_model(gen_model, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.save_dir, '{:d}.tar'.format(epoch))
            save_model(gen_model, outfile)

        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1) / stop_epoch)))

    # add loss to summary
    for i, loss in enumerate(loss_all):
        writer.add_scalar('train_loss', loss, i)

    writer.close()
    print("train finish !!")
    return loss_all


def save_model(model, outfile):
    state = {
            'state_dict': model.state_dict() ,
            'hyperparameters':hyperparameters,
            'encoder':{},
            'decoder':{}
        }
    for d in model.all_data_sources:
        state['encoder'][d] = model.encoder[d].state_dict()
        state['decoder'][d] = model.decoder[d].state_dict()
    torch.save(state, outfile)
    print('>> saved')


def load_fsl_model(params, attr_dim):
    # classifier few shot models
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)   # 5 way, 5 shot
    if params.method == 'am3_protonet':
        fsl_model = AM3_ProtoNet(model_dict[params.model], word_dim=attr_dim, **few_shot_params)
    else:
        raise ValueError("no such clf few shot mdoel")
    
    # modelfile   = get_best_file(params.checkpoint_dir)
    # if modelfile is not None:
    #     tmp = torch.load(modelfile)
    #     fsl_model.load_state_dict(tmp['state'])

    return fsl_model



def get_hyperparams(params):
    hyperparameters = {
    'num_shots': 0,
    'device': 'cuda',
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CADA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': 0.25,
                                           'end_epoch': 93,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': 2.37,
                                                           'end_epoch': 75,
                                                           'start_epoch': 21},
                                  'distance': {'factor': 8.13,
                                               'end_epoch': 22,
                                               'start_epoch': 6}}},

    'lr_gen_model': 0.00015,
    'generalized': True,
    'xyu_samples_per_class': {'SUN': (200, 0, 400, 0),
                              'APY': (200, 0, 400, 0),
                              'CUB': (200, 0, 400, 0),
                              'AWA2': (200, 0, 400, 0),
                              'FLO': (200, 0, 400, 0),
                              'AWA1': (200, 0, 400, 0)},
    'epochs': 100,
    'loss': 'l1',
    'auxiliary_data_source' : 'attributes',
    'lr_cls': 0.001,
    'dataset': 'CUB',
    'hidden_size_rule': {'resnet_features': (1560, 1660),
                        'attributes': (1450, 665),
                        'sentences': (1450, 665) },
    'latent_size': 64,
    'n_episodes' : 100,
    'batch_size' : 1

    }
    hyperparameters['dataset'] = params.dataset
    hyperparameters['num_ways'] = params.train_n_way



    return hyperparameters

# n_way, n_shot
if __name__ == "__main__":
    params = parse_args('train_gen')
    print(params)
    acc_all = []
    # split_list = ['base', 'val']
    iter_num = 600
    aux = params.aux
    miss_rate = params.miss_rate
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)   # 5 way, 5 shot

    # params.checkpoint_dir = '%s/checkpoints/%s/%s' %(configs.save_dir, params.dataset, "vae")
    # params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)

    params.load_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    params.save_dir = '%s/checkpoints/%s/%s' %(configs.save_dir, params.dataset, "vae")
    if params.train_aug:
        params.load_dir += '_aug'
    if aux:
        params.load_dir += '_aux'

    if not params.method in ['baseline', 'baseline++'] :
        params.load_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
        params.save_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    params.save_dir += '_%.1fmissing' % (params.miss_rate)

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    # print(params.save_dir)
    # print(params.load_dir)
    # exit(0)


    # load data
    # feature_dir = 'features/CUB/ResNet10_protonet_aug_5way_5shot'
    # base_file = os.path.join(feature_dir, 'base_best.hdf5')
    # val_file = os.path.join(feature_dir, 'val_best.hdf5')
    base_file = os.path.join(params.load_dir.replace("checkpoints","features"),  "base_best.hdf5")
    val_file = os.path.join(params.load_dir.replace("checkpoints","features"),  "val_best.hdf5")
    attr_file = 'filelists/CUB/attr_array.npy'

    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small ,16

    # dataloader 
    base_dataset = dataloader(params.dataset, base_file, attr_file, params.train_n_way, params.n_shot)
    val_dataset = dataloader(params.dataset, val_file, attr_file, params.test_n_way, params.n_shot + n_query) 

    hyperparameters = get_hyperparams(params)
    hyperparameters['img_dim'] = base_dataset.img_dim
    hyperparameters['attr_dim'] = base_dataset.attr_dim
    
    # scratch vae model and pre_trained fsl mdoel
    gen_model = Model(hyperparameters)
    fsl_model = load_fsl_model(params, base_dataset.attr_dim)
    fsl_model.n_query = n_query

    gen_model = gen_model.cuda()
    fsl_model = fsl_model.cuda()

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch


    train_vae(base_dataset, val_dataset, gen_model, fsl_model, start_epoch, stop_epoch, miss_rate, params)



    # train_vae(model, )

        