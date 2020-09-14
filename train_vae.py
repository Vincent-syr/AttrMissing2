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
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file, get_trlog_vae, save_fig
from utils import Timer
from data.dataloader_vae import DATA_LOADER as dataloader
from methods.vae_proto import Model
# from methods.am3_protonet import AM3_ProtoNet
from methods.am3 import AM3



def train_vae(base_dataset, val_dataset, gen_model, fsl_model, start_epoch, stop_epoch, miss_rate, params):
    loss_all = []
    max_acc = 0      

    trlog = get_trlog_vae(params)
    trlog_path = os.path.join(params.trlog_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))   # '20200909-185444'


    timer = Timer()

    # summary
    # writer = SummaryWriter(log_dir=params.writer_dir)  # default ./runs/***
    tm = time.strftime("%Y%m%d-%H%M%S", time.localtime())   # '20200903-205935'
    # writer = SummaryWriter(logdir=('%s/%s' % (params.writer_dir, tm)))

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
        # syn_acc, raw_acc, miss_acc = gen_model.test_loop(val_dataset, fsl_model, miss_rate)
        syn_acc, raw_acc, miss_acc, none_acc, (lambda_syn, lambda_raw, lambda_miss, lambda_none) = gen_model.test_loop(val_dataset, fsl_model, miss_rate)

        end = time.time()
        print("test time = %.2f s" % (end-start))
        trlog['syn_acc'].append(syn_acc)
        trlog['raw_acc'].append(raw_acc)
        trlog['miss_acc'].append(miss_acc)
        trlog['none_acc'].append(none_acc)
        trlog['lambda_syn'].append(lambda_syn)
        trlog['lambda_raw'].append(lambda_raw)
        trlog['lambda_miss'].append(lambda_miss)
        trlog['lambda_none'].append(lambda_none)
        


        # writer.add_scalars('validation_acc', {'syn attr':syn_acc,
        #                                     'raw attr':raw_acc,
        #                                     'miss attr':miss_acc,
        #                                       'none attr':none_acc}, epoch)


        # writer.add_scalars('lambda', {'lambda_syn':lambda_syn,
        #                              'lambda_raw':lambda_raw,
        #                              'lambda_miss':lambda_miss,
        #                              'lambda_none':lambda_none}, epoch)

        if syn_acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = syn_acc
            outfile = os.path.join(params.save_dir, 'best_model.tar')
            save_model(gen_model, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.save_dir, '{:d}.tar'.format(epoch))
            save_model(gen_model, outfile)
            torch.save(trlog, trlog_path)

        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1) / stop_epoch)))

    # add loss to summary
    # for i, loss in enumerate(loss_all):
    #     writer.add_scalar('train_loss', loss, i)
    for loss in loss_all:
        trlog['train_loss'].append(loss)
    torch.save(trlog, trlog_path)
    # writer.close()
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
    # few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)   # 5 way, 5 shot
    train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot, n_query=params.n_query) 

    if params.method == 'am3':
        fsl_model = AM3(model_dict[params.model], params=params, word_dim=attr_dim,  **train_few_shot_params)
        # fsl_model = AM3_ProtoNet(model_dict[params.model], params=params, word_dim=attr_dim, **few_shot_params)
    else:
        raise ValueError("no such clf few shot mdoel")
    
    modelfile   = get_best_file(params.load_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        fsl_model.load_state_dict(tmp['state'])

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
    aux = True if params.method=='am3' else False

    params.miss_rate = float(params.miss_rate)


    params.checkpoints = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    params.save_dir = '%s/checkpoints/%s/%s' %(configs.save_dir, params.dataset, "vae")

    if params.train_aug:
        params.checkpoints += '_aug'
        params.save_dir += '_aug'


    params.checkpoints += '_lr%s_%s' % (str(params.init_lr), params.lr_anneal)
    params.save_dir += '_lr%s_%s' % (str(params.init_lr), params.lr_anneal)
    # params.writer_dir += '_lr%s_%s' % (str(params.init_lr), params.lr_anneal)

    if not params.method in ['baseline', 'baseline++'] :
        params.checkpoints += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
        params.save_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
        # params.writer_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    # params.trlog_dir = os.path.join(params.load_dir, 'trlog')
    params.save_dir += '_%.1fmissing' % (params.miss_rate)
    params.trlog_dir = os.path.join(params.save_dir, 'trlog')
    params.save_dir = os.path.join(params.save_dir, 'model')
    params.load_dir = os.path.join(params.checkpoints, 'model')
    # params.writer_dir += '_%.1fmissing' % (params.miss_rate)

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)
    if not os.path.isdir(params.trlog_dir):
        os.makedirs(params.trlog_dir)


    # print('save_dir = ', params.save_dir)
    print('load_dir = ', params.load_dir)
    # print('trlog_dir = ', params.trlog_dir)
    # exit(0)
    # load data
    base_file = os.path.join(params.checkpoints.replace("checkpoints","features"),  "base_best.hdf5")
    val_file = os.path.join(params.checkpoints.replace("checkpoints","features"),  "val_best.hdf5")
    if params.dataset == 'CUB':
        base_attr_file = 'filelists/CUB/attr_array.npy'
        val_attr_file = 'filelists/CUB/attr_array.npy'
    elif params.dataset == 'miniImagenet':
        base_attr_file = '/test/0Dataset_others/Dataset/mini-imagenet-am3/few-shot-wordemb-base.npz'
        val_attr_file = '/test/0Dataset_others/Dataset/mini-imagenet-am3/few-shot-wordemb-val.npz'


    # dataloader 
    base_dataset = dataloader(params.dataset, base_file, base_attr_file, params.train_n_way, params.n_shot)
    val_dataset = dataloader(params.dataset, val_file, val_attr_file, params.test_n_way, params.n_shot + params.n_query) 

    hyperparameters = get_hyperparams(params)
    hyperparameters['img_dim'] = base_dataset.img_dim
    hyperparameters['attr_dim'] = base_dataset.attr_dim
    
    # scratch vae model and pre_trained fsl mdoel
    gen_model = Model(hyperparameters)
    fsl_model = load_fsl_model(params, base_dataset.attr_dim)
    gen_model = gen_model.cuda()
    fsl_model = fsl_model.cuda()

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    train_vae(base_dataset, val_dataset, gen_model, fsl_model, start_epoch, stop_epoch, params.miss_rate, params)



