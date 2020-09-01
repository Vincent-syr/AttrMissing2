'''
    test novel classes, use pre-train setting.
    non attribute missing
'''


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
import data.feature_loader as feat_loader
from data.dataloader_vae import DATA_LOADER as dataloader
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file
from methods.protonet import ProtoNet
from methods.am3_protonet import AM3_ProtoNet



def feature_evaluation(test_set, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):

    label, data_from_modalities = test_set.next_batch()
    if len(data_from_modalities) == 2:  # use attribute
        img_feat = data_from_modalities[0].cuda()  # (n*b, k+q, f1)
        attr_feat = data_from_modalities[1].cuda()  # (n*b, f2)

        attr_feat_exp = torch.unsqueeze(attr_feat, 1).expand(-1, img_feat.shape[1], -1)  # (n*b, k+q, f2)
        z_all = [ img_feat, attr_feat_exp]

    else: # only image feature
        _, z_all = test_set.next_batch()
        z_all = z_all.cuda()

    scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100
    return acc



if __name__ == '__main__':
    params = parse_args('test')
    print(params)
    acc_all = []
    iter_num = 600
    word_dim = 312
    image_size = 224
    n_query = 15
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)   # 5 way, 5 shot
    if params.method == 'am3_protonet':
        model = AM3_ProtoNet(model_dict[params.model], params=params, word_dim=word_dim, **few_shot_params)

    elif params.method == 'protonet':
        model = ProtoNet(model_dict[params.model], **few_shot_params)
    else:
       raise ValueError('Unknown method')

    model.n_query = n_query
    model = model.cuda()

    extra = '%s_%s' % (params.model, params.method)
    # checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    checkpoint_dir = '%s/checkpoints/%s/' % (configs.save_dir, params.dataset)

    if params.train_aug:
        extra += '_aug'
    if params.aux:
        extra += '_aux'
    if not params.method in ['baseline', 'baseline++'] :
        extra += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    checkpoint_dir += extra
    print("checkpoint_dir: ", checkpoint_dir)
    # exit(0)

    writer = SummaryWriter(log_dir='runs/%s_%s' % (params.dataset, extra))

    modelfile   = get_best_file(checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        loadfile = configs.data_dir[params.dataset] + 'novel.json'

    split_str = params.split
    novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                              split_str + "_best.hdf5")  # defaut split = novel, but you can also test base or val classes
    attr_file = None
    if params.aux:
        attr_file = 'filelists/CUB/attr_array.npy'
    # cl_data_file = feat_loader.init_loader(novel_file)


    test_set = dataloader(params.dataset, novel_file, attr_file,
                          params.test_n_way, params.n_shot + n_query)

    for i in range(iter_num):
        # n_way = 5, n_support = 5, n_query = 15
        acc = feature_evaluation(test_set, model, n_query=n_query, adaptation=params.adaptation, **few_shot_params)
        acc_all.append(acc)
        writer.add_scalar('test acc', acc, i)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    writer.close()
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))


    record_file ='record/%s_%s_results.txt' % (params.dataset, extra)
    print('record_file = ', record_file)
    with open(record_file , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        if params.method in ['baseline', 'baseline++'] :
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )