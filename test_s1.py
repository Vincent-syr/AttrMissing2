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
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file, save_fig, get_trlog_test
# from methods.protonet import ProtoNet
# from methods.am3_protonet import AM3_ProtoNet
from methods.am3 import AM3
from methods.protonet_multi_gpu import ProtoNetMulti


def feature_evaluation(test_set, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):

    _, data_from_modalities = test_set.next_batch()
    if len(data_from_modalities) == 2:  # use attribute
        img_feat = data_from_modalities[0].cuda()  # (n*b, k+q, f1)
        attr_feat = data_from_modalities[1].cuda()  # (n*b, f2)

        # attr_feat_exp = torch.unsqueeze(attr_feat, 1).expand(-1, img_feat.shape[1], -1)  # (n*b, k+q, f2)
        z_all = [ img_feat, attr_feat]

    else: # only image feature
        # _, z_all = test_set.next_batch()
        z_all = data_from_modalities.cuda()

    # scores = model.set_forward(z_all, is_feature=True)
    correct_this, count_this = model.correct_quick(z_all)
    
    acc = correct_this/float(count_this) * 100
    # pred = scores.data.cpu().numpy().argmax(axis = 1)
    # y = np.repeat(range( n_way ), n_query )
    # acc = np.mean(pred == y)*100
    return acc



def image_evaluation():




if __name__ == '__main__':
    params = parse_args('test')
    print(params)
    # acc_all = []
    acc_all = {'base':[], 'val':[], 'novel':[]}
    iter_num = 600
    if params.dataset == 'CUB':
        image_size = 224
        word_dim = 312

    elif params.dataset == 'miniImagenet':
        image_size = 84
        word_dim = 300

    n_query = params.n_query



    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot, n_query=params.n_query)   # 5 way, 5 shot
    if params.method == 'protonet':
        params.aux = False
        model = ProtoNetMulti(model_dict[params.model], params=params, **few_shot_params)
    elif params.method == 'am3':
        params.aux = True
        model = AM3(model_dict[params.model], params=params, word_dim=word_dim,  **few_shot_params)
    else:
        raise ValueError('Unknown method')
    
    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)

    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_lr%s_%s' % (str(params.init_lr), params.lr_anneal)

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    params.model_dir = os.path.join(params.checkpoint_dir, 'model')
    params.record_dir = params.checkpoint_dir.replace("checkpoints", "record")
    if not os.path.isdir(params.record_dir):
        os.makedirs(params.record_dir)

    record_file = os.path.join(params.record_dir, 'results.txt')

    # trlog = {}
    # trlog['acc'] = []
    # trlog['script'] = 'test'
    # trlog['args'] = params
    trlog = get_trlog_test(params)
    trlog_path = os.path.join(params.record_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))

    modelfile   = get_best_file(params.model_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        print('load model ')

    # loadfile = configs.data_dir[params.dataset] + 'novel.json'
    # split = params.split
    split_list = ['base', 'val', 'novel']


    for split in split_list:

        novel_file = os.path.join(params.checkpoint_dir.replace("checkpoints", "features"),
                                split + "_best.hdf5")  # defaut split = novel, but you can also test base or val classes
        attr_file = None
        if params.aux:
            if params.dataset == 'CUB':
                attr_file = 'filelists/CUB/attr_array.npy'
            elif params.dataset == 'miniImagenet':
                attr_file = '/test/0Dataset_others/Dataset/mini-imagenet-am3/few-shot-wordemb-{:}.npz'.format(split)

    #  dataset, data_file, attr_file, n_way, k_shot,aux_datasource='attributes'
        test_set = dataloader(params.dataset, novel_file, attr_file,  
                            params.test_n_way, params.n_shot + n_query)



        for i in range(iter_num):
            # n_way = 5, n_support = 5, n_query = 15
            acc = feature_evaluation(test_set, model, adaptation=params.adaptation, **few_shot_params)
            acc_all[split].append(acc)
            trlog['%s_acc' % split].append(acc)
            # writer.add_scalar('test acc', acc, i)
            # break

        acc_array = np.asarray(acc_all[split])
        acc_mean = np.mean(acc_array)
        acc_std = np.std(acc_all[split])
        # writer.close()
        # print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print('%d %s Acc = %4.2f%% +- %4.2f%%' % (iter_num, split, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        with open(record_file , 'a') as f:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            if params.method in ['baseline', 'baseline++'] :
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
            else:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
            acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
            f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )



    torch.save(trlog, trlog_path)
    

