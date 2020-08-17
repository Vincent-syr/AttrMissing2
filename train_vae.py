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

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.protonet import ProtoNet
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file














if __name__ == "__main__":
    params = parse_args('test')

    acc_all = []
    split_list = ['base', 'val']
    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)   # 5 way, 5 shot

    model           = ProtoNet( model_dict[params.model], **few_shot_params )
    model = model.cuda()
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    # load  pre-trained-model params
    modelfile   = get_best_file(checkpoint_dir)
    tmp = torch.load(modelfile)
    model.load_state_dict(tmp['state'])
    for split_str in split_list:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +"_best.hdf5") #defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)



        