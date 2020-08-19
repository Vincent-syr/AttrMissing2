import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet
from methods.am3_protonet import AM3_ProtoNet
# from methods.matchingnet import MatchingNet
# from methods.relationnet import RelationNet
# from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file  
from utils import Timer
import warnings

warnings.filterwarnings('ignore')



def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, max_acc=0):
    optimizer = torch.optim.Adam(model.parameters())
    # max_acc = 0
    timer = Timer()

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        start = time.time()
        model.train_loop(epoch, base_loader,  optimizer, aux=params.aux ) #model are called by reference, no need to return 
        end = time.time()
        print("train time = %.2f s" % (end-start))        
        model.eval()

        start = time.time()
        acc = model.test_loop(val_loader, aux=aux)
        end = time.time()
        print("test time = %.2f s" % (end-start))

        # print(" val acc = ", acc)
        print("max_acc = ", max_acc)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc':max_acc}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': max_acc}, outfile)

        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1) / stop_epoch)))
    return model



if __name__=='__main__':

    np.random.seed(10)
    params = parse_args('train')
    print(params)

    aux = params.aux

    base_file = configs.data_dir[params.dataset] + 'base.json' 
    val_file   = configs.data_dir[params.dataset] + 'val.json' 
    word_dim = 312
    
    if aux:
        attr_file = configs.data_dir[params.dataset] + 'attr_array.npy'
        base_file = [base_file, attr_file]
        val_file = [val_file, attr_file]
        

    image_size = 224
    params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
    optimization = 'Adam'

    # meta learning method to pre-training
    if params.n_shot == 1:
        params.stop_epoch = 600
    elif params.n_shot == 5:
        params.stop_epoch = 400     # default
    else:
        params.stop_epoch = 600    


    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small ,16
    # print("n_query = ", n_query)
    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
    # base_datamgr            = SetDataManager(image_size, n_query = n_query,   **train_few_shot_params)
    base_datamgr            = SetDataManager(image_size, n_query = n_query, aux=aux,   **train_few_shot_params)        
    base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        
    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
    # val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
    val_datamgr             = SetDataManager(image_size, n_query = n_query, aux=aux, **test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        

    if params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **train_few_shot_params ) 
    elif params.method == 'am3_protonet':
        model = AM3_ProtoNet(model_dict[params.model], word_dim=word_dim,  **train_few_shot_params)
    else:
        raise ValueError('Unknown method')


    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)


    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if aux:
        params.checkpoint_dir += '_aux'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        # resume_file = get_resume_file(params.checkpoint_dir)
        resume_file = os.path.join(params.checkpoint_dir, str(params.start_epoch) +'.tar')
        # print(resume_file)
        max_acc = 0
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            if 'max_acc' in tmp:
                max_acc = tmp['max_acc']
            model.load_state_dict(tmp['state'])


    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params, max_acc)
