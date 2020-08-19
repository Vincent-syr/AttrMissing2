import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 
import warnings

warnings.filterwarnings('ignore')



def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    
    f.close()




if __name__ == '__main__':
    params = parse_args('save_features')
    print(params)

    image_size = 224
    aux = params.aux
    # split = params.split   # split == novel
    loadfile_list = [configs.data_dir[params.dataset] + 'base.json', configs.data_dir[params.dataset] + 'val.json', configs.data_dir[params.dataset] + 'novel.json']
    split_list = ['base', 'val', 'novel']
    # loadfile = configs.data_dir[params.dataset] + split + '.json'
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if aux:
        params.checkpoint_dir += '_aux'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
        
    modelfile   = get_best_file(params.checkpoint_dir)
    model = model_dict[params.model]()    # resnet10
    model = model.cuda()


    # load pre-trained model
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
            
    model.load_state_dict(state)
    model.eval()
    
    for i, loadfile in enumerate(loadfile_list):   # base, val, novel
        
        outfile = os.path.join(params.checkpoint_dir.replace("checkpoints","features"), split_list[i] + '_best' + ".hdf5")  # './features/miniImagenet/Conv4_baseline_aug/novel.hdf5'
        datamgr         = SimpleDataManager(image_size, batch_size = 64)
        data_loader      = datamgr.get_data_loader(loadfile, aug = False)

        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        print('begin save feature in ', split_list[i])
        save_features(model, data_loader, outfile)
