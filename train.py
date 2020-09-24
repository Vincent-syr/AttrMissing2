import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from tensorboardX import SummaryWriter

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet
# from methods.am3_protonet import AM3_ProtoNet
from methods.am3 import AM3
from methods.protonet_multi_gpu import ProtoNetMulti
from io_utils import model_dict, parse_args, get_resume_file, get_trlog, save_fig
from utils import Timer
import warnings

warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

use_gpu = torch.cuda.is_available()

def adjust_learning_rate(params, optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    if params.lr_anneal == 'const':
        pass
    elif params.lr_anneal == 'pwc':
        for param_group in optimizer.param_groups:
            if epoch >=200 and epoch < 250:
                param_group['lr'] = init_lr*0.1
            elif epoch>=250:
                param_group['lr'] = init_lr*0.01
            # elif epoch ==300:
            #     param_group['lr'] = init_lr*0.001
    
    elif params.lr_anneal == 'exp':
        pass



    






def train(base_loader, val_loader, model, start_epoch, stop_epoch, params, max_acc=0):
    """[summary] train with single GPU

    Args:
        base_loader ([type]): [description]
        val_loader ([type]): [description]
        model ([type]): [description]
        start_epoch ([type]): [description]
        stop_epoch ([type]): [description]
        params ([type]): [description]
        max_acc (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
# %%
    trlog = get_trlog(params)
    trlog_dir = os.path.join(params.checkpoint_dir, 'trlog')
    if not os.path.isdir(trlog_dir):
        os.makedirs(trlog_dir)
    trlog_path = os.path.join(trlog_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))   # '20200909-185444'
    init_lr = params.init_lr
    if params.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.001)
    elif params.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError('Unknown Optimizer !!')

    # max_acc = 0
    timer = Timer()
    print_freq = 20

    if not os.path.isdir(params.model_dir):
        os.makedirs(params.model_dir)

    tmp = params.checkpoint_dir.split('/')[-1]
    writer_dir = 'runs/pretrain_%s_%s' % (params.dataset, tmp)
    writer = SummaryWriter(log_dir=writer_dir)
    # %%
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        start = time.time()
        cum_loss=0
        acc_all = []
        iter_num = len(base_loader)

        for i, (x,_ ) in enumerate(base_loader, 1):

            if params.aux:
                # both x[0] and x[1]: shape(n_way*(n_shot+query), 3, 224,224)
                x[0] = x[0].cuda()   # shape(n_way*(n_shot+query), 3, 224,224)
                x[1] = x[1].view(model.n_way, -1, x[1].shape[-1]).cuda()
                x[1] = x[1].mean(1)   # (n_way, feat_dim)
                z_all, lambda_c, attr_proj = model.forward(x)
                scores = model.compute_score(z_all, lambda_c, attr_proj)
                # correct_this, count_this = model.correct(z_all, lambda_c, attr_proj)
            else:
                # x = x.view(-1, *x.shape[2:]).cuda()    # shape(n_way*(n_shot+query), 3, 224,224)
                x = x.cuda()
                z_all = model.forward(x)

                scores = model.compute_score(z_all)
            
            correct_this, count_this = model.correct(scores)

            y_query = torch.from_numpy(np.repeat(range(model.n_way ), model.n_query))
            y_query = Variable(y_query.cuda())
            loss = model.loss_fn(scores, y_query )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cum_loss = cum_loss + loss.data[0]
            avg_loss = cum_loss/float(i)

            acc_all.append(correct_this/ float(count_this)*100)
            acc_mean = np.array(acc_all).mean()
            acc_std = np.std(acc_all)
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc {:.2f}%'.format(epoch, i, len(base_loader), avg_loss, acc_mean))
        
        print('Train Acc = %4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        end = time.time()
        print("train time = %.2f s" % (end-start))
        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.add_scalar('train_acc', acc_mean, epoch)
        trlog['train_loss'].append(avg_loss)
        trlog['train_acc'].append(acc_mean)
        torch.cuda.empty_cache()

        # %%
        model.eval()
        start = time.time()
        iter_num = len(val_loader)
        with torch.no_grad():
            acc_all = []
            for x,_ in val_loader:
                if params.aux:
                    x[0] = x[0].cuda()   # shape(n_way*(n_shot+query), 3, 224,224)
                    x[1] = x[1].view(model.n_way, -1, x[1].shape[-1]).cuda()
                    x[1] = x[1].mean(1)   # (n_way, feat_dim)            
                    z_all, lambda_c, attr_proj = model.forward(x)
                    scores = model.compute_score(z_all, lambda_c, attr_proj)

                else:
                    # x = x.view(-1, *x.shape[2:]).cuda()    # shape(n_way*(n_shot+query), 3, 224,224)
                    x = x.cuda()
                    z_all = model.forward(x)
                    scores = model.compute_score(z_all)

                correct_this, count_this = model.correct(scores)
                acc_all.append(correct_this/ float(count_this)*100)

            acc_all  = np.array(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
            # acc = acc_mean
            writer.add_scalar('val_acc', acc_mean, epoch)
            trlog['val_acc'].append(acc_mean)

        end = time.time()
        print("validation time = %.2f s" % (end-start))

        # %%
        if acc_mean > trlog['max_acc'] : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            trlog['max_acc_epoch'] = epoch
            trlog['max_acc']  = acc_mean

            print("best model! save...")
            outfile = os.path.join(params.model_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc':max_acc}, outfile)

        # save model and trlog regularly
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.model_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': max_acc}, outfile)
            torch.save(trlog, trlog_path)

        torch.cuda.empty_cache()  
        # best epoch and val acc
        print('best epoch = %d, best val acc = %.2f%%' % (int(trlog['max_acc_epoch']), trlog['max_acc']))
        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1) / stop_epoch)))

        print('epoch: ', epoch, 'lr: ', optimizer.param_groups[0]['lr'])
        trlog['lr'].append(optimizer.param_groups[0]['lr'])
        adjust_learning_rate(params, optimizer, epoch, init_lr)
        # %%
    writer.close()
    save_fig(trlog_path)
    return model



def train_multi_gpu(base_loader, val_loader, model, start_epoch, stop_epoch, params, max_acc=0):
# %%
    trlog = get_trlog(params)
    trlog_dir = os.path.join(params.checkpoint_dir, 'trlog')
    if not os.path.isdir(trlog_dir):
        os.makedirs(trlog_dir)
    trlog_path = os.path.join(trlog_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))   # '20200909-185444'

    init_lr = params.init_lr
    if params.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.001)
    elif params.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError('Unknown Optimizer !!')

    # max_acc = 0
    timer = Timer()
    print_freq = 50

    if not os.path.isdir(params.model_dir):
        os.makedirs(params.model_dir)

    tmp = params.checkpoint_dir.split('/')[-1]
    writer_dir = 'runs/pretrain_%s_%s' % (params.dataset, tmp)
    writer = SummaryWriter(log_dir=writer_dir)
    # %%
    for epoch in range(start_epoch,stop_epoch):
        adjust_learning_rate(params, optimizer, epoch, init_lr)

        model.train()
        start = time.time()
        cum_loss=0
        acc_all = []
        lambda_c_list = []
        attr_ratio_list = []
        iter_num = len(base_loader)

        for i, (x,_ ) in enumerate(base_loader, 1):

            if params.aux:
                # both x[0] and x[1]: shape(n_way*(n_shot+query), 3, 224,224)
                x[0] = x[0].cuda()   # shape(n_way*(n_shot+query), 3, 224,224)
                x[1] = x[1].view(model.module.n_way, -1, x[1].shape[-1]).cuda()
                x[1] = x[1].mean(1)   # (n_way, feat_dim)
                z_all, lambda_c, attr_proj = model.forward(x)
                scores = model.module.compute_score(z_all, lambda_c, attr_proj)
                attr_ratio = model.module.attr_ratio
            else:
                x = x.cuda()
                z_all = model.forward(x)

                scores = model.module.compute_score(z_all)
            
            correct_this, count_this = model.module.correct(scores)

            y_query = torch.from_numpy(np.repeat(range(model.module.n_way ), model.module.n_query))
            y_query = Variable(y_query.cuda())
            loss = model.module.loss_fn(scores, y_query )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cum_loss = cum_loss + loss.data[0]
            avg_loss = cum_loss/float(i)

            acc_all.append(correct_this/ float(count_this)*100)
            lambda_c_list.append(lambda_c.mean())
            attr_ratio_list.append(attr_ratio.item())

            acc_mean = np.array(acc_all).mean()
            acc_std = np.std(acc_all)

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc {:.2f} | lambda {:.2f}| attr_ratio {:.2f}%'.format(epoch, i, len(base_loader), avg_loss, acc_mean, lambda_c.mean(), attr_ratio))
                # print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc {:.2f}%'.format(epoch, i, len(base_loader), avg_loss, acc_mean))



        lambda_c_mean = np.array(lambda_c_list).mean()
        attr_ratio_mean = np.array(attr_ratio_list).mean()
        print('Train Acc = %4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        end = time.time()
        print("train time = %.2f s" % (end-start))
        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.add_scalar('train_acc', acc_mean, epoch)
        trlog['train_loss'].append(avg_loss)
        trlog['train_acc'].append(acc_mean)
        trlog['epoch'].append(epoch)
        trlog['attr_ratio'].append(attr_ratio_mean)
        trlog['lambda_c'].append(lambda_c_mean)
        torch.cuda.empty_cache()

        # %%
        model.eval()
        start = time.time()
        iter_num = len(val_loader)
        with torch.no_grad():
            acc_all = []
            for x,_ in val_loader:
                if params.aux:
                    x[0] = x[0].cuda()   # shape(n_way*(n_shot+query), 3, 224,224)
                    x[1] = x[1].view(model.module.n_way, -1, x[1].shape[-1]).cuda()
                    x[1] = x[1].mean(1)   # (n_way, feat_dim)            
                    z_all, lambda_c, attr_proj = model.forward(x)
                    scores = model.module.compute_score(z_all, lambda_c, attr_proj)

                else:
                    x = x.cuda()
                    z_all = model.forward(x)
                    scores = model.module.compute_score(z_all)

                correct_this, count_this = model.module.correct(scores)
                acc_all.append(correct_this/ float(count_this)*100)

            acc_all  = np.array(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
            # acc = acc_mean
            writer.add_scalar('val_acc', acc_mean, epoch)
            trlog['val_acc'].append(acc_mean)

        end = time.time()
        print("validation time = %.2f s" % (end-start))

        # %%
        if acc_mean > trlog['max_acc'] : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            trlog['max_acc_epoch'] = epoch
            trlog['max_acc']  = acc_mean

            print("best model! save...")
            outfile = os.path.join(params.model_dir, 'best_model.tar')
            # https://docs.python.org/zh-cn/3/tutorial/errors.html
            try:
                torch.save({'epoch':epoch, 'state':model.module.state_dict(), 'max_acc':max_acc}, outfile)
            except Exception as inst:
                print(inst) 

        # save model and trlog regularly
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.model_dir, '{:d}.tar'.format(epoch))
            try:
                torch.save({'epoch':epoch, 'state':model.module.state_dict(), 'max_acc':max_acc}, outfile)
                torch.save(trlog, trlog_path)

            except Exception as inst:
                print(inst)             

        torch.cuda.empty_cache()  
        # best epoch and val acc
        print('best epoch = %d, best val acc = %.2f%%' % (int(trlog['max_acc_epoch']), trlog['max_acc']))
        # cumulative cost time / total time predicted
        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch+1-start_epoch) / float(stop_epoch-start_epoch))))

        print('epoch: ', epoch, 'lr: ', optimizer.param_groups[0]['lr'])
        trlog['lr'].append(optimizer.param_groups[0]['lr'])
        # %%
    writer.close()
    # save_fig(trlog_path)
    return model




if __name__=='__main__':

    np.random.seed(10)
    params = parse_args('train')
    print(params)
    if params.method == 'am3':
        params.aux = True
    else:
        params.aux = False
    aux = params.aux
    
    base_file = configs.data_dir[params.dataset] + 'base.json' 
    val_file   = configs.data_dir[params.dataset] + 'val.json' 
    
    if aux:
        attr_file = configs.data_dir[params.dataset] + 'attr_array.npy'
        base_file = [base_file, attr_file]
        val_file = [val_file, attr_file]
        
    if params.dataset == 'CUB':
        image_size = 224
        word_dim = 312

    elif params.dataset == 'miniImagenet':
        image_size = 84
        word_dim = 300
    print('image_size = ', image_size)

    # params.n_query = 8
    # params.n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce params.n_query to keep batch size small ,16
    print("n_query = ", params.n_query)
    # train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
    train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot, n_query=params.n_query) 

    # print('base_file = ', base_file)
    # exit(0)

    base_datamgr            = SetDataManager(image_size, aux=aux, n_episode=params.n_episode, **train_few_shot_params)
    base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        
    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, n_query = params.n_query) 
    val_datamgr             = SetDataManager(image_size, aux=aux, n_episode=params.n_episode , **test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader(val_file, aug = False) 

    if params.method == 'protonet':
        # model           = ProtoNet( model_dict[params.model], **train_few_shot_params ) 
        model = ProtoNetMulti(model_dict[params.model], params=params, **train_few_shot_params)
    elif params.method == 'am3':
        model = AM3(model_dict[params.model], params=params, word_dim=word_dim,  **train_few_shot_params)
    else:
        raise ValueError('Unknown method')
    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)


    if params.train_aug:
        params.checkpoint_dir += '_aug'
        
    params.checkpoint_dir += '_lr%s_%s' % (str(params.init_lr), params.lr_anneal)

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    print('checkpoint_dir = ', params.checkpoint_dir)
    params.model_dir = os.path.join(params.checkpoint_dir, 'model')

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    max_acc = 0
    # if params.resume:
    if params.start_epoch != 0:
        # resume_file = get_resume_file(params.checkpoint_dir)
        resume_file = os.path.join(params.model_dir, str(params.start_epoch) +'.tar')
        # print(resume_file)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            if 'max_acc' in tmp:
                max_acc = tmp['max_acc']
            model.load_state_dict(tmp['state'])
            print('resume training')
    
    # if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))  
    print('gpu device: ', list(range(torch.cuda.device_count())))

    model = train_multi_gpu(base_loader, val_loader,  model, start_epoch, stop_epoch, params, max_acc)


    # if torch.cuda.device_count() > 1:
    #     model = train_multi_gpu(base_loader, val_loader,  model, start_epoch, stop_epoch, params, max_acc)
    # else:
    #     model = train(base_loader, val_loader,  model, start_epoch, stop_epoch, params, max_acc)
