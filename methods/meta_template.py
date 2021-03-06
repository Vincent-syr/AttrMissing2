import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

class MetaTemplate(nn.Module):
# class MetaTemplate(nn.DataParallel):
    def __init__(self, model_func, n_way, n_support, n_query, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        # self.n_query    = -1 #(change depends on input) 
        self.n_query    = n_query
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 

            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
            
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x, is_feature=False):

        scores = self.set_forward(x, is_feature)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)



    def train_loop(self, epoch, train_loader, optimizer, aux=False ):
        print_freq = 20

        avg_loss=0
        # if not aux:
        for i, (x,_ ) in enumerate(train_loader, 1):
            if aux:  # x = (img, attr)   
                # (n_way * (k+q), feat)  --> (n_way, k+q, feat)
                x[0] = x[0].cuda()
                x[0] = x[0].view(self.n_way, -1, x[0].shape[1], x[0].shape[2], x[0].shape[3])
                x[1] = x[1].cuda()
                x[1] = x[1].view(self.n_way, -1, x[1].shape[-1])
                # print("x[0].shape = ", x[0].shape)
                # print("x[1].shape = ", x[1].shape)
                # self.n_query = x[0].size(1) - self.n_support
            else:
                # self.n_query = x.size(1) - self.n_support           
                pass


            # if self.change_way:
            #     self.n_way  = x.size(0)
                
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.data[0]

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
        return avg_loss

    def test_loop(self, test_loader, aux=False ,record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):

            if aux:  # x = (img, attr)   
                # (n_way * (k+q), feat)  --> (n_way, k+q, feat)
                x[0] = x[0].view(self.n_way, -1, x[0].shape[1], x[0].shape[2], x[0].shape[3]).cuda()
                x[1] = x[1].view(self.n_way, -1, x[1].shape[-1]).cuda()
                # print("x[0].shape = ", x[0].shape)
                # print("x[1].shape = ", x[1].shape)
                # self.n_query = x[0].size(1) - self.n_support
            else:
                x = x.cuda()
                # self.n_query = x.size(1) - self.n_support        

            # if self.change_way:
            #     self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
