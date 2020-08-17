import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class AM3_ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, word_dim=None):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.word_dim = word_dim 
        self.img_feature = model_func
        self.proj = nn.Linear(self.word_dim,  self.feature.final_feat_dim)
        self.mixNet = nn.Sequential(
                        nn.Linear(self.word_dim, 1),
                        nn.Sigmoid()
                      )
        
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
        img_feat, attr_feat = x
        # attribute 
        attr_proj = self.proj(attr_feat)   # shape: (n_way, img_feat_dim=512)
        lambda_c = self.mixNet(attr_proj)

        
        z_support, z_query  = self.parse_feature(img_feat ,is_feature)     # the shape of z is [n_data, n_dim]

        z_support   = z_support.contiguous()
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )  # (n_way*n_query, feat_dim)

        img_proto   = z_support.view(self.n_way, self.n_support, -1 ).mean(1)   # (n_way, feat_dim)
        z_proto = lambda_c * img_proto + (1-lambda_c) * attr_proj   

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )