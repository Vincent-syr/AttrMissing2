# change am3 convex combination to cancatation 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate

class AM3_ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, params, word_dim=None):
        super(AM3_ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.word_dim = word_dim 
        self.img_feature = model_func
        self.concateNet = nn.Sequential(
            nn.Linear(self.word_dim + self.feature.final_feat_dim, 300),
            nn.ReLU(),
            nn.Dropout(1.0 - params.mlp_dropout),
            nn.Linear(300, self.feature.final_feat_dim)
        )
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature=False):
        img_feat, attr_feat = x    # attr_feat.shape: (n_way, k_shot+query, feat_dim=312)
                                    # img_feat_(n_way, k_shot+query, img_dim=512)

        fuse_feat = torch.cat((img_feat, attr_feat), dim=2)

        # print("img_feat.shape = ", img_feat.shape)
        z_support, z_query  = self.parse_feature(img_feat ,is_feature)     # the shape of z is [n_data, n_dim]
        # print('model.n_shot = ', self.n_support)
        # print('model.n_query = ', self.n_query)
        # print('z_query',z_query.shape)

        z_support   = z_support.contiguous()
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )  # (n_way*n_query, feat_dim)
        img_proto   = z_support.view(self.n_way, self.n_support, -1 ).mean(1)   # (n_way, feat_dim)
        z_proto = lambda_c * img_proto + (1-lambda_c) * attr_proj   

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores