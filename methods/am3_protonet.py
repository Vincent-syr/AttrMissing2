import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate
from backbone import init_layer
from utils import euclidean_dist



class AM3_ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, n_query, params, word_dim=None):
        super(AM3_ProtoNet, self).__init__( model_func,  n_way, n_support, n_query)
        self.word_dim = word_dim
        # self.img_feature = model_func
        self.transformer = nn.Sequential(
            nn.Linear(self.word_dim, 300),
            nn.ReLU(),
            nn.Dropout(1.0 - params.mlp_dropout),
            nn.Linear(300, self.feature.final_feat_dim)
        )

        self.mixNet = nn.Sequential(
            nn.Linear(self.feature.final_feat_dim, 300),
            nn.ReLU(),
            nn.Dropout(1.0 - params.mlp_dropout),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )
        
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature=False):
        img_feat, attr_feat = x    # attr_feat.shape: (n_way, k_shot+query, feat_dim=312)
                                    # img_feat_(n_way, k_shot+query, img_dim=512)
        # attribute
        # print('img_feat.shape = ', img_feat.shape)
        # print('attr_feat.shape = ', attr_feat.shape)

        attr_feat = attr_feat.mean(1)   # (n_way, feat_dim)
        attr_proj = self.transformer(attr_feat)   # (n_way, img_feat_dim)
        lambda_c = self.mixNet(attr_proj)   # (n_way, 1)
        # print("lambda_c = ", lambda_c)
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


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


    def get_coef(self, attr_feat):
        """
        :param attr_feat: shape: (n_way, k_shot+query, feat_dim=312)
        :return: coefficient params lambda_c, with the attribute just used

        """
        attr_feat = attr_feat.mean(1)   # (n_way, feat_dim)
        attr_proj = self.transformer(attr_feat)   # (n_way, img_feat_dim)
        lambda_c = self.mixNet(attr_proj)   # (n_way, 1)
        lambda_c = lambda_c.mean()
        return float(lambda_c)

        