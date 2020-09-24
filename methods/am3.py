# totally am3, nothing to do with meta_template,
# used for multi_gpu traning



import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate
from backbone import init_layer
from utils import euclidean_dist





class AM3(nn.Module):
    def __init__(self, model_func,  n_way, n_support, n_query, params, word_dim=None):
        super(AM3, self).__init__()
        self.word_dim = word_dim

        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query
        self.feature = model_func()
        self.feat_dim   = self.feature.final_feat_dim

        self.transformer = nn.Sequential(
            nn.Linear(self.word_dim, 300),
            nn.ReLU(),
            nn.Dropout(1.0 - params.mlp_dropout),
            nn.Linear(300, self.feature.final_feat_dim)
        )

        # self.mixNet = nn.Sequential(
        #     nn.Linear(self.feature.final_feat_dim, 300),
        #     nn.ReLU(),
        #     nn.Dropout(1.0 - params.mlp_dropout),
        #     nn.Linear(300, 1),
        #     nn.Sigmoid()
        # )

        self.mixNet = nn.Sequential()


        self.attr_ratio = None

        self.loss_fn = nn.CrossEntropyLoss()


    def parse_feature(self, x, is_feature=False):
        if is_feature:
            z_all = x
        else:
            # x = x.view(self.n_way*(self.n_support+self.n_query), *x.size()[2:])
            # x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 

            z_all       = self.feature.forward(x)
            z_all       = z_all.view(self.n_way, self.n_support + self.n_query, -1)
            
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query


    # def correct(self, x, is_feature=False):
    def correct(self, scores):

        # z_all, lambda_c, attr_proj = self.forward(x)
        # scores = self.compute_score(z_all, lambda_c, attr_proj)

        y_query = np.repeat(range(self.n_way ), self.n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)


    # def correct(self, z_all, lambda_c, attr_proj):
        
    #     y_query = np.repeat(range( self.n_way ), self.n_query )


# syn_correct, syn_count = fsl_model.correct([img_feat, syn_attr], is_feature=True)

                # z_all, lambda_c, attr_proj = model.forward(x)
                # scores = model.compute_score(z_all, lambda_c, attr_proj)
                # correct_this, count_this = model.correct(scores)
                
    def correct_quick(self, x):
        """[summary] 

        Args:
            x ([type]): z_all, attr_feat, and z_all is img feat extracted from backbone

        Returns:
            [type]: correct_this, count_this
        """
        z_all, attr_feat = x
        attr_proj = self.transformer(attr_feat)   # (n_way, img_feat_dim)
        lambda_c = self.mixNet(attr_proj)   # (n_way, 1)
        # z_all, lambda_c, attr_proj = self.forward(x)
        scores = self.compute_score(z_all, lambda_c, attr_proj)
        correct_this, count_this = self.correct(scores)
        return correct_this, count_this

    # def forward(self, img_feat, attr_feat, is_feature=False):
    def forward(self, x):
        """[summary]

        Args:
            x[0] ([type]): image:  (n_way*(k_shot+query), 3, 224, 224)
            x[1] ([type]): attribute: (n_way, feat_dim=312)
        """

        img_feat, attr_feat = x
        attr_proj = self.transformer(attr_feat)   # (n_way, img_feat_dim)
        # lambda_c = self.mixNet(attr_proj)   # (n_way, 1)
        lambda_c = 0.85 * torch.ones((len(attr_proj), 1)).cuda()

        z_all       = self.feature.forward(img_feat)

        return z_all, lambda_c, attr_proj






    def compute_score(self, z_all, lambda_c, attr_proj):
        z_all       = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]
        # z_query     = z_query.view(self.n_way* self.n_query, -1 )
        z_query     = z_query.contiguous().view(-1, z_query.shape[-1])
        
        img_proto   = z_support.mean(1)   # (n_way, feat_dim)

        
        z_proto = lambda_c * img_proto + (1-lambda_c) * attr_proj
        
        z_proto_abs = lambda_c * img_proto.abs() + (1-lambda_c) * attr_proj.abs()
        self.attr_ratio = ((1-lambda_c) * attr_proj.abs()).mean() / z_proto_abs.mean()
        # self.attr_ratio = ((1-lambda_c) * attr_proj.abs()).mean() / (img_proto.abs().mean() + )

        # print("attr_ratio = ", self.attr_ratio.data)
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

        



    def get_coef(self, attr_feat):
        """
        :param attr_feat: shape: (n_way, k_shot+query, feat_dim=312)
        :return: coefficient params lambda_c, with the attribute just used

        """
        # attr_feat = attr_feat.mean(1)   # (n_way, feat_dim)
        attr_proj = self.transformer(attr_feat)   # (n_way, img_feat_dim)
        lambda_c = self.mixNet(attr_proj)   # (n_way, 1)
        lambda_c = lambda_c.mean()
        return float(lambda_c)

    # def compute_ratio(self, z_all, lambda_c, attr_proj):
    #     # img_proto   = z_all.mean(1)   # (n_way, feat_dim)
    #     z_all       = z_all.view(self.n_way, self.n_support + self.n_query, -1)
    #     z_support   = z_all[:, :self.n_support]
    #     img_proto   = z_support.mean(1)   # (n_way, feat_dim)

    #     z_mean = (z_all * (1-lambda_c)).mean()
    #     attr_mean = (attr_proj * (lambda_c)).mean()
    #     attr_ratio = attr_mean / (z_mean + attr_mean)
    #     return attr_ratio

