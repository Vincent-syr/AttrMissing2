#vae proto model
import copy
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import methods.vae_backbone as models

import warnings
import copy

warnings.filterwarnings('ignore')


class Model(nn.Module):

    def __init__(self,hyperparameters):
        super(Model, self).__init__()

        # self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']   # 'attributes'
        self.all_data_sources  = ['resnet_features',self.auxiliary_data_source]  # ['resnet feature', 'attributes']
        self.DATASET = hyperparameters['dataset']   # 'CUB'

        self.img_dim = hyperparameters['img_dim']
        self.attr_dim = hyperparameters['attr_dim']

        self.k_shot = hyperparameters['num_shots']
        self.n_way = hyperparameters['num_ways']
        self.batch_size = hyperparameters['batch_size']  # 1
        self.n_episodes = hyperparameters['n_episodes']    # 100

        self.latent_size = hyperparameters['latent_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.classifier_batch_size = 32

        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        # self.cls_train_epochs = hyperparameters['cls_train_steps']   # 23

        self.reparameterize_with_noise = True
        

        feature_dimensions = [self.img_dim, self.attr_dim]

        # Here, the encoders and decoders for all modalities are created and put into dict

        self.encoder = {}  # encoder both image and attribute

        for datatype, dim in zip(self.all_data_sources, feature_dimensions):

            self.encoder[datatype] = models.encoder_template(dim,self.latent_size,self.hidden_size_rule[datatype]).cuda()

            # print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.hidden_size_rule[datatype]).cuda()
       

        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize +=  list(self.encoder[datatype].parameters())
            parameters_to_optimize +=  list(self.decoder[datatype].parameters())
        self.optimizer  = optim.Adam(parameters_to_optimize ,lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=True)

        elif self.reco_loss_function=='l1':   # l1
            self.reconstruction_criterion = nn.L1Loss(size_average=True)




    def train_vae(self, base_dataset, current_epoch):
        # turn into  trian mode
        losses = []   # loss list
        print_freq = 20
        avg_loss=0   # average list
        for key, value in self.encoder.items():
            self.encoder[key].train()
        for key, value in self.decoder.items():
            self.decoder[key].train()

        for iters in range(0, self.n_episodes):
            _, data_from_modalities = base_dataset.next_batch(self.batch_size)

            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].cuda()
                data_from_modalities[j].requires_grad = False

            # CA loss + DA loss + VAE loss
            loss = self.trainstep(data_from_modalities[0], data_from_modalities[1], current_epoch)        
            avg_loss += loss

            if iters % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(current_epoch, iters, self.n_episodes, avg_loss/float(iters+1)))

            if iters % print_freq==0 and iters>0:
                losses.append(loss)

        return losses



    def test_loop(self, val_dataset, fsl_model, miss_rate):
        """use missing rate of attribute
        """
        correct =0
        count = 0
        syn_acc_all = []
        raw_acc_all = []
        iter_num = self.n_episodes

        # turn into val mode
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()     

        for i in range(iter_num):

            label, data_from_modalities = val_dataset.next_batch(self.batch_size)
            img_feat = data_from_modalities[0].cuda()      # (n*b, k+q, f1)
            attr_feat = data_from_modalities[1].cuda()     # (n*b, f2)
            
            syn_idx = torch.randperm(img_feat.shape[0])[:int(miss_rate*img_feat.shape[0])]    # (n2)
            tmp_attr = self.generate_attr(img_feat[syn_idx])    # (n2, f2)
            # synthesis attr and raw attr use for validation test
            syn_attr = copy.deepcopy(attr_feat)
            syn_attr[syn_idx] = tmp_attr
            # print("syn_idx = ", syn_idx)
            raw_attr = attr_feat

            syn_attr_exp = torch.unsqueeze(syn_attr, 1).expand(-1, img_feat.shape[1], -1)   # (n*b, k+q, f2)
            raw_attr_exp = torch.unsqueeze(raw_attr, 1).expand(-1, img_feat.shape[1], -1)
            # print("img_feat.shape = ", img_feat.shape)
            # print('syn_attr_exp.shape = ', syn_attr_exp.shape)

            # scores = fsl_model.set_forward([img_feat, syn_attr_exp], is_feature=True)
            # pred = scores.data.cpu().numpy().argmax(axis=1)
            # y = np.repeat(range(fsl_model.n_way), fsl_model.n_query)
            # raw_acc = np.mean(pred == y) * 100




            syn_correct, syn_count = fsl_model.correct([img_feat, syn_attr_exp], is_feature=True)
            raw_correct, raw_count = fsl_model.correct([img_feat, raw_attr_exp], is_feature=True)

            syn_acc_all.append( syn_correct / syn_count * 100 )
            raw_acc_all.append( raw_correct / raw_count * 100 )

        syn_acc_all  = np.asarray(syn_acc_all)
        raw_acc_all  = np.asarray(raw_acc_all)

        syn_acc_mean = np.mean(syn_acc_all)
        raw_acc_mean = np.mean(raw_acc_all)
        syn_acc_std  = np.std(syn_acc_all)
        raw_acc_std  = np.std(raw_acc_all)

        print('%d Syn_Attribute: Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  syn_acc_mean, 1.96* syn_acc_std/np.sqrt(iter_num)))
        print('%d Raw_Attribute: Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  raw_acc_mean, 1.96* raw_acc_std/np.sqrt(iter_num)))

        return syn_acc_mean, raw_acc_mean


    def generate_attr(self, img):
        """generate attribute from image

        Args:
            img ([type]): (n*b, k, f1)

        Returns:
            attr_from_img   (n*b, f2)
        """
        mu_img, logvar_img = self.encoder['resnet_features'](img)  # (30,5,64), (30,5,64)
        mu_proto, logvar_proto = mu_img.mean(1), logvar_img.mean(1)   # (30,64), (30,64)

        z_from_proto = self.reparameterize(mu_proto, logvar_proto)   # (30,64)
    
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_proto)   # (n*b, 312)

        return att_from_img


    def trainstep(self, img, att, current_epoch):
        """[summary]

        Args:
            img ([type]): (n*b, k, f1)
            att ([type]): (n*b, f2)
        """
        ##############################################
        # Encode image features and additional
        # features
        ##############################################

        mu_img, logvar_img = self.encoder['resnet_features'](img)  # (30,5,64), (30,5,64)
        mu_proto, logvar_proto = mu_img.mean(1), logvar_img.mean(1)   # (30,64), (30,64)
        
        z_from_img = self.reparameterize(mu_img, logvar_img)
        z_from_proto = self.reparameterize(mu_proto, logvar_proto)   # (30,64)
    
        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)  # expect ouput: (n*b k, 64), (n*b k, 64)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################
        img_from_img = self.decoder['resnet_features'](z_from_img)
        # exit(0)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)


        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        img_from_att = self.decoder['resnet_features'](z_from_att)   # (n*b, 2048)
        img_from_att = img_from_att.unsqueeze(1).expand(img.size())   # (n*b, k, 2048)

        att_from_img = self.decoder[self.auxiliary_data_source](z_from_proto)   # (n*b, 312)
        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)
        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_proto - mu_proto.pow(2) - logvar_proto.exp()))
        KLD = KLD / logvar_att.shape[0]   # n*b

        # KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
        #       + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(torch.sum((mu_proto - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_proto.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
        # distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
        #                       torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

        distance = distance.sum()
        distance = distance / logvar_att.shape[0]   # n*b

        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################
        f1 = 1.0*(current_epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])])

        f2 = 1.0 * (current_epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

        f3 = 1.0*(current_epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        loss = reconstruction_loss - beta * KLD

        if cross_reconstruction_loss>0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor >0:
            loss += distance_factor*distance

        loss.backward()

        self.optimizer.step()

        return loss.item()


    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            # eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps = torch.cuda.FloatTensor(logvar.size()[:-1]).normal_(0,1)
            # eps  = eps.expand(sigma.size())
            eps = torch.unsqueeze(eps,len(sigma.size())-1).expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu