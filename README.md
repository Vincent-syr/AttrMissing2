version 2020.0817
args: 
Namespace(dataset='CUB', method='baseline', model='ResNet10', n_shot=5, num_classes=200, resume=False, save_freq=50, start_epoch=0, stop_epoch=-1, test_n_way=5, train_aug=True, train_n_way=5, warmup=False)

# only image feature
## stage 1:  pretrain the model and feature extractor
pretrain the model use CUB base category and validate on val category. only use image, no attribute. 

code: python train.py  

saved model dir: checkpoints/

code: save_features.py

saved feature dir: features/

training time: about 24 hours




# stage 2: using attribute and image feature


python save_feature.py --aux=True

Namespace(aux='True', dataset='CUB', method='am3_protonet', model='ResNet10', n_shot=5, save_iter=-1, split='novel', test_n_way=5, train_aug=False, train_n_way=5)

training time: about 11 mins