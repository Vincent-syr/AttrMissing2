version 2020.0817
args: 
Namespace(dataset='CUB', method='baseline', model='ResNet10', n_shot=5, num_classes=200, resume=False, save_freq=50, start_epoch=0, stop_epoch=-1, test_n_way=5, train_aug=True, train_n_way=5, warmup=False)

初始化：
am3: 
word_transformer
    weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor)
    bias: onstant_initializer(0.0)
self_attention: 
    weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor)
    bias: onstant_initializer(0.0)


# stage 1:  pretrain the model and feature extractor
pretrain the model use CUB base category and validate on val category. only use image, no attribute. 


saved model dir: checkpoints/




## 1.1 pretrain the model with attributes and image
python train.py --aux=True --train_aug

before
Namespace(aux='True', dataset='CUB', method='am3_protonet', model='ResNet10', n_shot=5, num_classes=200, resume=False, save_freq=30, start_epoch=0, stop_epoch=-1, test_n_way=5, train_aug=True, train_n_way=5, warmup=False)
training time: about 20 hours

08.24  add mlp_dropout, stop_epoch=100
Namespace(aux='True', dataset='CUB', method='am3_protonet', mlp_dropout=0.7, model='ResNet10', n_shot=5, num_classes=200, resume=False, save_freq=30, start_epoch=0, stop_epoch=-1, test_n_way=5, train_aug=True, train_n_way=5, warmup=False)
training time: about 5 hours

08.27 resume training
python train.py --aux=True --train_aug --resume --start_epoch=199 --stop_epoch=300

save feature:
python save_features.py --aux=True --train_aug

test:
python test_s1.py --aux=True --train_aug

## 1.2 pretrain only use image
python train.py --method=protonet --train_aug

Namespace(aux=False, dataset='CUB', method='protonet', model='ResNet10', n_shot=5, num_classes=200, resume=False, save_freq=30, start_epoch=0, stop_epoch=-1, test_n_way=5, train_aug=True, train_n_way=5, warmup=False)

training time: about 18.3h

08.27 resume training
python train.py --method=protonet --train_aug --resume --start_epoch=199 --stop_epoch=300

save feature:
python save_features.py --method=protonet --train_aug

test:
python test_s1.py --method=protonet --train_aug

## stage 1.3: save feature


python save_feature.py --aux=True --train_aug

Namespace(aux='True', dataset='CUB', method='am3_protonet', model='ResNet10', n_shot=5, save_iter=-1, split='novel', test_n_way=5, train_aug=False, train_n_way=5)

training time: about 11 mins


#stage 2 train_ave
