import torch
import matplotlib.pyplot as plt
from io_utils import save_fig
# from io_utils import model_dict, parse_args, get_resume_file, get_trlog


# https://blog.csdn.net/TH_NUM/article/details/86105609



trlog_path = '/test/5Mycode/AttrMissing2/checkpoints/CUB/ResNet12_am3_aug_lr0.01_pwc_5way_5shot/trlog/20200923-224249'

# trlog_path = '/test/5Mycode/AttrMissing2/record/miniImagenet/ResNet12_am3_aug_lr0.01_pwc_5way_5shot/20200914-155652'
# # trlog = torch.load(trlog_path)

save_fig(trlog_path)

# a = '/test/5Mycode/AttrMissing2/checkpoints/miniImagenet/ResNet12_protonet_aug_lr0.01_pwc_5way_5shot/trlog/20200914-161644'
# trlog1 = torch.load(a)
# b = '/test/5Mycode/AttrMissing2/checkpoints/miniImagenet/ResNet12_protonet_aug_lr0.01_pwc_5way_5shot/trlog/20200914-181456'
# trlog2 = torch.load(b)
# c = '/test/5Mycode/AttrMissing2/checkpoints/miniImagenet/ResNet12_protonet_aug_lr0.01_pwc_5way_5shot/trlog/20200914-185224'
# trlog3 = torch.load(c)
# trlog_list = [trlog2, trlog3]
# trlog1['train_loss'] += trlog2['train_loss']  + trlog3['train_loss']    
# trlog1
        
    