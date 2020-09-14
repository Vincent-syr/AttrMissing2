import torch
import matplotlib.pyplot as plt
from io_utils import save_fig
# from io_utils import model_dict, parse_args, get_resume_file, get_trlog


# https://blog.csdn.net/TH_NUM/article/details/86105609



# trlog_path = '/test/5Mycode/AttrMissing2/checkpoints/CUB/ResNet12_am3_aug_aux_5way_5shot/trlog/20200909-205958'
trlog_path = '/test/5Mycode/AttrMissing2/checkpoints/miniImagenet/ResNet12_am3_aug_lr0.01_pwc_5way_5shot/trlog/20200913-033545'
# trlog = torch.load(trlog_path)

save_fig(trlog_path)


