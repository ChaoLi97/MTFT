import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
from utils import list_to_tensor,Shuffle_data,Rand_Mask
import copy
from FMSTF import FMSTF
from Args import args
import random
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


'-----------------------------------------------------------------------------------'
with open(args.path_test, 'rb') as file:
    dataset_test= pickle.load(file)
scale_mask=torch.load(args.path_mask).cuda() #scale mask


'------------------------------------------------------------------------------------'
model=FMSTF().cuda()
model.load_state_dict(torch.load('../PretrainParams/HighD/FMSTF_54.pth'))
loss_mse=nn.MSELoss()

'-------------------------------------------------------------------------------------------------'
loss_test1, loss_test2, loss_test3, loss_test4, loss_test5 = [], [], [], [], []
model.eval()
for t in tqdm(range(int(len(dataset_test[0])/args.batch_size))):
    # (1) Load data and convert it to tensor
    his_list, fut_list, clasX_list, clasY_list = dataset_test[0][t * args.batch_size:(t + 1) * args.batch_size],dataset_test[1][t * args.batch_size:(t + 1) * args.batch_size], \
                                                 dataset_test[2][t * args.batch_size:(t + 1) * args.batch_size],dataset_test[3][t * args.batch_size:(t + 1) * args.batch_size]
    his, fut_label, clasX_label, clasY_label, point, social_mask = list_to_tensor(his_list, fut_list, clasX_list,clasY_list)

    # (2) masking and prediction
    his_train, obs_mask = Rand_Mask(copy.deepcopy(his), copy.deepcopy(scale_mask), point,rate_low=0.0, rate_high=0.3)
    fut_pre = model(his_train, obs_mask, social_mask, point)

    # (3) calculate loss
    loss_test1.append(loss_mse(fut_pre[:, 0:5 * 1, :].float(), fut_label[:, 0:5 * 1, :].float()).item())
    loss_test2.append(loss_mse(fut_pre[:, 0:5 * 2, :].float(), fut_label[:, 0:5 * 2, :].float()).item())
    loss_test3.append(loss_mse(fut_pre[:, 0:5 * 3, :].float(), fut_label[:, 0:5 * 3, :].float()).item())
    loss_test4.append(loss_mse(fut_pre[:, 0:5 * 4, :].float(), fut_label[:, 0:5 * 4, :].float()).item())
    loss_test5.append(loss_mse(fut_pre[:, 0:5 * 5, :].float(), fut_label[:, 0:5 * 5, :].float()).item())


print('test_loss1:',np.sqrt(sum(loss_test1)/len(loss_test1)))
print('test_loss2:',np.sqrt(sum(loss_test2)/len(loss_test2)))
print('test_loss3:',np.sqrt(sum(loss_test3)/len(loss_test3)))
print('test_loss4:',np.sqrt(sum(loss_test4)/len(loss_test4)))
print('test_loss5:',np.sqrt(sum(loss_test5)/len(loss_test5)))

