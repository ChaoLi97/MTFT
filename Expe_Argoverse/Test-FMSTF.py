import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
from utils import list_to_tensor,Shuffle_data,calc_ED_error,Rand_Mask
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


'-----------------------------------------------------------------------------------'
model=FMSTF().cuda()
model.load_state_dict(torch.load('../PretrainParams/Argoverse/FMSTF_189.pth'))


'------------------------------------------------------------------------------------------------'
preds=torch.zeros((1,30,2)).cuda()
gts=torch.zeros((1,30,2)).cuda()
model.eval()
with torch.no_grad():
    for t in tqdm(range(int(len(dataset_test[0])/args.batch_size))):
        # (1) Load data and convert it to tensor
        his_list, fut_list, clasX_list, clasY_list = dataset_test[0][t * args.batch_size:(t + 1) * args.batch_size],dataset_test[1][t * args.batch_size:(t + 1) * args.batch_size], \
                                                    dataset_test[2][t * args.batch_size:(t + 1) * args.batch_size],dataset_test[3][t * args.batch_size:(t + 1) * args.batch_size]
        his, fut_label, clasX_label, clasY_label, point, social_mask = list_to_tensor(his_list, fut_list, clasX_list,clasY_list)

        # (2) masking and prediction
        his_train, obs_mask = Rand_Mask(copy.deepcopy(his), copy.deepcopy(scale_mask), point,rate_low=0.0, rate_high=0.3)
        fut_pre= model(his_train, obs_mask, social_mask, point)

        # (3) Save predictions and gt
        preds = torch.cat((preds, fut_pre), 0)
        gts = torch.cat((gts, fut_label), 0)


preds=preds[1:,:,:]
gts=gts[1:,:,:]
ade, fde,MR = calc_ED_error(preds,gts)
print(ade.item(),fde.item(),MR.item())




