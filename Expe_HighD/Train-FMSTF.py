import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
from utils import Rand_Mask,list_to_tensor,Shuffle_data
import copy
from FMSTF import FMSTF
from Args import args
from torch.optim.lr_scheduler import StepLR
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"



'1 读取训练数据与验证数据------------------------------------------------------------------------------------'
with open(args.path_train, 'rb') as file:
    dataset_train= pickle.load(file)
with open(args.path_val, 'rb') as file:
    dataset_val= pickle.load(file)
scale_mask=torch.load(args.path_mask).cuda() #scale mask


'2 定义模型、优化器、损失函数等------------------------------------------------------------------------------------'
model=FMSTF().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.6*args.lr)
scheduler = StepLR(optimizer,step_size=25, gamma=0.5)
loss_mse=nn.MSELoss()


'3 模型训练与验证-------------------------------------------------------------------------------------------------'
loss_train,loss_val=[],[]
min_val=100 #Initialize minimum validation loss失
for i in range(args.epochs):

    #3.1 Training----------------------------------------------------------------------------------------
    model.train()
    dataset_train=Shuffle_data(dataset_train)
    print('Start Training epoch: ', i)
    for t in tqdm(range(int(len(dataset_train[0])/args.batch_size))):
        #(1)  Load data and convert it to tensor
        his_list,fut_list,clasX_list,clasY_list=dataset_train[0][t*args.batch_size:(t+1)*args.batch_size],dataset_train[1][t*args.batch_size:(t+1)*args.batch_size],\
                                                dataset_train[2][t*args.batch_size:(t+1)*args.batch_size],dataset_train[3][t*args.batch_size:(t+1)*args.batch_size]
        his,fut_label,_,_,point,social_mask=list_to_tensor(his_list,fut_list,clasX_list,clasY_list)

        #(2) masking and prediction
        his_train,obs_mask=Rand_Mask(copy.deepcopy(his),copy.deepcopy(scale_mask),point,rate_low=0.0,rate_high=0.9)
        fut_pre=model(his_train,obs_mask,social_mask,point)

        #(3) Calculate losses
        loss_reg = loss_mse(fut_pre.float(), fut_label.float())
        loss = loss_reg

        #(4) Record losses and update model parameters
        loss_train.append(loss_reg.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()  #  Dynamic adjustment of learning rate
    print('\r', 'End Training epoch: ', i, ' rmse:', np.sqrt(sum(loss_train) / len(loss_train)), '\n')
    loss_train.clear()

    #3.2 Validation-----------------------------------------------------------------------------------------
    model.eval()
    print('Start Val epoch: ', i)
    with torch.no_grad():
        for t in tqdm(range(int(len(dataset_val[0])/args.batch_size))):
            # (1)  Load data and convert it to tensor
            his_list, fut_list, clasX_list, clasY_list = dataset_val[0][t * args.batch_size:(t + 1) * args.batch_size],dataset_val[1][t * args.batch_size:(t + 1) * args.batch_size], \
                                                        dataset_val[2][t * args.batch_size:(t + 1) * args.batch_size],dataset_val[3][t *args.batch_size:(t + 1) * args.batch_size]
            his,fut_label,_,_, point, social_mask = list_to_tensor(his_list, fut_list, clasX_list,clasY_list)

            # (2) masking and prediction
            his_val, obs_mask = Rand_Mask(copy.deepcopy(his), copy.deepcopy(scale_mask),point,rate_low=0.0,rate_high=0.9)
            fut_pre= model(his_val, obs_mask, social_mask, point)

            # (3) Calculate losses
            loss_reg = loss_mse(fut_pre.float(), fut_label.float())
            loss_val.append(loss_reg.item())
        print('\r', 'End Val epoch: ', i, ' rmse:', np.sqrt(sum(loss_val) / len(loss_val)))
        if np.sqrt(sum(loss_val) / len(loss_val)) < min_val:
            min_val = np.sqrt(sum(loss_val) / len(loss_val))
            torch.save(model.state_dict(), args.path_save+f'FMSTF_{i}.pth')
        print('min_val_rmse:', min_val, '\n')
        loss_val.clear()
