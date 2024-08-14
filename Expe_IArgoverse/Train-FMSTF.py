import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
from utils import Rand_Mask,list_to_tensor,Shuffle_data,calc_ED_error,Augmentation
import copy
from FMSTF import FMSTF
from Args import args
from torch.optim.lr_scheduler import StepLR
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"



'------------------------------------------------------------------------------------'
with open(args.path_train, 'rb') as file:
    dataset_train= pickle.load(file)
with open(args.path_val, 'rb') as file:
    dataset_val= pickle.load(file)
scale_mask=torch.load(args.path_mask).cuda() #scale mask


'-----------------------------------------------------------------------------------'
model=FMSTF().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.6*args.lr)
scheduler = StepLR(optimizer,step_size=25, gamma=0.5)
loss_mae=nn.L1Loss()


'-------------------------------------------------------------------------------------------------'
loss_train_ade,loss_train_fde,loss_val_ade,loss_val_fde=[],[],[],[]
min_val_ade,min_val_fde=100,100 ##Initialize minimum validation loss
for i in range(args.epochs):

    #Training----------------------------------------------------------------------------------------
    model.train()
    dataset_train=Shuffle_data(dataset_train)  #
    print('Start Training epoch: ', i)
    for t in tqdm(range(int(len(dataset_train[0])/args.batch_size))):
        #(1) Load data and convert it to tensor
        his_list,fut_list,clasX_list,clasY_list=dataset_train[0][t*args.batch_size:(t+1)*args.batch_size],dataset_train[1][t*args.batch_size:(t+1)*args.batch_size],\
                                                dataset_train[2][t*args.batch_size:(t+1)*args.batch_size],dataset_train[3][t*args.batch_size:(t+1)*args.batch_size]
        his,fut_label,_,_,point,social_mask=list_to_tensor(his_list,fut_list,clasX_list,clasY_list)

        #(2) masking and prediction
        his_train,obs_mask=Rand_Mask(copy.deepcopy(his),copy.deepcopy(scale_mask))
        his_train,fut_label=Augmentation(copy.deepcopy(his_train),copy.deepcopy(fut_label),point)
        fut_pre=model(his_train,obs_mask,social_mask,point)

        #(3) Calculate losses
        loss_ade=loss_mae(fut_pre.float(), fut_label.float())
        loss_fde=loss_mae(fut_pre[:,-1,:].float(), fut_label[:,-1,:].float())
        loss = loss_ade+loss_fde

        #(4) Record losses and update model parameters
        loss_train_ade.append(loss_ade.item())
        loss_train_fde.append(loss_fde.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()  #  Dynamic adjustment of learning rate
    print('\r', 'End Training epoch: ', i, ' mae_vag:', sum(loss_train_ade) / len(loss_train_ade),' mae_fde:', sum(loss_train_fde) / len(loss_train_fde), '\n')
    loss_train_ade.clear()
    loss_train_fde.clear()

    #3.2 Validation----------------------------------------------------------------------------------------
    model.eval()
    print('Start Val epoch: ', i)
    with torch.no_grad():
        for t in tqdm(range(int(len(dataset_val[0])/args.batch_size))):
            # (1) Load data and convert it to tensor
            his_list, fut_list, clasX_list, clasY_list = dataset_val[0][t * args.batch_size:(t + 1) * args.batch_size],dataset_val[1][t * args.batch_size:(t + 1) * args.batch_size], \
                                                        dataset_val[2][t * args.batch_size:(t + 1) * args.batch_size],dataset_val[3][t *args.batch_size:(t + 1) * args.batch_size]
            his,fut_label,_,_, point, social_mask = list_to_tensor(his_list, fut_list, clasX_list,clasY_list)

            # (2) masking and prediction
            his_val, obs_mask = Rand_Mask(copy.deepcopy(his), copy.deepcopy(scale_mask))
            his_val,fut_label=Augmentation(copy.deepcopy(his_val),copy.deepcopy(fut_label),point)
            fut_pre= model(his_val, obs_mask, social_mask, point)

            # (3) Calculate losses
            loss_ade,loss_fde,_=calc_ED_error(fut_pre.float(), fut_label.float())
            loss_val_ade.append(loss_ade.item())
            loss_val_fde.append(loss_fde.item())
        print('\r', 'End Val epoch: ', i, ' mae_ade:', sum(loss_val_ade) / len(loss_val_ade), ' mae_fde:',sum(loss_val_fde) / len(loss_val_fde))
        if (sum(loss_val_ade) / len(loss_val_ade) + sum(loss_val_fde) / len(loss_val_fde)) < (min_val_ade + min_val_fde):
            min_val_ade = sum(loss_val_ade) / len(loss_val_ade)
            min_val_fde = sum(loss_val_fde) / len(loss_val_fde)
            torch.save(model.state_dict(), args.path_save+f'FMSTF_{i}.pth')
        print('min_val_ade:', min_val_ade, ' min_val_fde:', min_val_fde, '\n')
        loss_val_ade.clear()
        loss_val_fde.clear()


