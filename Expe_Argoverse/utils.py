import torch
import numpy as np
import random
import torch.nn as nn
from operator import itemgetter
from Args import args

'Random masking'
def Rand_Mask(his,scale_mask,point,rate_low,rate_high): #Historical trajectory, scale mask matrix, minimum mask rate, maximum mask rate

    obs_mask=scale_mask.unsqueeze(0).repeat(his.shape[0], 1, 1, 1) #observation matrix
    length=his.shape[1]
    low_num,high_num=int(length*rate_low),int(length*rate_high)
    hist_masked_num = torch.randint(low=low_num, high=high_num, size=(his.shape[0], 1))

    for i in range(0, his.shape[0]):
        if ((i in point)&(bool(hist_masked_num[i, 0]!=0))):
            loc = np.asarray(random.sample(range(0, length), hist_masked_num[i, 0]))
            his[i, loc, :] = 0.0
            obs_mask[i, :, :, loc] = 1.0
        else:
            loc=torch.where(his[i,:,0]==torch.tensor(float('-inf')))[0]
            if len(loc)!=0:
                his[i, loc, :] = 0.0
                obs_mask[i, :, :, loc] = 1.0
    return his, obs_mask



'Position encoding'
def PositionEmbedding(sample_num,n_position, d_model):
    PE=torch.zeros((n_position,d_model))
    for i in range(d_model): PE[:,i]=i # 赋予列值
    for pos in range(n_position):
        PE[pos,0::2]=np.sin(pos/np.power(1000, PE[pos,0::2]/d_model))
        PE[pos,1::2]=np.cos(pos/np.power(1000,(2*(PE[pos,1::2]//2))/d_model ))
    PE=PE.unsqueeze(0).expand(sample_num,PE.shape[0],PE.shape[1])
    return PE.cuda()


"Multi-scale fusion network"
class MultiScaleFus(nn.Module):
    def __init__(self):
        super(MultiScaleFus, self).__init__()
        self.W_Q = nn.Linear(args.hidd_dim,args.hidd_dim)
        self.W_K = nn.Linear(args.hidd_dim,args.hidd_dim)
        self.W_V = nn.Linear(args.hidd_dim, args.hidd_dim)
    def forward(self,Q_inputs,K_inputs,V_inputs):

        Q,K,V=self.W_Q(Q_inputs),self.W_K(K_inputs),self.W_V(V_inputs)
        Q=Q.unsqueeze(-2).unsqueeze(1).repeat(1,args.look_back,1,1,1)
        K=K.transpose(2,1).unsqueeze(-1)
        V=V.transpose(2,1)

        scores=torch.matmul(Q,K).squeeze(-1).squeeze(-1)/ np.sqrt(args.hidd_dim)
        atten=nn.Softmax(dim=-1)(scores).unsqueeze(-1)

        fus_enc=torch.sum(torch.mul(V,atten),dim=-2)

        return fus_enc

"list2tensor"
def list_to_tensor(his_list,fut_list,clasX_list,clasY_list):

    his,fut_label=torch.empty((args.batch_size*30,args.look_back,2)),torch.empty((args.batch_size,args.pre_len,2))
    clasX_label,clasY_label=torch.empty((args.batch_size)),torch.empty((args.batch_size))
    social_mask,point=torch.ones((1,1,30*args.batch_size,30*args.batch_size)),[]
    num_all=0

    for sample in range(args.batch_size):
        point.append(num_all)
        num_sample=his_list[sample].shape[0]
        his[num_all:num_all + num_sample, :, :] =his_list[sample]
        fut_label[sample,:,:]=fut_list[sample]
        clasX_label[sample],clasY_label[sample]=clasX_list[sample],clasY_list[sample]
        social_mask[:, :, num_all:num_all + num_sample, num_all:num_all + num_sample] = 0.0
        num_all = num_all + num_sample

    return his[0:num_all,:,:].cuda(),fut_label.cuda(),clasX_label.cuda(),clasY_label.cuda(),point,social_mask[:,:,0:num_all,0:num_all].cuda()


"Shuffle"
def Shuffle_data(dataset):
    num = len(dataset[0])
    random_permutation = tuple(np.random.permutation(num))
    dataset[0] = list(itemgetter(*random_permutation)(dataset[0]))
    dataset[1] = list(itemgetter(*random_permutation)(dataset[1]))
    dataset[2] = list(itemgetter(*random_permutation)(dataset[2]))
    dataset[3] = list(itemgetter(*random_permutation)(dataset[3]))
    return dataset


"loss"
def calc_ED_error(pred_trajs, future_traj):

    error_ADE = torch.sqrt(torch.sum((pred_trajs - future_traj) ** 2, dim=-1))
    error_FDE = torch.sqrt(torch.sum((pred_trajs - future_traj) ** 2, dim=-1))[:,-1]
    num_miss=sum(error_FDE>2)

    return torch.mean(error_ADE),torch.mean(error_FDE),num_miss/error_FDE.shape[0]



"Data Augmentation"
def Augmentation(X,Y,point):

    X,Y=X.unsqueeze(-2),Y.unsqueeze(-2)
    for i in range(0,len(point)):
        theta=torch.from_numpy(2*np.pi*np.random.random(size=1)).cuda()
        trans=torch.zeros((1,2,2)).cuda()
        trans[0,0,0],trans[0,0,1],trans[0,1,0],trans[0,1,1]=torch.cos(theta),-torch.sin(theta),torch.sin(theta),torch.cos(theta)
        if i!=(len(point)-1):
            start,end=point[i],point[i+1]
            num_sample=end-start
            trans_X=trans.unsqueeze(1).repeat(num_sample,X.shape[1],1,1)
            trans_Y=trans.repeat(Y.shape[1], 1, 1)
            X[start:end,:,:,:]=torch.matmul(X[start:end,:,:,:].double(),trans_X.double())
            Y[i,:,:,:]=torch.matmul(Y[i,:,:,:].double(),trans_Y.double())
        else:
            start=point[i]
            num_sample=X.shape[0]-start
            trans_X=trans.unsqueeze(1).repeat(num_sample,X.shape[1],1,1)
            trans_Y=trans.repeat(Y.shape[1], 1, 1)
            X[start:,:,:,:]=torch.matmul(X[start:,:,:,:].double(),trans_X.double())
            Y[i,:,:,:]=torch.matmul(Y[i,:,:,:].double(),trans_Y.double())

    return X.squeeze(-2),Y.squeeze(-2)




