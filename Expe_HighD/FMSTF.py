import torch
import torch.nn as nn
import numpy as np
from utils import PositionEmbedding,MultiScaleFus
from Args import args

'sequential coding layer'
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn =MultiHeadAttention()
        self.ffn =FeedForward()
    def forward(self,Q_inputs, K_inputs, V_inputs,attn_mask):

        Multi_out,attn= self.enc_self_attn(Q_inputs, K_inputs, V_inputs, attn_mask)

        Ffd_out = self.ffn(Multi_out) # enc_outputs: [batch_size 5*x len_q x d_model]
        return Ffd_out,attn

'temporal feedforward layer'
class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.Linear1 = nn.Linear(in_features=args.hidd_dim,out_features=2*args.hidd_dim)
        self.Linear2 = nn.Linear(in_features=2*args.hidd_dim,out_features=args.hidd_dim)
        self.layer_norm = nn.LayerNorm(args.hidd_dim)
    def forward(self,Multi_out):
        residual = Multi_out # [B,S,d_model]
        out= nn.ReLU()(self.Linear1(Multi_out)) #[B,S,d_ff]
        out= self.Linear2(out) # [B,S,d_model]
        Ffd_out=self.layer_norm(out + residual)
        return Ffd_out

'Multi-scale Attention Head--For temporal modelling'
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(args.hidd_dim, args.hidd_dim* 5)
        self.W_K = nn.Linear(args.hidd_dim, args.hidd_dim* 5)
        self.W_V = nn.Linear(args.hidd_dim, args.hidd_dim* 5)
        self.layer_norm = nn.LayerNorm(args.hidd_dim)

    def forward(self, Q_inputs, K_inputs, V_inputs, attn_mask):

        Q, K, V = self.W_Q(Q_inputs), self.W_K(K_inputs), self.W_V(V_inputs)

        q_n = Q.reshape(Q.shape[0],Q.shape[1],-1,args.hidd_dim).transpose(1, 2)
        k_n =K.reshape(K.shape[0],K.shape[1],-1,args.hidd_dim).transpose(1, 2)
        v_n =V.reshape(V.shape[0],V.shape[1],-1,args.hidd_dim).transpose(1, 2)

        residual= q_n

        contex, atten = ScaledDotProductAttention()(q_n, k_n, v_n, attn_mask)
        Multi_out = self.layer_norm(contex+ residual)
        return Multi_out, atten

'ScaledDotProductAttention'
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self,q_n,k_n,v_n,attn_mask):

        scores = torch.matmul(q_n,k_n.transpose(-1, -2)) / np.sqrt(args.hidd_dim) # scores : [batch_size x n_heads x S × S]
        scores.masked_fill_(attn_mask.bool(), -1e9)


        attn = nn.Softmax(dim=-1)(scores) #[batch_size × n_heads × S × S]
        context = torch.matmul(attn,v_n) #[batch_size,n_heads,S,S]*[batch_size,n_heads,S,d_v]=[batch_size,n_heads,S,d_k]
        return context, attn



class FMSTF(nn.Module):
    def __init__(self):
        super(FMSTF, self).__init__()
        self.enc_l1=nn.Linear(2,args.hidd_dim)
        self.TempEL=EncoderLayer()
        self.MultiScaleFus=MultiScaleFus()
        self.enc_l2=nn.Linear(5*args.hidd_dim,args.hidd_dim)
        self.enc_l3=nn.Linear((args.look_back+5)*args.hidd_dim,int((args.look_back+5)*args.hidd_dim/2))
        self.enc_l4= nn.Linear(int((args.look_back+5)*args.hidd_dim/2),args.hidd_dim)
        self.GlobalSocial = MultiHeadAttention()
        self.enc_l5= nn.Linear(5*args.hidd_dim,args.hidd_dim)

        self.dec_LT = nn.LSTMCell(input_size=args.hidd_dim, hidden_size=args.hidd_dim)
        self.dec_l1=nn.Linear(args.hidd_dim,int(args.hidd_dim/2))
        self.activate=nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(p=args.drop)
        self.dec_l2 = nn.Linear(int(args.hidd_dim/2),int(args.hidd_dim/4))
        self.dec_l3= nn.Linear(int(args.hidd_dim/4),2)

    def forward(self,his,obs_mask,social_mask,point):

        "1. encoding"
        "1.1 temporal encoding"
        his=self.enc_l1(his.float())+PositionEmbedding(his.shape[0],args.look_back,args.hidd_dim)
        for l in range(3):
            his,atten=self.TempEL(his,his,his,obs_mask)
            his=self.enc_l2(his.transpose(1,2).reshape(his.shape[0],args.look_back,-1))
        his,atten=self.TempEL(his,his,his,obs_mask)


        "1.2 Calculate Continuity representation"
        InfoInc=torch.sum(1-obs_mask,dim=-1) #Calculate information increment
        IncAtten=nn.Softmax(dim=-1)(InfoInc).unsqueeze(-1) #AcrossAtten
        Cont=torch.sum(torch.mul(IncAtten,his),dim=-2) #Continuity representation

        "1.3 Fusion"
        fus_enc=self.MultiScaleFus(Cont,his,his)
        enc_temp=self.enc_l4(self.activate(self.enc_l3(torch.cat((fus_enc,Cont),1).reshape(fus_enc.shape[0],-1))))

        "1.4 Interaction modeling"
        inputs=enc_temp.unsqueeze(0)
        social_mask=social_mask.repeat(1,5,1,1)
        global_social,_=self.GlobalSocial(inputs,inputs,inputs,social_mask)# inputs-[batch,sq_len,input_size],mask-[batch,5,sq_len,sq_len]
        enc_tar=global_social[0,0,point,:]

        "2.decoding"
        h, c = enc_tar,enc_tar
        H = torch.empty(enc_tar.shape[0],args.pre_len,args.hidd_dim).cuda()
        for t in range(args.pre_len):
            h_t, c_t = self.dec_LT(h, (h, c))
            H[:,t, :] = h_t
            h, c = h_t, c_t
        out = self.drop(self.activate(self.dec_l1(H)))
        out = self.activate(self.dec_l2(out))
        pre = self.dec_l3(out)


        "3. return the predictions"
        return pre

