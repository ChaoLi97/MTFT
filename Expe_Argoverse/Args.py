import argparse


'Define model parameters-------------------------------------------------------------------------------------- -'
p = argparse.ArgumentParser()
p.add_argument('--look_back',default=20,type=int,help='the length of  observation')
p.add_argument('--pre_len',default=30,type=int,help='the length of  prediction')
p.add_argument('--batch_size',default=256,type=int,help='the size of a batch')
p.add_argument('--epochs',default=200,type=int,help='the num of epoch')
p.add_argument('--lr',default=1e-4,type=float,help='the learning rate')
p.add_argument('--hidd_dim',default=128,type=int,help='the dimension of hidden layers')
p.add_argument('--path_train',default='../data/Argoverse/dataset_train.pkl',type=str,help='the path of train set')
p.add_argument('--path_val',default='../data/Argoverse/dataset_val.pkl',type=str,help='the path of validation set')
p.add_argument('--path_test',default='../data/Argoverse/dataset_test.pkl',type=str,help='the path of test set')
p.add_argument('--path_mask',default='../data/Argoverse/mask.pt',type=str,help='the path of scalee mask')
p.add_argument('--path_save',default='../Params/Argoverse/',type=str,help='the path of scale mask')
p.add_argument('--drop',default=0.1,type=float,help='the value of dropout')
args = p.parse_args() 
