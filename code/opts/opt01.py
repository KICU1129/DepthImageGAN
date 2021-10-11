import numpy as np
import torch
import random


class UNetOpts():
    def __init__(self,seed=1234):
        self.fix_seed(seed)
        self.experience_ver="cyclegan_unpaired_ver4.0.0"
        self.version="0.0.0"
        self.memo="paper[5]のモデルをまんまコピー"
        # self.dataroot = r"E:\KISUKE\SUNRGBD\SUNRGBD\kv1\b3dodata/"
        self.dataroot = "./dataset/SUNRGBD/SUNRGBD/kv1/NYUdata/"
        self.depth_name="depth_bfx"

        self.start_epoch = 0
        self.n_epochs = 100
        self.batch_size = 2
        self.display_iter=1000
        self.save_epoch=10
        self.lr = 0.0002
        self.decay_epoch = 200
        #coefficient of cycle consist loss
        self.lamda_a= 1
        self.lamda_b= 1
        self.isIdentify=False

        self.D_model="UD" # or ["D","UD","SND"] 
        self.G_model="UG" # or ["G","UG"]

        self.size = (64 ,48)
        self.domainA_nc = 3
        self.domainB_nc = 1

        self.cpu = False
        self.n_cpu = 0
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device =  torch.device(self.device_name) 
        self.load_weight = False

        #Discrimatorの学習頻度
        self.d_freq=1
        #Generatorの学習頻度
        self.g_freq=1

        #pair or unpair ? if unpair True
        # self.unaligned=False
        self.unaligned=True


    def fix_seed(self,seed):
        # Numpy
        np.random.seed(seed)
        random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
if __name__ =="__main__":
    opt=UNetOpts()
    print(vars(opt))

