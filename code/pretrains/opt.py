import numpy as np
import torch
import random


class PretrainOpts():
    def __init__(self,seed=1234):
        self.fix_seed(seed)
        self.experience_ver="pretrain_gemerator_ver0.0"
        self.version="0.0.0"
        self.memo="あまりに上手いこといかないから事前学習して試してみる"
        self.dataroot = r"E:\KISUKE\SUNRGBD\SUNRGBD\kv1\NYUdata/"
        # self.dataroot = "./dataset/SUNRGBD/SUNRGBD/kv1/NYUdata/"
        self.depth_name="depth_bfx"

        self.start_epoch = 0
        self.n_epochs = 2000
        self.batch_size = 8
        self.display_iter=1000
        self.save_epoch=20
        self.lr = 0.0002
        self.decay_epoch = 200
        #coefficient of cycle consist loss
        self.lamda_a= 1*1
        self.lamda_b= 1*1
        self.isIdentify=False

        self.D_model="SNUD" # or ["D","UD","SNUD","SND"] 
        self.G_model="UG" # or ["G","UG"]

        self.size = (320,240)#(640 ,480)
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
        self.unaligned=False

        #data length limit
        self.limit=None


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

