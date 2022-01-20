import numpy as np
import torch
import random


class Opts():
    def __init__(self,seed=2021):
        self.fix_seed(seed)
        self.is_mlflow=True
        self.experience_ver="cyclegan_unpaired_ver5.2.0"

        self.version="0.0.0"
        self.memo="U-Netで試す 簡易 "

        # self.dataroot = r"E:\KISUKE\SUNRGBD\SUNRGBD\kv1\NYUdata/"
        # self.subdataroot = r"E:\KISUKE\SUNRGBD\SUNRGBD\kv1\b3dodata/"
        self.dataroot = "./dataset/SUNRGBD/SUNRGBD/kv1/NYUdata/"
        self.subdataroot = "./dataset/SUNRGBD/SUNRGBD/kv1/b3dodata/"
        self.depth_name="depth_bfx"

        self.start_epoch = 0
        self.n_epochs = 1000
        self.init_epochs=100
        self.batch_size = 1
        self.display_iter=1000
        self.save_epoch=20
        self.lr = 0.0002
        self.decay_epoch = 100
        self.pool_size=10
        #coefficient of cycle consist loss
        self.lamda_a= 1*10
        self.lamda_b= 1*10
        self.lamda_i=1*0
        self.isIdentify=False
        self.is_semi=False

        self.D_model="PD" # or ["D","SND",SNUD,PD] 
        self.G_model="PG" # or ["G","UG",PG]

        self.size = (32,32)
        self.domainA_nc = 3
        self.domainB_nc = 1

        self.cpu = False
        self.n_cpu = 0
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device =  torch.device(self.device_name) 
        self.load_weight = False
        self.is_pretrain=False

        #Discrimatorの学習頻度
        self.d_freq=1
        #Generatorの学習頻度
        self.g_freq=1
         #Semi Supervisedの学習頻度
        self.semi_freq=1
        #Frequency of Cycle Loss
        self.cycle_freq=1

        #pair or unpair ? if True , dataset is selected in unpair 
        # self.unaligned=False
        self.unaligned=True

        self.limit=100


    def fix_seed(self,seed):
        # Numpy
        np.random.seed(seed)
        random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class AnimeOpts():
    def __init__(self,seed=2021):
        self.fix_seed(seed)
        self.is_mlflow=True
        self.experience_ver="cyclegan_Animeface_ver0.0.0"

        self.version="0.0.0"
        self.memo="写真→アニメでCycleGANを試す "

        self.dataroot = r"E:\KISUKE\Dataset"
        # self.dataroot = r"E:\KISUKE\SUNRGBD\SUNRGBD\kv1\b3dodata/"
        # self.dataroot = "../dataset/SUNRGBD/SUNRGBD/kv1/b3dodata/"
        self.depth_name="depth_bfx"

        self.start_epoch = 0
        self.n_epochs = 1000
        self.batch_size = 1
        self.display_iter=1000
        self.save_epoch=10
        self.lr = 0.0002
        self.decay_epoch = 200
        #coefficient of cycle consist loss
        self.lamda_a= 1*1
        self.lamda_b= 1*1
        self.lamda_i=1*1
        self.isIdentify=False

        self.D_model="SNUD" # or ["D","SND",SNUD] 
        self.G_model="UG" # or ["G","UG"]

        self.size = (160,240)
        self.domainA_nc = 3
        self.domainB_nc = 3

        self.cpu = False
        self.n_cpu = 0
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device =  torch.device(self.device_name) 
        self.load_weight = False
        self.is_pretrain=False

        #Discrimatorの学習頻度
        self.d_freq=1
        #Generatorの学習頻度
        self.g_freq=1

        #pair or unpair ? if unpair True
        # self.unaligned=False
        self.unaligned=True

        self.limit=2000


    def fix_seed(self,seed):
        # Numpy
        np.random.seed(seed)
        random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
class Pix2PixOpts():
    def __init__(self,seed=1234):
        self.fix_seed(seed)
        self.experience_ver="pix2pix_ver1.0.0"
        self.version="0.0.1"
        self.start_epoch = 0
        self.n_epochs = 50
        self.batch_size = 1
        self.dataroot = "../dataset/SUNRGBD/SUNRGBD/kv1/b3dodata/"
        self.lr = 0.0002
        self.decay_epoch = 200
        self.size = 256
        self.domainA_nc = 3
        self.domainB_nc = 3
        self.cpu = False
        self.n_cpu = 0
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device =  torch.device(self.device_name) 
        self.load_weight = False
        #Discrimatorの学習頻度
        self.k=1
        self.unaligned=False

    def fix_seed(self,seed):
        # Numpy
        np.random.seed(seed)
        random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class TestOpts():
    def __init__(self,seed=1234):
        self.fix_seed(seed)
        self.experience_ver="cyclegan_pared_ver0.0.0"
        self.version="0.0.0"
        self.start_epoch = 0
        self.n_epochs = 1000
        self.batch_size = 1
        self.dataroot = "../dataset/SUNRGBD/SUNRGBD/kv1/b3dodata/"
        self.lr = 0.0002
        self.decay_epoch = 200
        self.size = 256
        self.domainA_nc = 3
        self.domainB_nc = 3
        self.cpu = False
        self.n_cpu = 4
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device =  torch.device(self.device_name) 
        self.load_weight = False

    def fix_seed(self,seed):
        # Numpy
        np.random.seed(seed)
        random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


if __name__ =="__main__":
    opt=Opts()
    print(vars(opt))

