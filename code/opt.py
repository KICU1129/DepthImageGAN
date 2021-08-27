import numpy as np
import torch
import random


class Opts():
    def __init__(self,seed=1234):
        self.fix_seed(seed)
        self.experience_ver="cyclegan_paired_ver1.0.0"
        self.version="0.0.0"
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
        self.k=5
        #pair or unpair ?
        self.unaligned=False
        # self.unaligned=True


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
        self.n_epochs = 1
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

