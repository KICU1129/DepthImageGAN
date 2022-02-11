from typing import Optional
import numpy as np
import torch
import random

class SegmentationOpts():
    def __init__(self,seed=1234):
        self.fix_seed(seed)
        self.experience_ver="segmentaion_unet_ver1.0.0"
        self.version="fake_depth_0.0.0"

        # self.dataroot = "../../dataset/SUNRGBD/SUNRGBD/kv1/b3dodata/"
        self.dataroot = r"../../dataset/SUNRGBD/SUNRGBD/kv2/kinect2data_segmentation/tain_images_path.txt"
        self.test_root=r"../../dataset/SUNRGBD/SUNRGBD/kv2/kinect2data_segmentation/test_images_path.txt"
        self.train_label=r"../../dataset/SUNRGBD/SUNRGBD/kv2/kinect2data_segmentation/train13labels"
        self.test_label=r"../../dataset/SUNRGBD/SUNRGBD/kv2/kinect2data_segmentation/test13labels"

        self.isColor=False
        self.input_nc=6

        #Model Parameters
        self.ENCODER = 'resnet34'
        self.ENCODER_WEIGHTS = 'imagenet'
        self.CLASS = [
            "Bed","Books","Ceiling","Chair","Floor","Furniture",
            "Objects","Picture","Sofa","Table","TV","Wall","Window"
            ]
        self.ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DECODER = 'unet'
        self.num_class=13
        self.aux_params=dict(
                    pooling='max',             # one of 'avg', 'max'
                    dropout=0.5,               # dropout ratio, default is None
                    activation='sigmoid',      # activation function, default is None
                    classes=self.num_class,                 # define number of output labels
                )        

        #train parameters
        self.n_epochs = 200
        self.batch_size = 32
        self.sample_num=500
        self.limit=None#1000
        self.lr = 0.0002
        self.size = 256
        self.loss_name="dice"
        # self.loss_name="jaccard"
        self.opt_name="adam"

        #Others
        self.cpu = False
        self.n_cpu = 0# self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device =  torch.device(self.DEVICE) 
        self.load_weight = False

        self.depth="cycle_paired"
        # self.depth="depth"


    def fix_seed(self,seed):
        # Numpy
        np.random.seed(seed)
        random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True






