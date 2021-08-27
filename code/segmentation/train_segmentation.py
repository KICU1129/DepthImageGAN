import sys
sys.path.append('../')
from opt import SegmentationOpts
# from code.utils import *
from database import SegmentDataset

import numpy as np
import gc
import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

import itertools
from PIL import Image

import mlflow

""" --- Initial Setting  ---"""
opt=SegmentationOpts()

root_path="../../output/"
model_path=root_path+f"model/{opt.experience_ver}/"
record_path=root_path+f"record/{opt.experience_ver}/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(record_path):
    os.mkdir(record_path)


# mlflow.set_experiment("segmentation_unet_{}".format(opt.experience_ver))
# mlflow.start_run()
# for _ ,(key , item) in enumerate(vars(opt).items()):
#     mlflow.log_param(key,item)



""" --- Call Models ---"""

# データローダー
transforms_ = [ transforms.Resize((int(opt.size),int(opt.size))), 
                # transforms.RandomCrop(opt.size), 
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
mask_transforms_ = [ transforms.Resize((int(opt.size),int(opt.size))), 
                # transforms.RandomCrop(opt.size), 
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                 ]
dataset=SegmentDataset(root=opt.dataroot, transforms_=transforms_,mask_transforms=mask_transforms_, limit=None,isColor=opt.isColor,depth=opt.depth)
dataloader = DataLoader(dataset[:opt.sample_num], 
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

#Dataset for sampling 
sample_images=[dataset[i] for i in range(opt.sample_num) ]



    

#Save generate image from netG
# image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,input_A,input_B)

#Save Record with Mlflow
# for i in range(len(image_pathes['image-A2B'])):
#     mlflow.log_artifact(image_pathes["image-A2B"][i])
#     mlflow.log_artifact(image_pathes["image-B2A"][i])



# mlflow.end_run()





