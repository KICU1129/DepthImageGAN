from model import Generator,Discriminator
from opt import TestOpts
from utils import *
from database import ImageDataset

import numpy as np
import gc
import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

import itertools
from PIL import Image

# import mlflow

""" --- Initial Setting  ---"""
opt=TestOpts()

root_path="../output/"
model_path=root_path+f"model/{opt.experience_ver}/"
record_path=root_path+f"generate/{opt.experience_ver}/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(record_path):
    os.mkdir(record_path)

recorder=Recoder(opt.version,root=record_path)

# mlflow.set_experiment("depthimage-gan_{}".format(opt.experience_ver))
# mlflow.start_run()
# for _ ,(key , item) in enumerate(vars(opt).items()):
#     mlflow.log_param(key,item)

""" --- Call Models ---"""


# 生成器
netG_A2B = Generator(opt.domainA_nc, opt.domainB_nc)
netG_B2A = Generator(opt.domainB_nc, opt.domainB_nc)

# 識別器
netD_A = Discriminator(opt.domainA_nc)
netD_B = Discriminator(opt.domainB_nc)

# GPU
if not opt.cpu:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

# 重みパラメータ初期化
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# 保存したモデルのロード
if opt.load_weight is True:
    netG_A2B.load_state_dict(torch.load(model_path+"netG_A2B.pth", map_location="cuda:0"), strict=False)
    netG_B2A.load_state_dict(torch.load(model_path+"netG_B2A.pth", map_location="cuda:0"), strict=False)
    netD_A.load_state_dict(torch.load(model_path+"netD_A.pth", map_location="cuda:0"), strict=False)
    netD_B.load_state_dict(torch.load(model_path+"netD_B.pth", map_location="cuda:0"), strict=False)


# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)

# 入出力メモリ確保
Tensor = torch.cuda.FloatTensor if not opt.cpu else torch.Tensor
input_A = Tensor(opt.batch_size, opt.domainA_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.domainB_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

# 過去データ分のメモリ確保
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# データローダー
transforms_ = [ 
                transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC), 
                # transforms.RandomCrop(opt.size), 
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                ]
dataset=ImageDataset(root=opt.dataroot, transforms_=transforms_, unaligned=False,limit=None)
dataloader = DataLoader(dataset, 
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

#Dataset for sampling 
sample_images=[dataset[i] for i in range(10) ]

print("num dataloader= {}".format(len(dataloader)))

#Release Memory
del netD_A,netD_B,fake_A_buffer,fake_B_buffer
gc.collect()


#Save generate image from netG
image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,input_A,input_B)

# mlflow.end_run()





