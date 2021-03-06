from pickle import FALSE
from models.model01 import UNetGenerator,UNetDiscriminator,SNDiscriminator,SNUNetDiscriminator
from models.model import Generator,SNDiscriminator
from pretrains.opt import PretrainOpts
from utils import *
from pretrains.database import ImageDataset

import numpy as np
import gc
import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
 
import itertools
from PIL import Image
import time
import mlflow

""" --- Initial Setting  ---"""
opt=PretrainOpts()

root_path="./output/"
model_path=root_path+f"model/{opt.experience_ver}/"
record_path=root_path+f"record/{opt.experience_ver}/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(record_path):
    os.mkdir(record_path)

recorder=Recoder(opt.version,root=record_path,epoch=opt.start_epoch)

mlflow.set_experiment("depthimage-gan_{}".format(opt.experience_ver))
mlflow.start_run()
for _ ,(key , item) in enumerate(vars(opt).items()):
    mlflow.log_param(key,item)

""" --- Call Models ---"""

# 生成器
if opt.G_model=="UG":
    netG_A2B = UNetGenerator(opt.domainA_nc, opt.domainB_nc)
    netG_B2A = UNetGenerator(opt.domainB_nc, opt.domainA_nc)
if opt.G_model=="G":
    netG_A2B = Generator(opt.domainA_nc, opt.domainB_nc)
    netG_B2A = Generator(opt.domainB_nc, opt.domainA_nc)

# 識別器
if opt.D_model=="UD":
    netD_A = UNetDiscriminator(opt.domainA_nc*2)
    netD_B = UNetDiscriminator(opt.domainB_nc*2)

if opt.D_model=="SNUD":
    netD_A = SNUNetDiscriminator(opt.domainA_nc)
    netD_B = SNUNetDiscriminator(opt.domainB_nc)
if opt.D_model=="SND":
    netD_A = SNDiscriminator(opt.domainA_nc*2)
    netD_B = SNDiscriminator(opt.domainB_nc*2)

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

# 損失関数
criterion_GAN = torch.nn.MSELoss()#torch.nn.BCELoss()#
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

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
input_A = Tensor(opt.batch_size, opt.domainA_nc, opt.size[1], opt.size[0])
input_B = Tensor(opt.batch_size, opt.domainB_nc, opt.size[1], opt.size[0])
input_C = Tensor(opt.batch_size, opt.domainB_nc, opt.size[1], opt.size[0])
input_D = Tensor(opt.batch_size, opt.domainA_nc, opt.size[1], opt.size[0])
input_A_test = Tensor(1, opt.domainA_nc, opt.size[1], opt.size[0])
input_B_test = Tensor(1, opt.domainB_nc, opt.size[1], opt.size[0])
input_C_test = Tensor(1, opt.domainB_nc, opt.size[1], opt.size[0])
input_D_test = Tensor(1, opt.domainA_nc, opt.size[1], opt.size[0])
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

# 過去データ分のメモリ確保
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# データローダー
transforms_ = [ 
                transforms.Lambda(lambda x: resize(x,opt.size)),
                # transforms.Lambda(normalize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) 
                ]

dataset=ImageDataset(depth_name=opt.depth_name,depth_gray=opt.domainB_nc==1,root=opt.dataroot,image_name=opt.depth_name,
                     transforms_=transforms_, limit=opt.limit,unaligned=opt.unaligned)

test_dataset=ImageDataset(depth_name=opt.depth_name,depth_gray=opt.domainB_nc==1,root=opt.dataroot,image_name=opt.depth_name,
                     transforms_=transforms_, limit=100,unaligned=False)

dataloader = DataLoader(dataset,
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

#Dataset for sampling 
sample_images=[test_dataset[i] for i in range(10) ]

print("num dataloader= {}".format(len(dataloader)))


""" --- Let's Training !! --- """
for epoch in range(opt.start_epoch, opt.n_epochs):
    s=time.time()
    #For Test generate images
    # image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,input_A_test,input_B_test)
    for i, batch in enumerate(dataloader):
        if np.shape(batch["A"])[0]!=opt.batch_size:
            break
        # モデルの入力
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        real_C = Variable(input_C.copy_(batch['C']))
        real_D = Variable(input_D.copy_(batch['D']))


        if i%opt.g_freq==0:

            ##### 生成器A2B、B2Aの処理 #####
            optimizer_G.zero_grad()

            # 敵対的損失（GAN loss）
            fake_B = netG_A2B(real_A)
            loss_A2B = criterion_cycle(fake_B, real_B)


            loss_A2B.backward()

            # 敵対的損失（GAN loss）
            fake_D = netG_B2A(real_C)
            loss_C2D = criterion_cycle(fake_D, real_D)


            loss_C2D.backward()
            
            optimizer_G.step()

     

        if i % opt.display_iter == 0:
            # print('Epoch[{}]({}/{}) loss_G: {:.4f} loss_G_identity: {:.4f} loss_G_GAN: {:.4f} loss_G_cycle: {:.4f} loss_D: {:.4f}'.format(
            #     epoch, i, len(dataloader), loss_G, (loss_identity_A + loss_identity_B),
            #     (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)
            #     ))
            print('Epoch[{}]({}/{}) loss_G: {:.4f}   '.format(
                epoch, i, len(dataloader), loss_A2B,))
        

            train_info = {
                'epoch': epoch, 
                'batch_num': i, 
                'lossG': loss_A2B.item(),
                }

            #Save Metrics with Mlflow
            for _ ,(key , item) in enumerate(train_info.items()):
                mlflow.log_metric(key,item)



        batches_done = (epoch - 1) * len(dataloader) + i


    # Update learning rates
    lr_scheduler_G.step()

    if epoch % opt.save_epoch==0:
        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), model_path+f'netG_A2B_{epoch}.pth')
        torch.save(netG_B2A.state_dict(), model_path+f'netG_B2A_{epoch}.pth')
        torch.save(netD_A.state_dict(), model_path+f'netD_A_{epoch}.pth')
        torch.save(netD_B.state_dict(), model_path+f'netD_B_{epoch}.pth')
        image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,input_A_test,input_B_test)

    print(f"Epoch {epoch}/{opt.n_epochs}  ETA : {round(time.time()-s)}[sec]"  )
    

#Release Memory
del netD_A,netD_B,fake_A_buffer,fake_B_buffer
gc.collect()

#Save generate image from netG
# image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,input_A,input_B)

#Save Record with Mlflow
# for i in range(len(image_pathes['image-A2B'])):
#     mlflow.log_artifact(image_pathes["image-A2B"][i])
#     mlflow.log_artifact(image_pathes["image-B2A"][i])


#Save generate image from netG
image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,input_A_test,input_B_test)

mlflow.end_run()





