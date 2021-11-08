from typing import IO

from matplotlib.image import pil_to_array
from model import Generator,Discriminator,SNDiscriminator
from model_unet import UNetGenerator,SNUNetDiscriminator,UNetDiscriminator
from opt import Opts
from utils import *
from database import ImageDataset
from networks import *

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
import matplotlib.pyplot as plt 

""" --- Initial Setting  ---"""
opt=Opts()
torch.backends.cudnn.benchmark = True
root_path="./output/"
model_path=root_path+f"model/{opt.experience_ver}/"
record_path=root_path+f"record/{opt.experience_ver}/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(record_path):
    os.mkdir(record_path)

recorder=Recoder(opt.version,root=record_path,epoch=opt.start_epoch)

if opt.is_mlflow:

    mlflow.set_experiment("depthimage-gan_{}".format(opt.experience_ver))
    mlflow.start_run()
    for _ ,(key , item) in enumerate(vars(opt).items()):
        mlflow.log_param(key,item)
    print("Mlflow OK!")

""" --- Call Models ---"""

# 生成器
if opt.G_model=="G":
    netG_A2B = Generator(opt.domainA_nc, opt.domainB_nc)
    netG_B2A = Generator(opt.domainB_nc, opt.domainA_nc)
if opt.G_model=="UG":
    netG_A2B = UNetGenerator(opt.domainA_nc, opt.domainB_nc)
    netG_B2A = UNetGenerator(opt.domainB_nc, opt.domainA_nc)
if opt.G_model=="PG":
    netG_A2B = define_G(input_nc=opt.domainA_nc,output_nc=opt.domainB_nc,ngf=64,netG='resnet_9blocks')
    netG_B2A = define_G(input_nc=opt.domainB_nc,output_nc=opt.domainA_nc,ngf=64,netG='resnet_9blocks')


# 識別器
if opt.D_model=="D":
    netD_A = Discriminator(opt.domainA_nc)
    netD_B = Discriminator(opt.domainB_nc)
if opt.D_model=="SND":
    netD_A = SNDiscriminator(opt.domainA_nc)
    netD_B = SNDiscriminator(opt.domainB_nc)
if opt.D_model=="UD":
    netD_A = UNetDiscriminator(opt.domainA_nc)
    netD_B = UNetDiscriminator(opt.domainB_nc)

if opt.D_model=="SNUD":
    netD_A = SNUNetDiscriminator(opt.domainA_nc)
    netD_B = SNUNetDiscriminator(opt.domainB_nc)
if opt.D_model=="PD":
    netD_A = define_D(input_nc=opt.domainA_nc,ndf=64,netD="basic")
    netD_B = define_D(input_nc=opt.domainB_nc,ndf=64,netD="basic")

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

# 保存したモデルのロード
if opt.is_pretrain is True:
    netG_A2B.load_state_dict(torch.load(r"C:\Users\Keisoku\Desktop\KISUKE\Project\DepthImageGAN\output\model\pretrain\netG_A2B_1980.pth", map_location="cuda:0"), strict=False)
    netG_B2A.load_state_dict(torch.load(r"C:\Users\Keisoku\Desktop\KISUKE\Project\DepthImageGAN\output\model\pretrain\netG_B2A_1980.pth", map_location="cuda:0"), strict=False)
   
# 損失関数
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
# lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)

# 入出力メモリ確保
Tensor = torch.cuda.FloatTensor if not opt.cpu else torch.Tensor
input_A = Tensor(opt.batch_size, opt.domainA_nc, opt.size[1], opt.size[0])
input_B = Tensor(opt.batch_size, opt.domainB_nc, opt.size[1], opt.size[0])
test_input_A = Tensor(1, opt.domainA_nc, opt.size[1], opt.size[0])
test_input_B = Tensor(1, opt.domainB_nc, opt.size[1], opt.size[0])
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

# 過去データ分のメモリ確保
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# データローダー
transforms_ = [ transforms.Lambda(normalize),
                transforms.Lambda(lambda x: resize(x,opt.size)),
                # transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC), 
                # transforms.RandomCrop(opt.size), 
                # transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                ]

dataset=ImageDataset(depth_name=opt.depth_name,depth_gray=opt.domainB_nc==1,root=opt.dataroot,
                     transforms_=transforms_, limit=opt.limit,unaligned=opt.unaligned)
test_dataset=ImageDataset(depth_name=opt.depth_name,depth_gray=opt.domainB_nc==1,root=opt.dataroot,
                     transforms_=transforms_, limit=100,unaligned=False)

dataloader = DataLoader(dataset,
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
test_dataloader = DataLoader(test_dataset,
                        batch_size=1, shuffle=False, num_workers=opt.n_cpu)
#Dataset for sampling 
sample_images=[test_dataset[i] for i in range(10) ]

print("num dataloader= {}".format(len(dataloader)))


""" --- Let's Training !! --- """
for epoch in range(opt.start_epoch, opt.n_epochs):
    s=time.time()

    for i, batch in enumerate(dataloader):
        if np.shape(batch["A"])[0]<opt.batch_size:
            break
        # モデルの入力
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))


        if i%opt.g_freq==0:

            ##### 生成器A2B、B2Aの処理 #####
            netD_A,netD_B= set_requires_grad([netD_A,netD_B],requires_grad=False)
            optimizer_G.zero_grad()

            if opt.isIdentify:
                # 同一性損失の計算（Identity loss)
                # G_A2B(B)はBと一致
                real_B_=real_B
                if opt.domainA_nc != opt.domainB_nc:
                    real_B_=torch.cat([real_B,real_B,real_B],dim=1)
                    
                same_B = netG_A2B(real_B_)
                loss_identity_B = criterion_identity(same_B, real_B)*opt.lamda_i
                # G_B2A(A)はAと一致
                real_A_=real_A
                if opt.domainA_nc != opt.domainB_nc:
                    real_A_=real_A[:,0,:,:,]*0.3+real_A[:,1,:,:,]*0.59+real_A[:,2,:,:]*0.11
                    
                same_A = netG_B2A(real_A_.unsqueeze(1))
                loss_identity_A = criterion_identity(same_A, real_A)*opt.lamda_i
            else :
                loss_identity_A=None
                loss_identity_B=None
            # 敵対的損失（GAN loss）
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            # loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
            loss_GAN_A2B = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) 

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            # loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
            loss_GAN_B2A = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) 

            # サイクル一貫性損失（Cycle-consistency loss）
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*opt.lamda_a if (i%opt.cycle_freq)==0 else 0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*opt.lamda_b if (i%opt.cycle_freq)==0 else 0

            # 生成器の合計損失関数（Total loss）
            if opt.isIdentify:
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A +loss_cycle_ABA +loss_cycle_BAB
            else:
                loss_G =  loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()

        if i % opt.d_freq == 0:
            ##### ドメインAの識別器 #####
            netD_A,netD_B= set_requires_grad([netD_A,netD_B],requires_grad=True)
            optimizer_D.zero_grad()

            # ドメインAの本物画像の識別結果（Real loss）
            pred_real = netD_A(real_A)
            # loss_D_real = criterion_GAN(pred_real, target_real)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            # ドメインAの生成画像の識別結果（Fake loss）
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            # loss_D_fake = criterion_GAN(pred_fake, target_fake)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            # 識別器（ドメインA）の合計損失（Total loss）
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            ##### ドメインBの識別器 #####
            # optimizer_D_B.zero_grad()

            # ドメインBの本物画像の識別結果（Real loss）
            pred_real = netD_B(real_B)
            # loss_D_real = criterion_GAN(pred_real, target_real)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # ドメインBの生成画像の識別結果（Fake loss）
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            # loss_D_fake = criterion_GAN(pred_fake, target_fake)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            # 識別器（ドメインB）の合計損失（Total loss）
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            # optimizer_D_B.step()

            optimizer_D.step()
            ###################################

        if i % opt.display_iter == 0:
            print('Epoch[{}]({}/{}) loss_G: {:.4f} loss_G_identity: {:.4f} loss_G_GAN: {:.4f} loss_G_cycle: {:.4f} loss_D: {:.4f}'.format(
                epoch, i, len(dataloader), loss_G, ( loss_identity_A + loss_identity_B) if opt.isIdentify else -1,
                (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)
                ))
            # print('Epoch[{}]({}/{}) loss_G: {:.4f}  loss_G_GAN: {:.4f} loss_G_cycle: {:.4f} loss_D: {:.4f}'.format(
            #     epoch, i, len(dataloader), loss_G,
            #     (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)
            #     ))
        

            train_info = {
                'epoch': epoch, 
                'batch_num': i, 
                'lossG': loss_G.item(),
                'lossG_identity':  (loss_identity_A.item() + loss_identity_B.item())if opt.isIdentify else -1,
                'lossG_GAN': (loss_GAN_A2B.item() + loss_GAN_B2A.item()) ,
                'lossG_cycle': (loss_cycle_ABA.item() + loss_cycle_BAB.item()),
                'lossG_cycle_A2B':loss_cycle_ABA.item() ,
                'lossG_cycle_B2A':  loss_cycle_BAB.item(),
                'lossD': (loss_D_A.item() + loss_D_B.item()), 
                }

            #Save Metrics with Mlflow
            if opt.is_mlflow:
                for _ ,(key , item) in enumerate(train_info.items()):
                    mlflow.log_metric(key,item)



        batches_done = (epoch - 1) * len(dataloader) + i

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    # lr_scheduler_D_A.step()
    # lr_scheduler_D_B.step()


    if epoch % opt.save_epoch==0:
        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), model_path+f'netG_A2B_{epoch}.pth')
        torch.save(netG_B2A.state_dict(), model_path+f'netG_B2A_{epoch}.pth')
        torch.save(netD_A.state_dict(), model_path+f'netD_A_{epoch}.pth')
        torch.save(netD_B.state_dict(), model_path+f'netD_B_{epoch}.pth')
        image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,test_input_A,test_input_B)

    print(f"Epoch {epoch}/{opt.n_epochs}  ETA : {round(time.time()-s)}[sec]"  )
    

#Release Memory
del netD_A,netD_B,fake_A_buffer,fake_B_buffer
gc.collect()

#Save generate image from netG
image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,test_input_A,test_input_B)

if opt.is_mlflow:
    mlflow.end_run()





