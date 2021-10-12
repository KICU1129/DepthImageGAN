from model import Generator,Discriminator,SNDiscriminator
from opt import Opts
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
import time
import mlflow

""" --- Initial Setting  ---"""
opt=Opts()

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
if opt.G_model=="G":
    netG_A2B = Generator(opt.domainA_nc, opt.domainB_nc)
    netG_B2A = Generator(opt.domainB_nc, opt.domainA_nc)

# 識別器
if opt.D_model=="D":
    netD_A = Discriminator(opt.domainA_nc)
    netD_B = Discriminator(opt.domainB_nc)
if opt.D_model=="SND":
    netD_A = SNDiscriminator(opt.domainA_nc)
    netD_B = SNDiscriminator(opt.domainB_nc)

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
criterion_GAN = torch.nn.MSELoss()
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
input_A = Tensor(opt.batch_size, opt.domainA_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.domainB_nc, opt.size, opt.size)
test_input_A = Tensor(1, opt.domainA_nc, opt.size, opt.size)
test_input_B = Tensor(1, opt.domainB_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

# 過去データ分のメモリ確保
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# データローダー
transforms_ = [ transforms.Lambda(normalize),
                transforms.Lambda(resize),
                # transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC), 
                # transforms.RandomCrop(opt.size), 
                # transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                ]

dataset=ImageDataset(depth_name=opt.depth_name,depth_gray=opt.domainB_nc==1,root=opt.dataroot,
                     transforms_=transforms_, limit=None,unaligned=opt.unaligned)
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
    #For Test generate images
    # image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,test_input_A,test_input_B)
    for i, batch in enumerate(dataloader):
        if np.shape(batch["A"])[0]<opt.batch_size:
            break
        # モデルの入力
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))


        if i%opt.g_freq==0:

            ##### 生成器A2B、B2Aの処理 #####
            optimizer_G.zero_grad()

            if opt.isIdentify:
                # 同一性損失の計算（Identity loss)
                # G_A2B(B)はBと一致
                same_B = netG_A2B(real_B)
                loss_identity_B = criterion_identity(same_B, real_B)*5.0
                # G_B2A(A)はAと一致
                same_A = netG_B2A(real_A)
                loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # 敵対的損失（GAN loss）
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # サイクル一貫性損失（Cycle-consistency loss）
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # 生成器の合計損失関数（Total loss）
            if opt.isIdentify:
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A +opt.lamda_a* loss_cycle_ABA + opt.lamda_b* loss_cycle_BAB
            else:
                loss_G =  loss_GAN_A2B + loss_GAN_B2A +opt.lamda_a* loss_cycle_ABA + opt.lamda_b* loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()

        if i % opt.d_freq == 0:
            ##### ドメインAの識別器 #####
            optimizer_D_A.zero_grad()

            # ドメインAの本物画像の識別結果（Real loss）
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # ドメインAの生成画像の識別結果（Fake loss）
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # 識別器（ドメインA）の合計損失（Total loss）
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            ##### ドメインBの識別器 #####
            optimizer_D_B.zero_grad()

            # ドメインBの本物画像の識別結果（Real loss）
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # ドメインBの生成画像の識別結果（Fake loss）
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # 識別器（ドメインB）の合計損失（Total loss）
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

        if i % opt.display_iter == 0:
            # print('Epoch[{}]({}/{}) loss_G: {:.4f} loss_G_identity: {:.4f} loss_G_GAN: {:.4f} loss_G_cycle: {:.4f} loss_D: {:.4f}'.format(
            #     epoch, i, len(dataloader), loss_G, (loss_identity_A + loss_identity_B),
            #     (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)
            #     ))
            print('Epoch[{}]({}/{}) loss_G: {:.4f}  loss_G_GAN: {:.4f} loss_G_cycle: {:.4f} loss_D: {:.4f}'.format(
                epoch, i, len(dataloader), loss_G,
                (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)
                ))
        

            train_info = {
                'epoch': epoch, 
                'batch_num': i, 
                'lossG': loss_G.item(),
                # 'lossG_identity': (loss_identity_A.item() + loss_identity_B.item()),
                'lossG_GAN': (loss_GAN_A2B.item() + loss_GAN_B2A.item()),
                'lossG_cycle': (loss_cycle_ABA.item() + loss_cycle_BAB.item()),
                'lossD': (loss_D_A.item() + loss_D_B.item()), 
                }

            #Save Metrics with Mlflow
            for _ ,(key , item) in enumerate(train_info.items()):
                mlflow.log_metric(key,item)



        batches_done = (epoch - 1) * len(dataloader) + i

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()


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
# image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,input_A,input_B)

#Save Record with Mlflow
# for i in range(len(image_pathes['image-A2B'])):
#     mlflow.log_artifact(image_pathes["image-A2B"][i])
#     mlflow.log_artifact(image_pathes["image-B2A"][i])


#Save generate image from netG
image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,test_input_A,test_input_B)

mlflow.end_run()





