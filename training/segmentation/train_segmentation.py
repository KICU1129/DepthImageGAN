from operator import gt
import sys
# from albumentations.core.serialization import save

from numpy.core.fromnumeric import size
sys.path.append('../')
from opt import SegmentationOpts
# from code.utils import *
from database import SegmentDataset
from segmentation_models_pytorch import Unet
import segmentation_models_pytorch as smp
# import albumentations as albu

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
import albumentations as albu
import  matplotlib.pyplot as plt

""" --- Initial Setting  ---"""
opt=SegmentationOpts()

root_path="../../output/"
model_path=root_path+f"model/{opt.experience_ver}_{opt.version}/"
record_path=root_path+f"record/{opt.experience_ver}_{opt.version}/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(record_path):
    os.mkdir(record_path)

mlflow.set_experiment("segmentation_unet_{}".format(opt.experience_ver))
mlflow.start_run()
for _ ,(key , item) in enumerate(vars(opt).items()):
    mlflow.log_param(key,item)

# テンソル化
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

# 前処理
def get_preprocessing(preprocessing_fn):
    _transform = [
            # albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    return albu.Compose(_transform)

# データ確認用
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def save_result(epoch):
    #SHOW RESULT
    show_dataset=SegmentDataset(
        root=opt.test_root,CLASS=opt.CLASS,label_path=opt.test_label,size=opt.size,
        augmentation=None,preprocessing=None,
        limit=opt.limit,isColor=True,depth=opt.depth
        )
    plt.title("Train Result")
    fig=plt.figure(figsize=(20,20))
    for i in range(5):
        n = np.random.choice(len(test_dataset))
        
        image_vis = test_dataset[n][0]
        image, gt_mask = show_dataset[n]
        
        # gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image_vis).to(opt.DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask=pr_mask.transpose(1, 2, 0)

        gt_mask=np.argmax(gt_mask,axis=2)
        pr_mask=np.argmax(pr_mask,axis=2)

        ax=fig.add_subplot(3,5,i+1)
        ax.set_title("Image")
        ax.imshow(image)

        ax=fig.add_subplot(3,5,i+1+5)
        ax.set_title("GT")
        ax.imshow(gt_mask)

        ax=fig.add_subplot(3,5,i+1+10)
        ax.set_title("PR")
        ax.imshow(pr_mask)
    plt.savefig(f"{record_path}/train_result_{epoch}.png")



""" --- Call Models ---"""

#DFINE PREPROCESS
preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.ENCODER, opt.ENCODER_WEIGHTS)

train_dataset=SegmentDataset(
    root=opt.dataroot,CLASS=opt.CLASS,label_path=opt.train_label,size=opt.size,
    augmentation=None,preprocessing=get_preprocessing(preprocessing_fn),
    limit=opt.limit,isColor=opt.isColor,depth=opt.depth
    )
train_dataloader = DataLoader(train_dataset, 
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

test_dataset=SegmentDataset(
    root=opt.test_root,CLASS=opt.CLASS,label_path=opt.test_label,size=opt.size ,
    augmentation=None,preprocessing=get_preprocessing(preprocessing_fn),
    limit=opt.sample_num,isColor=opt.isColor,depth=opt.depth
    )
test_dataloader = DataLoader(test_dataset, 
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)



#Define 

model = smp.Unet(
    encoder_name=opt.ENCODER,
    encoder_weights=opt.ENCODER_WEIGHTS,
    classes=opt.num_class,
    activation=opt.ACTIVATION,
    # aux_params=opt.aux_params,
    in_channels = opt.input_nc,
    )

model.to(opt.DEVICE)

#Load model if loadWeihgt is True
if opt.load_weight:
    model.load_state_dict(torch.load(f'{model_path}{opt.DECODER}_{opt.ENCODER}.pth', map_location=opt.DEVICE), strict=False)

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5),
    smp.utils.metrics.Accuracy(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
    smp.utils.metrics.Precision(threshold=0.5),
]

# ロス
if opt.loss_name=="dice":
    loss = smp.utils.losses.DiceLoss()
elif opt.loss_name=="jaccard":
    loss = smp.utils.losses.JaccardLoss()

# 最適化関数
if opt.opt_name=="adam":
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=opt.lr),
    ])
# elif opt.opt_name=="sgd":
#     optimizer = torch.optim.Adam([
#         dict(params=model.parameters(), lr=0.001),
#     ])
    
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=opt.DEVICE,
    verbose=True,
)
valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=opt.DEVICE,
    verbose=True,
)

    # visualize(
    #     image=image, 
    #     ground_truth_mask=gt_mask, 
    #     predicted_mask=pr_mask
    # )
# print("Print")
# save_result(999)
# exit()

#########################################################
# OK LET'S TRAIN !!
#########################################################

max_score = 0
for i in range(opt.n_epochs):
 
    print('\nEpoch: {}'.format(i))

    try:
        train_logs = train_epoch.run(train_dataloader)
        val_logs = valid_epoch.run(test_dataloader)
    except Exception as e:
        print("Error Occored !!")
        print(e)
    
    # print(train_logs)
    print("/n [train log]  iou :{}  [valid log]  iou : {}  ".format(train_logs["iou_score"],val_logs["iou_score"]) )
    
    # do something (save model, change lr, etc.)
    if max_score < val_logs['iou_score']:
        max_score = val_logs['iou_score']
        torch.save(model.state_dict(), f'{model_path}{opt.DECODER}_{opt.ENCODER}.pth')
        print('Model saved!')

    if i%10==0:
        save_result(i)
 
    if i == 50:
        try:
            optimizer.param_groups[0]['lr'] *= 0.5#1e-4
        except:
            pass
        print('Decrease decoder learning rate to 1e-4!')

    
    train_info = {
            'epoch': i, 
            "train_iou"  :train_logs["iou_score"],
            "train_fscore"  :train_logs["fscore"],
            "train_loss"  :train_logs["dice_loss"],
            "train_accuracy"  :train_logs["accuracy"],
            "train_recall"  :train_logs["recall"],
            "train_precision"  :train_logs["precision"],

            "val_iou"  :val_logs["iou_score"],
            "val_fscore"  :val_logs["fscore"],
            "val_loss"  :val_logs["dice_loss"],
            "val_accuracy"  :val_logs["accuracy"],
            "val_recall"  :val_logs["recall"],
            "val_precision"  :val_logs["precision"],

            }

    #Save Metrics with Mlflow
    for _ ,(key , item) in enumerate(train_info.items()):
        mlflow.log_metric(key,item)

    print("---------------------------------------------------------------------------------------")
###############################################################################################################
save_result(opt.n_epochs)
torch.save(model.state_dict(), f'{model_path}{opt.DECODER}_{opt.ENCODER}.pth')
print('Model saved!')

#Save generate image from netG
# image_pathes=recorder.save_image(netG_A2B,netG_B2A,sample_images,input_A,input_B)

#Save Record with Mlflow
# for i in range(len(image_pathes['image-A2B'])):
#     mlflow.log_artifact(image_pathes["image-A2B"][i])
#     mlflow.log_artifact(image_pathes["image-B2A"][i])



mlflow.end_run()





