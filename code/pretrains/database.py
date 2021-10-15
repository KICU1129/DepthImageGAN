from sys import path

from matplotlib.pyplot import imshow
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np
import  torch
import matplotlib.pyplot as plt
from utils import *

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None,transforms_1dim=None,depth_name="depth_color",image_name="image",depth_gray=False, unaligned=False, mode='train',limit=None):
        super(ImageDataset,self).__init__()
        self.transform = transforms.Compose(transforms_)
        self.transform_1dim = transforms.Compose(transforms_1dim)
        self.unaligned=unaligned
        self.depth_gray=depth_gray
        image_folders=[f for f in os.listdir(root) if os.path.isdir("{}{}".format(root,f))]

        self.files_A=["{}{}/{}/".format(root,f,depth_name) for f in image_folders]
        self.files_A=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_A]
        # self.files_B=["{}{}/depth_bfx/".format(root,f) for f in image_folders]
        self.files_B=["{}{}/{}/".format(root,f,depth_name) for f in image_folders]
        self.files_B=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_B]


        self.files_C=["{}{}/image/".format(root,f) for f in image_folders]
        self.files_C=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_C]
        # self.files_B=["{}{}/depth_bfx/".format(root,f) for f in image_folders]
        self.files_D=["{}{}/image/".format(root,f) for f in image_folders]
        self.files_D=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_D]


        if not limit==None:
            self.files_A=self.files_A[:limit]
            self.files_B=self.files_B[:limit]

        

    def __getitem__(self, index):

        item_A = self.transform(cv2.imread(self.files_A[index % len(self.files_A)]))
        item_B=self._depth_norm(cv2.imread(self.files_B[index % len(self.files_B)],0))
        item_B = self.transform(item_B)
        
        item_D = self.transform(cv2.imread(self.files_D[index % len(self.files_D)]))
        item_C=self._depth_norm(cv2.imread(self.files_C[index % len(self.files_C)],0))
        item_C = self.transform(item_C)

        return {'A': item_A, 'B': item_B,"C":item_C,"D":item_D}
    
    def _depth_norm(self,image):
        return image

        # num_img=np.array(image)
        # min_p=np.min(num_img)
        # max_p=np.max(num_img)
        # # print(f"min={min_p} , max={max_p}")
        # img=255*(num_img-min_p)/(max_p-min_p)

        # return Image.fromarray(img)

    def __len__(self):
        return max(len(self.files_A),len(self.files_B))
        

if __name__ =="__main__":
    transforms_ = [ transforms.Lambda(normalize),
                transforms.Lambda(resize),
                # transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC), 
                # transforms.RandomCrop(opt.size), 
                # transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                ]
    dataroot = r"E:\KISUKE\SUNRGBD\SUNRGBD\kv1\b3dodata/"
    dataset=ImageDataset(depth_name="depth",depth_gray=True,root=dataroot,
                        transforms_=transforms_, limit=None,unaligned=False)
    dataset[0]
