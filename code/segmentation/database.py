from sys import path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np
import  torch
        
class SegmentDataset(Dataset):
    def __init__(self, root, transforms_=None,mask_transforms=None, mode='train',depth="cycle",isColor=False,limit=None):
        super(SegmentDataset,self).__init__()
        self.transform = transforms.Compose(transforms_)
        self.mask_tranform=transforms.Compose( mask_transforms)
        image_folders=[f for f in os.listdir(root) if os.path.isdir("{}{}".format(root,f))]
        
        self.files_A=["{}{}/image/".format(root,f) for f in image_folders]
        self.files_A=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_A]
        # self.files_B=["{}{}/depth_bfx/".format(root,f) for f in image_folders]
        self.files_B=[f"{root}{f}/{depth}/" for f in image_folders]
        self.files_B=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_B]

        self.files_M=[f"{root}{f}/mask/" for f in image_folders]
        self.files_M=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_M]

        self.isColor=isColor

        if not limit==None:
            self.files_A=self.files_A[:limit]
            self.files_B=self.files_B[:limit]
            self.files_M=self.files_M[:limit]

    def __getitem__(self, index):

        item_C = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        if not self.isColor:
            item_C=torch.cat([item_C,item_B],2)

        item_M=Image.open(self.files_M[index % len(self.files_M)])
        item_M=self.mask_tranform(item_M)

        # B=self._depth_norm(Image.open(self.files_B[index % len(self.files_B)]))
        # print(np.array(B))
        # print(np.shape(B))
        # print(f"C : {np.shape(item_C)} M: {np.shape(item_M)}")

        return {'C': item_C, 'M': item_M}
    
    def __len__(self):
        return max(len(self.files_A),len(self.files_B))
