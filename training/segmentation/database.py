from sys import path
from numpy.core.defchararray import lower
# from torch._C import device, int32
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np
import  torch
import torchvision.transforms.functional as TF
import  cv2
# import albumentations as albu

# def get_preprocessing(preprocessing_fn):
#     _transform = [
#             albu.Lambda(image=preprocessing_fn),
#             # albu.Lambda(image=to_tensor, mask=to_tensor),
#         ]
#     return albu.Compose(_transform)
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result
        
class SegmentDataset(Dataset):
    def __init__(
        self, 
        root,label_path,CLASS, 
        augmentation=None,
        preprocessing=None,
        size=256,
         mode='train',
         depth="cycle",
         isColor=False,
         limit=None,
         cuda_name="cuda"
         ):
        super(SegmentDataset,self).__init__()
        self.device=cuda_name
        self.size=size
        self.augmentation = augmentation#transforms.Compose(transforms_)
        self.preprocessing=preprocessing#transforms.Compose( mask_transforms)
        # image_folders=[f for f in os.listdir(root) if os.path.isdir("{}{}".format(root,f))]
        image_paths=np.loadtxt(root,dtype=str)
        image_folders=["../../dataset/SUNRGBD/{}".format(r[:r.rfind("/image")]) for r in image_paths]
        # print(image_folders)
        self.files_A=[f"{f}/image/" for f in image_folders]
        self.files_A=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_A]
        
        self.files_B=[f"{f}/{depth}/" for f in image_folders]
        self.files_B=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_B]

        # self.files_M=[f"{f}/mask/" for f in image_folders]
        # self.files_M=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_M]
        self.files_M=[f"{label_path}/{l}" for l in os.listdir(label_path)]

        self.isColor=isColor

        # convert str names to class values on masks
        CLASS=[c.lower() for c in CLASS]
        self.class_values = [CLASS.index(cls.lower()) for cls in CLASS]

        if not limit==None:
            self.files_A=self.files_A[:limit]
            self.files_B=self.files_B[:limit]
            self.files_M=self.files_M[:limit]

    def __getitem__(self, index):

        # item_C = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        # item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        
        item_C=cv2.imread(self.files_A[index % len(self.files_A)])
        item_B=cv2.imread(self.files_B[index % len(self.files_B)])
        # print(np.shape(item_C))
        item_C=cv2.cvtColor(item_C, cv2.COLOR_BGR2RGB)
        item_B=cv2.cvtColor(item_B, cv2.COLOR_BGR2RGB)

        item_C=cv2.resize(item_C,(self.size,self.size))
        item_B=cv2.resize(item_B,(self.size,self.size))

        # cv2.imwrite("./color.png",item_C)



        if not self.isColor:
            # item_C=torch.cat([item_C,item_B],2)
            item_C=np.concatenate([item_C,item_B],2)
            

        # print("eofoah : "+self.files_M[index % len(self.files_M)])
        item_M=cv2.imread(self.files_M[index % len(self.files_M)],0)#Image.open(self.files_M[index % len(self.files_M)])
        # item_M= self.mask_tranform(item_M)
        masks = [(item_M == v) for v in self.class_values]
        item_M= np.stack(masks, axis=-1).astype('float')
        # item_M =torch.Tensor( np.stack(masks, axis=-1).astype(np.int32)[0]).permute(2, 0, 1).squeeze()

        item_M=cv2.resize(item_M,(self.size,self.size))
        # print(np.shape(item_C))
        # print(np.shape(item_M))


         # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=item_C, mask=item_M)
            item_C, item_M = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=item_C, mask=item_M)
            item_C, item_M = sample['image'], sample['mask']

        item_C=min_max(item_C)


        # item_C=torch.Tensor(item_C)
        # item_M=torch.Tensor(item_M)



        return item_C, item_M
    
    def __len__(self):
        return max(len(self.files_A),len(self.files_M))
