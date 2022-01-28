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

def get_transform( opt, params=None, grayscale=None, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    
    osize = [int(opt.size[1]*1.1),int(opt.size[0]*1.1)]
    transform_list.append(transforms.Resize(osize, method))
    
    transform_list.append(transforms.RandomCrop([opt.size[-1-i] for i in range(len(opt.size))])) #256
    

    # if not opt.no_flip:
    #     if params is None:
    #         transform_list.append(transforms.RandomHorizontalFlip())
    #     elif params['flip']:
    #         transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class ImageDataset(Dataset):
    def __init__(self,opt,root,  unaligned=False, mode='train',limit=None,datamode="none"):
        super(ImageDataset,self).__init__()
        self.unaligned=unaligned
        self.depth_gray=opt.domainB_nc==1
        image_folders=[f for f in os.listdir(root) if os.path.isdir("{}{}".format(root,f))]

        self.files_A=["{}{}/image/".format(root,f) for f in image_folders]
        self.files_A=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_A]
        # self.files_B=["{}{}/depth_bfx/".format(root,f) for f in image_folders]
        self.files_B=["{}{}/{}/".format(root,f,opt.depth_name) for f in image_folders]
        self.files_B=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_B]

        self.transform_A = get_transform(opt,grayscale=False)
        self.transform_B = get_transform(opt,grayscale=True)

        if not limit==None and datamode!="test":
            self.files_A=self.files_A[:limit]
            self.files_B=self.files_B[:limit]
        elif not limit==None and datamode=="test":
            self.files_A=self.files_A[len(self.files_A)-limit:]
            self.files_B=self.files_B[len(self.files_B)-limit:]

        

    def __getitem__(self, index):

        item_A = self.transform_A(Image.fromarray( cv2.imread(self.files_A[index % len(self.files_A)]) ))

        if self.unaligned :
            if self.depth_gray:
                item_B=cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)],0)
                item_B = self.transform_B(Image.fromarray( item_B))
            else:
                item_B = self.transform_B(self._depth_norm(cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])))
        else:
            if self.depth_gray:
                item_B = self.transform_B(self._depth_norm(cv2.imread(self.files_B[index % len(self.files_B)],0)))
            else:
                item_B = self.transform_B(self._depth_norm(cv2.imread(self.files_B[index % len(self.files_B)])))
        
        return {'A': item_A, 'B': item_B}
    
    def _depth_norm(self,image):
        return Image.fromarray(image)

        # num_img=np.array(image)
        # min_p=np.min(num_img)
        # max_p=np.max(num_img)
        # # print(f"min={min_p} , max={max_p}")
        # img=255*(num_img-min_p)/(max_p-min_p)

        # return Image.fromarray(img)

    def __len__(self):
        return max(len(self.files_A),len(self.files_B))

class AnimeDataset(Dataset):
    def __init__(self, root, transforms_=None,transforms_1dim=None,depth_name="depth_color",depth_gray=False, unaligned=False, mode='train',limit=None):
        super(AnimeDataset,self).__init__()
        self.transform = transforms.Compose(transforms_)
        self.transform_1dim = transforms.Compose(transforms_1dim)
        self.unaligned=unaligned
        self.depth_gray=depth_gray
        image_folders=[f for f in os.listdir(root) if os.path.isdir("{}{}".format(root,f))]
        anime_folder=f"{root}/animeface-character-dataset/animeface-character-dataset/thumb/"
        anime_folders=os.listdir(anime_folder)
        face_folders=f"{root}/img_align_celeba/img_align_celeba/"

        files_A=[[f"{anime_folder}/{f}/{im}" for im in os.listdir(f"{anime_folder}/{f}")] for f in anime_folders]
        self.files_A=[i  for f in files_A for i in f if ".png" in i] 
        # self.files_B=["{}{}/depth_bfx/".format(root,f) for f in image_folders]
        self.files_B=[f"{face_folders}{f}" for f in os.listdir(face_folders) if ".jpg" in f]
        

        if not limit==None:
            self.files_A=self.files_A[:limit]
            self.files_B=self.files_B[:limit]

        

    def __getitem__(self, index):

        item_A = self.transform(cv2.imread(self.files_A[index % len(self.files_A)]))

        if self.unaligned :
            item_B = self.transform(self._depth_norm(cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])))
        else:
            item_B = self.transform(self._depth_norm(cv2.imread(self.files_B[index % len(self.files_B)])))
        

        return {'A': item_A, 'B': item_B}
    
    def _depth_norm(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB )

        # num_img=np.array(image)
        # min_p=np.min(num_img)
        # max_p=np.max(num_img)
        # # print(f"min={min_p} , max={max_p}")
        # img=255*(num_img-min_p)/(max_p-min_p)

        # return Image.fromarray(img)

    def __len__(self):
        return max(len(self.files_A),len(self.files_B))
        

if __name__ =="__main__":
    path="../dataset/SUNRGBD/SUNRGBD/kv1/b3dodata/"
    dataset=ImageDataset(path,transforms_=[transforms.ToTensor()])
    print(dataset[0])
