from sys import path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train',limit=None):
        super(ImageDataset,self).__init__()
        self.transform = transforms.Compose(transforms_)
        self.unaligned=unaligned
        image_folders=[f for f in os.listdir(root) if os.path.isdir("{}{}".format(root,f))]
        
        self.files_A=["{}{}/image/".format(root,f) for f in image_folders]
        self.files_A=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_A]
        # self.files_B=["{}{}/depth_bfx/".format(root,f) for f in image_folders]
        self.files_B=["{}{}/depth_color/".format(root,f) for f in image_folders]
        self.files_B=["{}{}".format(f,os.listdir(f)[0]) for f in self.files_B]

        if not limit==None:
            self.files_A=self.files_A[:limit]
            self.files_B=self.files_B[:limit]

        

    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(self._depth_norm(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])).convert('RGB'))
        else:
            item_B = self.transform(self._depth_norm(Image.open(self.files_B[index % len(self.files_B)])).convert('RGB'))

        # B=self._depth_norm(Image.open(self.files_B[index % len(self.files_B)]))
        # print(np.array(B))
        # print(np.shape(B))
        return {'A': item_A, 'B': item_B}
    
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
    path="../dataset/SUNRGBD/SUNRGBD/kv1/b3dodata/"
    dataset=ImageDataset(path,transforms_=[transforms.ToTensor()])
    print(dataset[0])
