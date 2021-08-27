import torch
from torch.autograd import Variable
import random
import os
import datetime
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np

# 過去の生成データ(50iter分)を保持しておく
class ReplayBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            #
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class Recoder:
    def __init__(self,version,root="../output/record") :
        #self.image_root=os.path.join(root,"images/")#os.path.join(root,"{}-version{}/".format(datetime.date.today(),version))
        self.image_root=os.path.join(root,"{}-version{}/".format(datetime.date.today(),version))

        self.image_num=0
        if not os.path.exists(self.image_root):
            os.mkdir(self.image_root)

    def save_image(self,netG_A2B,netG_B2A,dataloader,input_A,input_B,n_sample=1):
        c=0
        img_path_list={"image-A2B":[],"image-B2A" : []}

        for i, batch in enumerate(dataloader):
            # image_B=to_pil_image(batch["B"])
            # image_B.save("./imageB.png")
            # image_A=to_pil_image(batch["A"])
            # image_A.save("./imageA.png")


            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # Generate output
            fake_B = 0.5*(netG_A2B(real_A).data + 1.0)

            out_img1 = torch.cat([real_A, fake_B,real_B], dim=2)

            if netG_B2A!=None and input_B!=None:
                fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
                out_img2 = torch.cat([real_B, fake_A,real_A], dim=2)
                imgA2B_path="{}image_A2B_{}-{}.png".format(self.image_root,self.image_num,c)
                img_path_list["image-A2B"].append(imgA2B_path)

                

            # Save image files
            imgB2A_path="{}image_B2A_{}-{}.png".format(self.image_root,self.image_num,c)
            save_image(out_img1,imgA2B_path )
            save_image(out_img2, imgB2A_path)
            img_path_list["image-B2A"].append(imgB2A_path)
            c+=1
            if i>n_sample:
                break
        self.image_num+=1
        return img_path_list
        

