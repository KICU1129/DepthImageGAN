from sys import path
from matplotlib.colors import cnames
import torch
from torch.autograd import Variable
import random
import os
import datetime
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def disnorm(img):
    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    return 0.5*(img + 1.0)

def normalize(data):
    # # data=np.array(data)
    # plt.imshow(data,cmap="gray")
    # plt.title("Bugs!!")
    # plt.show()
    data=(data-np.min(data))/(np.max(data)-np.min(data))
    
    return data

def resize(data , size=(256,256)):
    data=np.array(data)
    return cv2.resize(data,size)

def rgb2gray(im,s):
    im=torch.reshape(im,[1,s[-2],s[-1]])
    print(f"max : {torch.max(im)} min : {torch.min(im)}")
    return torch.concat([im, im,im], 0)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 
            image_numpy = cv2.cvtColor(image_numpy,cv2.COLOR_BGR2GRAY)
            return image_numpy
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



# 過去の生成データ(50iter分)を保持しておく
class ReplayBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            
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

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
        return nets


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('ConvBlock') == -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Recoder:
    def __init__(self,version,root="../output/record",epoch=0) :
        #self.image_root=os.path.join(root,"images/")#os.path.join(root,"{}-version{}/".format(datetime.date.today(),version))
        self.image_root=os.path.join(root,"{}-version{}/".format(datetime.date.today(),version))

        self.image_num=epoch
        if not os.path.exists(self.image_root):
            os.mkdir(self.image_root)
    
    def _save(self,img_list,path,fig_path,cmode="jet"):
        fig=plt.figure(figsize=(15,30))
        plt.title("Summary ",fontsize=20)
        plt.axis("off")
        k=len(img_list["real_A"])
        for i in range(k):

            ### 画像 ###
            ax=fig.add_subplot(6,k,i+1)
            ax.set_title("Real A",fontsize=20)
            ax.imshow( img_list["real_A"][i])
            ax.axis("off")

            ax=fig.add_subplot(6,k,i+1+k)
            ax.set_title("Real B",fontsize=20)
            a=ax.imshow(img_list["real_B"][i], cmap=cmode)
            ax.axis("off")
            fig.colorbar(a, ax=ax)
            

            ax=fig.add_subplot(6,k,i+1+2*k)
            ax.set_title("Fake A",fontsize=20)
            ax.imshow(img_list["fake_A"][i])
            ax.axis("off")

            ax=fig.add_subplot(6,k,i+1+3*k)
            ax.set_title("Fake B",fontsize=20)
            a=ax.imshow(img_list["fake_B"][i], cmap=cmode)
            ax.axis("off")
            fig.colorbar(a, ax=ax)

            ax=fig.add_subplot(6,k,i+1+4*k)
            ax.set_title("Cycle A",fontsize=20)
            ax.imshow(img_list["cycle_A"][i])
            ax.axis("off")

            ax=fig.add_subplot(6,k,i+1+5*k)
            ax.set_title("Cycle B",fontsize=20)
            a=ax.imshow(img_list["cycle_B"][i], cmap=cmode)
            ax.axis("off")
            fig.colorbar(a, ax=ax)

        plt.savefig(path)

        ### 図 ###
        # plt.clf()
        # ax=fig.add_subplot(2,1,1)
        # ax.set_title("Graph A",fontsize=20)
        # ax.hist( np.reshape( img_list["real_A"][0],-1),label="real_A", bins=100, alpha=0.3, histtype='stepfilled')
        # ax.hist( np.reshape( img_list["cycle_A"][0],-1),label="cycle_A", bins=100, alpha=0.3, histtype='stepfilled')
        # ax.hist( np.reshape( img_list["fake_A"][0],-1),label="fake_A", bins=100, alpha=0.3, histtype='stepfilled')
        # ax.legend()

        # ax=fig.add_subplot(2,1,2)
        # ax.set_title("Graph B",fontsize=20)
        # ax.hist( np.reshape( img_list["real_B"][0],-1),label="real_B", bins=100, alpha=0.3, histtype='stepfilled')
        # ax.hist( np.reshape( img_list["cycle_B"][0],-1),label="cycle_B", bins=100, alpha=0.3, histtype='stepfilled')
        # ax.hist( np.reshape( img_list["fake_B"][0],-1),label="fake_B", bins=100, alpha=0.3, histtype='stepfilled')
        # ax.legend()
        # plt.savefig(fig_path)

    def _save_fig(self,img_list,fig_path):
        fig=plt.figure(figsize=(15,30))
        ### 図 ###
        plt.clf()
        ax=fig.add_subplot(2,1,1)
        ax.set_title("Graph A",fontsize=20)
        ax.hist( np.reshape( img_list["real_A"],-1),label="real_A", bins=100, alpha=0.3, histtype='stepfilled')
        ax.hist( np.reshape( img_list["cycle_A"],-1),label="cycle_A", bins=100, alpha=0.3, histtype='stepfilled')
        ax.hist( np.reshape( img_list["fake_A"],-1),label="fake_A", bins=100, alpha=0.3, histtype='stepfilled')
        ax.legend()

        ax=fig.add_subplot(2,1,2)
        ax.set_title("Graph B",fontsize=20)
        ax.hist( np.reshape( img_list["real_B"],-1),label="real_B", bins=100, alpha=0.3, histtype='stepfilled')
        ax.hist( np.reshape( img_list["cycle_B"],-1),label="cycle_B", bins=100, alpha=0.3, histtype='stepfilled')
        ax.hist( np.reshape( img_list["fake_B"],-1),label="fake_B", bins=100, alpha=0.3, histtype='stepfilled')
        ax.legend()
        plt.savefig(fig_path)
            



    def save_image(self,netG_A2B,netG_B2A,dataloader,input_A,input_B,n_sample=1):
        c=0
        img_path_list={"image-A2B":[],"image-B2A" : []}
        img_list={"real_A":[],"real_B":[],"fake_A":[],"fake_B":[],"cycle_A":[],"cycle_B":[],}
        fig_list={"real_A":[],"real_B":[],"fake_A":[],"fake_B":[],"cycle_A":[],"cycle_B":[],}

        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))
            # A=np.array(to_pil_image(batch["A"]))
            A=tensor2im(real_A)
            # B=np.array(to_pil_image(batch["B"]))
            B=tensor2im(real_B)
            

            img_list["real_A"].append(A)
            img_list["real_B"].append(B)
            fig_list["real_A"]=real_A.data.cpu().float().numpy()
            fig_list["real_B"]=real_B.data.cpu().float().numpy()

            # Generate output
            # fake_B = 0.5*(netG_A2B(real_A).cpu().data + 1.0)
            # #fake_B=rgb2gray(fake_B,np.shape(fake_B))#cv2.cvtColor(fake_B, cv2.COLOR_GRAY2BGR)
            # fake_shape=[int(i) for i in  np.shape(fake_B)]
            # fake_B=np.array(to_pil_image(torch.reshape(fake_B,(fake_shape[1],fake_shape[2],fake_shape[3]))))
            fake_B=netG_A2B(real_A)
            fig_list["fake_B"]=fake_B.data.cpu().float().numpy()
            fake_B=tensor2im(fake_B)
            img_list["fake_B"].append(fake_B)

            # cycle_A = 0.5*(netG_B2A( netG_A2B(real_A)).cpu().data + 1.0)
            # fake_shape=[int(i) for i in  np.shape(cycle_A)]
            # cycle_A=np.array(to_pil_image(torch.reshape(cycle_A,(fake_shape[-3],fake_shape[-2],fake_shape[-1]))))
            cycle_A=netG_B2A( netG_A2B(real_A))
            fig_list["cycle_A"]=cycle_A.data.cpu().float().numpy()
            cycle_A=tensor2im(cycle_A)
            img_list["cycle_A"].append(cycle_A)


            # out_img1 = torch.cat([real_A, fake_B,real_B], dim=2)

            if netG_B2A!=None and input_B!=None:
                # fake_A = 0.5*(netG_B2A(real_B).cpu().data + 1.0)
                # fake_shape=[int(i) for i in  np.shape(fake_A)]
                # fake_A=np.array(to_pil_image(torch.reshape(fake_A,(fake_shape[1],fake_shape[2],fake_shape[3]))))
                fake_A=netG_B2A(real_B)
                fig_list["fake_A"]=fake_A.data.cpu().float().numpy()
                fake_A=tensor2im(fake_A)
                img_list["fake_A"].append(fake_A)

                # cycle_B = 0.5*(netG_A2B(netG_B2A(real_B)).cpu().data + 1.0)
                # fake_shape=[int(i) for i in  np.shape(cycle_B)]
                # cycle_B=np.array(to_pil_image(torch.reshape(cycle_B,(fake_shape[1],fake_shape[2],fake_shape[3]))))
                cycle_B=netG_A2B(netG_B2A(real_B))
                fig_list["cycle_B"]=cycle_B.data.cpu().float().numpy()
                cycle_B=tensor2im(cycle_B)
                img_list["cycle_B"].append(cycle_B)

            
            c+=1
            if i>n_sample:
                break
        im_path=f"{self.image_root}record_epoch_{self.image_num}.png"
        fig_path=f"{self.image_root}fig_epoch_{self.image_num}.png"
        self._save(img_list,im_path,fig_path,cmode="jet")
        self._save_fig(fig_list,fig_path)
        self.image_num+=1
        return img_path_list
        

