from os import O_TRUNC
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules import pooling
from torch.nn.modules.activation import Tanh
import  numpy as np


# Define  ReidualBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        """

        """
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3,padding=1,stride=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3,padding=1,stride=1),
            nn.BatchNorm2d(in_features),
            # nn.ReLU(inplace=True),

        )
        self.out=nn.ReLU()

    def forward(self, x):
        return self.out( x+self.conv_block(x))


class ResidualNet(nn.Module):
    def __init__(self, n_block, in_feature):
        super(ResidualNet, self).__init__()

        self.blocks = nn.ModuleList(
            [ResidualBlock(in_features=in_feature) for _ in range(n_block)])

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_features,out_features,is_cat=False):
        super(ConvBlock, self).__init__()
        self.is_cat=is_cat
        # self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3,padding=1,stride=1),
            nn.BatchNorm2d(in_features),
            # nn.ReLU(inplace=True)

        )
    def forward(self,x):
        if self.is_cat:
            x1=self.conv_block(x)
            return torch.cat([x,x1],dim=1)

        return self.conv_block(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_features,out_features,is_cat=False):
        super(UpConvBlock, self).__init__()
        self.is_cat=is_cat
        # self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3,padding=1,stride=1),
            nn.BatchNorm2d(out_features),
            # nn.ReLU(inplace=True)

        )
    def forward(self,x):
        return self.conv_block(x)

class UpBlock(nn.Module):
    def __init__(self, in_nc,out_nc,isLast=False):
        super(UpBlock, self).__init__()
        self.isLast=isLast

        self.Up=nn.ConvTranspose2d(in_nc, in_nc*2, 2, stride=2, padding=0)
    
        self.conv1=UpConvBlock(in_nc*3, out_nc)
        self.conv2=UpConvBlock(out_nc, out_nc)

    def forward(self,x1,x2):
        up=self.Up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # print(x1.size())
        # print(up.size())
        # print(x2.size())
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        c1=self.conv1(torch.cat([x2,up],dim=1))
        if self.isLast:
            return c1
        # print(c1.size())
        # print()
        return self.conv2(c1)

        

class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc,
                 n_residual_block=6, in_features=512):

        super(UNetGenerator, self).__init__()
        self.pool=nn.MaxPool2d(kernel_size=2)

        self.e1=nn.Sequential(
            nn.Conv2d(input_nc, 32, 3,padding=1,stride=1),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True)

        )#ConvBlock(input_nc,32)
        self.e2=ConvBlock(32,64,is_cat=True)
        self.p1=self.pool
        self.e3=ConvBlock(64,64)
        self.e4=ConvBlock(64,128,is_cat=True)
        self.p2=self.pool
        self.e5=ConvBlock(128,128)
        self.e6=ConvBlock(128,256,is_cat=True)
        self.p3=self.pool
        self.e7=ConvBlock(256,256)
        self.e8=ConvBlock(256,in_features,is_cat=True)
        self.p4=self.pool

        self.blocks = ResidualNet(n_residual_block, in_features)

        self.d1=UpBlock(in_features,256)
        self.d2=UpBlock(256,128)
        self.d3=UpBlock(128,64)
        self.d4=UpBlock(64,32,isLast=True)
        self.d5=nn.Sequential(
            nn.Conv2d(32, output_nc, 3,padding=1,stride=1),
            nn.BatchNorm2d(output_nc),
            # nn.ReLU(inplace=True)

        )#ConvBlock(32,output_nc)


    def forward(self, x):
        e1=self.e1(x)
        e1=self.e2(e1)
        p1=self.p1(e1)
        e2=self.e3(p1)
        e2=self.e4(e2)
        p2=self.p2(e2)

        e3=self.e5(p2)
        e3=self.e6(e3)
        p3=self.p3(e3)

        e4=self.e7(p3)
        e4=self.e8(e4)
        p4=self.p4(e4)

        block=self.blocks(p4)

        d1=self.d1(block,e4)
        d2=self.d2(d1,e3)
        d3=self.d3(d2,e2)
        d4=self.d4(d3,e1)

        return self.d5(d4)


class UNetDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(UNetDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 32, 3, stride=1, padding=1),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 1, 3,stride=1, padding=0),
            nn.Tanh()

        )

    def forward(self, x):
        x = self.model(x)
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return x


class SNUNetDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(SNUNetDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 32, 3, stride=1, padding=1),

            nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
            # nn.BatchNorm2d(64),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            # nn.BatchNorm2d(128),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),
            # nn.BatchNorm2d(256),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, stride=2, padding=1)),
            # nn.BatchNorm2d(512),

            nn.Conv2d(512, 1, 3,stride=1, padding=0),
            nn.Tanh()

        )

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class SpectralNorm(nn.Module):
    def __init__(self) :
        super(SpectralNorm,self).__init__()
    def forward(self,input):
        return nn.utils.spectral_norm(input)

class SNDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(SNDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_nc, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1)

        )

    def forward(self, x):
        x = self.model(x)
        
        return x#F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# See Model Structure
if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = UNetGenerator(3, 1)
    netD = UNetDiscriminator(2)
    # netSD=SNDiscriminator(3)

    print("--- the Structure of the Generator model ---")
    print(netG)
    print()
    print("--- the Structure of the Discriminator model ---")
    print(netD)

    print("IS CUDA AVAILABLE = {}".format(torch.cuda.is_available()))

    summary(netG.to(device), input_size=(3, 640 ,480 ))
    summary(netD.to(device), input_size=(2, 640 ,480 ))

    dummy_input_G=torch.randn(1,3,640,480).to(device)#ダミーの入力を用意する
    dummy_input_D=torch.randn(1,2,640,480).to(device)#ダミーの入力を用意する
    input_names = [ "input"]
    output_names = [ "output" ]

    torch.onnx.export(netG, dummy_input_G, "./test_model_G.onnx", verbose=True,input_names=input_names,output_names=output_names)
    torch.onnx.export(netD, dummy_input_D, "./test_model_D.onnx", verbose=True,input_names=input_names,output_names=output_names)
