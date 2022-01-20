import torch.nn.functional as F
import torch.nn as nn
import torch


# Define  ReidualBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        """

        """
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)

        )

    def forward(self, x):
        return x+self.conv_block(x)


class ResidualNet(nn.Module):
    def __init__(self, n_block, in_feature):
        super(ResidualNet, self).__init__()

        self.blocks = nn.ModuleList(
            [ResidualBlock(in_features=in_feature) for _ in range(n_block)])

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc,
                 n_residual_block=9, in_features=256):

        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.blocks = ResidualNet(n_residual_block, in_features)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1)

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
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# See Model Structure
if __name__ == "__main__":
    netG = Generator(3, 3)
    netD = Discriminator(3)
    netSD=SNDiscriminator(3)

    print("--- the Structure of the Generator model ---")
    print(netG)
    print()
    print("--- the Structure of the Discriminator model ---")
    print(netD)
    print("--- the Structure of the SND model ---")
    print(netSD)

    print("IS CUDA AVAILABLE = {}".format(torch.cuda.is_available()))
