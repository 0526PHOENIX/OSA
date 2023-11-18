"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from Unet import Unet


"""
====================================================================================================
Generator
====================================================================================================
"""
class Generator(Unet):

    def __init__(self):

        super().__init__()


"""
====================================================================================================
Initialization Block
====================================================================================================
"""
class Init(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.filters = filters

        self.init_block = nn.Sequential(nn.Conv2d(2, filters, 4, stride = 2, padding = 1),
                                        nn.LeakyReLU(0.01))

    def forward(self, img_in):

        img_out = self.init_block(img_in)

        return img_out


"""
====================================================================================================
Discriminator Block
====================================================================================================
"""
class Dis(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.filters = filters

        self.dis_block = nn.Sequential(nn.Conv2d(filters // 2, filters, 4, stride = 2, padding = 1),
                                       nn.InstanceNorm2d(filters * 2),
                                       nn.LeakyReLU(0.01))

    def forward(self, img_in):

        img_out = self.dis_block(img_in)

        return img_out


"""
====================================================================================================
Final Block
====================================================================================================
"""
class Final(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.filters = filters

        self.final_block = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)),
                                         nn.Conv2d(512, 1, 4, padding = 1, bias = False))

    def forward(self, img_in):

        img_out = self.final_block(img_in)

        return img_out
    

"""
====================================================================================================
Discriminator
====================================================================================================
"""
class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.filters = [64, 128, 256, 512]

        self.init = Init(self.filters[0])

        self.dis1 = Dis(self.filters[1])
        self.dis2 = Dis(self.filters[2])
        self.dis3 = Dis(self.filters[3])

        self.final = Final(self.filters[3])

    def forward(self, img_in_1, img_in_2):

        img_in = torch.cat((img_in_1, img_in_2), 1)

        init = self.init(img_in)

        dis1 = self.dis1(init)
        dis2 = self.dis2(dis1)
        dis3 = self.dis3(dis2)

        final = self.final(dis3)

        return final


"""
====================================================================================================
Test CycleGAN
====================================================================================================
"""
def test_cycle(image):

    gen1 = Generator()
    dis1 = Discriminator()

    gen2 = Generator()        
    dis2 = Discriminator()        

    fake1 = gen1(image)
    result1 = dis1(image, fake1) 

    fake2 = gen2(fake1)
    result2 = dis2(fake1, fake2)

    print(fake1.shape)
    print(fake2.shape)
    print(result1.shape)
    print(result2.shape)


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('\n' + 'Training on device: ' + str(device) + '\n')
    
    # model = Generator().to(device = device)
    # print(summary(model, input_size = (1, 512, 512), batch_size = 2))

    # model = Discriminator().to(device = device)
    # print(summary(model, input_size = [(1, 512, 512), (1, 512, 512)], batch_size = 2))

    # image = torch.randn(2, 1, 512, 512)
    # test_cycle(image)