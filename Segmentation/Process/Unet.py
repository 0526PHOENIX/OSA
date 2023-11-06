"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


"""
====================================================================================================
Residual Block
====================================================================================================
"""
class Res(nn.Module):

    def __init__(self, filters):

        super().__init__()
        # critical parameter
        self.filters = filters

        # (normalization -> activation -> convolution) * 2
        self.res_block = nn.Sequential(nn.BatchNorm2d(num_features = filters), 
                                       nn.LeakyReLU(0.01),
                                       nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False),
                                       nn.BatchNorm2d(num_features = filters),
                                       nn.LeakyReLU(0.01),
                                       nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False))

    def forward(self, img_in):

        img_out = self.res_block(img_in)

        # jump connection
        return img_in + img_out


"""
====================================================================================================
Initialization Block
====================================================================================================
"""
class Init(nn.Module):

    def __init__(self, num_channel, filters):

        super().__init__()
        # critical parameter
        self.num_channel = num_channel
        self.filters = filters

        # convolution -> dropout -> res block
        self.conv = nn.Conv2d(num_channel, filters, kernel_size = 3, padding = 1, bias = False)
        self.drop = nn.Dropout2d(0.2)
        self.res = Res(filters)

    def forward(self, img_in):

        img_out = self.conv(img_in)
        img_out = self.drop(img_out)
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Downsampling Block
====================================================================================================
"""
class Down(nn.Module):

    def __init__(self, filters):

        super().__init__()
        # critical parameter
        self.filters = filters

        # downsampling -> resblock
        self.down = nn.Conv2d(filters // 2, filters, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.res = Res(filters)

    def forward(self, img_in):

        img_out = self.down(img_in)
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Upsampling Block
====================================================================================================
"""
class Up(nn.Module):

    def __init__(self, filters):

        super().__init__()
        # critical parameter
        self.filters = filters

        # convolution -> upsampling -> res block
        self.conv = nn.Conv2d(filters * 2, filters, kernel_size = 1, padding = 0, bias = False)
        self.up = nn.ConvTranspose2d(filters, filters, kernel_size = 2, stride = 2, bias = False)
        self.res = Res(filters)
    
    def forward(self, img_in_1, img_in_2):

        img_out = self.conv(img_in_1)
        img_out = self.up(img_out)
        # jump connection
        img_out += img_in_2
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Final Block
====================================================================================================
"""
class Final(nn.Module):

    def __init__(self, filters, num_class):

        super().__init__()
        # critical parameter
        self.num_class = num_class
        self.filters = filters

        if (self.num_class == 2):
            # one class: sigmoid
            self.final_block = nn.Sequential(nn.Conv2d(filters, num_class - 1, kernel_size = 1, bias = False),
                                             nn.Sigmoid())
        else:
            # mutiple class: softmax
            self.final_block = nn.Sequential(nn.Conv2d(filters, num_class, kernel_size = 1, bias = False),
                                             nn.Softmax2d())
    
    def forward(self, img_in):

        img_out = self.final_block(img_in)

        if (self.num_class == 2):
            # ouput directly
            return img_out
        else:
            # combine to one image
            img_out = torch.argmax(img_out, dim = 1)
            return img_out.unsqueeze(1)


"""
====================================================================================================
Unet
====================================================================================================
"""
class Unet(nn.Module):

    def __init__(self, num_channel, num_class):

        super().__init__()
        # critical parameter
        self.num_channel = num_channel
        self.num_class = num_class
        # number of filters
        self.filters = [16, 32, 64, 128, 256]

        # initialization
        self.init = Init(num_channel, self.filters[0])

        # downsampling
        self.down_1 = Down(self.filters[1])
        self.down_2 = Down(self.filters[2])
        self.down_3 = Down(self.filters[3])
        self.down_4 = Down(self.filters[4])

        # upsampling
        self.up_4 = Up(self.filters[3])
        self.up_3 = Up(self.filters[2])
        self.up_2 = Up(self.filters[1])
        self.up_1 = Up(self.filters[0])

        # ouput
        self.final = Final(self.filters[0], num_class)
    
    def forward(self, img_in):
        
        # initialization
        init = self.init(img_in)

        # downsampling
        down_1 = self.down_1(init)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)

        # upsampling
        up_4 = self.up_4(down_4, down_3)
        up_3 = self.up_3(up_4, down_2)
        up_2 = self.up_2(up_3, down_1)
        up_1 = self.up_1(up_2, init)

        # ouput
        img_out = self.final(up_1)

        return img_out


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('\n' + 'Training on device: ' + str(device) + '\n')
    
    model = Unet(num_channel = 7, num_class = 3).to(device = device)
    print(summary(model, input_size = (7, 512, 512), batch_size = 2))