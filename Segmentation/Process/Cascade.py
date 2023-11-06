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
        self.res_1 = Res(filters)
        self.res_2 = Res(filters)

    def forward(self, img_in):

        img_out = self.down(img_in)
        img_out = self.res_1(img_out)
        img_out = self.res_2(img_out)

        return img_out


"""
====================================================================================================
Upsampling Block: Transpose Convolution
====================================================================================================
"""
class Up_Trans(nn.Module):

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
Upsampling Block: Bilinear Interpolation
====================================================================================================
"""
class Up_Bilinear(nn.Module):

    def __init__(self, filters):

        super().__init__()
        # critical parameter
        self.filters = filters

        # convolution -> upsampling -> res block
        self.conv = nn.Conv2d(filters * 2, filters, kernel_size = 1, padding = 0, bias = False)
        self.up = F.interpolate
        self.res = Res(filters)
    
    def forward(self, img_in_1, img_in_2):

        img_out = self.conv(img_in_1)
        img_out = self.up(img_out, scale_factor = 2, mode = 'bilinear')
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
Cascade
====================================================================================================
"""
class Cascade(nn.Module):

    def __init__(self, num_channel, num_class):

        super().__init__()
        # critical parameter
        self.num_channel = num_channel
        self.num_class = num_class
        # number of filters for stage 1
        self.filters_1 = [16, 32, 64, 128, 256]
        # number of filters for stage 2
        self.filters_2 = [32, 64, 128, 256, 512]

        """
        ============================================================================================
        Stage 1
        ============================================================================================
        """
        # initialization
        self.init1 = Init(num_channel, self.filters_1[0])

        # downsampling
        self.down1_1 = Down(self.filters_1[1])
        self.down1_2 = Down(self.filters_1[2])
        self.down1_3 = Down(self.filters_1[3])
        self.down1_4 = Down(self.filters_1[4])

        # upsampling: trans convolution
        self.up1_4 = Up_Trans(self.filters_1[3])
        self.up1_3 = Up_Trans(self.filters_1[2])
        self.up1_2 = Up_Trans(self.filters_1[1])
        self.up1_1 = Up_Trans(self.filters_1[0])

        # ouput
        self.final1 = Final(self.filters_1[0], num_class)
                
        """
        ============================================================================================
        Stage 2
        ============================================================================================
        """
        # initialization
        self.init2 = Init(num_channel, self.filters_2[0])

        # downsampling
        self.down2_1 = Down(self.filters_2[1])
        self.down2_2 = Down(self.filters_2[2])
        self.down2_3 = Down(self.filters_2[3])
        self.down2_4 = Down(self.filters_2[4])

        # upsampling: trans convolution
        self.up2_1_4 = Up_Trans(self.filters_2[3])
        self.up2_1_3 = Up_Trans(self.filters_2[2])
        self.up2_1_2 = Up_Trans(self.filters_2[1])
        self.up2_1_1 = Up_Trans(self.filters_2[0])

        # output: trans convolution
        self.final2_1 = Final(self.filters_2[0], num_class)

        # upsampling: bilinear interpolation
        self.up2_2_4 = Up_Bilinear(self.filters_2[3])
        self.up2_2_3 = Up_Bilinear(self.filters_2[2])
        self.up2_2_2 = Up_Bilinear(self.filters_2[1])
        self.up2_2_1 = Up_Bilinear(self.filters_2[0])

        # output: bilinear interpolation
        self.final2_2 = Final(self.filters_2[0], num_class)
    
    def forward(self, img_in_1):

        """
        ============================================================================================
        Stage 1
        ============================================================================================
        """
        # initialization
        init1 = self.init1(img_in_1)

        # downsampling
        down1_1 = self.down1_1(init1)
        down1_2 = self.down1_2(down1_1)
        down1_3 = self.down1_3(down1_2)
        down1_4 = self.down1_4(down1_3)

        # upsampling: trans convolution
        up1_4 = self.up1_4(down1_4, down1_3)
        up1_3 = self.up1_3(up1_4, down1_2)
        up1_2 = self.up1_2(up1_3, down1_1)
        up1_1 = self.up1_1(up1_2, init1)

        # ouput
        img_out1 = self.final1(up1_1)

        """
        ============================================================================================
        Stage 2
        ============================================================================================
        """
        # input: jump connection
        img_in_2 = img_in_1 + img_out1

        # initialization
        init_2 = self.init2(img_in_2)

        # downsampling
        down2_1 = self.down2_1(init_2)
        down2_2 = self.down2_2(down2_1)
        down2_3 = self.down2_3(down2_2)
        down2_4 = self.down2_4(down2_3)

        # upsampling: trans convolution
        up2_1_4 = self.up2_1_4(down2_4, down2_3)
        up2_1_3 = self.up2_1_3(up2_1_4, down2_2)
        up2_1_2 = self.up2_1_2(up2_1_3, down2_1)
        up2_1_1 = self.up2_1_1(up2_1_2, init_2)

        # output: trans convolution
        img_out2_1 = self.final2_1(up2_1_1)

        # upsampling: bilinear interpolation
        up2_2_4 = self.up2_2_4(down2_4, down2_3)
        up2_2_3 = self.up2_2_3(up2_2_4, down2_2)
        up2_2_2 = self.up2_2_2(up2_2_3, down2_1)
        up2_2_1 = self.up2_2_1(up2_2_2, init_2)

        # output: bilinear interpolation
        img_out2_2 = self.final2_2(up2_2_1)

        return (img_out1, img_out2_1, img_out2_2)


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('\n' + 'Training on device: ' + str(device) + '\n')
    
    model = Cascade(num_channel = 7, num_class = 3).to(device = device)
    print(summary(model, input_size = (7, 512, 512), batch_size = 2))