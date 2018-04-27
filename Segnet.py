

import torch
import torch.nn as nn
import os
import numpy as np
import torchvision.utils as vutils
import math
# import torch.nn.functional as Funct



class down_block(nn.Module):
    #using the input channels I specify the channels for repeated use of this block
    def __init__(self, channels, num_of_convs = 2):
        super(down_block,self).__init__()

        self.num_of_convs = num_of_convs

        # Declare operations with learning features
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=0, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=0, bias=True)
        self.batchnorm2 = nn.BatchNorm2d(channels[1])
        if(num_of_convs == 3):
            self.conv3 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=0, bias=True)
            self.batchnorm3 = nn.BatchNorm2d(channels[1])

        # Declare operations without learning features
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2, return_indices = True)
        
        # Initialize Kernel weights for the encoder section with vgg weights
        # this will be done on another python file after an instance of the model network is created
                
    #forward function through the block
    def forward(self, x):
        input_size = x.size()
        
        fwd_map = self.conv1(x)
        fwd_map = self.batchnorm1(fwd_map)
        self.relu(fwd_map)

        fwd_map = self.conv2(fwd_map)
        fwd_map = self.batchnorm2(fwd_map)
        self.relu(fwd_map)

        if(self.num_of_convs == 3):
            fwd_map = self.conv3(fwd_map)
            fwd_map = self.batchnorm3(fwd_map)
            self.relu(fwd_map)

        #Saving the tensor and for unpooling tensor size & indeces to map it to the layers deeper in the model
        output_size = fwd_map.size()
        fwd_map, indices = self.maxpool(fwd_map)
        
        size = {"input_size": input_size, "b4max": output_size}
        return (fwd_map, indices, size)



class up_block(nn.Module):

    def __init__(self,channels,num_of_convs = 2):
        super(up_block,self).__init__()
        
        self.num_of_convs = num_of_convs
        
        self.unpooled = nn.MaxUnpool2d(kernel_size=(2,2) , stride=2)
        self.upsample = nn.Upsample(mode="bilinear")

        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3), stride=1, padding=1, dilation=1, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(channels[1])
        
        if(num_of_convs== 2):
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=0, bias=True)
        elif(num_of_convs == 3):
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=0, bias=True)
            self.batchnorm2 = nn.BatchNorm2d(channels[1])
            self.conv3 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=0, bias=True)
        
        self.batchnorm_for_last_conv = nn.BatchNorm2d(channels[1])

        self.relu = nn.ReLU(inplace=True)
        
        
        # Initialize Kernel weights for the decoder section with normally distributed weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                
    #forward function through the block
    def forward(self, x, indices, size):

        #print("Before upsampling: ", x.size())
        fwd_map = self.unpooled(x, indices, output_size=size)
        fwd_map = self.upsample(fwd_map)
        
        fwd_map = self.conv1(fwd_map)
        fwd_map = self.batchnorm1(fwd_map)
        self.relu(fwd_map)
        
        if(self.num_of_convs == 2):
            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm_for_last_conv(fwd_map)
            self.relu(fwd_map)

        elif(self.num_of_convs == 3):
            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm2(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv3(fwd_map)
            fwd_map = self.batchnorm_for_last_conv(fwd_map)
            self.relu(fwd_map)

        #print("down block after convs: ", fwd_map.size())
        
        return fwd_map

class network(nn.Module):

    def __init__(self, num_classes):
        super(network,self).__init__()
        self.layer1 = down_block((3,64), 2)              
        self.layer2 = down_block((64,128), 2)
        self.layer3 = down_block((128,256), 3)
        self.layer4 = down_block((256,512), 3)
        self.layer5 = down_block((512,1024), 3)
        
        #self.layer6 = up_block((inp,curr,next), 3)
        self.layer6 = up_block((512,1024), 3)
        self.layer7 = up_block((512,256), 3)
        self.layer8 = up_block((256,128), 3)
        self.layer9 = up_block((128,64), 2)
        self.layer10 = up_block((64,64), 2)
        
        self.conv1x1 = nn.Conv2d(64, 35, kernel_size=(1,1), stride=1, padding=0, dilation=0, bias=False)

        self.softmax = nn.Softmax()

    def forward(self,x):

        print("\nLayer1...")
        out1, indices1, size1= self.layer1(x)
        print(out1.size())
        print("\nLayer2...")
        out2, indices2, size2 = self.layer2(out1)
        print(out2.size())
        print("\nLayer3...")
        out3, indices3, size3= self.layer3(out2)
        print(out3.size())
        print("\nLayer4...")
        out4, indices4,size4 = self.layer4(out3)
        print(out4.size())
        print("\nLayer5...")
        out5, indices5, size5 = self.layer5(out4)
        print(out5.size())

        print("\nLayer6...")
        out6 = self.layer6(out5, indices5, size5['b4max'])
        print(out6.size())
        print("\nLayer7...")
        out7 = self.layer7(out6, indices4, size4['b4max'])
        print(out7.size())
        print("\nLayer8...")
        out8 = self.layer8(out7, indices3, size3['b4max'])
        print(out8.size())
        print("\nLayer9...")
        out9 = self.layer9(out8, indices2, size2['b4max'])
        print(out9.size())
        print("\nLayer10...")
        out10 = self.layer10(out9, indices1, size1['b4max'])
        print(out10.size())
        
        print("\nconv1x1")
        out_conv1x1 = self.conv1x1(out10)
        print("size of out_conv1x1:", out_conv1x1.size())
        
        
        print("\nSoftmax Layer...")
        #res = Funct.softmax(out10)
        res = self.softmax(out_conv1x1, dim=2)
        print(res.size())
        return res

