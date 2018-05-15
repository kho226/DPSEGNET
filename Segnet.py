import torch
import torch.nn as nn
import os
import numpy as np
import torchvision.utils as vutils
from torchvision import models
import math
from torch.autograd import Variable
# import torch.nn.functional as Funct



class down_block(nn.Module):
    #using the input channels I specify the channels for repeated use of this block
    def __init__(self, channels, num_of_convs = 2):
        super(down_block,self).__init__()

        self.num_of_convs = num_of_convs

        # Declare operations with learning features
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=1, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=1, bias=True)
        self.batchnorm2 = nn.BatchNorm2d(channels[1])
        if(num_of_convs == 3):
            self.conv3 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=0, dilation=1, bias=True)
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
        
        self.unpool = nn.MaxUnpool2d(kernel_size=(2,2) , stride=2)
        # self.upsample = nn.Upsample(mode="bilinear")

        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3), stride=1, padding=2, dilation=1, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(channels[1])
        
        if(num_of_convs== 2):
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=2, dilation=1, bias=True)
        elif(num_of_convs == 3):
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=2, dilation=1, bias=True)
            self.batchnorm2 = nn.BatchNorm2d(channels[1])
            self.conv3 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3), stride=1, padding=2, dilation=1, bias=True)
        
        self.batchnorm_for_last_conv = nn.BatchNorm2d(channels[1])

        self.relu = nn.ReLU(inplace=True)
        
        
        # Initialize Kernel weights for the decoder section with normally distributed weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                
    #forward function through the block
    def forward(self, x, indices, size):

        # print("\nInside upblock...")
        fwd_map = self.unpool(x, indices, output_size=size)
        # fwd_map = self.unpool(x, indices)
        
        # print(fwd_map.size())
        # fwd_map = self.upsample(fwd_map)
        
        fwd_map = self.conv1(fwd_map)
        # print(fwd_map.size())
        fwd_map = self.batchnorm1(fwd_map)
        self.relu(fwd_map)
        
        if(self.num_of_convs == 2):
            fwd_map = self.conv2(fwd_map)
            # print(fwd_map.size())
            fwd_map = self.batchnorm_for_last_conv(fwd_map)
            self.relu(fwd_map)

        elif(self.num_of_convs == 3):
            fwd_map = self.conv2(fwd_map)
            # print(fwd_map.size())
            fwd_map = self.batchnorm2(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv3(fwd_map)
            # print(fwd_map.size())
            fwd_map = self.batchnorm_for_last_conv(fwd_map)
            self.relu(fwd_map)

        #print("down block after convs: ", fwd_map.size())
        
        return fwd_map

class network(nn.Module):

    def __init__(self, num_classes, init_weights=False):
        super(network,self).__init__()
        
        self.layer1 = down_block((3,64), 2)              
        self.layer2 = down_block((64,128), 2)
        self.layer3 = down_block((128,256), 3)
        self.layer4 = down_block((256,512), 3)
        self.layer5 = down_block((512,1024), 3)
        
        self.layer6 = up_block((1024, 512), 3)
        self.layer7 = up_block((512,256), 3)
        self.layer8 = up_block((256,128), 3)
        self.layer9 = up_block((128,64), 2)
        self.layer10 = up_block((64,64), 2)
        
        if(init_weights):
            self.init_encoder_weights()
            self.init_decoder_weights(self.layer6, self.layer7, self.layer8, self.layer9, self.layer10)

        self.conv1x1 = nn.Conv2d(64, 35, kernel_size=(1,1), stride=1, padding=0, dilation=1, bias=False)

        self.softmax = nn.Softmax(dim=1)
    
    def init_encoder_weights(self):
        vgg = models.vgg19_bn(pretrained=True)
        layers = list(vgg.features.children())
        
        conv_ctr = 1; bn_ctr = 1;
        vgg_conv_layers_weights = []; vgg_bn_layers_weights  = []

        for i in range(len(layers)):
            if isinstance(layers[i], torch.nn.Conv2d):
                vgg_conv_layers_weights.append(layers[i].weight)
                conv_ctr += 1
            elif isinstance(layers[i], torch.nn.BatchNorm2d):
                vgg_bn_layers_weights.append(layers[i].weight)
                bn_ctr += 1   
        
        print (vgg_conv_layers_weights)
        print (vgg_bn_layers_weights)
        self.init_all_downblock_weights(vgg_conv_layers_weights, vgg_bn_layers_weights)
       


    def init_decoder_weights(self, *up_blocks):
        for block in up_blocks:
            for layer in block.children():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.zero_()
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.weight.data.fill_(1)
                    layer.bias.data.zero_()
        print("Initlalized Up block weights")
    
    def init_all_downblock_weights(self, vgg_conv_layers_weights, vgg_bn_layers_weights):
        #Down block 1
        self.layer1.conv1.weight.data      = vgg_conv_layers_weights[0].data
        self.layer1.batchnorm1.weight.data = vgg_bn_layers_weights[0].data
        self.layer1.conv2.weight.data      = vgg_conv_layers_weights[1].data
        self.layer1.batchnorm2.weight.data = vgg_bn_layers_weights[1].data
        #Down block 2
        self.layer2.conv1.weight.data      = vgg_conv_layers_weights[2].data
        self.layer2.batchnorm1.weight.data = vgg_bn_layers_weights[2].data
        self.layer2.conv2.weight.data      = vgg_conv_layers_weights[3].data
        self.layer2.batchnorm2.weight.data = vgg_bn_layers_weights[3].data
        #Down block 3    
        self.layer3.conv1.weight.data      = vgg_conv_layers_weights[4].data
        self.layer3.batchnorm1.weight.data = vgg_bn_layers_weights[4].data
        self.layer3.conv2.weight.data      = vgg_conv_layers_weights[5].data
        self.layer3.batchnorm2.weight.data = vgg_bn_layers_weights[5].data
        self.layer3.conv3.weight.data      = vgg_conv_layers_weights[6].data
        self.layer3.batchnorm3.weight.data = vgg_bn_layers_weights[6].data
        #Down block 4
        self.layer4.conv1.weight.data      = vgg_conv_layers_weights[7].data
        self.layer4.batchnorm1.weight.data = vgg_bn_layers_weights[7].data    
        self.layer4.conv2.weight.data      = vgg_conv_layers_weights[8].data
        self.layer4.batchnorm2.weight.data = vgg_bn_layers_weights[8].data
        self.layer4.conv3.weight.data      = vgg_conv_layers_weights[9].data
        self.layer4.batchnorm3.weight.data = vgg_bn_layers_weights[9].data
        print("Initlalized Down block weights")
        
    def save_checkpoint(self, relative_path, val):
        '''
            save_path: type: string ~> provide a path to the directory to save the parameters
            will append the current working directory to relative_path

            torch.save(...,os.getcwd() + relative_path)
        '''
        c_wd = os.getcwd()
        abs_path = c_wd + relative_path
        
        if not os.path.exists(abs_path):    # if path doesn't exist, create it
            os.makedirs(abs_path)   

        with open(abs_path + '/network_state_checkpoint{}.pth.tar'.format(val), 'wb') as f: 
            torch.save(self.state_dict(), f)
    
    def forward(self,x):

        # print("\nLayer1 Output size - ", end="")
        out1, indices1, size1= self.layer1(x)
        # print(out1.size())
        # print("\nLayer2 Output size - ", end="")
        out2, indices2, size2 = self.layer2(out1)
        # print(out2.size())
        # print("\nLayer3 Output size - ", end="")
        out3, indices3, size3= self.layer3(out2)
        # print(out3.size())
        # print("\nLayer4 Output size - ", end="")
        out4, indices4,size4 = self.layer4(out3)
        # print(out4.size())
        # print("\nLayer5 Output size - ", end="")
        # out5, indices5, size5 = self.layer5(out4)
        # print(out5.size())

        # print("\nLayer6 Output size - ", end="")
        # out6 = self.layer6(out5, indices5, size5['b4max'])
        # print(out6.size())
        
        # print("\nLayer7 Output size - ", end="")
        out7 = self.layer7(out4, indices4, size4['b4max'])
        # print(out7.size())
        
        # print("\nLayer8 Output size - ", end="")
        out8 = self.layer8(out7, indices3, size3['b4max'])
        # print(out8.size())
        
        # print("\nLayer9 Output size - ", end="")
        out9 = self.layer9(out8, indices2, size2['b4max'])
        # print(out9.size())
        
        # print("\nLayer10 Output size - ", end="")
        out10 = self.layer10(out9, indices1, size1['b4max'])
        # print(out10.size())
        
        # print("\nconv1x1 Output size - ", end="")
        out_conv1x1 = self.conv1x1(out10)
        # print(out_conv1x1.size())
        
        
        # print("\nSoftmax Output size - ", end="")
        #res = Funct.softmax(out10)
        softmax_out = self.softmax(out_conv1x1)
        # print(softmax_out.size())
        
        # print("\nID-fied Output size - ", end="")
        _ , ind = torch.max(softmax_out,1)
        out = ind.data.numpy()
        res = np.reshape(out, (out.shape[0], 1, out.shape[1], out.shape[2]))
        res = Variable(torch.from_numpy(res + 1).float()  , requires_grad = True)
        print(res.size())
        
        return res

