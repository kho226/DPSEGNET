import os
import torch
from torch.utils.data import Dataset 
from torchvision import transforms, utils
from PIL import Image
import random
from torchvision import datasets,models,transforms
import numpy as np

trans = transforms.ToTensor()

class data_loader_seg(Dataset):

    def __init__(self, root_dir, trans=None, c2id=None):
        self.root_dir = root_dir
        self.files = [fn for fn in os.listdir(root_dir + 'original/') if fn.endswith('.png')]
        self.trans = trans 
        self.c2id = c2id
        # to  - do ~> implement multi-threaded processing
        #self.pool = mp.Pool(processes=num_processes)

    def __len__(self):
        return len(self.files)
    
    def IDfy(self, pil_img) :
        kitti_image_label_np      = np.array(pil_img)
        kitti_image_label_IDfied = np.zeros((kitti_image_label_np.shape[0],kitti_image_label_np.shape[1]))

        for i in range(kitti_image_label_np.shape[0]):
            for j in range(kitti_image_label_np.shape[1]):
                kitti_image_label_IDfied[i][j] = self.c2id[ tuple(kitti_image_label_np[i][j][:]) ].id

        s = kitti_image_label_IDfied.shape
        kitti_image_label_IDfied = kitti_image_label_IDfied.reshape((s[0], s[1], 1))


        return kitti_image_label_IDfied

    def __getitem__(self,idx):
        image = Image.open(self.root_dir + 'original/' + self.files[idx])
        image_seg = Image.open(self.root_dir + 'segmented/' + self.files[idx])
        image_seg_idfy = self.IDfy(image_seg)
        # THIS PART PRE-PROCESSES THE LABELED IMAGE FOR A BINARY SEGMENTATION (Remove when segmenting more than 2)
        # image_seg = image_seg.convert('L')
        #image_seg = image_seg.resize((388,388))
        # image_seg = np.array(image_seg)
        # image_seg[image_seg>=100] = 1
        # image_seg[image_seg<100] = 0
        # image_seg = Image.fromarray(image_seg.astype('uint8'))
        
        # image_seg_idfy = Image.fromarray(image_seg_idfy)


        if self.trans:
            image = self.trans(image)
            image_seg_idfy = self.trans(image_seg_idfy)
        
        image = np.array(image)

        #print("This is the image shape: {}".format(image.shape))
        #print("this is the segmented image shape: {}".format(image_seg_idfy.shape))

        image = image.reshape((3,image.shape[0],image.shape[1]))
        image_seg_idfy = image_seg_idfy.reshape((1,image_seg_idfy.shape[0],image_seg_idfy.shape[1]))

        image = image[:, :350 , :1230]
        image_seg_idfy = image_seg_idfy[:, : 350 , :1230]

        image = torch.from_numpy(image).float()
        image_seg_idfy = torch.from_numpy(image_seg_idfy).float()


        #print("image shape ~> {}".format(image.shape))
        #print("image_seg_idfy shape ~> {}".format(image_seg_idfy.shape))


        return {'image': image, 'image_seg': image_seg_idfy}
