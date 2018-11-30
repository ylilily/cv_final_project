# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

########################################################################
# Define the Convolution Neural Networks for colorization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class Net(nn.Module):
    def __init__(self):
        """
        Composing the layers for a sequential network
        Max-pooling is not used as such a pooling layer will distort the image
        Instead, we use a stride of 2 to improve info density (see: https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/)
        Intiutively, the conv filter "moves" through the image in 2-pixel steps instead of 1
        
        The input images are of size 224 * 224.
        The number of classes for the classification network is 365 (dataset: http://places2.csail.mit.edu/)
        """
        super(Net, self).__init__()
        
        h = 224
        num_classes = 365
        # Low-level features network
        layers = []
        layers.append(nn.Conv2d(1, int(h/4), 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(int(h/4)))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(int(h/4), int(h/2), 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(int(h/2)))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(int(h/2), int(h/2), 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(int(h/2)))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(int(h/2), int(h), 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(int(h)))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(int(h), int(h), 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(int(h)))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(int(h), int(h*2), 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(int(h*2)))
        layers.append(nn.ReLU())
        self.low_features = nn.Sequential(*layers)

        # Global features network
        global_conv_layers = []
        global_conv_layers.append(nn.Conv2d(int(h*2), int(h*2), 3, stride=2, padding=1))
        global_conv_layers.append(nn.BatchNorm2d(int(h*2)))
        global_conv_layers.append(nn.ReLU())
        global_conv_layers.append(nn.Conv2d(int(h*2), int(h*2), 3, stride=1, padding=1))
        global_conv_layers.append(nn.BatchNorm2d(int(h*2)))
        global_conv_layers.append(nn.ReLU())
        global_conv_layers.append(nn.Conv2d(int(h*2), int(h*2), 3, stride=2, padding=1))
        global_conv_layers.append(nn.BatchNorm2d(int(h*2)))
        global_conv_layers.append(nn.ReLU())
        global_conv_layers.append(nn.Conv2d(int(h*2), int(h*2), 3, stride=1, padding=1))
        global_conv_layers.append(nn.BatchNorm2d(int(h*2)))
        global_conv_layers.append(nn.ReLU())
        self.global_conv_features = nn.Sequential(*global_conv_layers)
        
        global_fc_layers = []
        global_fc_layers.append(nn.Linear(int(h/32)*int(h/32)*int(h*2), int(h*4)))
        global_fc_layers.append(nn.ReLU())
        global_fc_layers.append(nn.Linear(int(h*4), int(h*2)))
        global_fc_layers.append(nn.ReLU())
        self.global_fc_features = nn.Sequential(*global_fc_layers)
        
        final_fc_layers = []
        final_fc_layers.append(nn.Linear(int(h*2), int(h)))
        final_fc_layers.append(nn.ReLU())
        self.final_fc_features = nn.Sequential(*final_fc_layers)
        
        # Classification network
        class_layers = []
        class_layers.append(nn.Linear(int(h*2), int(h)))
        class_layers.append(nn.ReLU())
        class_layers.append(nn.Linear(int(h), num_classes))
        self.class_features = nn.Sequential(*class_layers)
        
        # Mid-Level features network
        mid_features_layers = []
        mid_features_layers.append(nn.Conv2d(int(h*2), int(h*2), 3, stride=1, padding=1))
        mid_features_layers.append(nn.BatchNorm2d(int(h*2)))
        mid_features_layers.append(nn.ReLU())
        mid_features_layers.append(nn.Conv2d(int(h*2), int(h), 3, stride=1, padding=1))
        mid_features_layers.append(nn.BatchNorm2d(int(h)))
        mid_features_layers.append(nn.ReLU())
        self.mid_features = nn.Sequential(*mid_features_layers)

        # Colorization network
        fusion_layers = []
        fusion_layers.append(nn.Linear(int(h*2), int(h)))
        fusion_layers.append(nn.ReLU())
        self.fusion_features = nn.Sequential(*fusion_layers)
        
        color_layers1 = []
        color_layers1.append(nn.Conv2d(int(h), int(h/2), 3, stride=1, padding=1))
        color_layers1.append(nn.BatchNorm2d(int(h/2)))
        color_layers1.append(nn.ReLU())
        self.color1 = nn.Sequential(*color_layers1)

        color_layers2 = []
        color_layers2.append(nn.Conv2d(int(h/2), int(h/4), 3, stride=1, padding=1))
        color_layers2.append(nn.BatchNorm2d(int(h/4)))
        color_layers2.append(nn.ReLU())
        color_layers2.append(nn.Conv2d(int(h/4), int(h/4), 3, stride=1, padding=1))
        color_layers2.append(nn.BatchNorm2d(int(h/4)))
        color_layers2.append(nn.ReLU())
        self.color2 = nn.Sequential(*color_layers2)

        color_layers3 = []
        color_layers3.append(nn.Conv2d(int(h/4), int(h/8), 3, stride=1, padding=1))
        color_layers2.append(nn.BatchNorm2d(int(h/8)))
        color_layers3.append(nn.ReLU())
        color_layers3.append(nn.Conv2d(int(h/8), 2, 3, stride=1, padding=1))
        color_layers3.append(nn.Sigmoid())
        self.color3 = nn.Sequential(*color_layers3)
        
    def forward(self, img):
        img = img.permute(0, 3, 1, 2)
        img = self.low_features(img)
        global_img = self.global_conv_features(img)
        global_img = global_img.view(global_img.size(0), -1)
        global_img = self.global_fc_features(global_img)
        label_preds = self.class_features(global_img)
        global_img = self.final_fc_features(global_img)
        img = self.mid_features(img)
        fusion = torch.cat((img.permute(2, 3, 0, 1), global_img.repeat(img.size(2), img.size(3), 1, 1)), dim=-1)
        fusion = self.fusion_features(fusion)
        fusion = fusion.permute(2, 3, 0, 1)
        combined = self.color1(fusion)
        combined = F.interpolate(combined, scale_factor=2, mode='nearest')
        combined = self.color2(combined)
        combined = F.interpolate(combined, scale_factor=2, mode='nearest')
        combined = self.color3(combined)
        color_preds = F.interpolate(combined, scale_factor=2, mode='nearest')
        color_preds = color_preds.permute(0, 2, 3, 1)
        return color_preds, label_preds
    