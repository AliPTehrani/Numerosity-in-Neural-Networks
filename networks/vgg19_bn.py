from __future__ import division
from torchvision import models
import torch
import torch.nn as nn

vgg_feat_list = ['Conv2d_1', 'BatchNorm2d_1', 'ReLU_1', 'Conv2d_2', 'BatchNorm2d_2', 'ReLU_2', 'MaxPool2d_1', 'Conv2d_3',
                 'BatchNorm2d_3', 'ReLU_3', 'Conv2d_4', 'BatchNorm2d_4', 'ReLU_4', 'MaxPool2d_2', 'Conv2d_5',
                 'BatchNorm2d_5', 'ReLU_5', 'Conv2d_6', 'BatchNorm2d_6', 'ReLU_6', 'Conv2d_7', 'BatchNorm2d_7',
                 'ReLU_7', 'Conv2d_8', 'BatchNorm2d_8', 'ReLU_8', 'MaxPool2d_3', 'Conv2d_9', 'BatchNorm2d_9',
                 'ReLU_9', 'Conv2d_10', 'BatchNorm2d_10', 'ReLU_10', 'Conv2d_11', 'BatchNorm2d_11', 'ReLU_11',
                 'Conv2d_12', 'BatchNorm2d_12', 'ReLU_12', 'MaxPool2d_4', 'Conv2d_13', 'BatchNorm2d_13', 'ReLU_13',
                 'Conv2d_14', 'BatchNorm2d_14', 'ReLU_14', 'Conv2d_15', 'BatchNorm2d_15', 'ReLU_15', 'Conv2d_16',
                 'BatchNorm2d_16', 'ReLU_16', 'MaxPool2d_5']

#ÄNDERN

vgg_classifier_list = ['fc6','ReLU6','Dropout6','fc7','ReLU7','Dropout7','fc8']

class VGG19_bnNet(nn.Module):
    def __init__(self,is_pretrained):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGG19_bnNet, self).__init__()
        self.select_feats = ['MaxPool2d_1', 'MaxPool2d_2', 'MaxPool2d_3', 'MaxPool2d_4', 'MaxPool2d_5']
        #self.select_feats = vgg_feat_list
        self.select_classifier = ['fc6' , 'fc7', 'fc8']
        #self.select_classifier = vgg_classifier_list

        self.feat_list = self.select_feats + self.select_classifier
        self.vgg_feats = models.vgg19_bn(pretrained=is_pretrained).features
        self.vgg_classifier = models.vgg19_bn(pretrained=is_pretrained).classifier
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        """Extract multiple feature maps."""
        # 1. Put the given picture into the layer function and get the activation after it ran through all the feature layers
        # => we get an image and run it through every layer , and the activation of all layers are collected in features
        # => At the beginning its an image, which then gets converted to layer actiavtions. thats why x is being overwritten
        # 2. at the end is x the activation of the final layer, which is averarges and sized
        # 3. We put the final layer through all the classifiers we have and add these "layers" also to the features list
        # 4. At the end we have activation from one image of feature layers + classifier layers
        features = []
        for name, layer in self.vgg_feats._modules.items():
            x = layer(x)
            if vgg_feat_list[int(name)] in self.select_feats:
                features.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        for name, layer in self.vgg_classifier._modules.items():
            x = layer(x)
            if vgg_classifier_list[int(name)] in self.select_classifier:
                features.append(x)
        return features
