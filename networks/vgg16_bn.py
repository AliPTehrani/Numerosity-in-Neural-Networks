from __future__ import division
from torchvision import models
import torch
import torch.nn as nn
vgg_feat_list = ['Conv2d_1', 'BatchNorm2d_1', 'ReLU_1', 'Conv2d_2', 'BatchNorm2d_2', 'ReLU_2', 'MaxPool2d_1',
                 'Conv2d_3', 'BatchNorm2d_3', 'ReLU_3', 'Conv2d_4', 'BatchNorm2d_4', 'ReLU_4', 'MaxPool2d_2',
                 'Conv2d_5', 'BatchNorm2d_5', 'ReLU_5', 'Conv2d_6', 'BatchNorm2d_6', 'ReLU_6', 'Conv2d_7',
                 'BatchNorm2d_7', 'ReLU_7', 'MaxPool2d_3', 'Conv2d_8', 'BatchNorm2d_8', 'ReLU_8', 'Conv2d_9',
                 'BatchNorm2d_9', 'ReLU_9', 'Conv2d_10', 'BatchNorm2d_10', 'ReLU_10', 'MaxPool2d_4', 'Conv2d_11',
                 'BatchNorm2d_11', 'ReLU_11', 'Conv2d_12', 'BatchNorm2d_12', 'ReLU_12', 'Conv2d_13', 'BatchNorm2d_13',
                 'ReLU_13', 'MaxPool2d_5']

vgg_classifier_list = ['fc6','ReLU6','Dropout6','fc7','ReLU7','Dropout7','fc8']

class VGG16_bnNet(nn.Module):
    def __init__(self,is_pretrained):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGG16_bnNet, self).__init__()
        self.select_feats = ['MaxPool2d_1', 'MaxPool2d_2', 'MaxPool2d_3', 'MaxPool2d_4', 'MaxPool2d_5']
        # self.select_feats = vgg_feat_list
        self.select_classifier = ['fc6' , 'fc7', 'fc8']
        # self.select_classifier = vgg_classifier_list

        self.feat_list = self.select_feats + self.select_classifier
        self.vgg_feats = models.vgg16_bn(pretrained=is_pretrained).features
        self.vgg_classifier = models.vgg16_bn(pretrained=is_pretrained).classifier
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        # 1. Alle Namen aufschreibenn die es gibt
        # 2. Namen deklarieren in self. die wir wollen
        # 3. hinzufügen wenn wir sie wollen.
        # 4. das gleiche auch für classifier!

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
