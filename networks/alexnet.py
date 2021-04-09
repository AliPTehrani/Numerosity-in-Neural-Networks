from __future__ import division
from torchvision import models
import torch.nn as nn

alex_feat_list = ['conv1','ReLU1','maxpool1',
                  'conv2','ReLU2','maxpool2',
                  'conv3','ReLU3',
                  'conv4','ReLU4',
                  'conv5','ReLU5','maxpool5',
                  ]
alex_classifier_list = ['Dropout6','fc6','ReLU6','Dropout7','fc7','ReLU7','fc8']


class AlexNet(nn.Module):
    def __init__(self,is_pretrained):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(AlexNet, self).__init__()

        # Eigene Attribute wo wir sagebn
        self.select_feats = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        self.select_classifier = ['fc6' , 'fc7', 'fc8']

        self.feat_list = self.select_feats + self.select_classifier

        # VON MODLES ALEXNET BEKOMMEN

        self.alex_feats = models.alexnet(pretrained=is_pretrained).features
        self.alex_classifier = models.alexnet(pretrained=is_pretrained).classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        """Extract multiple feature maps."""
        # 1. Put the given picture into the layer function and get the activation after it ran through all the feature layers
        # => we get an image and run it through every layer , and the activation of all layers are collected in features
        # => At the beginning its an image, which then gets converted to layer actiavtions. thats why x is being overwritten
        # 2. at the end is x the activation of the final layer, which is averarges and sized
        # 3. We put the final layer through all the classifiers we have and add these "layers" also to the features list
        # 4. At the end we have activation from one image of feature layers + classifier layers


        # Alexnet speichert die layer namen als zahlen 1-12 backbone und 1-7 für fc
        # Wir haben für uns den layern in alex_feat_list und alex_classifier_list namen gegeben
        # in feat_list speichern wir die layer von denen wir die features haben wollen
        # Wir lesen dann in einer for schleife für JEDES layer die features aus. Sollte name(die zahl)
        #  in unserem feat_list drin sein, speicher wirs!
        # 1. Alle Namen aufschreibenn die es gibt
        # 2. Namen deklarieren in self. die wir wollen
        # 3. hinzufügen wenn wir sie wollen.
        # 4. das gleiche auch für classifier!

        features = []
        for name, layer in self.alex_feats._modules.items():
            x = layer(x)
            if alex_feat_list[int(name)] in self.feat_list:
                features.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        for name, layer in self.alex_classifier._modules.items():
            x = layer(x)
            if alex_classifier_list[int(name)] in self.feat_list:
                features.append(x)
        return features
