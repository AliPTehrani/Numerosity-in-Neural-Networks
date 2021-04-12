import torch
import numpy as np
import os
import torchvision.models as models
from torchvision import transforms as trn
from PIL import Image
import torch.nn as nn
from networks.alexnet import *
from torch.autograd import Variable as V
import glob
import helper

'''
This file is used to generate the features of the Neural Network from all stimuli
Functions:
preprocess_image : preprocessing an image for the neural network
get_images : get the path for all images
run_alexnet : generate the features for all images

'''


def preprocess_image(image):
    """function to resize images"""
    centre_crop = trn.Compose([
        trn.Resize((224, 224)),  # resize to 224 x 224 pixels
        trn.ToTensor(),  # transform to tensor
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize according to ImageNet
    ])
    img = Image.open(image)
    input_img = V(centre_crop(img).unsqueeze(0))  # apply resizing
    if torch.cuda.is_available():
        input_img = input_img.cuda()
    return input_img


def get_images():
    """Return list of all stimuli"""
    path = "stimuli_jpg"
    images = glob.glob(path + "/**/*.jpg", recursive=True) # list all files
    images.sort()
    print(images)
    return images


def run_torchvision_model(model, result_path):
    """
    This function is used to run the Neural Network and save the features as .npz files
    Runs all torchvision architectures
    """
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    #directory of results
    images = get_images()

    for image in images:

        sep = helper.check_platform()  # depending on platform we have different seperators
        filename = image.split(sep)[-1].split(".")[0]
        filetype = image.split(sep)[-1].split(".")[1]
        filefolder = image.split(sep)[0] + image.split(sep)[1]
        net_save_dir = os.path.join(result_path, image.split(sep)[1])

        if filetype == "jpg":
            valid_type = True
            input_img = preprocess_image(image)  # preprocess the image
            x = model.forward(input_img)  # forward function of model

        if valid_type == True:
            """Save everything"""
            save_path = os.path.join(net_save_dir, filename + ".npz")
            feats = {}
            for i, feat in enumerate(x):
                if filetype == "jpg":
                    feats[model.feat_list[i]] = feat.data.cpu().numpy()  # bring back to cpu

            if not os.path.exists(net_save_dir):
                os.makedirs(net_save_dir)
            np.savez(save_path, **feats)  # save features

            """Delete Cache"""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del x
            del feats


