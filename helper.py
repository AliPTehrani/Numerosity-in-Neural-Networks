from sys import platform
import numpy as np
import glob
import os

'''
This file is used for variables or functions that are used all over the project.

Implemented:
sub_list : all subjects from 04 - 23
test_cases : dictionary with all test cases , used to calculate average [value, frequency of testcase]
check_platform : returns the separator of the current platform
get_layers_ncondns : get information about the neural network
load_npz : loads npz file
'''


def get_sub_list():

    sub_list = ["sub04", "sub05", "sub06", "sub07", "sub08", "sub09", "sub10", "sub11", "sub12", "sub13", "sub14",
                "sub15", "sub16", "sub17", "sub18", "sub19", "sub20", "sub21", "sub22", "sub23"]
    return sub_list


def get_test_cases():

    test_cases = {
        'N1_S1_TFA1': [0, 0],
        'N1_S1_TFA2': [0, 0],
        'N1_S2_TFA1': [0, 0],
        'N1_S2_TFA2': [0, 0],
        'N1_S3_TFA1': [0, 0],
        'N1_S3_TFA2': [0, 0],
        'N2_S1_TFA1': [0, 0],
        'N2_S1_TFA2': [0, 0],
        'N2_S2_TFA1': [0, 0],
        'N2_S2_TFA2': [0, 0],
        'N2_S3_TFA1': [0, 0],
        'N2_S3_TFA2': [0, 0],
        'N3_S1_TFA1': [0, 0],
        'N3_S1_TFA2': [0, 0],
        'N3_S2_TFA1': [0, 0],
        'N3_S2_TFA2': [0, 0],
        'N3_S3_TFA1': [0, 0],
        'N3_S3_TFA2': [0, 0]
    }

    return test_cases


def check_platform():
    """depending on its platform we have different separators"""
    if platform == "linux" or platform == "linux2":
        sep = "/"
    elif platform == "darwin":
        sep = "\\"
    else:
        sep = "\\"

    return sep


def get_layers_ncondns(feat_dir):
    """Function to return facts about the npz-file
    Returns: Amount of Layers
    Returns: Names of Layers
    Returns: Amount of Images"""
    activations = glob.glob(feat_dir + "/*.npz")
    num_condns = len(activations)
    feat = np.load(activations[0], allow_pickle=True)

    num_layers = 0
    layer_list = []
    for key in feat:
        if "__" in key:  # key: __header__, __version__, __globals__
            continue
        else:
            num_layers+=1
            layer_list.append(key)  # collect all layer names
    #Liste: ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    return num_layers,layer_list,num_condns


def loadnpz(npzfile):
    """load in npz format"""
    return np.load(npzfile, allow_pickle=True)


def delete_files(result_path):
    npz_files = glob.glob(result_path + "/**/*.npz", recursive=True)
    for npz_file in npz_files:
        os.remove(npz_file)

#def sort_results():


