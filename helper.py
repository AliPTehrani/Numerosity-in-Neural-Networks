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


def get_lower_triangular(rdm):
    """Get lower triangle of a RDM"""
    num_conditions = rdm.shape[0]
    return rdm[np.tril_indices(num_conditions, 1)]


def remove_diagonal(rdm_lower_triangle):
    """Removes the diagonal of zeros and one value before the zero from the lower triangle of the rdm"""

    index = 1
    counter = 2
    rdm_lower_triangle = np.delete(rdm_lower_triangle, 0)
    loops = 1

    while loops != 18:
        rdm_lower_triangle = np.delete(rdm_lower_triangle, index)
        rdm_lower_triangle = np.delete(rdm_lower_triangle, index)
        index += counter
        counter += 1
        loops += 1

    return rdm_lower_triangle

def get_brain_regions_npz(option):
    """
    Loads the RDMs for the brain_region given one of the three options :
    :param option:  Int representing  Option = 0 taskBoth / 1 TaskNum / 2 TaskSize
    :return: brain_region_dict  : Keys : Brain regions , Values : Loaded npz files of brain region RDM
    """

    # Get option as filter to search for correct npz files
    sep = check_platform()
    option = ["taskBoth", "taskNum", "taskSize"][option]
    cwd = os.getcwd()
    search_path = os.getcwd() + sep + "RSA_matrices"
    all_rsa_files = glob.glob(search_path + "/*" + ".npz")
    brain_region_dict = {}
    filtered_files = []
    # Filter them on the correct option
    for file in all_rsa_files:
        if (option in file):
            filtered_files.append(file)

    for filtered_file in filtered_files:
        average_rdm = []
        loadednpz = loadnpz(filtered_file)
        filename = filtered_file.split(check_platform())[-1]
        brain_region = filename.split("_")
        brain_region = brain_region[-1].split(".")[0]
        brain_region_dict[brain_region] = loadednpz

    return brain_region_dict




