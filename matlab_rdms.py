import scipy.io
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

import RDM_Evaluation
import helper


'''
This file is used to convert and save the matlab files from RSA_Matrices directory as .npz files
'''


def visualize_rdm(rdm, npz_save_dir, filename):

    number_of_matrix = 1
    for matrix in rdm:

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap="inferno")
        plt.title("Matrix of " + filename + "Number: " + str(number_of_matrix))
        plt.colorbar()
        save_path = os.path.join(npz_save_dir,filename + "Matrix " + str(number_of_matrix))
        plt.savefig(save_path)
        number_of_matrix += 1


def read_in_rdms_matlab():
    """Reads in all matlab files from RSA_Matrices and saves them as .npz files"""

    path = "RSA_matrices"  # Path of the RSA matrices that evelyn has sent
    mat_files = glob.glob(path + "/**/*.mat", recursive=True)

    for mat_file in mat_files:
        sep = helper.check_platform()  # depending on platform we have different seperators
        filename = mat_file.split(sep)[-1].split(".")[0]
        file_type = mat_file.split(sep)[-1].split(".")[1]
        file_folder = mat_file.split(sep)[0]
        npz_save_dir = file_folder + "\\" + filename
        if file_type == "mat":
            rdm_dict = scipy.io.loadmat(mat_file)
            for key in rdm_dict:
                if isinstance(rdm_dict[key], np.ndarray):
                    if not os.path.exists(npz_save_dir):
                        os.makedirs(npz_save_dir)
                    save_path = os.path.join(npz_save_dir, filename + ".npz")
                    rdm = rdm_dict[key]
                    rdm2 = np.transpose(rdm, (2, 0, 1))    # Matrix reconfiguration 18x18x20 to 20x18x18
                    np.savez(save_path, rdm2)  # save features
                    visualize_rdm(rdm2, npz_save_dir, filename)


"""This function only has to be executed once, it is not implemented into the main function"""

def multiple_regression_graph(option):
    task = ["taskBoth", "taskNum", "taskSize"][option]
    sep = helper.check_platform()
    brain_regions = ["IPS15", "V3ABV7", "V13", "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1"]
    result_dict = {}
    for brain_region in brain_regions:
        path = "RSA_matrices" + sep + "RDM_allsub_" + task + "_" + brain_region + ".npz"
        brain_rdm = helper.loadnpz(path)
        brain_rdm = brain_rdm.f.arr_0
        brain_rdm = np.mean(brain_rdm,axis = 0)
        result = RDM_Evaluation.multiple_regression(brain_rdm)
        result_dict[brain_region] = result
        RDM_Evaluation.visualize_multiple_regression(result_dict,"Alexnet pretrained results",str(option),[])


#read_in_rdms_matlab()
multiple_regression_graph(2)