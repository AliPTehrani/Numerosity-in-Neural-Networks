import scipy.io
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import RDM_Evaluations.MultipleRegression as MulitpleRegression
import helper


'''
This file is used to convert and save the matlab files from the directory "RSA_Matrices" as .npz files
The files in "RSA_Matrices" are the given fmri RDMs given in matlab format.
All this functios only had to be executed once. Because of that they are not implemented in the User Interface (main).
The results of the fmri RDMs can be found in the directory "Multiple Regression on fmRI"
The RDMs from the fmRI Data are stored in the directory "RSA_matrices"
'''


def visualize_rdm(rdm, npz_save_dir, filename):
    """
    Visualize an given RDM and save it into given directory and filename
    :param rdm: Given RDM
    :param npz_save_dir: Directory to save the files
    :param filename: Name of the file
    """
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


def multiple_regression_graph_averages(option):
    """
    Multiple regression on the average fmri RDMs as an safety check for the implemented multiple regression function
    :param option: Option of integer 0-2 for the different tasks taskBoth,taskNum,taskSize
    :return:
    """
    task = ["taskBoth", "taskNum", "taskSize"][option]
    sep = helper.check_platform()
    brain_regions = ["IPS15", "V3ABV7", "V13", "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1"]
    brain_regions.reverse()
    result_dict = {}
    for brain_region in brain_regions:
        path = "RSA_matrices" + sep + "RDM_allsub_" + task + "_" + brain_region + ".npz"
        brain_rdm = helper.loadnpz(path)
        brain_rdm = brain_rdm.f.arr_0
        brain_rdm = np.mean(brain_rdm, axis=0)
        result = MulitpleRegression.multiple_regression(brain_rdm)
        result_dict[brain_region] = result
        MulitpleRegression.visualize_multiple_regression(result_dict,"Alexnet pretrained results",str(option),[])

def multiple_regression_graph_subjects(option):
    """
    Multiple regression for every subject fmri RDMs as an safety check for the implemented multiple regression function
    The multiple regression results will be averaged after multiple regression was performed for every subject.
    :param option: Option of integer 0-2 for the different tasks taskBoth,taskNum,taskSize
    :return:
    """
    task = ["taskBoth", "taskNum", "taskSize"][option]
    sep = helper.check_platform()
    brain_regions = [ "IPS345", "IPS12", "IPS0", "V3AB", "V3", "V2", "V1","IPS15", "V3ABV7", "V13"]
    brain_regions.reverse()
    result_dict = {}
    for brain_region in brain_regions:
        path = "RSA_matrices" + sep + "RDM_allsub_" + task + "_" + brain_region + ".npz"
        brain_rdms = helper.loadnpz(path)
        brain_rdms = brain_rdms.f.arr_0
        for brain_rdm in brain_rdms:
            result = MulitpleRegression.multiple_regression(brain_rdm)
            if brain_region not in result_dict:
                result_dict[brain_region] = result
            else:
                result_dict[brain_region] += result
        result_dict[brain_region] = result_dict[brain_region] / 20
        MulitpleRegression.visualize_multiple_regression(result_dict,"Alexnet pretrained results",str(option),[])



#read_in_rdms_matlab()
#multiple_regression_graph_subjects(0)
#multiple_regression_graph_subjects(1)
#multiple_regression_graph_subjects(2)
#multiple_regression_graph(0)
#multiple_regression_graph(1)
#multiple_regression_graph(2)


